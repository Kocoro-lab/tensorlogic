#!/usr/bin/env python3
"""
Shakespeare character-level language model training script using TensorLogic Transformers.

Features:
- MPS/CUDA/CPU support with automatic device detection
- Learning rate warmup and cosine decay
- Gradient clipping and accumulation
- Checkpoint saving with best model tracking
- Temperature-controlled generation
- Memory-efficient training for M-series Macs

Based on nanoGPT's proven hyperparameters, adapted for TensorLogic.
"""

import argparse
import os
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict
import torch
import torch.nn.functional as F
from tensorlogic.transformers import DecoderOnlyLM


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Combined model and training configuration."""
    # Model
    vocab_size: int = None  # Set from data
    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    dim_feedforward: int = None  # Defaults to 4 * d_model
    block_size: int = 128
    dropout: float = 0.2
    pos_encoding: str = 'learned'
    tie_weights: bool = True

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    max_iters: int = 5000
    lr_decay_iters: int = 5000
    grad_clip: float = 1.0

    # Evaluation
    eval_interval: int = 250
    eval_iters: int = 200

    # Generation
    temperature: float = 0.8
    top_k: int = 200

    # System
    device: str = 'auto'  # 'auto', 'cuda', 'mps', or 'cpu'
    compile: bool = False  # PyTorch 2.0 compile
    dtype: str = 'float32'
    seed: int = 1337

    # I/O
    data_path: str = 'input.txt'
    checkpoint_dir: str = 'checkpoints/shakespeare'
    save_interval: int = 1000

    def __post_init__(self):
        if self.dim_feedforward is None:
            self.dim_feedforward = 4 * self.d_model
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'


# ============================================================================
# Data Loading
# ============================================================================

class CharDataset:
    """Character-level dataset for language modeling."""

    def __init__(self, text_path, block_size):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Build vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Encode text
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.block_size = block_size

        # Train/val split (90/10)
        n = len(self.data)
        n_train = int(0.9 * n)
        self.train_data = self.data[:n_train]
        self.val_data = self.data[n_train:]

        print(f"Dataset loaded: {self.vocab_size} unique chars, {n:,} total chars")
        print(f"Train: {len(self.train_data):,} | Val: {len(self.val_data):,}")
        print(f"Block size: {block_size}")

    def get_batch(self, split='train', batch_size=32, device='cpu'):
        """Get a batch of data."""
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x.to(device), y.to(device)

    def encode(self, text):
        """Encode text to token indices."""
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, indices):
        """Decode token indices to text."""
        return ''.join([self.itos.get(i, '') for i in indices])


# ============================================================================
# Training Utilities
# ============================================================================

def get_lr(iter_num, config):
    """Learning rate schedule with warmup and cosine decay."""
    # Linear warmup
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters
    # Past decay period
    if iter_num > config.lr_decay_iters:
        return config.min_lr
    # Cosine decay
    decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model, dataset, config):
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = dataset.get_batch(split, config.batch_size, config.device)
            logits = model(X)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = Y.view(B*T)
            loss = F.cross_entropy(logits, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def generate_text(model, dataset, config, prompt="", max_new_tokens=100):
    """Generate text from the model."""
    model.eval()

    # Encode prompt or start with newline
    if prompt:
        indices = dataset.encode(prompt)
        context = torch.tensor([indices], dtype=torch.long, device=config.device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

    # Ensure we don't exceed block_size
    max_new_tokens = min(max_new_tokens, config.block_size - context.shape[1])

    # Generate
    generated = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k
    )

    model.train()
    return dataset.decode(generated[0].tolist())


# ============================================================================
# Main Training Loop
# ============================================================================

def train(config):
    """Main training function."""

    print("=" * 70)
    print("Shakespeare Character-Level Language Model Training")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Config: {config.n_layer} layers, {config.n_head} heads, {config.d_model} dim")
    print(f"Training: {config.max_iters} iters, batch size {config.batch_size}")
    print("=" * 70)

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Load dataset
    dataset = CharDataset(config.data_path, config.block_size)
    config.vocab_size = dataset.vocab_size

    # Initialize model
    model = DecoderOnlyLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layer=config.n_layer,
        n_head=config.n_head,
        dim_feedforward=config.dim_feedforward,
        max_seq_len=config.block_size,
        dropout=config.dropout,
        pos_encoding=config.pos_encoding,
        tie_weights=config.tie_weights,
    ).to(config.device)

    # Compile model (PyTorch 2.0+)
    if config.compile and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Training state
    best_val_loss = float('inf')
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    t0 = time.time()

    for iter_num in range(config.max_iters):
        # Set learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, dataset, config)
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num:4d} | loss train {losses['train']:.4f} val {losses['val']:.4f} | lr {lr:.2e} | {dt:.1f}s")

            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': asdict(config),
                        'iter': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    torch.save(checkpoint, checkpoint_dir / 'best.pt')
                    print(f"  → Saved best model (val loss: {best_val_loss:.4f})")

            # Generate sample
            if iter_num > 0 and iter_num % config.save_interval == 0:
                print("\n--- Sample Generation ---")
                text = generate_text(model, dataset, config, max_new_tokens=200)
                print(text)
                print("--- End Sample ---\n")

        # Training step
        optimizer.zero_grad(set_to_none=True)

        # Gradient accumulation
        for micro_step in range(config.gradient_accumulation_steps):
            X, Y = dataset.get_batch('train', config.batch_size, config.device)
            logits = model(X)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = Y.view(B*T)
            loss = F.cross_entropy(logits, targets)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Optimizer step
        optimizer.step()

    # Final evaluation
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)

    losses = estimate_loss(model, dataset, config)
    print(f"\nFinal loss: train {losses['train']:.4f} val {losses['val']:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")

    # Save final model
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': asdict(config),
        'iter': config.max_iters,
        'final_train_loss': losses['train'],
        'final_val_loss': losses['val'],
    }
    torch.save(checkpoint, checkpoint_dir / 'final.pt')
    print(f"\nFinal model saved to {checkpoint_dir / 'final.pt'}")

    # Generate final samples
    print("\n" + "=" * 70)
    print("Final text generation samples:")
    print("=" * 70)

    for temp in [0.5, 0.8, 1.0]:
        config.temperature = temp
        print(f"\n--- Temperature {temp} ---")
        text = generate_text(model, dataset, config, max_new_tokens=300)
        print(text)

    print("\n✅ Training complete!")

    # Check if loss is reasonable
    if losses['val'] < 0.5:
        print("\n⚠️ WARNING: Loss is suspiciously low! Check for mask or data issues.")
    elif losses['val'] > 3.0:
        print("\n⚠️ WARNING: Loss is high. Model might need more training.")
    else:
        print(f"\n✓ Loss looks reasonable for character-level modeling: {losses['val']:.4f}")


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train a Shakespeare character-level language model')

    # Model arguments
    parser.add_argument('--d_model', type=int, default=384, help='Model dimension')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--block_size', type=int, default=128, help='Context length')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_iters', type=int, default=5000, help='Max training iterations')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate')
    parser.add_argument('--warmup_iters', type=int, default=100, help='Warmup iterations')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--eval_interval', type=int, default=250, help='Evaluation interval')

    # System arguments
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/mps/cpu)')
    parser.add_argument('--compile', action='store_true', help='Use PyTorch 2.0 compile')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')

    # I/O arguments
    parser.add_argument('--data_path', type=str, default='input.txt', help='Path to training data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/shakespeare',
                       help='Directory for checkpoints')

    args = parser.parse_args()

    # Create config from arguments
    config = Config(**vars(args))

    # Train
    train(config)


if __name__ == '__main__':
    main()