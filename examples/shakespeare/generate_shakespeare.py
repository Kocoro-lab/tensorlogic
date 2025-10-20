#!/usr/bin/env python3
"""
Generate text from a trained TensorLogic Shakespeare model.

Usage:
    # Train a model first
    python train_shakespeare.py --max_iters 5000

    # Then generate text from the best checkpoint
    python generate_shakespeare.py checkpoints/shakespeare/best.pt

    # Or generate with custom parameters
    python generate_shakespeare.py checkpoints/shakespeare/final.pt --temperature 0.8 --length 500

Supports various generation strategies and temperature control.
"""

import argparse
import torch
from pathlib import Path

from tensorlogic.transformers import DecoderOnlyLM


def load_model(checkpoint_path, device='cpu'):
    """Load model from checkpoint."""
    # PyTorch 2.6+ requires weights_only=False for complex objects
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both old format (model_config) and new format (config)
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        raise ValueError("Checkpoint missing configuration")

    # Reconstruct model
    model = DecoderOnlyLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        dim_feedforward=config.get('dim_feedforward', 4 * config['d_model']),
        max_seq_len=config.get('block_size', config.get('max_seq_len', 256)),
        dropout=0.0,  # No dropout for inference
        pos_encoding=config.get('pos_encoding', 'learned'),
        norm_first=config.get('norm_first', True),
        mode='continuous',
        tie_weights=config.get('tie_weights', True),
    )

    # Load weights (handle both formats)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        raise ValueError("Checkpoint missing model weights")
    model.to(device)
    model.eval()

    return model, config


def build_char_maps(text_path='input.txt'):
    """Build character vocabulary from training data."""
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    return stoi, itos


@torch.no_grad()
def generate(model, prompt, stoi, itos, device,
            max_new_tokens=500,
            temperature=0.8,
            top_k=40,
            top_p=0.9):
    """Generate text from prompt."""

    # Encode prompt
    if prompt:
        prompt_ids = [stoi.get(c, 0) for c in prompt]
    else:
        # Start with newline if no prompt
        prompt_ids = [stoi.get('\n', 0)]

    # Convert to tensor
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Generate
    print(f"Generating with temperature={temperature}, top_k={top_k}...")
    print("="*60)

    # Generate tokens
    output = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        # Note: top_p not implemented in current version
    )

    # Decode
    generated_ids = output[0].tolist()
    generated_text = ''.join([itos[i] for i in generated_ids])

    return generated_text


def interactive_mode(model, stoi, itos, device):
    """Interactive generation mode."""
    print("Interactive mode - Type 'quit' to exit")
    print("Commands: temp=0.8, topk=40, len=200")
    print("="*60)

    temperature = 0.8
    top_k = 40
    max_len = 200

    while True:
        prompt = input("\nPrompt> ").strip()

        if prompt.lower() == 'quit':
            break

        # Check for commands
        if prompt.startswith('temp='):
            try:
                temperature = float(prompt.split('=')[1])
                print(f"Temperature set to {temperature}")
                continue
            except:
                pass

        if prompt.startswith('topk='):
            try:
                top_k = int(prompt.split('=')[1])
                print(f"Top-k set to {top_k}")
                continue
            except:
                pass

        if prompt.startswith('len='):
            try:
                max_len = int(prompt.split('=')[1])
                print(f"Max length set to {max_len}")
                continue
            except:
                pass

        # Generate
        text = generate(model, prompt, stoi, itos, device,
                       max_new_tokens=max_len,
                       temperature=temperature,
                       top_k=top_k)
        print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='input.txt',
                       help='Path to training data (for vocabulary)')
    parser.add_argument('--prompt', type=str, default='',
                       help='Starting prompt')
    parser.add_argument('--max_tokens', type=int, default=500,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k sampling')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda/mps)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive generation mode')
    parser.add_argument('--show_equations', action='store_true',
                       help='Show model as tensor equations')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')

    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    # Build vocabulary
    stoi, itos = build_char_maps(args.data)
    print(f"Vocabulary: {len(stoi)} characters")

    # Show equations if requested
    if args.show_equations:
        print("\n" + "="*60)
        print("Model as Tensor Equations:")
        print("="*60)
        equations = model.to_tensor_equations()
        for eq in equations:
            print(eq)
        print("="*60 + "\n")

    # Interactive or single generation
    if args.interactive:
        interactive_mode(model, stoi, itos, device)
    else:
        text = generate(
            model, args.prompt, stoi, itos, device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print(text)


if __name__ == '__main__':
    main()