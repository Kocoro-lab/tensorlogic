#!/usr/bin/env python3
"""
FB15k-237 Knowledge Graph Link Prediction Benchmark using TensorLogic RESCAL.

Trains a RESCAL bilinear model on FB15k-237 and evaluates with standard
filtered metrics: MRR, Hits@1, Hits@3, Hits@10.

Features:
- Auto-download of FB15k-237 dataset
- 1vsAll training: scores all entities per batch via efficient matmul
- Cross-entropy loss (head + tail prediction)
- Adagrad optimizer (proven best for RESCAL embeddings)
- Dropout on entity and relation embeddings
- Efficient all-entity scoring via einsum + matmul
- Filtered evaluation with proper exclusion of known triples
- Checkpoint saving with best model tracking and resume support
- CUDA/MPS/CPU support with automatic device detection

Usage:
    python examples/fb15k237_benchmark.py                    # Full training
    python examples/fb15k237_benchmark.py --epochs 2 --eval_interval 1  # Quick test
    python examples/fb15k237_benchmark.py --resume           # Resume training
    python examples/fb15k237_benchmark.py --eval_only        # Evaluate checkpoint
"""

import argparse
import os
import time
import math
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Set, Tuple
from urllib.request import urlretrieve

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Training and model configuration."""
    # Model
    rank: int = 200
    dropout: float = 0.3
    # Training
    epochs: int = 1000
    batch_size: int = 512
    lr: float = 0.001
    min_lr: float = 1e-5
    weight_decay: float = 0.0
    label_smoothing: float = 0.1
    warmup_epochs: int = 10
    # Evaluation
    eval_interval: int = 25
    eval_batch_size: int = 64
    # Early stopping
    patience: int = 30
    # System
    device: str = 'auto'
    seed: int = 42
    # I/O
    data_dir: str = 'data/fb15k237'
    checkpoint_dir: str = 'checkpoints/fb15k237'
    resume: bool = False
    eval_only: bool = False

    def __post_init__(self):
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'


# ============================================================================
# Model
# ============================================================================

class RESCALBenchmark(nn.Module):
    """RESCAL with dropout for benchmarking.

    score(i, r, j) = e_i^T W_r e_j

    Uses same init as RESCALModel (randn*0.01, normalized E) which is proven
    to work well. Adds dropout on embeddings during training.
    """

    def __init__(self, num_entities: int, num_relations: int, rank: int, dropout: float = 0.3):
        super().__init__()
        self.num_objects = num_entities
        self.num_relations = num_relations
        self.rank = rank

        # Same init as RESCALModel: randn*0.01, normalize E
        E = torch.randn(num_entities, rank) * 0.01
        E = F.normalize(E, p=2, dim=1)
        self.E = nn.Parameter(E)
        self.W = nn.Parameter(torch.randn(num_relations, rank, rank) * 0.01)

        self.dropout = nn.Dropout(dropout)

    def score_triples(self, heads: torch.Tensor, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """Compute scores for batches of triples. Returns [B] logits."""
        e_h = self.E[heads]
        e_t = self.E[tails]
        W_r = self.W[rels]
        mid = torch.einsum("bd,bdk->bk", e_h, W_r)
        return torch.einsum("bk,bk->b", mid, e_t)


# ============================================================================
# Dataset
# ============================================================================

BASE_URL = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/"
SPLITS = ["train.txt", "valid.txt", "test.txt"]


def download_fb15k237(data_dir: str) -> None:
    """Download FB15k-237 dataset if not present."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        dest = data_path / split
        if dest.exists():
            continue
        url = BASE_URL + split
        print(f"Downloading {split}...")
        urlretrieve(url, dest)
        print(f"  Saved to {dest}")


def load_triples(path: str) -> list:
    """Load TSV triples file. Format: head<TAB>relation<TAB>tail"""
    triples = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


class FB15k237Dataset:
    """FB15k-237 dataset with entity/relation indexing."""

    def __init__(self, data_dir: str):
        download_fb15k237(data_dir)
        data_path = Path(data_dir)

        raw_train = load_triples(str(data_path / "train.txt"))
        raw_valid = load_triples(str(data_path / "valid.txt"))
        raw_test = load_triples(str(data_path / "test.txt"))

        entities = set()
        relations = set()
        for triples in [raw_train, raw_valid, raw_test]:
            for h, r, t in triples:
                entities.add(h)
                entities.add(t)
                relations.add(r)

        self.entity2id = {e: i for i, e in enumerate(sorted(entities))}
        self.relation2id = {r: i for i, r in enumerate(sorted(relations))}
        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.relation2id)

        self.train = self._to_tensor(raw_train)
        self.valid = self._to_tensor(raw_valid)
        self.test = self._to_tensor(raw_test)

        all_raw = raw_train + raw_valid + raw_test
        self.all_true_triples = set()
        for h, r, t in all_raw:
            self.all_true_triples.add(
                (self.entity2id[h], self.relation2id[r], self.entity2id[t])
            )
        self.hr_to_t, self.rt_to_h = build_filter_dicts(self.all_true_triples)

        print(f"FB15k-237 loaded:")
        print(f"  Entities:  {self.num_entities:,}")
        print(f"  Relations: {self.num_relations}")
        print(f"  Train:     {self.train.shape[0]:,}")
        print(f"  Valid:     {self.valid.shape[0]:,}")
        print(f"  Test:      {self.test.shape[0]:,}")

    def _to_tensor(self, raw_triples: list) -> torch.LongTensor:
        rows = []
        for h, r, t in raw_triples:
            rows.append([self.entity2id[h], self.relation2id[r], self.entity2id[t]])
        return torch.tensor(rows, dtype=torch.long)


# ============================================================================
# Filtered Ranking
# ============================================================================

def build_filter_dicts(
    all_triples: Set[Tuple[int, int, int]]
) -> Tuple[Dict, Dict]:
    """Build (head, rel) -> set(tails) and (rel, tail) -> set(heads) dicts."""
    hr_to_t = defaultdict(set)
    rt_to_h = defaultdict(set)
    for h, r, t in all_triples:
        hr_to_t[(h, r)].add(t)
        rt_to_h[(r, t)].add(h)
    return dict(hr_to_t), dict(rt_to_h)


def score_all_tails(model, heads: torch.Tensor, rels: torch.Tensor) -> torch.Tensor:
    """Score all entities as tails: [B, N]."""
    e_h = model.E[heads]
    W_r = model.W[rels]
    mid = torch.einsum("bd,bdk->bk", e_h, W_r)
    return mid @ model.E.data.T


def score_all_heads(model, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
    """Score all entities as heads: [B, N]."""
    e_t = model.E[tails]
    W_r = model.W[rels]
    mid = torch.einsum("bdk,bk->bd", W_r, e_t)
    return mid @ model.E.data.T


@torch.no_grad()
def run_filtered_ranking(
    model,
    triples: torch.Tensor,
    hr_to_t: Dict,
    rt_to_h: Dict,
    num_entities: int,
    device: str,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Filtered ranking: MRR, Hits@1, Hits@3, Hits@10."""
    model.eval()

    ranks = []
    n = triples.shape[0]

    for start in tqdm(range(0, n, batch_size), desc="Ranking", leave=False):
        batch = triples[start:start + batch_size].to(device)
        heads = batch[:, 0]
        rels = batch[:, 1]
        tails = batch[:, 2]
        B = batch.shape[0]

        # --- Tail prediction ---
        tail_scores = score_all_tails(model, heads, rels)
        for i in range(B):
            h, r, t = heads[i].item(), rels[i].item(), tails[i].item()
            target_score = tail_scores[i, t].clone()
            known_tails = hr_to_t.get((h, r), set())
            filter_ids = [x for x in known_tails if x != t]
            if filter_ids:
                tail_scores[i, filter_ids] = float('-inf')
            rank = (tail_scores[i] > target_score).sum().item() + 1
            ranks.append(rank)

        # --- Head prediction ---
        head_scores = score_all_heads(model, rels, tails)
        for i in range(B):
            h, r, t = heads[i].item(), rels[i].item(), tails[i].item()
            target_score = head_scores[i, h].clone()
            known_heads = rt_to_h.get((r, t), set())
            filter_ids = [x for x in known_heads if x != h]
            if filter_ids:
                head_scores[i, filter_ids] = float('-inf')
            rank = (head_scores[i] > target_score).sum().item() + 1
            ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float)
    mrr = (1.0 / ranks).mean().item()
    hits1 = (ranks <= 1).float().mean().item()
    hits3 = (ranks <= 3).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()

    return {"MRR": mrr, "Hits@1": hits1, "Hits@3": hits3, "Hits@10": hits10}


# ============================================================================
# Training
# ============================================================================

def get_lr(epoch: int, config: Config) -> float:
    """LR schedule: linear warmup then cosine decay to min_lr."""
    if epoch < config.warmup_epochs:
        return config.lr * (epoch + 1) / config.warmup_epochs
    progress = (epoch - config.warmup_epochs) / max(config.epochs - config.warmup_epochs, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.lr - config.min_lr) * coeff


class FB15k237Trainer:
    """1vsAll RESCAL trainer with Adam + LR warmup/cosine decay.

    For each triple (h, r, t):
      - Tail: scores = dropout(E[h]) @ W[r] @ E.T → [B, N], target = t
      - Head: scores = (W[r] @ dropout(E[t]))^T @ E.T → [B, N], target = h
    Uses cross-entropy loss with optional label smoothing.
    """

    def __init__(self, model, config: Config):
        self.model = model.to(config.device)
        self.device = config.device
        self.config = config
        self.optim = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    def _score_all_tails_train(self, heads, rels):
        """Score with dropout applied to query (head) embeddings."""
        e_h = self.model.dropout(self.model.E[heads])
        W_r = self.model.W[rels]
        mid = torch.einsum("bd,bdk->bk", e_h, W_r)
        return mid @ self.model.E.T  # gradient flows through E

    def _score_all_heads_train(self, rels, tails):
        """Score with dropout applied to query (tail) embeddings."""
        e_t = self.model.dropout(self.model.E[tails])
        W_r = self.model.W[rels]
        mid = torch.einsum("bdk,bk->bd", W_r, e_t)
        return mid @ self.model.E.T  # gradient flows through E

    def train_epoch(self, pos_triples: torch.Tensor, num_entities: int) -> float:
        """One training epoch with 1vsAll scoring."""
        cfg = self.config
        device = self.device
        self.model.train()
        smooth = cfg.label_smoothing

        perm = torch.randperm(pos_triples.shape[0])
        pos_triples = pos_triples[perm].to(device)

        total_loss = 0.0
        n_batches = 0

        for i in range(0, pos_triples.shape[0], cfg.batch_size):
            batch = pos_triples[i:i + cfg.batch_size]
            if batch.shape[0] == 0:
                continue

            heads = batch[:, 0]
            rels = batch[:, 1]
            tails = batch[:, 2]
            B = batch.shape[0]

            self.optim.zero_grad()

            # Tail prediction
            tail_logits = self._score_all_tails_train(heads, rels)
            if smooth > 0:
                tail_targets = torch.full((B, num_entities), smooth / (num_entities - 1), device=device)
                tail_targets.scatter_(1, tails.unsqueeze(1), 1.0 - smooth)
                tail_loss = -(tail_targets * F.log_softmax(tail_logits, dim=1)).sum(dim=1).mean()
            else:
                tail_loss = F.cross_entropy(tail_logits, tails)

            # Head prediction
            head_logits = self._score_all_heads_train(rels, tails)
            if smooth > 0:
                head_targets = torch.full((B, num_entities), smooth / (num_entities - 1), device=device)
                head_targets.scatter_(1, heads.unsqueeze(1), 1.0 - smooth)
                head_loss = -(head_targets * F.log_softmax(head_logits, dim=1)).sum(dim=1).mean()
            else:
                head_loss = F.cross_entropy(head_logits, heads)

            loss = (tail_loss + head_loss) / 2.0
            loss.backward()
            self.optim.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)


# ============================================================================
# Main
# ============================================================================

def train(config: Config):
    """Main training and ranking loop."""
    print("=" * 70)
    print("FB15k-237 Knowledge Graph Completion Benchmark")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Config: rank={config.rank}, epochs={config.epochs}, "
          f"batch_size={config.batch_size}, lr={config.lr}")
    print(f"Training: 1vsAll + Adam (warmup={config.warmup_epochs}, cosine decay)")
    print(f"  dropout={config.dropout}, label_smoothing={config.label_smoothing}")
    print("=" * 70)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    dataset = FB15k237Dataset(config.data_dir)
    print(f"\nEntities: {dataset.num_entities:,} | Relations: {dataset.num_relations} | Rank: {config.rank}")

    model = RESCALBenchmark(dataset.num_entities, dataset.num_relations, config.rank, config.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    latest_path = ckpt_dir / "latest.pt"

    trainer = FB15k237Trainer(model, config)

    start_epoch = 0
    best_mrr = 0.0
    patience_counter = 0

    if config.eval_only:
        if not best_path.exists():
            print(f"No checkpoint found at {best_path}")
            return
        ckpt = torch.load(best_path, map_location=config.device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.to(config.device)
        print(f"\nLoaded best checkpoint (epoch {ckpt.get('epoch', '?')})")
        print("\nRunning test ranking...")
        results = run_filtered_ranking(
            model, dataset.test, dataset.hr_to_t, dataset.rt_to_h,
            dataset.num_entities, config.device, config.eval_batch_size,
        )
        print_results("Test", results)
        print_reference()
        return

    if config.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=config.device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        trainer.optim.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mrr = ckpt.get("best_mrr", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        model.to(config.device)
        print(f"\nResumed from epoch {start_epoch} (best MRR: {best_mrr:.4f})")

    print(f"\nStarting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, config.epochs):
        # Update learning rate
        lr = get_lr(epoch, config)
        for pg in trainer.optim.param_groups:
            pg['lr'] = lr

        t0 = time.time()
        avg_loss = trainer.train_epoch(dataset.train, dataset.num_entities)
        dt = time.time() - t0

        print(f"Epoch {epoch + 1:3d}/{config.epochs} | loss={avg_loss:.4f} | lr={lr:.2e} | {dt:.1f}s")

        if (epoch + 1) % config.eval_interval == 0 or epoch == config.epochs - 1:
            results = run_filtered_ranking(
                model, dataset.valid, dataset.hr_to_t, dataset.rt_to_h,
                dataset.num_entities, config.device, config.eval_batch_size,
            )
            print_results("Valid", results)

            if results["MRR"] > best_mrr:
                best_mrr = results["MRR"]
                patience_counter = 0
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_mrr": best_mrr,
                    "config": asdict(config),
                }, best_path)
                print(f"  -> Saved best model (MRR: {best_mrr:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{config.patience})")

            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        torch.save({
            "model": model.state_dict(),
            "optimizer": trainer.optim.state_dict(),
            "epoch": epoch,
            "best_mrr": best_mrr,
            "patience_counter": patience_counter,
            "config": asdict(config),
        }, latest_path)

    print("\n" + "=" * 70)
    print("Training complete. Loading best checkpoint for test ranking...")
    print("=" * 70)

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=config.device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.to(config.device)
        print(f"Loaded best model from epoch {ckpt.get('epoch', '?')} (MRR: {ckpt.get('best_mrr', '?')})")
    else:
        print("No best checkpoint saved; using final model weights.")

    results = run_filtered_ranking(
        model, dataset.test, dataset.hr_to_t, dataset.rt_to_h,
        dataset.num_entities, config.device, config.eval_batch_size,
    )
    print_results("Test", results)
    print_reference()


def print_results(split: str, results: Dict[str, float]):
    print(f"\n  {split} Results (filtered):")
    print(f"    MRR:     {results['MRR']:.4f}")
    print(f"    Hits@1:  {results['Hits@1']:.4f}")
    print(f"    Hits@3:  {results['Hits@3']:.4f}")
    print(f"    Hits@10: {results['Hits@10']:.4f}")


def print_reference():
    print(f"\n  Reference RESCAL (LibKGE): MRR=0.304, H@1=0.242, H@3=0.331, H@10=0.419")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FB15k-237 KG link prediction benchmark with TensorLogic RESCAL"
    )
    parser.add_argument('--rank', type=int, default=200, help='Embedding rank')
    parser.add_argument('--dropout', type=float, default=0.3, help='Embedding dropout')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Peak learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='LR warmup epochs')
    parser.add_argument('--eval_interval', type=int, default=25, help='Epochs between validation')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Ranking batch size')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/mps/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='data/fb15k237', help='Dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/fb15k237',
                        help='Checkpoint directory')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--eval_only', action='store_true', help='Only run test ranking')

    args = parser.parse_args()
    config = Config(**vars(args))
    train(config)


if __name__ == '__main__':
    main()
