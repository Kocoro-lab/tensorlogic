"""
Decomposition-based predicate invention (RESCAL-style factorization).

Provides a simple RESCAL model and trainer over sparse triples, utilities to
extract candidate invented predicates (NxN Boolean adjacencies), and helpers to
inject them into the language layer or analyze them downstream.
"""

from typing import Iterable, List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RESCALModel(nn.Module):
    """
    RESCAL-style bilinear factorization:
        score(i, r, j) = e_i^T W_r e_j

    Parameters:
        E: [N, d]
        W: [R, d, d]
    """

    def __init__(self, num_objects: int, num_relations: int, rank: int):
        super().__init__()
        self.num_objects = num_objects
        self.num_relations = num_relations
        self.rank = rank

        E = torch.randn(num_objects, rank) * 0.01
        E = F.normalize(E, p=2, dim=1)
        self.E = nn.Parameter(E)
        self.W = nn.Parameter(torch.randn(num_relations, rank, rank) * 0.01)

    def score_triples(self, heads: torch.Tensor, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """
        Compute scores for batches of triples.
        Inputs are 1D Long tensors of same length B.
        Returns [B] scores (logits).
        """
        e_h = self.E[heads]  # [B, d]
        e_t = self.E[tails]  # [B, d]
        W_r = self.W[rels]   # [B, d, d]
        # e_h^T W_r e_t -> [B]
        mid = torch.einsum("bd, bdk -> bk", e_h, W_r)
        scores = torch.einsum("bk, bk -> b", mid, e_t)
        return scores


class RESCALTrainer:
    """SGD-style training for RESCAL with negative sampling and BCE loss."""

    def __init__(
        self,
        model: RESCALModel,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        self.bce = nn.BCEWithLogitsLoss()
        self.use_amp = use_amp
        self._amp_device_type = torch.device(device).type
        self._scaler = torch.amp.GradScaler(device=device, enabled=use_amp)

    def _sample_negatives(self, pos: torch.Tensor, num_objects: int, k: int) -> torch.Tensor:
        """Negative sampling by corrupting tails (simple and fast)."""
        B = pos.shape[0]
        heads = pos[:, 0].unsqueeze(1).expand(B, k)
        rels = pos[:, 1].unsqueeze(1).expand(B, k)
        tails = torch.randint(low=0, high=num_objects, size=(B, k), device=pos.device)
        triples = torch.stack([heads, rels, tails], dim=-1)  # [B, K, 3]
        return triples.reshape(-1, 3)

    def train_epoch(
        self,
        pos_triples: torch.Tensor,  # [M, 3] (head, rel, tail)
        num_objects: int,
        negative_k: int = 5,
        batch_size: int = 1024,
        verbose: bool = True,
    ) -> float:
        device = self.device
        self.model.train()

        # Shuffle
        perm = torch.randperm(pos_triples.shape[0])
        pos_triples = pos_triples[perm].to(device)

        total = 0.0
        n = 0

        for i in range(0, pos_triples.shape[0], batch_size):
            batch = pos_triples[i:i+batch_size]
            if batch.shape[0] == 0:
                continue
            heads = batch[:, 0]
            rels = batch[:, 1]
            tails = batch[:, 2]

            neg = self._sample_negatives(batch, num_objects, k=negative_k)
            nh = neg[:, 0]
            nr = neg[:, 1]
            nt = neg[:, 2]

            self.optim.zero_grad()
            if self.use_amp:
                with torch.amp.autocast(device_type=self._amp_device_type):
                    pos_logits = self.model.score_triples(heads, rels, tails)
                    pos_targets = torch.ones_like(pos_logits)
                    neg_logits = self.model.score_triples(nh, nr, nt)
                    neg_targets = torch.zeros_like(neg_logits)
                    loss = self.bce(pos_logits, pos_targets) + self.bce(neg_logits, neg_targets)
                self._scaler.scale(loss).backward()
                if self.max_grad_norm is not None:
                    self._scaler.unscale_(self.optim)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._scaler.step(self.optim)
                self._scaler.update()
            else:
                pos_logits = self.model.score_triples(heads, rels, tails)
                pos_targets = torch.ones_like(pos_logits)
                neg_logits = self.model.score_triples(nh, nr, nt)
                neg_targets = torch.zeros_like(neg_logits)
                loss = self.bce(pos_logits, pos_targets) + self.bce(neg_logits, neg_targets)
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optim.step()

            total += loss.item()
            n += 1

            if verbose and (i // batch_size) % 50 == 0:
                print(f"  Batch {i//batch_size:4d}: loss={loss.item():.4f}")

        return total / max(n, 1)


def extract_predicates_from_embeddings(E: torch.Tensor, top_k: Optional[int] = None, percentile: float = 95.0) -> List[torch.Tensor]:
    """
    From E [N, d], produce candidate NxN Boolean adjacencies via outer products
    of each latent dimension and percentile thresholding.
    """
    N, d = E.shape
    dims = list(range(d)) if top_k is None else list(range(min(top_k, d)))
    cands: List[torch.Tensor] = []
    for k in dims:
        scores = torch.ger(E[:, k], E[:, k])  # [N, N]
        thr = torch.quantile(scores.flatten(), percentile / 100.0)
        adj = (scores >= thr).float()
        cands.append(adj)
    return cands


def latent_outer_scores(E: torch.Tensor, top_k: Optional[int] = None) -> List[torch.Tensor]:
    """
    Return continuous score matrices S_k = e[:,k] e[:,k]^T for each latent k
    (or top_k dims). Useful for AUC/Hits@K ranking before thresholding.
    """
    N, d = E.shape
    dims = list(range(d)) if top_k is None else list(range(min(top_k, d)))
    scores: List[torch.Tensor] = []
    for k in dims:
        scores.append(torch.ger(E[:, k], E[:, k]))
    return scores


def triples_from_adjacency(adj: torch.Tensor, rel_id: int) -> torch.Tensor:
    """Convert NxN Boolean adjacency to triples [M,3] with fixed relation id."""
    idx = (adj > 0).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return torch.empty((0, 3), dtype=torch.long)
    rel = torch.full((idx.shape[0], 1), rel_id, dtype=torch.long, device=adj.device)
    return torch.cat([idx.long(), rel], dim=1)[:, [0, 2, 1]]  # (i, rel, j)


def rank_predicates_by_support(candidates: List[torch.Tensor]) -> List[Tuple[int, int]]:
    """Return list of (index, support) sorted by support descending."""
    scores = []
    for i, adj in enumerate(candidates):
        scores.append((i, int((adj > 0).sum().item())))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
