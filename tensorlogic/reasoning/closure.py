"""
Utilities to derive self-supervised training signals from base facts by
computing k-hop closures (Boolean compositions) over relations.
"""

from typing import Dict, Iterable, List, Tuple

import torch


def boolean_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Boolean matrix multiply (2D): (A @ B) > 0 -> 1.0 else 0.0"""
    C = A @ B
    return (C > 0).to(A.dtype)


def boolean_power(A: torch.Tensor, k: int) -> torch.Tensor:
    """Compute Boolean power A^k for 2D adjacency A."""
    assert k >= 1
    if k == 1:
        return (A > 0).to(A.dtype)
    out = (A > 0).to(A.dtype)
    for _ in range(k - 1):
        out = boolean_matmul(out, A)
    return out


def compose_sequence(relations: Dict[str, torch.Tensor], sequence: Iterable[str]) -> torch.Tensor:
    """
    Compose a named sequence of 2D Boolean adjacency matrices by Boolean matmul.
    Example: sequence ['parent','parent'] yields grandparent adjacency.
    """
    seq = list(sequence)
    assert len(seq) >= 1
    result = (relations[seq[0]] > 0).to(relations[seq[0]].dtype)
    for name in seq[1:]:
        result = boolean_matmul(result, relations[name])
    return result


def khop_union(relations: Dict[str, torch.Tensor], L: int) -> torch.Tensor:
    """Union of all paths up to length L across all relations (Boolean)."""
    keys = list(relations.keys())
    # Start with zero adjacency
    N = relations[keys[0]].shape[0]
    union = torch.zeros((N, N), dtype=relations[keys[0]].dtype, device=relations[keys[0]].device)

    # Enumerate sequences up to L (small L recommended)
    def dfs(path: List[str], depth: int):
        nonlocal union
        if depth == 0:
            return
        # Compose current path
        comp = compose_sequence(relations, path)
        union = (union + comp).clamp_max_(1.0)
        # Extend
        if depth > 1:
            for k in keys:
                dfs(path + [k], depth - 1)

    for k in keys:
        dfs([k], L)
    return union


def sample_pairs_from_adjacency(adj: torch.Tensor, num_samples: int) -> List[Tuple[int, int]]:
    """Sample (i,j) pairs from a Boolean adjacency matrix."""
    idx = (adj > 0).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return []
    if idx.shape[0] <= num_samples:
        return [(int(i.item()), int(j.item())) for i, j in idx]
    sel = torch.randperm(idx.shape[0])[:num_samples]
    chosen = idx[sel]
    return [(int(i.item()), int(j.item())) for i, j in chosen]


def negative_pairs_from_adjacency(adj: torch.Tensor, num_samples: int) -> List[Tuple[int, int]]:
    """Sample non-edges as negatives under closed-world assumption."""
    N = adj.shape[0]
    negatives: List[Tuple[int, int]] = []
    while len(negatives) < num_samples:
        i = torch.randint(0, N, (1,)).item()
        j = torch.randint(0, N, (1,)).item()
        if i != j and adj[i, j].item() == 0:
            negatives.append((i, j))
    return negatives

