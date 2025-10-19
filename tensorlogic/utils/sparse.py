"""
Sparse utilities: build sparse adjacency from pairs and aggregate messages.
"""

from typing import Iterable, Tuple

import torch


def adjacency_from_pairs_sparse(num_objects: int, pairs: Iterable[Tuple[int, int]]) -> torch.Tensor:
    """Create an (N x N) COO sparse adjacency from (i,j) pairs."""
    if not isinstance(pairs, list):
        pairs = list(pairs)
    if len(pairs) == 0:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty((0,), dtype=torch.float32)
    else:
        idx = torch.tensor(pairs, dtype=torch.long)
        indices = idx.t().contiguous()  # [2, E]
        values = torch.ones(idx.shape[0], dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (num_objects, num_objects))


def sparse_mm_aggregate(adj: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    """
    Aggregate neighbor embeddings via sparse adjacency multiplication.

    Args:
        adj: [N, N] sparse COO adjacency
        embeddings: [N, D] dense embedding matrix
    Returns:
        [N, D] aggregated messages
    """
    if not adj.is_sparse:
        raise ValueError("adj must be a sparse tensor")
    return torch.sparse.mm(adj, embeddings)

