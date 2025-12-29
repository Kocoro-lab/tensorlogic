from typing import Optional

import torch


def causal_mask(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create a causal mask for self-attention.
    Returns a mask where True means "mask out" (set to -inf).
    For causal attention, we mask out future positions (upper triangle).

    Position i can attend to positions 0...i (inclusive).
    Position i cannot attend to positions i+1...size-1 (future).
    """
    idx = torch.arange(size, device=device)
    # Create mask: mask[i,j] = True if position i cannot attend to position j
    # Position i cannot attend to j if j > i (j is in the future)
    mask = idx.unsqueeze(0) < idx.unsqueeze(1)  # mask[i,j] = (i < j)
    # Transpose to get the right shape: we want mask[i,j] = (j > i)
    mask = mask.T
    return mask.unsqueeze(0)
