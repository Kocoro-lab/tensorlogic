from typing import Optional

import torch
import torch.nn.functional as F

from tensorlogic.ops.logical import logical_join


def attention_scores(q: torch.Tensor, k: torch.Tensor, scale: bool = True) -> torch.Tensor:
    """Compute scaled dot-product attention scores.

    Shapes:
        q: [B, H, Lq, D]
        k: [B, H, Lk, D]
        returns: [B, H, Lq, Lk]
    """
    scores = logical_join(q, k, equation="bhid,bhjd->bhij")
    if scale:
        d = q.size(-1)
        scores = scores / (d ** 0.5)
    return scores


def attention_weights(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mode: str = "continuous",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Normalize attention scores to weights.

    scores: [B, H, Lq, Lk]
    mask: broadcastable to scores. True means mask out (set to -inf), False means keep.
    This follows PyTorch convention for attention masks.
    """
    if mask is not None:
        mask = mask.to(dtype=torch.bool)
        scores = scores.masked_fill(mask, float("-inf"))  # Changed from ~mask to mask

    if mode == "boolean":
        idx = torch.argmax(scores, dim=-1)
        one_hot = F.one_hot(idx, num_classes=scores.size(-1)).to(dtype=scores.dtype)
        return one_hot

    return F.softmax(scores / max(temperature, 1e-6), dim=-1)


def apply_attention(weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply attention weights to values.

    weights: [B, H, Lq, Lk]
    v: [B, H, Lk, D]
    returns: [B, H, Lq, D]
    """
    return logical_join(weights, v, equation="bhij,bhjd->bhid")

