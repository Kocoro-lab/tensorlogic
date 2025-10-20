from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tensor_equations import attention_scores, attention_weights, apply_attention


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention.

    Expects inputs with shape [B, L, E]. Returns [B, L, E].
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        mode: str = "continuous",
        return_attention_weights: bool = False,
    ) -> None:
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.d_model = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.mode = mode
        self.return_attention_weights = return_attention_weights

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        x = x.view(b, l, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, l, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(b, l, h * d)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        need_weights: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if key is None:
            key = query
        if value is None:
            value = key

        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))

        scores = attention_scores(q, k, scale=True)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.to(device=scores.device)

        weights = attention_weights(scores, mask=mask, mode=self.mode)
        weights = self.dropout(weights)
        attended = apply_attention(weights, v)
        out = self._merge(attended)
        out = self.out_proj(out)

        ret_w = need_weights or self.return_attention_weights
        if ret_w:
            return out, weights
        return out, None

