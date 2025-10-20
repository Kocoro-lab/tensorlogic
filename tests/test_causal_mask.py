import torch

from tensorlogic.transformers.utils import causal_mask
from tensorlogic.transformers.tensor_equations import attention_weights
from tensorlogic.transformers.attention import MultiHeadAttention


def test_causal_mask_upper_triangle_true():
    L = 7
    m = causal_mask(L)
    assert m.shape == (1, L, L)
    assert m.dtype == torch.bool

    # True exactly on upper triangle (j > i)
    expected = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    assert torch.equal(m[0], expected)


def test_attention_weights_zero_future_positions():
    L = 6
    # Simple increasing scores for readability
    scores = torch.arange(L * L, dtype=torch.float32).reshape(1, 1, L, L)
    m = causal_mask(L)

    w = attention_weights(scores, mask=m, mode="continuous")
    assert w.shape == (1, 1, L, L)

    # Future positions (j > i) must have exactly zero probability
    for i in range(L):
        if i + 1 < L:
            assert torch.all(w[0, 0, i, i + 1:] == 0)
        # Probabilities on allowed keys should sum to 1
        assert torch.allclose(w[0, 0, i, : i + 1].sum(), torch.tensor(1.0))


def test_multihead_attention_respects_causality():
    B, L, E, H = 1, 5, 16, 4
    x = torch.randn(B, L, E)
    attn = MultiHeadAttention(embedding_dim=E, num_heads=H, dropout=0.0, mode="continuous")

    m = causal_mask(L)  # [1, L, L]
    y, w = attn(x, mask=m, need_weights=True)

    assert y.shape == (B, L, E)
    assert w is not None and w.shape == (B, H, L, L)

    # Check masked positions are zero across all heads
    for i in range(L):
        if i + 1 < L:
            assert torch.all(w[0, :, i, i + 1:] == 0)

