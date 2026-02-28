"""Tests for Transformer encoder/decoder norm placement and basic forward pass."""

import torch

from tensorlogic.transformers.layers import (
    TransformerEncoder, TransformerDecoder, Transformer,
)


class TestTransformerEncoderNorm:
    def test_prenorm_has_final_norm(self):
        enc = TransformerEncoder(num_layers=2, d_model=16, nhead=2, norm_first=True)
        assert enc.norm is not None

    def test_postnorm_no_final_norm(self):
        enc = TransformerEncoder(num_layers=2, d_model=16, nhead=2, norm_first=False)
        assert enc.norm is None

    def test_forward_shape(self):
        enc = TransformerEncoder(num_layers=2, d_model=16, nhead=2)
        x = torch.randn(2, 5, 16)
        out = enc(x)
        assert out.shape == (2, 5, 16)

    def test_forward_with_attention(self):
        enc = TransformerEncoder(num_layers=2, d_model=16, nhead=2)
        x = torch.randn(2, 5, 16)
        out, attn = enc(x, return_attention=True)
        assert out.shape == (2, 5, 16)
        assert len(attn) == 2


class TestTransformerDecoderNorm:
    def test_prenorm_has_final_norm(self):
        dec = TransformerDecoder(num_layers=2, d_model=16, nhead=2, norm_first=True)
        assert dec.norm is not None

    def test_postnorm_no_final_norm(self):
        dec = TransformerDecoder(num_layers=2, d_model=16, nhead=2, norm_first=False)
        assert dec.norm is None

    def test_forward_shape(self):
        dec = TransformerDecoder(num_layers=2, d_model=16, nhead=2)
        tgt = torch.randn(2, 4, 16)
        memory = torch.randn(2, 6, 16)
        out = dec(tgt, memory)
        assert out.shape == (2, 4, 16)


class TestTransformerFull:
    def test_forward(self):
        model = Transformer(d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1)
        src = torch.randn(2, 5, 16)
        tgt = torch.randn(2, 4, 16)
        out = model(src, tgt)
        assert out.shape == (2, 4, 16)

    def test_encode_decode_separate(self):
        model = Transformer(d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1)
        src = torch.randn(2, 5, 16)
        tgt = torch.randn(2, 4, 16)
        memory = model.encode(src)
        out = model.decode(tgt, memory)
        assert out.shape == (2, 4, 16)

    def test_prenorm_and_postnorm_produce_different_outputs(self):
        """Sanity check that norm_first actually changes behavior."""
        torch.manual_seed(42)
        m1 = Transformer(d_model=16, nhead=2, num_encoder_layers=1,
                         num_decoder_layers=1, norm_first=True)
        torch.manual_seed(42)
        m2 = Transformer(d_model=16, nhead=2, num_encoder_layers=1,
                         num_decoder_layers=1, norm_first=False)
        src = torch.randn(1, 3, 16)
        tgt = torch.randn(1, 3, 16)
        o1 = m1(src, tgt)
        o2 = m2(src, tgt)
        # They share init but different norm order → different outputs
        assert not torch.allclose(o1, o2, atol=1e-4)
