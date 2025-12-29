from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import TransformerEncoder
from .positional import LearnedPositionalEncoding, SinusoidalPositionalEncoding
from .utils import causal_mask


class DecoderOnlyLM(nn.Module):
    """
    Minimal decoder-only language model (GPT-like) using TensorLogic components.

    Architecture:
    - Token embedding + learned positional encoding
    - N × TransformerEncoderLayer (masked self-attention only)
    - Final LayerNorm
    - LM head with optional weight tying
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        dim_feedforward: Optional[int] = None,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pos_encoding: str = "learned",  # or "sinusoidal"
        norm_first: bool = True,
        mode: str = "continuous",
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if pos_encoding == "learned":
            self.pos_emb = LearnedPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        else:
            self.pos_emb = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        self.encoder = TransformerEncoder(
            num_layers=n_layer,
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout,
            norm_first=norm_first,
            mode=mode,
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] token indices
            causal: If True, apply causal mask
        Returns:
            logits: [B, L, vocab_size]
        """
        b, l = input_ids.shape
        assert l <= self.max_seq_len, "Sequence length exceeds max_seq_len"
        x = self.tok_emb(input_ids)  # [B, L, D]
        x = self.pos_emb(x)

        src_mask = causal_mask(l, device=input_ids.device) if causal else None
        x = self.encoder(x, src_mask=src_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Greedy/top-k sampling without KV cache (O(T^2) per step).
        Uses sliding window to handle sequences longer than max_seq_len.
        """
        device = input_ids.device
        out = input_ids
        for _ in range(max_new_tokens):
            # Crop context if it exceeds max_seq_len (sliding window)
            context = out if out.size(1) <= self.max_seq_len else out[:, -self.max_seq_len:]
            logits = self.forward(context)  # [B, L, V]
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            out = torch.cat([out, next_token], dim=1)

            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
        return out

    def to_tensor_equations(self) -> List[str]:
        """
        Export decoder-only LM as tensor equations.

        Returns a readable list of einsum-style equations describing:
        - Token and position embeddings
        - Masked self-attention (per layer)
        - Feed-forward networks
        - Language model head projection
        """
        eqs = []
        eqs.append("# Decoder-Only Language Model Tensor Equations")
        eqs.append("# ============================================")
        eqs.append("")

        # Embeddings
        eqs.append("# Token + Position Embeddings")
        eqs.append("E_tok[b,i,d] := TokenEmbed[tokens[b,i], d]")
        eqs.append("E_pos[i,d] := PosEmbed[i, d]  # Learned or sinusoidal")
        eqs.append("X_0[b,i,d] := E_tok[b,i,d] + E_pos[i,d]")
        eqs.append("")

        # Per-layer equations
        eqs.append("# For each transformer layer l ∈ [1, N_layers]:")
        eqs.append("# -----------------------------------------------")
        eqs.append("")

        # Masked self-attention
        eqs.append("# Masked Self-Attention (layer l)")
        eqs.append("X_norm[b,i,d] := LayerNorm(X_{l-1}[b,i,d])")
        eqs.append("Q[b,h,i,d] := X_norm[b,i,e] × W_q^l[h,e,d]")
        eqs.append("K[b,h,j,d] := X_norm[b,j,e] × W_k^l[h,e,d]")
        eqs.append("V[b,h,j,d] := X_norm[b,j,e] × W_v^l[h,e,d]")
        eqs.append("Scores[b,h,i,j] := Q[b,h,i,d] × K[b,h,j,d] / sqrt(d)")
        eqs.append("# Apply causal mask: Scores[b,h,i,j] = -∞ if j > i")
        eqs.append("A[b,h,i,j] := softmax(masked(Scores[b,h,i,j]), dim=j)")
        eqs.append("Out[b,h,i,d] := A[b,h,i,j] × V[b,h,j,d]")
        eqs.append("MultiHead[b,i,e] := concat(Out[b,:,i,:]) × W_o^l[h*d,e]")
        eqs.append("X_attn[b,i,d] := X_{l-1}[b,i,d] + Dropout(MultiHead[b,i,e])")
        eqs.append("")

        # Feed-forward
        eqs.append("# Feed-Forward Network (layer l)")
        eqs.append("X_norm2[b,i,d] := LayerNorm(X_attn[b,i,d])")
        eqs.append("H[b,i,f] := activation(X_norm2[b,i,e] × W_1^l[e,f] + b_1^l[f])")
        eqs.append("FFN[b,i,e] := H[b,i,f] × W_2^l[f,e] + b_2^l[e]")
        eqs.append("X_l[b,i,d] := X_attn[b,i,d] + Dropout(FFN[b,i,e])")
        eqs.append("")

        # Final layer norm and LM head
        eqs.append("# Final Layer Norm and Language Model Head")
        eqs.append("X_final[b,i,d] := LayerNorm(X_N[b,i,d])")
        eqs.append("logits[b,i,v] := X_final[b,i,d] × W_lm[d,v]")
        eqs.append("# Note: If weight tying, W_lm = TokenEmbed^T")
        eqs.append("")

        # Generation equations
        eqs.append("# Generation (autoregressive decoding)")
        eqs.append("# -------------------------------------")
        eqs.append("# At each timestep t:")
        eqs.append("probs[b,v] := softmax(logits[b,t,v] / temperature)")
        eqs.append("# Optional top-k filtering: zero out all but top k values")
        eqs.append("next_token[b] := sample(probs[b,v])")
        eqs.append("# Append to sequence and repeat")

        return eqs
