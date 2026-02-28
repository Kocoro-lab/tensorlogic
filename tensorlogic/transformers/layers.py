from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "relu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            x = F.relu(self.fc1(x))
        elif self.activation == "gelu":
            x = F.gelu(self.fc1(x))
        else:
            x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
        mode: str = "continuous",
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, mode=mode)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout=dropout, activation="relu")

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if self.norm_first:
            x = src
            x = self.norm1(x)
            attn_out, attn_weights = self.self_attn(x, mask=src_mask, need_weights=return_attention)
            x = src + self.dropout(attn_out)
            y = self.norm2(x)
            y = self.ff(y)
            output = x + self.dropout(y)
            if return_attention:
                return output, attn_weights
            return output
        else:
            attn_out, attn_weights = self.self_attn(src, mask=src_mask, need_weights=return_attention)
            x = self.norm1(src + self.dropout(attn_out))
            y = self.ff(x)
            output = self.norm2(x + self.dropout(y))
            if return_attention:
                return output, attn_weights
            return output


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
        mode: str = "continuous",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=norm_first,
                mode=mode,
            )
            for _ in range(num_layers)
        ])
        # Pre-norm (norm_first=True) applies LayerNorm before each sublayer,
        # so the final layer's output is unnormalized after the residual
        # connection. A final LayerNorm is needed to normalize it.
        # Post-norm (norm_first=False) already normalizes after each sublayer,
        # so no extra norm is needed.
        self.norm = nn.LayerNorm(d_model) if norm_first else None

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        x = src
        attention_weights = []

        for layer in self.layers:
            if return_attention:
                x, attn_weights = layer(x, src_mask=src_mask, return_attention=True)
                attention_weights.append(attn_weights)
            else:
                x = layer(x, src_mask=src_mask)

        if self.norm is not None:
            x = self.norm(x)

        if return_attention:
            return x, attention_weights
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with self-attention and cross-attention.

    Tensor equations:
    1. Self-attention: y = MultiHeadAttention(x, x, x, mask=tgt_mask)
    2. Cross-attention: z = MultiHeadAttention(y, memory, memory, mask=memory_mask)
    3. Feed-forward: output = FFN(z)

    Each with residual connections and layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
        mode: str = "continuous",
    ) -> None:
        super().__init__()
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, mode=mode)
        # Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, mode=mode)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Feed-forward and dropout
        self.ff = FeedForward(d_model, dim_feedforward, dropout=dropout, activation="relu")
        self.dropout = nn.Dropout(dropout)

        self.norm_first = norm_first

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.

        Args:
            tgt: [batch_size, tgt_len, d_model] target sequence
            memory: [batch_size, src_len, d_model] encoder output
            tgt_mask: [batch_size, tgt_len, tgt_len] self-attention mask
            memory_mask: [batch_size, tgt_len, src_len] cross-attention mask
            return_attention: If True, return attention weights

        Returns:
            output: [batch_size, tgt_len, d_model]
            If return_attention: also returns (self_attn_weights, cross_attn_weights)
        """
        if self.norm_first:
            # Pre-norm architecture
            # Self-attention
            x = tgt
            x2 = self.norm1(x)
            self_attn_out, self_attn_weights = self.self_attn(x2, mask=tgt_mask, need_weights=True)
            x = x + self.dropout(self_attn_out)

            # Cross-attention
            x2 = self.norm2(x)
            # Query from decoder, Key/Value from encoder
            cross_attn_out, cross_attn_weights = self.cross_attn(
                x2, memory, memory, mask=memory_mask, need_weights=True
            )
            x = x + self.dropout(cross_attn_out)

            # Feed-forward
            x2 = self.norm3(x)
            ff_out = self.ff(x2)
            output = x + self.dropout(ff_out)
        else:
            # Post-norm architecture
            # Self-attention
            self_attn_out, self_attn_weights = self.self_attn(tgt, mask=tgt_mask, need_weights=True)
            x = self.norm1(tgt + self.dropout(self_attn_out))

            # Cross-attention
            cross_attn_out, cross_attn_weights = self.cross_attn(
                x, memory, memory, mask=memory_mask, need_weights=True
            )
            x = self.norm2(x + self.dropout(cross_attn_out))

            # Feed-forward
            ff_out = self.ff(x)
            output = self.norm3(x + self.dropout(ff_out))

        if return_attention:
            return output, self_attn_weights, cross_attn_weights
        return output


class TransformerDecoder(nn.Module):
    """
    Stack of transformer decoder layers.

    Implements the full decoder with multiple layers of:
    - Masked self-attention
    - Cross-attention to encoder output
    - Feed-forward networks
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
        mode: str = "continuous",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=norm_first,
                mode=mode,
            )
            for _ in range(num_layers)
        ])
        # Pre-norm needs final LayerNorm (see TransformerEncoder comment).
        # Post-norm already normalizes after each sublayer.
        self.norm = nn.LayerNorm(d_model) if norm_first else None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through decoder stack.

        Args:
            tgt: [batch_size, tgt_len, d_model] target sequence
            memory: [batch_size, src_len, d_model] encoder output
            tgt_mask: Optional self-attention mask
            memory_mask: Optional cross-attention mask
            return_attention: If True, return all attention weights

        Returns:
            output: [batch_size, tgt_len, d_model]
            If return_attention: also returns attention weights from all layers
        """
        x = tgt
        self_attn_weights_all = []
        cross_attn_weights_all = []

        for layer in self.layers:
            if return_attention:
                x, self_attn_w, cross_attn_w = layer(
                    x, memory, tgt_mask, memory_mask, return_attention=True
                )
                self_attn_weights_all.append(self_attn_w)
                cross_attn_weights_all.append(cross_attn_w)
            else:
                x = layer(x, memory, tgt_mask, memory_mask)

        # Final layer norm if using pre-norm
        if self.norm is not None:
            x = self.norm(x)

        if return_attention:
            return x, self_attn_weights_all, cross_attn_weights_all
        return x


class Transformer(nn.Module):
    """
    Full transformer model with encoder and decoder.

    Combines:
    - TransformerEncoder: Processes source sequence
    - TransformerDecoder: Generates target sequence with cross-attention to encoder

    This implements the standard transformer architecture from "Attention is All You Need",
    expressed as tensor equations in the TensorLogic framework.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
        mode: str = "continuous",
    ) -> None:
        super().__init__()

        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=norm_first,
            mode=mode,
        )

        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=norm_first,
            mode=mode,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full transformer forward pass.

        Args:
            src: [batch_size, src_len, d_model] source sequence
            tgt: [batch_size, tgt_len, d_model] target sequence
            src_mask: Optional encoder self-attention mask
            tgt_mask: Optional decoder self-attention mask (usually causal)
            memory_mask: Optional decoder cross-attention mask

        Returns:
            output: [batch_size, tgt_len, d_model] decoded sequence
        """
        # Encode source
        memory = self.encoder(src, src_mask)

        # Decode with cross-attention to encoder output
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)

        return output

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Just run the encoder."""
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Just run the decoder."""
        return self.decoder(tgt, memory, tgt_mask, memory_mask)

    def to_tensor_equations(self) -> List[str]:
        """Export high-level tensor equations for the Transformer.

        This provides a readable, symbolic description of the attention and FFN
        computations used by the encoder and decoder.
        """
        eqs = []
        d = "d"
        eqs.append("# Encoder self-attention (per layer)")
        eqs.append("Q[b,h,i,d] := X[b,i,e] × W_q[h,e,d]")
        eqs.append("K[b,h,j,d] := X[b,j,e] × W_k[h,e,d]")
        eqs.append("V[b,h,j,d] := X[b,j,e] × W_v[h,e,d]")
        eqs.append("Scores[b,h,i,j] := Q[b,h,i,d] × K[b,h,j,d] / sqrt(d)")
        eqs.append("A[b,h,i,j] := softmax(Scores[b,h,i,j], dim=j)")
        eqs.append("Out[b,h,i,d] := A[b,h,i,j] × V[b,h,j,d]")
        eqs.append("X′[b,i,e] := concat(Out[b,:,i,:]) × W_o[h*d,e]")
        eqs.append("# Encoder FFN")
        eqs.append("H[b,i,f] := act(X′[b,i,e] × W_1[e,f] + b_1[f])")
        eqs.append("X_enc[b,i,e] := H[b,i,f] × W_2[f,e] + b_2[e]")

        eqs.append("\n# Decoder self-attention (per layer)")
        eqs.append("Q_t[b,h,i,d] := T[b,i,e] × W_q^t[h,e,d]")
        eqs.append("K_t[b,h,j,d] := T[b,j,e] × W_k^t[h,e,d]")
        eqs.append("V_t[b,h,j,d] := T[b,j,e] × W_v^t[h,e,d]")
        eqs.append("Scores_t[b,h,i,j] := mask(Q_t[b,h,i,d] × K_t[b,h,j,d]/sqrt(d))")
        eqs.append("A_t[b,h,i,j] := softmax(Scores_t[b,h,i,j], dim=j)")
        eqs.append("Out_t[b,h,i,d] := A_t[b,h,i,j] × V_t[b,h,j,d]")
        eqs.append("T′[b,i,e] := concat(Out_t[b,:,i,:]) × W_o^t[h*d,e]")
        eqs.append("# Cross-attention")
        eqs.append("Q_c[b,h,i,d] := T′[b,i,e] × W_q^c[h,e,d]")
        eqs.append("K_c[b,h,j,d] := X_enc[b,j,e] × W_k^c[h,e,d]")
        eqs.append("V_c[b,h,j,d] := X_enc[b,j,e] × W_v^c[h,e,d]")
        eqs.append("Scores_c[b,h,i,j] := Q_c[b,h,i,d] × K_c[b,h,j,d] / sqrt(d)")
        eqs.append("A_c[b,h,i,j] := softmax(Scores_c[b,h,i,j], dim=j)")
        eqs.append("Out_c[b,h,i,d] := A_c[b,h,i,j] × V_c[b,h,j,d]")
        eqs.append("T″[b,i,e] := concat(Out_c[b,:,i,:]) × W_o^c[h*d,e]")
        eqs.append("# Decoder FFN")
        eqs.append("H_t[b,i,f] := act(T″[b,i,e] × W_1^t[e,f] + b_1^t[f])")
        eqs.append("Y[b,i,e] := H_t[b,i,f] × W_2^t[f,e] + b_2^t[e]")
        return eqs
