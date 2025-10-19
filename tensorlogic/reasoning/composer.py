"""
Gated multi-hop relation composer for embedding-space reasoning.

Learns to compose a bank of relation matrices over multiple hops, conditioned
on the subject/object embeddings. Produces a scalar compatibility score that
can be trained with contrastive or BCE-style objectives.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultiHopComposer(nn.Module):
    """
    Per-hop softmax gating over a bank of relation matrices.

    - Encodes a (subject, object) pair to a query vector
    - For each hop, produces a softmax over relations
    - Applies the weighted relation to the running embedding
    - Scores the final embedding against the object embedding

    Inputs:
        subject_emb: [batch, dim]
        object_emb:  [batch, dim]
        relation_bank: [num_rel, dim, dim]

    Output:
        scores (logits): [batch]
    """

    def __init__(self, embedding_dim: int, num_relations: int, num_hops: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.num_hops = num_hops

        self.query_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.hop_attention = nn.ModuleList(
            [nn.Linear(embedding_dim, num_relations) for _ in range(num_hops)]
        )

    def forward(
        self,
        subject_emb: torch.Tensor,
        object_emb: torch.Tensor,
        relation_bank: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits for (subject, object) pairs via multi-hop composition."""
        # Encode (s, o)
        query = torch.cat([subject_emb, object_emb], dim=-1)
        query_vec = self.query_encoder(query)  # [B, D]

        current = subject_emb  # [B, D]

        # relation_bank: [R, D, D]
        for hop_idx in range(self.num_hops):
            attn_logits = self.hop_attention[hop_idx](query_vec)  # [B, R]
            attn_weights = F.softmax(attn_logits, dim=-1)  # [B, R]

            # Weighted sum of relation matrices: [B, D, D]
            composed = torch.einsum("br,rij->bij", attn_weights, relation_bank)

            # Apply to current embedding: [B, D]
            current = torch.einsum("bij,bj->bi", composed, current)

        # Final dot with object embedding -> logits
        scores = (current * object_emb).sum(dim=-1)
        return scores

    def forward_with_attn(
        self,
        subject_emb: torch.Tensor,
        object_emb: torch.Tensor,
        relation_bank: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Like forward, but also returns a list of attention weight tensors
        (one per hop), each of shape [B, R].
        """
        query = torch.cat([subject_emb, object_emb], dim=-1)
        query_vec = self.query_encoder(query)

        current = subject_emb
        weights: List[torch.Tensor] = []

        for hop_idx in range(self.num_hops):
            logits = self.hop_attention[hop_idx](query_vec)
            if temperature is not None and temperature > 0:
                logits = logits / temperature
            attn = F.softmax(logits, dim=-1)
            weights.append(attn)
            composed = torch.einsum("br,rij->bij", attn, relation_bank)
            current = torch.einsum("bij,bj->bi", composed, current)

        scores = (current * object_emb).sum(dim=-1)
        return scores, weights


def stack_relation_bank(param_dict: "nn.ParameterDict", order: Optional[List[str]] = None) -> torch.Tensor:
    """Stack a ParameterDict of relation matrices into a [R, D, D] tensor.

    If order is provided, use that relation name order; otherwise sort by key for
    stable stacking.
    """
    keys = order if order is not None else sorted(list(param_dict.keys()))
    mats = [param_dict[k] for k in keys]
    return torch.stack(mats, dim=0)
