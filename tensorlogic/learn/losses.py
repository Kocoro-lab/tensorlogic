"""
Loss functions for embedding-space and composer-based training.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for relation learning with positive/negative pairs.

    Works with a generic scorer callable or a model that exposes a suitable
    interface. The scorer should accept (model, subjects, objects) and return
    probabilities in [0, 1] or logits if use_logits=True.
    """

    def __init__(self, margin: float = 0.5, negative_samples: int = 5, use_logits: bool = False):
        super().__init__()
        self.margin = margin
        self.negative_samples = negative_samples
        self.use_logits = use_logits
        self._bce = nn.BCEWithLogitsLoss() if use_logits else nn.BCELoss()

    def forward(
        self,
        model,
        positive_pairs: torch.Tensor,  # [B, 2]
        negative_pairs: Optional[torch.Tensor] = None,  # [B, K, 2]
        scorer: Optional[Callable] = None,
        num_objects: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            model: scorer's first argument
            positive_pairs: [B, 2] subject, object indices
            negative_pairs: [B, K, 2] optional; if None, auto-sample using num_objects
            scorer: callable(model, subjects, objects) -> probs or logits
            num_objects: required if negative_pairs is None
        """
        if scorer is None:
            raise ValueError("scorer callable is required for ContrastiveLoss")

        pos_subj = positive_pairs[:, 0]
        pos_obj = positive_pairs[:, 1]

        # Positives target=1
        pos_scores = scorer(model, pos_subj, pos_obj)  # [B]
        pos_targets = torch.ones_like(pos_scores)
        pos_loss = self._bce(pos_scores, pos_targets)

        # Negatives target=0
        if negative_pairs is None:
            if num_objects is None:
                raise ValueError("num_objects must be provided to auto-sample negatives")
            B = positive_pairs.shape[0]
            K = self.negative_samples
            neg_subj = pos_subj.unsqueeze(1).expand(-1, K)
            neg_obj = torch.randint(low=0, high=num_objects, size=(B, K), device=pos_subj.device)
        else:
            neg_subj = negative_pairs[:, :, 0]
            neg_obj = negative_pairs[:, :, 1]

        neg_subj_flat = neg_subj.reshape(-1)
        neg_obj_flat = neg_obj.reshape(-1)

        neg_scores = scorer(model, neg_subj_flat, neg_obj_flat)  # [B*K]
        neg_targets = torch.zeros_like(neg_scores)

        # Margin (optional): encourage neg_scores < margin by down-weighting small logits
        if self.use_logits:
            neg_loss = self._bce(neg_scores - self.margin, neg_targets)
        else:
            neg_loss = self._bce(neg_scores, neg_targets)

        return pos_loss + neg_loss

