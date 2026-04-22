"""Iris-region identity-preservation loss.

Cosine distance between the eye-encoder embedding of the refiner's
output and the ground-truth target. The encoder is frozen during
refiner training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from gazelock_ml.models.eye_encoder import EyeIdentityEncoder


class IdentityLoss(nn.Module):
    """1 - cosine similarity of embeddings."""

    def __init__(self, encoder: EyeIdentityEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder or EyeIdentityEncoder()
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_emb = self.encoder(pred)
        with torch.no_grad():
            tgt_emb = self.encoder(target)
        cos = F.cosine_similarity(pred_emb, tgt_emb, dim=1)
        return 1.0 - cos.mean()


__all__ = ["IdentityLoss"]
