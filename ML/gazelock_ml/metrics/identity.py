"""Identity cosine similarity metric (using the P2a eye encoder)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gazelock_ml.models.eye_encoder import EyeIdentityEncoder


def identity_cosine(
    pred: torch.Tensor,
    target: torch.Tensor,
    encoder: EyeIdentityEncoder | None = None,
) -> float:
    """Mean cosine similarity of (pred, target) eye embeddings."""
    enc = encoder or EyeIdentityEncoder()
    enc.eval()
    with torch.no_grad():
        ep = enc(pred)
        et = enc(target)
    cos = F.cosine_similarity(ep, et, dim=1)
    return float(cos.mean().item())


__all__ = ["identity_cosine"]
