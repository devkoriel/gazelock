"""Shape and parameter tests for the eye-identity encoder."""

from __future__ import annotations

import torch

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.models.eye_encoder import EyeIdentityEncoder
from gazelock_ml.models.refiner import count_parameters


def test_embedding_shape_and_norm() -> None:
    model = EyeIdentityEncoder()
    model.eval()
    x = torch.randn((4, 3, EYE_ROI_H, EYE_ROI_W)) * 0.1
    with torch.no_grad():
        emb = model(x)
    assert emb.shape == (4, EyeIdentityEncoder.EMBEDDING_DIM)
    # L2-normalised rows → each row norm ≈ 1
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_encoder_param_count_reasonable() -> None:
    params = count_parameters(EyeIdentityEncoder())
    # Target ~180k, allow 100k-300k
    assert 100_000 <= params <= 300_000, f"unexpected params: {params}"


def test_different_inputs_produce_different_embeddings() -> None:
    model = EyeIdentityEncoder()
    model.eval()
    x1 = torch.rand((1, 3, EYE_ROI_H, EYE_ROI_W))
    x2 = torch.rand((1, 3, EYE_ROI_H, EYE_ROI_W))
    with torch.no_grad():
        e1 = model(x1)
        e2 = model(x2)
    # Not required to be orthogonal, just not identical
    assert not torch.allclose(e1, e2, atol=1e-3)
