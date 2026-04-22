"""Smoke tests for the four loss modules."""

from __future__ import annotations

import torch

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.losses.identity import IdentityLoss
from gazelock_ml.losses.perceptual import VGGPerceptualLoss
from gazelock_ml.losses.reconstruction import L1ReconstructionLoss
from gazelock_ml.losses.temporal import TemporalConsistencyLoss


def _rand_pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    pred = torch.rand((2, 3, EYE_ROI_H, EYE_ROI_W))
    tgt = torch.rand((2, 3, EYE_ROI_H, EYE_ROI_W))
    return pred, tgt


def test_reconstruction_loss_is_zero_on_identical_inputs() -> None:
    loss_fn = L1ReconstructionLoss()
    x = torch.rand((1, 3, 10, 10))
    assert loss_fn(x, x).item() == 0.0


def test_reconstruction_loss_positive_on_different_inputs() -> None:
    pred, tgt = _rand_pair()
    assert L1ReconstructionLoss()(pred, tgt).item() > 0


def test_perceptual_loss_runs_and_returns_scalar() -> None:
    pred, tgt = _rand_pair()
    val = VGGPerceptualLoss()(pred, tgt)
    assert val.shape == torch.Size([])
    assert val.item() >= 0.0


def test_identity_loss_is_nearly_zero_on_identical_inputs() -> None:
    loss_fn = IdentityLoss()
    x = torch.rand((2, 3, EYE_ROI_H, EYE_ROI_W))
    val = loss_fn(x, x).item()
    # Cosine similarity of vector with itself is 1, so 1-cos ≈ 0
    assert val < 1e-4


def test_identity_loss_positive_on_different_inputs() -> None:
    pred, tgt = _rand_pair()
    assert IdentityLoss()(pred, tgt).item() > 0


def test_temporal_loss_zero_on_identical_outputs() -> None:
    x = torch.rand((1, 3, 8, 8))
    assert TemporalConsistencyLoss()(x, x).item() == 0.0


def test_temporal_loss_positive_on_different_outputs() -> None:
    torch.manual_seed(1)
    a = torch.rand((1, 3, 8, 8))
    b = torch.rand((1, 3, 8, 8))
    assert TemporalConsistencyLoss()(a, b).item() > 0
