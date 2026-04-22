"""Tests for ComposedLoss weighted-sum behavior."""

from __future__ import annotations

import torch

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.training.composed_loss import ComposedLoss, LossWeights


def _rand_pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    p = torch.rand((2, 3, EYE_ROI_H, EYE_ROI_W))
    t = torch.rand((2, 3, EYE_ROI_H, EYE_ROI_W))
    return p, t


def test_composed_returns_dict_with_all_keys() -> None:
    loss = ComposedLoss()
    p, t = _rand_pair()
    out = loss(p, t)
    assert set(out.keys()) == {"l1", "perceptual", "identity", "temporal", "total"}
    for v in out.values():
        assert v.shape == torch.Size([])


def test_temporal_is_zero_when_no_neighbour_provided() -> None:
    loss = ComposedLoss()
    p, t = _rand_pair()
    out = loss(p, t)
    assert out["temporal"].item() == 0.0


def test_total_matches_weighted_sum() -> None:
    weights = LossWeights(l1=1.0, perceptual=0.2, identity=0.5, temporal=0.1)
    loss = ComposedLoss(weights=weights)
    p, t = _rand_pair()
    neighbour = p + 0.01 * torch.randn_like(p)
    out = loss(p, t, pred_neighbour=neighbour)

    expected = (
        weights.l1 * out["l1"]
        + weights.perceptual * out["perceptual"]
        + weights.identity * out["identity"]
        + weights.temporal * out["temporal"]
    )
    torch.testing.assert_close(out["total"], expected, rtol=1e-5, atol=1e-5)


def test_total_has_gradient() -> None:
    loss = ComposedLoss()
    p, t = _rand_pair()
    p.requires_grad_(True)
    out = loss(p, t)
    out["total"].backward()
    assert p.grad is not None
    assert not torch.isnan(p.grad).any()
