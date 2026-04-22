"""Shape and parameter-count tests for the refiner UNet."""

from __future__ import annotations

import torch

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.models.refiner import RefinerUNet, count_parameters


def test_forward_shape_matches_input_spatial_dims() -> None:
    model = RefinerUNet()
    model.eval()
    x = torch.zeros((2, 6, EYE_ROI_H, EYE_ROI_W))  # batch=2
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 3, EYE_ROI_H, EYE_ROI_W)
    # Sigmoid output must be in [0, 1]
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


def test_parameter_count_within_target() -> None:
    model = RefinerUNet()
    params = count_parameters(model)
    # Spec §6.5 calls for ~120 k params. Allow ±30 % headroom:
    # 85 k ≤ params ≤ 160 k.
    assert 85_000 <= params <= 160_000, f"unexpected params: {params}"


def test_model_supports_single_sample_inference() -> None:
    model = RefinerUNet()
    model.eval()
    x = torch.zeros((1, 6, EYE_ROI_H, EYE_ROI_W))
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, EYE_ROI_H, EYE_ROI_W)
