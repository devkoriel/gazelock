"""Tests for PSNR / SSIM / identity / flicker."""

from __future__ import annotations

import torch

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.metrics.flicker import frame_to_frame_l1
from gazelock_ml.metrics.identity import identity_cosine
from gazelock_ml.metrics.psnr import psnr
from gazelock_ml.metrics.ssim import ssim


def test_psnr_infinite_on_identical_inputs() -> None:
    x = torch.rand((1, 3, 8, 8))
    assert psnr(x, x) >= 99.0  # "effectively infinite" sentinel


def test_psnr_finite_and_positive_on_different_inputs() -> None:
    torch.manual_seed(0)
    x = torch.rand((1, 3, 8, 8))
    y = torch.rand((1, 3, 8, 8))
    val = psnr(x, y)
    assert 0.0 < val < 99.0


def test_ssim_one_on_identical_inputs() -> None:
    x = torch.rand((1, 3, 16, 16))
    assert abs(ssim(x, x) - 1.0) < 1e-3


def test_ssim_lower_on_different_inputs() -> None:
    torch.manual_seed(1)
    x = torch.rand((1, 3, 16, 16))
    y = torch.rand((1, 3, 16, 16))
    assert ssim(x, y) < 0.95


def test_identity_cosine_close_to_one_on_identical_inputs() -> None:
    x = torch.rand((2, 3, EYE_ROI_H, EYE_ROI_W))
    assert identity_cosine(x, x) > 0.999


def test_flicker_zero_on_constant_sequence() -> None:
    frame = torch.rand((3, 8, 8))
    seq = frame.unsqueeze(0).expand(5, -1, -1, -1)
    assert frame_to_frame_l1(seq) == 0.0


def test_flicker_positive_on_varying_sequence() -> None:
    torch.manual_seed(2)
    seq = torch.rand((5, 3, 8, 8))
    assert frame_to_frame_l1(seq) > 0.0
