"""Simplified SSIM — 11x11 Gaussian window, per-channel mean."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

_K1 = 0.01
_K2 = 0.03


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)  # (size, size)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    window_size: int = 11,
) -> float:
    """Compute mean SSIM over batch + channels. Inputs in [0, max_val]."""
    assert pred.shape == target.shape
    channels = pred.shape[1]
    window = _gaussian_window(window_size).to(pred.device).to(pred.dtype)
    window = window.expand(channels, 1, window_size, window_size).contiguous()

    c1 = (_K1 * max_val) ** 2
    c2 = (_K2 * max_val) ** 2

    mu_x = functional.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu_y = functional.conv2d(target, window, padding=window_size // 2, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = functional.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu_x2
    sigma_y2 = functional.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu_y2
    sigma_xy = functional.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(torch.mean(num / den).item())


__all__ = ["ssim"]
