"""Peak Signal-to-Noise Ratio."""

from __future__ import annotations

import math

import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 100.0  # effectively infinite
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


__all__ = ["psnr"]
