"""Frame-to-frame temporal consistency loss.

Given two output tensors ``pred_a`` and ``pred_b`` produced from
synthetic near-neighbour inputs (same identity, tiny gaze perturbation),
penalise their difference so the refiner doesn't introduce flicker
across adjacent frames.
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalConsistencyLoss(nn.Module):
    """Mean L1 between two predicted outputs from near-neighbour inputs."""

    def forward(self, pred_a: torch.Tensor, pred_b: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred_a - pred_b))


__all__ = ["TemporalConsistencyLoss"]
