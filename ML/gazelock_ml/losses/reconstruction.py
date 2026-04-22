"""L1 pixel-reconstruction loss."""

from __future__ import annotations

import torch
from torch import nn


class L1ReconstructionLoss(nn.Module):
    """Mean absolute error between predicted and target tensors."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target))


__all__ = ["L1ReconstructionLoss"]
