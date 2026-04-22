"""Weighted composition of L1 + VGG + identity + temporal.

Weights per design spec §7.2:
    L1 reconstruction : 1.0
    VGG perceptual    : 0.2
    Identity          : 0.5
    Temporal          : 0.1
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gazelock_ml.losses.identity import IdentityLoss
from gazelock_ml.losses.perceptual import VGGPerceptualLoss
from gazelock_ml.losses.reconstruction import L1ReconstructionLoss
from gazelock_ml.losses.temporal import TemporalConsistencyLoss


@dataclass(frozen=True)
class LossWeights:
    l1: float = 1.0
    perceptual: float = 0.2
    identity: float = 0.5
    temporal: float = 0.1


class ComposedLoss(nn.Module):
    """Weighted sum of the four loss components.

    Usage:
        loss = composed(
            pred=refiner_output,
            target=ground_truth,
            pred_neighbour=refiner_output_on_shifted_input,  # optional
        )
    """

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.l1 = L1ReconstructionLoss()
        self.perceptual = VGGPerceptualLoss()
        self.identity = IdentityLoss()
        self.temporal = TemporalConsistencyLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_neighbour: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a dict with per-component + total scalars."""
        l_l1 = self.l1(pred, target)
        l_perc = self.perceptual(pred, target)
        l_id = self.identity(pred, target)

        if pred_neighbour is not None:
            l_temp = self.temporal(pred, pred_neighbour)
        else:
            l_temp = torch.zeros((), device=pred.device, dtype=pred.dtype)

        total = (
            self.weights.l1 * l_l1
            + self.weights.perceptual * l_perc
            + self.weights.identity * l_id
            + self.weights.temporal * l_temp
        )

        return {
            "l1": l_l1.detach(),
            "perceptual": l_perc.detach(),
            "identity": l_id.detach(),
            "temporal": l_temp.detach(),
            "total": total,  # the gradient source
        }


__all__ = ["ComposedLoss", "LossWeights"]
