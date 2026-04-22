"""VGG-16 perceptual feature loss.

We compare predicted vs target tensors at a small set of VGG-16 layer
activations. VGG weights come from torchvision's IMAGENET1K_V1. The
loss is mean MSE over the selected feature-map pairs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import VGG16_Weights, vgg16


_VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_VGG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGPerceptualLoss(nn.Module):
    """Mean MSE over VGG-16 feature-map pairs at layers 3, 8, 15."""

    LAYER_INDICES = (3, 8, 15)

    def __init__(self) -> None:
        super().__init__()
        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_features = features.eval()
        for p in self.vgg_features.parameters():
            p.requires_grad = False

        self.register_buffer("_mean", _VGG_MEAN)
        self.register_buffer("_std", _VGG_STD)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def _extract(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        out = self._normalize(x)
        for i, layer in enumerate(self.vgg_features):
            out = layer(out)
            if i in self.LAYER_INDICES:
                feats.append(out)
            if i >= max(self.LAYER_INDICES):
                break
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feats = self._extract(pred)
        with torch.no_grad():
            tgt_feats = self._extract(target)
        losses = [F.mse_loss(p, t) for p, t in zip(pred_feats, tgt_feats, strict=True)]
        return torch.stack(losses).mean()


__all__ = ["VGGPerceptualLoss"]
