"""Small CNN that embeds eye crops into an identity-preserving space.

The encoder is trained separately (Phase 2b) on FFHQ eye-crop pairs —
positive pairs from the same face, negatives from different faces —
with a contrastive objective. At training time for the refiner, the
encoder is frozen and we penalise cosine distance between the
embedding of the refiner's output and the ground-truth target.

Architecture choices:
    - Small MobileNetV2-style stack (~180 k params)
    - Global-average-pool → 128-dim embedding → L2-normalise
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _inv_residual(in_ch: int, out_ch: int, stride: int, expand: int) -> nn.Module:
    """MobileNetV2 inverted-residual block (no residual when stride>1)."""
    hidden = in_ch * expand
    layers: list[nn.Module] = []
    if expand != 1:
        layers.extend(
            [
                nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        )
    layers.extend(
        [
            nn.Conv2d(
                hidden, hidden, kernel_size=3, stride=stride, padding=1,
                groups=hidden, bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
    )

    block = nn.Sequential(*layers)

    class _Wrap(nn.Module):
        def __init__(self, inner: nn.Sequential, use_residual: bool) -> None:
            super().__init__()
            self.inner = inner
            self.use_residual = use_residual

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.inner(x)
            if self.use_residual:
                return x + out
            return out

    return _Wrap(block, use_residual=(stride == 1 and in_ch == out_ch))


class EyeIdentityEncoder(nn.Module):
    EMBEDDING_DIM = 128

    def __init__(self) -> None:
        super().__init__()
        self.stem = _conv_bn_relu(3, 32, stride=2)
        self.block1 = _inv_residual(32, 48, stride=2, expand=6)
        self.block2 = _inv_residual(48, 64, stride=2, expand=6)
        self.block3 = _inv_residual(64, 96, stride=1, expand=6)
        self.block4 = _inv_residual(96, 128, stride=1, expand=6)
        self.head = nn.Conv2d(128, self.EMBEDDING_DIM, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        return F.normalize(x, p=2, dim=1)


__all__ = ["EyeIdentityEncoder"]
