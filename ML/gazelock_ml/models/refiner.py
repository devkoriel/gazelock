"""Seam-hider refiner UNet.

Per design spec §6.5:
    - Input  : 6-channel tensor (warped-eye RGB ++ original-eye RGB)
    - Output : 3-channel refined eye RGB
    - Target : ~120 k parameters, fp16-deployable to Core ML / Neural Engine
    - Purpose: inpaint the ~5 % disoccluded sclera strip and hide
               warp-induced seams. *Not* a full eye generator.
"""

from __future__ import annotations

import torch
from torch import nn


def _depthwise_separable(in_ch: int, out_ch: int) -> nn.Sequential:
    """Depthwise + pointwise conv — the building block of our UNet."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _depthwise_separable(in_ch, out_ch)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        out = self.down(skip)
        return out, skip


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _depthwise_separable(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RefinerUNet(nn.Module):
    """Seam-hider refiner. See module docstring for spec references."""

    def __init__(self, in_channels: int = 6, out_channels: int = 3) -> None:
        super().__init__()
        self.stem = _depthwise_separable(in_channels, 32)
        self.enc1 = _EncoderBlock(32, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.bottleneck = _depthwise_separable(128, 128)
        self.dec2 = _DecoderBlock(in_ch=128, skip_ch=128, out_ch=64)
        self.dec1 = _DecoderBlock(in_ch=64, skip_ch=64, out_ch=32)
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        e1, s1 = self.enc1(stem)
        e2, s2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(b, s2)
        d1 = self.dec1(d2, s1)
        return torch.sigmoid(self.head(d1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["RefinerUNet", "count_parameters"]
