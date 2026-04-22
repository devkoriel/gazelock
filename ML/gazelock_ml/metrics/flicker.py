"""Frame-to-frame L1 — a flicker proxy for video sequences."""

from __future__ import annotations

import torch


def frame_to_frame_l1(sequence: torch.Tensor) -> float:
    """Given (N, 3, H, W), compute mean |frame_i - frame_{i-1}| over N-1 pairs."""
    assert sequence.dim() == 4, "expected (N, 3, H, W)"
    if sequence.shape[0] < 2:
        return 0.0
    diffs = torch.abs(sequence[1:] - sequence[:-1])
    return float(diffs.mean().item())


__all__ = ["frame_to_frame_l1"]
