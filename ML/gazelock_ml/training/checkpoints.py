"""Checkpoint save/load — model + optimizer state + step + metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class CheckpointMetadata:
    step: int
    loss_total: float
    git_sha: str | None = None
    notes: str = ""


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    metadata: CheckpointMetadata,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": asdict(metadata),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> CheckpointMetadata:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    md = payload["metadata"]
    return CheckpointMetadata(**md)


def write_run_manifest(path: Path, data: dict) -> None:
    """Persist a JSON manifest next to the checkpoint (step counts,
    losses, git SHA, etc.)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


__all__ = [
    "CheckpointMetadata",
    "save_checkpoint",
    "load_checkpoint",
    "write_run_manifest",
]
