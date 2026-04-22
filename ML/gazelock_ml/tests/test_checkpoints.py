"""Round-trip checkpoint save/load tests."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW

from gazelock_ml.models.refiner import RefinerUNet
from gazelock_ml.training.checkpoints import (
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
    write_run_manifest,
)


def test_roundtrip_preserves_weights(tmp_path: Path) -> None:
    model = RefinerUNet()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    torch.manual_seed(0)
    state_before = {k: v.clone() for k, v in model.state_dict().items()}

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        metadata=CheckpointMetadata(step=42, loss_total=1.5, git_sha="abc", notes="x"),
    )

    new_model = RefinerUNet()
    new_opt = AdamW(new_model.parameters(), lr=1e-4)
    md = load_checkpoint(ckpt_path, new_model, new_opt)

    assert md.step == 42
    assert md.loss_total == 1.5
    assert md.git_sha == "abc"
    for k, v in state_before.items():
        torch.testing.assert_close(new_model.state_dict()[k], v)


def test_manifest_writes_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_run_manifest(manifest_path, {"step": 1, "loss": 1.2})
    import json
    data = json.loads(manifest_path.read_text())
    assert data == {"step": 1, "loss": 1.2}
