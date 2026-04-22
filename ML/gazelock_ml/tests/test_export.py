"""Smoke test for Core ML export — 16-sample validation gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from gazelock_ml.export.coreml import export
from gazelock_ml.models.refiner import RefinerUNet


@pytest.mark.slow
def test_export_runs_and_passes_linf_gate(tmp_path: Path) -> None:
    model = RefinerUNet()
    model.eval()
    out_path = tmp_path / "refiner.mlpackage"
    result = export(
        model,
        out_path,
        num_validation_samples=16,  # Fast smoke — full export uses 1000
        tolerance=1e-3,
    )
    assert out_path.exists()
    assert result["linf"] <= 1e-3
