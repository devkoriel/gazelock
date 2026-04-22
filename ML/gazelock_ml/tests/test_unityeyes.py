"""Tests for UnityEyes loader — run against a synthetic mini-dataset.

We don't ship any real UnityEyes renders (they're multi-GB and
licensed for academic use only). This test fabricates the exact
on-disk layout a Unity render produces and verifies the loader reads
it correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gazelock_ml.data.unityeyes import UnityEyesDataset


@pytest.fixture
def synthetic_unityeyes_root(tmp_path: Path) -> Path:
    root = tmp_path / "unityeyes_mini"
    ident = root / "identity_0001"
    ident.mkdir(parents=True)

    img_arr = (np.random.default_rng(0).integers(0, 256, (600, 800, 3), dtype=np.uint8))
    Image.fromarray(img_arr, mode="RGB").save(ident / "0.jpg")

    meta = {
        "eye_details": {"look_vec": "(0.12 -0.08 -0.99)"},
        "head_pose": "(5.0 2.0 -1.0)",
        "interior_margin_2d": ["(100 300)", "(200 310)", "(300 320)"],
        "iris_2d": ["(195 308)", "(205 312)"],
    }
    (ident / "0.json").write_text(json.dumps(meta))
    return root


def test_loader_reads_single_identity(synthetic_unityeyes_root: Path) -> None:
    ds = UnityEyesDataset(synthetic_unityeyes_root)
    assert len(ds) == 1
    assert ds.identity_count == 1

    samples = list(ds.iter_identity(0))
    assert len(samples) == 1
    s = samples[0]
    assert s.image.shape == (600, 800, 3)
    assert s.image.dtype == np.uint8
    assert s.head_pose_deg == (5.0, 2.0, -1.0)
    # gaze yaw = atan2(0.12, 0.99) in degrees ≈ 6.9
    assert 6.0 < s.gaze_angle_deg[0] < 8.0
    # iris center ≈ average of (195,308) and (205,312) = (200, 310)
    assert s.iris_center_px == pytest.approx((200.0, 310.0))


def test_loader_raises_on_missing_root() -> None:
    with pytest.raises(FileNotFoundError):
        UnityEyesDataset(Path("/nonexistent/unityeyes_root"))
