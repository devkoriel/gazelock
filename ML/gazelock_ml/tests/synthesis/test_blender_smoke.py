"""End-to-end Blender integration smoke test.

Runs a 50-frame render and verifies output shape. Skipped if Blender is
not installed locally or if HDRI assets have not been fetched. (Eye
geometry is procedural — no eye assets needed.)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from shutil import which

import pytest

from gazelock_ml.synthesis.blender_eyes import BlenderEyeSource

REPO_ROOT = Path(__file__).resolve().parents[4]
HDRI_MANIFEST = (
    REPO_ROOT
    / "ML"
    / "gazelock_ml"
    / "synthesis"
    / "blender"
    / "assets"
    / "hdri"
    / "MANIFEST.json"
)
RENDER_SCRIPT = (
    REPO_ROOT
    / "ML"
    / "gazelock_ml"
    / "synthesis"
    / "blender"
    / "render_eyes.py"
)


def _blender_available() -> bool:
    return which("blender") is not None


def _hdris_present() -> bool:
    if not HDRI_MANIFEST.exists():
        return False
    manifest = json.loads(HDRI_MANIFEST.read_text())
    assets = manifest.get("assets", [])
    if not assets:
        return False
    for asset in assets:
        path = HDRI_MANIFEST.parent / asset["filename"]
        if not path.exists():
            return False
    return True


@pytest.mark.skipif(not _blender_available(), reason="blender CLI not installed")
@pytest.mark.skipif(not _hdris_present(), reason="CC0 HDRI assets not fetched")
def test_blender_smoke_50_frames(tmp_path: Path) -> None:
    cmd = [
        "blender",
        "--background",
        "--python",
        str(RENDER_SCRIPT),
        "--",
        "--count",
        "50",
        "--output",
        str(tmp_path / "smoke-out"),
        "--resolution",
        "128",
        "--samples",
        "16",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"Blender failed:\n{result.stderr}"

    src = BlenderEyeSource(tmp_path / "smoke-out")
    assert len(src) >= 1, "Smoke render produced 0 frames"

    patches = list(src)
    assert len(patches) >= 1
    for p in patches:
        assert p.shape[0] > 0 and p.shape[1] > 0
