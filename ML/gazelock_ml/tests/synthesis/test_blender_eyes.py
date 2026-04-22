"""BlenderEyeSource tests."""

from pathlib import Path

import pytest

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.synthesis.blender_eyes import BlenderEyeSource

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "blender_eyes_mini"


def test_blender_eyes_loader_finds_fixture() -> None:
    src = BlenderEyeSource(FIXTURE_ROOT)
    assert src.info.name == "blender_eye"
    assert len(src) == 3


def test_blender_eyes_yields_patches() -> None:
    src = BlenderEyeSource(FIXTURE_ROOT)
    patches = list(src)
    assert len(patches) == 3
    for p in patches:
        assert p.shape == (EYE_ROI_H, EYE_ROI_W, 3)
        assert p.dtype.name == "uint8"


def test_blender_eyes_missing_root_raises() -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        BlenderEyeSource(Path("/nonexistent/blender"))


def test_blender_eyes_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Manifest missing"):
        BlenderEyeSource(tmp_path)


def test_blender_eyes_max_samples() -> None:
    src = BlenderEyeSource(FIXTURE_ROOT, max_samples=2)
    assert len(src) == 2
