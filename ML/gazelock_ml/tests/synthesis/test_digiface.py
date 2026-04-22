"""DigiFaceSource tests using procedural fixtures."""

from pathlib import Path

import pytest

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.synthesis.digiface import DigiFaceSource

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "digiface_mini"


def test_digiface_loader_finds_fixtures() -> None:
    src = DigiFaceSource(FIXTURE_ROOT)
    assert src.info.name == "digiface"
    assert len(src) == 3


def test_digiface_loader_yields_eye_patches() -> None:
    src = DigiFaceSource(FIXTURE_ROOT)
    patches = list(src)
    assert len(patches) == 3
    for p in patches:
        assert p.shape == (EYE_ROI_H, EYE_ROI_W, 3)
        assert p.dtype.name == "uint8"


def test_digiface_missing_root_raises() -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        DigiFaceSource(Path("/nonexistent/path/to/digiface"))


def test_digiface_missing_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError, match="manifest missing"):
        DigiFaceSource(tmp_path)


def test_digiface_max_samples_limits_iteration() -> None:
    src = DigiFaceSource(FIXTURE_ROOT, max_samples=2)
    assert len(src) == 2
    patches = list(src)
    assert len(patches) == 2
