"""FaceSyntheticsSource tests."""

from pathlib import Path

import pytest

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.synthesis.facesynthetics import FaceSyntheticsSource

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "facesynthetics_mini"


def test_facesynthetics_loader_finds_fixtures() -> None:
    src = FaceSyntheticsSource(FIXTURE_ROOT)
    assert src.info.name == "facesynthetics"
    assert len(src) == 3


def test_facesynthetics_yields_eye_patches() -> None:
    src = FaceSyntheticsSource(FIXTURE_ROOT)
    patches = list(src)
    assert len(patches) == 3
    for p in patches:
        assert p.shape == (EYE_ROI_H, EYE_ROI_W, 3)
        assert p.dtype.name == "uint8"


def test_facesynthetics_missing_root_raises() -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        FaceSyntheticsSource(Path("/nonexistent/fs"))


def test_facesynthetics_empty_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No FaceSynthetics"):
        FaceSyntheticsSource(tmp_path)


def test_facesynthetics_max_samples() -> None:
    src = FaceSyntheticsSource(FIXTURE_ROOT, max_samples=2)
    assert len(src) == 2
