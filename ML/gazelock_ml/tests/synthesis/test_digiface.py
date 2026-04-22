"""DigiFaceSource tests using procedural fixtures.

The fixture ships a pre-populated .landmarks_cache.jsonl so tests never
invoke face-alignment. All DigiFaceSource instances use
build_cache_if_needed=False.
"""

from pathlib import Path

import pytest

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.synthesis.digiface import DigiFaceSource

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "digiface_mini"


def test_digiface_loader_finds_fixtures() -> None:
    src = DigiFaceSource(FIXTURE_ROOT, build_cache_if_needed=False)
    assert src.info.name == "digiface"
    assert len(src) == 6  # 3 subjects x 2 images each


def test_digiface_loader_yields_eye_patches() -> None:
    src = DigiFaceSource(FIXTURE_ROOT, build_cache_if_needed=False)
    patches = list(src)
    assert len(patches) == 6
    for p in patches:
        assert p.shape == (EYE_ROI_H, EYE_ROI_W, 3)
        assert p.dtype.name == "uint8"


def test_digiface_missing_root_raises() -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        DigiFaceSource(Path("/nonexistent/path/to/digiface"))


def test_digiface_no_images_raises(tmp_path: Path) -> None:
    # Empty directory — no <id>/<n>.png files
    with pytest.raises(FileNotFoundError, match="No DigiFace-1M images"):
        DigiFaceSource(tmp_path, build_cache_if_needed=False)


def test_digiface_max_samples_limits_iteration() -> None:
    src = DigiFaceSource(FIXTURE_ROOT, max_samples=4, build_cache_if_needed=False)
    assert len(src) == 4
    patches = list(src)
    assert len(patches) == 4
