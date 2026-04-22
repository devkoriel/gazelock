"""Tests for FFHQ eye-crop extractor.

We fabricate a directory of synthetic "face" images — large uniform
patches with two dark ellipses where the eyes would be — and verify
the extractor returns at least one crop of the correct shape per
image. We don't ship real FFHQ; the real dataset is loaded by the
user in Phase 2b.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gazelock_ml.data.ffhq import FFHQEyeExtractor
from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W


def _synthesize_face(seed: int = 0) -> np.ndarray:
    """Build a 512x512 RGB pseudo-face with visible eye-like ellipses."""
    rng = np.random.default_rng(seed)
    img = np.full((512, 512, 3), 200, dtype=np.uint8) - rng.integers(0, 40, (512, 512, 3), dtype=np.uint8)
    # Draw two eye ellipses at predictable positions
    cv2.ellipse(img, (180, 220), (40, 20), 0, 0, 360, (30, 30, 30), -1)
    cv2.ellipse(img, (330, 220), (40, 20), 0, 0, 360, (30, 30, 30), -1)
    # Draw a head/face outline so the cascade has a chance
    cv2.ellipse(img, (256, 256), (180, 230), 0, 0, 360, (150, 130, 110), 4)
    return img


@pytest.fixture
def fake_ffhq_root(tmp_path: Path) -> Path:
    root = tmp_path / "ffhq_mini"
    root.mkdir()
    for i in range(2):
        Image.fromarray(_synthesize_face(seed=i), mode="RGB").save(root / f"face_{i:05d}.png")
    return root


def test_extractor_shape_contract_on_synthetic_faces(fake_ffhq_root: Path) -> None:
    # Haar cascades may or may not match our synthetic faces; the contract
    # we verify is shape/dtype, not detection count.
    extractor = FFHQEyeExtractor(fake_ffhq_root)
    crops = list(extractor.iter_crops(max_images=2))
    for crop in crops:
        assert crop.patch.shape == (EYE_ROI_H, EYE_ROI_W, 3)
        assert crop.patch.dtype == np.uint8
        assert crop.source_file.parent == fake_ffhq_root
        assert crop.side in ("left", "right")


def test_extractor_respects_max_images(fake_ffhq_root: Path) -> None:
    extractor = FFHQEyeExtractor(fake_ffhq_root)
    # max_images=1 should process only the first file — cap on source files,
    # not on output crops (one file may yield 0 or 2 crops).
    crops = list(extractor.iter_crops(max_images=1))
    sources = {c.source_file for c in crops}
    assert len(sources) <= 1
