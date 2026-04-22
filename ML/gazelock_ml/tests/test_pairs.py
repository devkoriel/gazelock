"""Tests for training-pair synthesiser."""

from __future__ import annotations

import numpy as np

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W, make_fake_eye_patch
from gazelock_ml.data.pairs import synthesise_pair


def test_pair_shapes_match_input() -> None:
    patch = make_fake_eye_patch(seed=0)
    pair = synthesise_pair(patch, rng=np.random.default_rng(0))
    assert pair.warped.shape == (EYE_ROI_H, EYE_ROI_W, 3)
    assert pair.original.shape == (EYE_ROI_H, EYE_ROI_W, 3)
    assert pair.warped.dtype == np.uint8
    assert pair.original.dtype == np.uint8


def test_warped_differs_from_original_when_offset_nonzero() -> None:
    patch = make_fake_eye_patch(seed=1)
    rng = np.random.default_rng(42)  # produces a non-zero offset
    pair = synthesise_pair(patch, rng=rng, max_offset_px=8.0)
    assert pair.offset_px != (0.0, 0.0)
    assert not np.array_equal(pair.warped, pair.original)


def test_original_is_unchanged_from_input() -> None:
    patch = make_fake_eye_patch(seed=2)
    pair = synthesise_pair(patch, rng=np.random.default_rng(2))
    np.testing.assert_array_equal(pair.original, patch)


def test_determinism_with_same_rng_seed() -> None:
    patch = make_fake_eye_patch(seed=3)
    p1 = synthesise_pair(patch, rng=np.random.default_rng(7))
    p2 = synthesise_pair(patch, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(p1.warped, p2.warped)
    assert p1.offset_px == p2.offset_px
