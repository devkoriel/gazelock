"""Tests for procedural fixture generator."""

from __future__ import annotations

import numpy as np

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W, make_fake_eye_pair, make_fake_eye_patch


def test_patch_shape_and_dtype() -> None:
    patch = make_fake_eye_patch(seed=0)
    assert patch.shape == (EYE_ROI_H, EYE_ROI_W, 3)
    assert patch.dtype == np.uint8


def test_patch_is_deterministic_per_seed() -> None:
    p0 = make_fake_eye_patch(seed=0)
    p1 = make_fake_eye_patch(seed=0)
    p2 = make_fake_eye_patch(seed=1)
    np.testing.assert_array_equal(p0, p1)
    assert not np.array_equal(p0, p2)


def test_pair_differ_in_iris_region_only_approx() -> None:
    source, target = make_fake_eye_pair(seed=0, target_offset_px=(3, 0))
    assert source.shape == target.shape
    # Corners (background only) should be identical
    assert np.array_equal(source[:5, :5], target[:5, :5])
    assert np.array_equal(source[-5:, -5:], target[-5:, -5:])
    # Iris region should differ
    h_start = EYE_ROI_H // 2 - 10
    w_start = EYE_ROI_W // 2 - 10
    src_iris = source[h_start : h_start + 20, w_start : w_start + 20]
    tgt_iris = target[h_start : h_start + 20, w_start : w_start + 20]
    assert not np.array_equal(src_iris, tgt_iris)
