"""crop_eye correctness tests."""

import numpy as np
import pytest

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.synthesis.eye_crop import Eye, crop_eye


def _face_with_marker(h: int = 512, w: int = 512, colour: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[h // 3 : h // 2, w // 4 : w // 3] = colour
    return img


def _canonical_landmarks(h: int = 512, w: int = 512) -> np.ndarray:
    lm = np.zeros((68, 2), dtype=np.float32)
    for i, idx in enumerate(range(36, 42)):
        lm[idx] = [w // 4 + 5 + i * 4, h // 3 + 10 + (i % 2) * 5]
    for i, idx in enumerate(range(42, 48)):
        lm[idx] = [3 * w // 4 - 25 + i * 4, h // 3 + 10 + (i % 2) * 5]
    return lm


def test_crop_eye_output_shape_and_dtype() -> None:
    face = _face_with_marker()
    lm = _canonical_landmarks()
    roi = crop_eye(face, lm, Eye.LEFT)
    assert roi.shape == (EYE_ROI_H, EYE_ROI_W, 3)
    assert roi.dtype == np.uint8


def test_crop_eye_left_contains_marker_region() -> None:
    face = _face_with_marker(colour=(0, 255, 0))
    lm = _canonical_landmarks()
    roi = crop_eye(face, lm, Eye.LEFT)
    g_mean = roi[:, :, 1].mean()
    r_mean = roi[:, :, 0].mean()
    assert g_mean > r_mean


def test_crop_eye_rejects_wrong_dtype() -> None:
    face = np.zeros((128, 128, 3), dtype=np.float32)
    lm = _canonical_landmarks(128, 128)
    with pytest.raises(ValueError, match="uint8"):
        crop_eye(face, lm, Eye.LEFT)


def test_crop_eye_rejects_wrong_landmark_shape() -> None:
    face = _face_with_marker()
    lm = np.zeros((40, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(68, 2\)"):
        crop_eye(face, lm, Eye.LEFT)


def test_crop_eye_rejects_grayscale_face() -> None:
    face = np.zeros((128, 128), dtype=np.uint8)
    lm = _canonical_landmarks(128, 128)
    with pytest.raises(ValueError, match="shape"):
        crop_eye(face, lm, Eye.LEFT)


def test_crop_eye_handles_near_edge() -> None:
    face = np.full((128, 128, 3), 127, dtype=np.uint8)
    lm = np.zeros((68, 2), dtype=np.float32)
    for i, idx in enumerate(range(36, 42)):
        lm[idx] = [2 + i * 2, 5 + (i % 2) * 3]
    roi = crop_eye(face, lm, Eye.LEFT)
    assert roi.shape == (EYE_ROI_H, EYE_ROI_W, 3)
