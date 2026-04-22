"""Full-face -> eye-ROI extraction via landmarks.

DigiFace-1M and FaceSynthetics provide full-face images with 2D
landmark annotations. Our training pipeline consumes fixed-size eye
patches. This module does the crop + resize.
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W


class Eye(Enum):
    """Which eye to crop."""

    LEFT = "left"
    RIGHT = "right"


_IBUG_LEFT_EYE_IDX = list(range(36, 42))
_IBUG_RIGHT_EYE_IDX = list(range(42, 48))


def _landmark_indices(eye: Eye) -> list[int]:
    return _IBUG_LEFT_EYE_IDX if eye == Eye.LEFT else _IBUG_RIGHT_EYE_IDX


def _eye_bbox(
    landmarks_2d: np.ndarray,
    eye: Eye,
    margin_ratio: float = 0.4,
) -> tuple[int, int, int, int]:
    idx = _landmark_indices(eye)
    pts = landmarks_2d[idx]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    w = x_max - x_min
    h = y_max - y_min
    mx = w * margin_ratio
    my = h * margin_ratio
    target_ratio = EYE_ROI_W / EYE_ROI_H
    bbox_w = w + 2 * mx
    bbox_h = h + 2 * my
    if bbox_w / bbox_h < target_ratio:
        bbox_w = bbox_h * target_ratio
    else:
        bbox_h = bbox_w / target_ratio
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    x0 = round(cx - bbox_w / 2)
    y0 = round(cy - bbox_h / 2)
    x1 = round(cx + bbox_w / 2)
    y1 = round(cy + bbox_h / 2)
    return x0, y0, x1, y1


def crop_eye(
    face_bgr: np.ndarray,
    landmarks_2d: np.ndarray,
    eye: Eye,
) -> np.ndarray:
    """Crop + resize the given eye to (EYE_ROI_H, EYE_ROI_W, 3) uint8 BGR."""
    if face_bgr.dtype != np.uint8:
        raise ValueError(f"face_bgr dtype must be uint8; got {face_bgr.dtype}")
    if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
        raise ValueError(f"face_bgr shape must be (H, W, 3); got {face_bgr.shape}")
    if landmarks_2d.shape != (68, 2):
        raise ValueError(
            f"landmarks_2d shape must be (68, 2); got {landmarks_2d.shape}"
        )

    h, w = face_bgr.shape[:2]
    x0, y0, x1, y1 = _eye_bbox(landmarks_2d, eye)

    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(w, x1)
    y1c = min(h, y1)

    cropped = face_bgr[y0c:y1c, x0c:x1c]

    pad_top = y0c - y0
    pad_bottom = y1 - y1c
    pad_left = x0c - x0
    pad_right = x1 - x1c
    if any(v > 0 for v in (pad_top, pad_bottom, pad_left, pad_right)):
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
        )

    resized = cv2.resize(cropped, (EYE_ROI_W, EYE_ROI_H), interpolation=cv2.INTER_AREA)
    if resized.dtype != np.uint8:
        resized = resized.astype(np.uint8)
    return resized


__all__ = ["Eye", "crop_eye"]
