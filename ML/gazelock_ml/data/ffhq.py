"""FFHQ eye-crop extractor.

FFHQ (Karras et al. 2019) ships as a directory of square RGB face
images at various resolutions. We crop a fixed-size eye ROI (96x72)  # noqa: RUF002
centered on the iris using a lightweight landmark detector.

For reproducibility we use OpenCV's built-in Haar cascade as a fallback
when no stronger detector is available. In Phase 2b a dedicated face-
landmark model (e.g., 68-point MediaPipe or a compact ONNX model) can
replace the Haar step; the extract_eye_rois signature stays the same.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W


@dataclass(frozen=True)
class EyeCrop:
    """A single eye ROI cropped from an FFHQ face."""

    patch: np.ndarray  # (EYE_ROI_H, EYE_ROI_W, 3) uint8 RGB
    source_file: Path
    side: str  # "left" or "right" (w.r.t. the subject)
    roi_center_in_source: tuple[int, int]  # (x, y) in source image coords


class FFHQEyeExtractor:
    """Given a directory of FFHQ images, yield cropped eye ROIs."""

    def __init__(
        self,
        image_root: Path,
        face_cascade: cv2.CascadeClassifier | None = None,
        eye_cascade: cv2.CascadeClassifier | None = None,
    ) -> None:
        self._image_root = Path(image_root)
        self._face_cascade = face_cascade or _load_builtin_cascade(
            "haarcascade_frontalface_default.xml"
        )
        self._eye_cascade = eye_cascade or _load_builtin_cascade(
            "haarcascade_eye.xml"
        )

    def iter_crops(
        self,
        max_images: int | None = None,
    ) -> Iterator[EyeCrop]:
        images = sorted(self._image_root.glob("*.png")) + sorted(
            self._image_root.glob("*.jpg")
        )
        if max_images is not None:
            images = images[:max_images]
        for img_path in images:
            yield from self._extract_from_file(img_path)

    def _extract_from_file(self, img_path: Path) -> Iterator[EyeCrop]:
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 5)
        for fx, fy, fw, fh in faces:
            face_gray = gray[fy : fy + fh, fx : fx + fw]
            eyes = self._eye_cascade.detectMultiScale(face_gray, 1.1, 5)
            eyes = sorted(eyes, key=lambda e: e[0])  # sort by x ascending
            for idx, (ex, ey, ew, eh) in enumerate(eyes):
                cx = fx + ex + ew // 2
                cy = fy + ey + eh // 2
                patch = _crop_centered(rgb, cx, cy, EYE_ROI_W, EYE_ROI_H)
                if patch is None:
                    continue
                side = "right" if idx == 0 else "left"  # image right = subject's left
                yield EyeCrop(
                    patch=patch,
                    source_file=img_path,
                    side=side,
                    roi_center_in_source=(cx, cy),
                )


def _crop_centered(
    image: np.ndarray,
    cx: int,
    cy: int,
    w: int,
    h: int,
) -> np.ndarray | None:
    x0 = cx - w // 2
    y0 = cy - h // 2
    x1 = x0 + w
    y1 = y0 + h
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        return None
    return image[y0:y1, x0:x1].copy()


def _load_builtin_cascade(name: str) -> cv2.CascadeClassifier:
    data_dir = Path(cv2.data.haarcascades)  # type: ignore[attr-defined]
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"OpenCV cascade not found: {path}")
    return cv2.CascadeClassifier(str(path))


__all__ = ["EyeCrop", "FFHQEyeExtractor"]
