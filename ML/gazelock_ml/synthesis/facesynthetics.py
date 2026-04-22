"""Microsoft FaceSynthetics loader (MIT-licensed).

Expects the official FaceSynthetics release layout:

    <root>/
        <idx>.png
        <idx>_ldmks.txt     # 70 landmarks (68 iBUG + 2 pupils), one per line
        <idx>_seg.png       # per-pixel seg (unused here)

Consumes only the iBUG 68 (first 68 entries of _ldmks.txt), crops
one random eye.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from gazelock_ml.synthesis.base import SourceInfo, SyntheticFaceSource, _validate_patch
from gazelock_ml.synthesis.eye_crop import Eye, crop_eye


class FaceSyntheticsSource(SyntheticFaceSource):
    def __init__(
        self,
        root: Path,
        *,
        rng_seed: int = 0,
        max_samples: int | None = None,
    ) -> None:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"FaceSynthetics root does not exist: {root}")

        entries = sorted(root.glob("*_ldmks.txt"))
        if not entries:
            raise FileNotFoundError(
                f"No FaceSynthetics landmark files in {root}. Download from "
                "https://github.com/microsoft/FaceSynthetics"
            )
        if max_samples is not None:
            entries = entries[:max_samples]

        self._root = root
        self._entries = entries
        self._rng_seed = rng_seed
        self._info = SourceInfo(
            name="facesynthetics",
            root=root,
            sample_count=len(entries),
        )

    @staticmethod
    def _parse_landmarks(lm_path: Path) -> np.ndarray:
        coords: list[tuple[float, float]] = []
        with lm_path.open() as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 2:
                    coords.append((float(parts[0]), float(parts[1])))
                if len(coords) == 68:
                    break
        if len(coords) != 68:
            raise ValueError(f"{lm_path} has <68 points")
        return np.asarray(coords, dtype=np.float32)

    @property
    def info(self) -> SourceInfo:
        return self._info

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._rng_seed)
        for lm_path in self._entries:
            img_path = lm_path.with_name(lm_path.name.replace("_ldmks.txt", ".png"))
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            try:
                landmarks = self._parse_landmarks(lm_path)
            except ValueError:
                continue
            eye = Eye.LEFT if rng.random() < 0.5 else Eye.RIGHT
            try:
                patch = crop_eye(img, landmarks, eye)
            except ValueError:
                continue
            _validate_patch(patch)
            yield patch


__all__ = ["FaceSyntheticsSource"]
