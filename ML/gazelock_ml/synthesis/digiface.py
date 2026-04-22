"""DigiFace-1M loader (MIT-licensed synthetic faces).

Expects the official DigiFace-1M release layout:

    <root>/
        images/
            <id>/<img>.jpg
        landmarks2d.txt   # <img_path> x0 y0 x1 y1 ... x67 y67

Crops one random eye per face; yields EYE_ROI patches.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from gazelock_ml.synthesis.base import SourceInfo, SyntheticFaceSource, _validate_patch
from gazelock_ml.synthesis.eye_crop import Eye, crop_eye


class DigiFaceSource(SyntheticFaceSource):
    def __init__(
        self,
        root: Path,
        *,
        rng_seed: int = 0,
        max_samples: int | None = None,
    ) -> None:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"DigiFace root does not exist: {root}")
        manifest = root / "landmarks2d.txt"
        if not manifest.exists():
            raise FileNotFoundError(
                f"DigiFace manifest missing (expected {manifest}). "
                "Download from https://github.com/microsoft/DigiFace1M"
            )

        self._root = root
        self._rng_seed = rng_seed
        self._entries = self._parse_manifest(manifest, limit=max_samples)
        self._info = SourceInfo(
            name="digiface",
            root=root,
            sample_count=len(self._entries),
        )

    @staticmethod
    def _parse_manifest(manifest: Path, limit: int | None) -> list[tuple[str, np.ndarray]]:
        entries: list[tuple[str, np.ndarray]] = []
        with manifest.open() as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts:
                    continue
                path = parts[0]
                coords = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
                if coords.size != 136:
                    continue
                entries.append((path, coords.reshape(68, 2)))
                if limit is not None and len(entries) >= limit:
                    break
        if not entries:
            raise ValueError(f"DigiFace manifest {manifest} produced 0 valid entries")
        return entries

    @property
    def info(self) -> SourceInfo:
        return self._info

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._rng_seed)
        for rel_path, landmarks in self._entries:
            img_path = self._root / "images" / rel_path
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            eye = Eye.LEFT if rng.random() < 0.5 else Eye.RIGHT
            try:
                patch = crop_eye(img, landmarks, eye)
            except ValueError:
                continue
            _validate_patch(patch)
            yield patch


__all__ = ["DigiFaceSource"]
