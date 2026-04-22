"""Blender-render consumer.

Reads the manifest.jsonl written by render_eyes.py and yields
eye patches at the EYE_ROI_W x EYE_ROI_H canonical size.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.synthesis.base import SourceInfo, SyntheticFaceSource, _validate_patch


class BlenderEyeSource(SyntheticFaceSource):
    def __init__(self, root: Path, *, max_samples: int | None = None) -> None:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(
                f"Blender-eyes root does not exist: {root}. "
                "Run `make ml-render-smoke` to populate."
            )
        manifest_path = root / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest missing at {manifest_path}. "
                "Did the render pass complete?"
            )
        entries = [
            json.loads(line)
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        ]
        if max_samples is not None:
            entries = entries[:max_samples]
        if not entries:
            raise ValueError(f"Manifest {manifest_path} produced 0 entries")

        self._root = root
        self._entries = entries
        self._info = SourceInfo(
            name="blender_eye",
            root=root,
            sample_count=len(entries),
        )

    @property
    def info(self) -> SourceInfo:
        return self._info

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[np.ndarray]:
        for entry in self._entries:
            png_path = self._root / entry["png_path"]
            img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            resized = cv2.resize(
                img, (EYE_ROI_W, EYE_ROI_H), interpolation=cv2.INTER_AREA
            )
            if resized.dtype != np.uint8:
                resized = resized.astype(np.uint8)
            _validate_patch(resized)
            yield resized


__all__ = ["BlenderEyeSource"]
