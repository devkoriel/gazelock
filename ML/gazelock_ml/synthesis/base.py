"""SyntheticFaceSource abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W


@dataclass(frozen=True)
class SourceInfo:
    """Metadata describing a source instance."""

    name: str
    root: Path
    sample_count: int


class SyntheticFaceSource(ABC):
    """Iterable producer of (EYE_ROI_H, EYE_ROI_W, 3) uint8 BGR patches."""

    @property
    @abstractmethod
    def info(self) -> SourceInfo:
        """Lightweight metadata; does not load samples."""

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield eye patches."""

    @abstractmethod
    def __len__(self) -> int:
        """Total discrete samples available."""

    def sample(self, n: int, rng: np.random.Generator | None = None) -> Iterable[np.ndarray]:
        """Yield n patches drawn with replacement."""
        rng = rng or np.random.default_rng()
        samples = list(self)
        if not samples:
            raise ValueError(f"Source '{self.info.name}' produced 0 samples")
        indices = rng.integers(0, len(samples), size=n)
        for i in indices:
            yield samples[int(i)]


def _validate_patch(patch: np.ndarray) -> None:
    """Contract check — all sources emit patches of this shape + dtype."""
    if patch.shape != (EYE_ROI_H, EYE_ROI_W, 3):
        raise ValueError(
            f"Expected shape ({EYE_ROI_H}, {EYE_ROI_W}, 3); got {patch.shape}"
        )
    if patch.dtype != np.uint8:
        raise ValueError(f"Expected dtype uint8; got {patch.dtype}")


__all__ = ["SyntheticFaceSource", "SourceInfo", "_validate_patch"]
