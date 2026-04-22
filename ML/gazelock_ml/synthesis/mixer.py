"""Weighted mixer over multiple SyntheticFaceSource instances."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from gazelock_ml.synthesis.base import SyntheticFaceSource, _validate_patch


class SourceMixer:
    """Samples patches across N sources by fixed weights.

    Pre-materialises each source's patches on first iteration.
    Each ``__iter__`` returns a new infinite iterator drawing with
    replacement according to ``weights``.
    """

    def __init__(
        self,
        sources: list[SyntheticFaceSource],
        weights: dict[str, float] | None = None,
        seed: int = 0,
    ) -> None:
        if not sources:
            raise ValueError("SourceMixer requires at least 1 source")

        names = [s.info.name for s in sources]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate source names: {names}")

        weights = weights or {name: 1.0 / len(sources) for name in names}

        missing = set(names) - set(weights)
        extra = set(weights) - set(names)
        if missing or extra:
            raise ValueError(
                f"Weight/source name mismatch. missing={missing} extra={extra}"
            )

        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to > 0")

        self._sources = sources
        self._names = names
        self._probs = np.array([weights[n] / total for n in names], dtype=np.float64)
        self._seed = seed
        self._materialised: dict[str, list[np.ndarray]] | None = None

    def _materialise(self) -> None:
        self._materialised = {s.info.name: list(s) for s in self._sources}
        for name, patches in self._materialised.items():
            if not patches:
                raise ValueError(f"Source '{name}' produced 0 patches")

    @property
    def weights(self) -> dict[str, float]:
        return dict(zip(self._names, self._probs.tolist(), strict=True))

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._materialised is None:
            self._materialise()
        assert self._materialised is not None
        rng = np.random.default_rng(self._seed)
        n_sources = len(self._sources)
        while True:
            src_idx = int(rng.choice(n_sources, p=self._probs))
            name = self._names[src_idx]
            patches = self._materialised[name]
            patch_idx = int(rng.integers(0, len(patches)))
            patch = patches[patch_idx]
            _validate_patch(patch)
            yield patch


__all__ = ["SourceMixer"]
