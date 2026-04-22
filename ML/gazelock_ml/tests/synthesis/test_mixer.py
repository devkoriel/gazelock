"""SourceMixer correctness + reproducibility tests."""

from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.synthesis.base import SourceInfo, SyntheticFaceSource
from gazelock_ml.synthesis.mixer import SourceMixer


class _TaggedSource(SyntheticFaceSource):
    def __init__(self, name: str, colour: int, n: int = 4) -> None:
        self._name = name
        self._colour = colour
        self._n = n

    @property
    def info(self) -> SourceInfo:
        return SourceInfo(name=self._name, root=Path("/"), sample_count=self._n)

    def __iter__(self):
        for _ in range(self._n):
            yield np.full((EYE_ROI_H, EYE_ROI_W, 3), self._colour, dtype=np.uint8)

    def __len__(self) -> int:
        return self._n


def _identify_source(patch: np.ndarray) -> int:
    return int(patch[0, 0, 0])


def test_mixer_default_weights_equal() -> None:
    a = _TaggedSource("a", 10)
    b = _TaggedSource("b", 20)
    mixer = SourceMixer([a, b], seed=1)
    assert abs(mixer.weights["a"] - 0.5) < 1e-9
    assert abs(mixer.weights["b"] - 0.5) < 1e-9


def test_mixer_weights_honoured_approximately() -> None:
    a = _TaggedSource("a", 10)
    b = _TaggedSource("b", 20)
    mixer = SourceMixer([a, b], weights={"a": 0.8, "b": 0.2}, seed=42)
    drawn = list(islice(mixer, 10_000))
    counts = Counter(_identify_source(p) for p in drawn)
    assert 0.76 < counts[10] / 10_000 < 0.84


def test_mixer_rejects_duplicate_names() -> None:
    a = _TaggedSource("same", 1)
    b = _TaggedSource("same", 2)
    with pytest.raises(ValueError, match="Duplicate source names"):
        SourceMixer([a, b])


def test_mixer_rejects_weight_mismatch() -> None:
    a = _TaggedSource("a", 1)
    with pytest.raises(ValueError, match="mismatch"):
        SourceMixer([a], weights={"b": 1.0})


def test_mixer_rejects_zero_weights() -> None:
    a = _TaggedSource("a", 1)
    with pytest.raises(ValueError, match="sum to > 0"):
        SourceMixer([a], weights={"a": 0.0})


def test_mixer_seed_reproducibility() -> None:
    a = _TaggedSource("a", 10)
    b = _TaggedSource("b", 20)
    m1 = SourceMixer([a, b], seed=7)
    m2 = SourceMixer([a, b], seed=7)
    s1 = [_identify_source(p) for p in islice(m1, 50)]
    s2 = [_identify_source(p) for p in islice(m2, 50)]
    assert s1 == s2


def test_mixer_requires_nonempty_sources() -> None:
    class Empty(SyntheticFaceSource):
        @property
        def info(self) -> SourceInfo:
            return SourceInfo(name="empty", root=Path("/"), sample_count=0)

        def __iter__(self):
            return iter([])

        def __len__(self) -> int:
            return 0

    with pytest.raises(ValueError, match="0 patches"):
        mixer = SourceMixer([Empty()])
        next(iter(mixer))


def test_mixer_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError, match="at least 1 source"):
        SourceMixer([])
