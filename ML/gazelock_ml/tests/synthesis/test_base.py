"""Contract tests for SyntheticFaceSource."""

from pathlib import Path

import numpy as np
import pytest

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.synthesis.base import SourceInfo, SyntheticFaceSource, _validate_patch


class _FakeSource(SyntheticFaceSource):
    def __init__(self, n: int) -> None:
        self._n = n
        self._info = SourceInfo(name="fake", root=Path("/dev/null"), sample_count=n)

    @property
    def info(self) -> SourceInfo:
        return self._info

    def __iter__(self):
        rng = np.random.default_rng(0)
        for _ in range(self._n):
            yield rng.integers(0, 256, (EYE_ROI_H, EYE_ROI_W, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return self._n


def test_source_info_fields() -> None:
    src = _FakeSource(10)
    assert src.info.name == "fake"
    assert src.info.sample_count == 10


def test_source_yields_correct_shape() -> None:
    src = _FakeSource(3)
    patches = list(src)
    assert len(patches) == 3
    for p in patches:
        _validate_patch(p)


def test_source_len_matches_iteration() -> None:
    src = _FakeSource(7)
    assert len(src) == 7
    assert len(list(src)) == 7


def test_validate_patch_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="Expected shape"):
        _validate_patch(np.zeros((10, 10, 3), dtype=np.uint8))


def test_validate_patch_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="Expected dtype"):
        _validate_patch(np.zeros((EYE_ROI_H, EYE_ROI_W, 3), dtype=np.float32))


def test_sample_yields_requested_count() -> None:
    src = _FakeSource(5)
    drawn = list(src.sample(20))
    assert len(drawn) == 20


def test_sample_reproducible_with_rng() -> None:
    src = _FakeSource(10)
    r1 = np.random.default_rng(42)
    r2 = np.random.default_rng(42)
    a = list(src.sample(5, rng=r1))
    b = list(src.sample(5, rng=r2))
    for pa, pb in zip(a, b, strict=True):
        assert np.array_equal(pa, pb)
