"""Tests for EyePairDataset + DataLoader integration."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from gazelock_ml.data.dataset import EyePairDataset
from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W, make_fake_eye_patch


def _fixture_source(n: int = 5) -> Iterable[np.ndarray]:
    return (make_fake_eye_patch(seed=i) for i in range(n))


def test_iterdataset_yields_expected_shapes() -> None:
    ds = EyePairDataset(lambda: _fixture_source(3), rng_seed=0)
    batches = list(ds)
    assert len(batches) == 3
    for inp, tgt in batches:
        assert inp.shape == (6, EYE_ROI_H, EYE_ROI_W)
        assert tgt.shape == (3, EYE_ROI_H, EYE_ROI_W)
        assert inp.dtype == torch.float32
        assert tgt.dtype == torch.float32
        assert inp.min().item() >= 0.0 and inp.max().item() <= 1.0


def test_dataloader_batches_correctly() -> None:
    ds = EyePairDataset(lambda: _fixture_source(8), rng_seed=1)
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    batches = list(loader)
    # 8 samples, batch_size=4 → 2 batches of 4
    assert len(batches) == 2
    for inp, tgt in batches:
        assert inp.shape == (4, 6, EYE_ROI_H, EYE_ROI_W)
        assert tgt.shape == (4, 3, EYE_ROI_H, EYE_ROI_W)
