"""End-to-end smoke test: 10 steps on procedural fixtures."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from gazelock_ml.data.dataset import EyePairDataset
from gazelock_ml.data.fixtures import make_fake_eye_patch
from gazelock_ml.models.refiner import RefinerUNet
from gazelock_ml.training.composed_loss import ComposedLoss
from gazelock_ml.training.loop import pick_device, train_for_n_steps


def _source(n: int = 64) -> Iterable[np.ndarray]:
    return (make_fake_eye_patch(seed=i) for i in range(n))


def test_10_step_training_runs_without_nan() -> None:
    device = pick_device()
    torch.manual_seed(0)
    np.random.seed(0)

    model = RefinerUNet().to(device)
    loss_fn = ComposedLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    ds = EyePairDataset(lambda: _source(32), rng_seed=0)
    loader = DataLoader(ds, batch_size=4, num_workers=0)

    history = train_for_n_steps(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        loader=loader,
        device=device,
        n_steps=10,
    )

    assert len(history) == 10
    for r in history:
        assert r.loss_total == r.loss_total  # not NaN (NaN != NaN)
        assert r.loss_total > 0
        assert r.lr == 1e-4
