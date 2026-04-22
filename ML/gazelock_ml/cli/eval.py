"""CLI: gazelock-eval — compute objective metrics on a held-out set."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gazelock_ml.data.dataset import EyePairDataset
from gazelock_ml.data.fixtures import make_fake_eye_patch
from gazelock_ml.metrics.flicker import frame_to_frame_l1
from gazelock_ml.metrics.identity import identity_cosine
from gazelock_ml.metrics.psnr import psnr
from gazelock_ml.metrics.ssim import ssim
from gazelock_ml.models.refiner import RefinerUNet
from gazelock_ml.training.checkpoints import load_checkpoint


def _fixture_source(n: int, start_seed: int) -> Iterable[np.ndarray]:
    return (make_fake_eye_patch(seed=start_seed + i) for i in range(n))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the GazeLock refiner.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    model = RefinerUNet()
    load_checkpoint(args.checkpoint, model)
    model.eval()

    ds = EyePairDataset(
        lambda: _fixture_source(args.batches * args.batch_size, start_seed=10_000),
        rng_seed=100,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    psnrs: list[float] = []
    ssims: list[float] = []
    ids: list[float] = []
    last_preds: list[torch.Tensor] = []
    with torch.no_grad():
        for inp, tgt in loader:
            pred = model(inp)
            psnrs.append(psnr(pred, tgt))
            ssims.append(ssim(pred, tgt))
            ids.append(identity_cosine(pred, tgt))
            last_preds.append(pred[0])
    preds_stack = torch.stack(last_preds, dim=0) if last_preds else torch.zeros((0, 3, 1, 1))
    flicker = frame_to_frame_l1(preds_stack) if preds_stack.shape[0] >= 2 else 0.0

    print(f"[eval] PSNR={sum(psnrs) / len(psnrs):.2f}")
    print(f"[eval] SSIM={sum(ssims) / len(ssims):.4f}")
    print(f"[eval] identity_cos={sum(ids) / len(ids):.4f}")
    print(f"[eval] flicker_l1={flicker:.4f}")


if __name__ == "__main__":
    main()
