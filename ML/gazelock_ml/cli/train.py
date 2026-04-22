"""CLI: gazelock-train — kick off a training run."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from gazelock_ml.data.dataset import EyePairDataset
from gazelock_ml.data.fixtures import make_fake_eye_patch
from gazelock_ml.models.refiner import RefinerUNet
from gazelock_ml.synthesis.base import SyntheticFaceSource
from gazelock_ml.synthesis.blender_eyes import BlenderEyeSource
from gazelock_ml.synthesis.digiface import DigiFaceSource
from gazelock_ml.synthesis.facesynthetics import FaceSyntheticsSource
from gazelock_ml.synthesis.mixer import SourceMixer
from gazelock_ml.training.checkpoints import (
    CheckpointMetadata,
    save_checkpoint,
    write_run_manifest,
)
from gazelock_ml.training.composed_loss import ComposedLoss
from gazelock_ml.training.loop import pick_device, train_for_n_steps


def _fixture_source(n: int) -> Iterable[np.ndarray]:
    return (make_fake_eye_patch(seed=i) for i in range(n))


def _parse_mixer_weights(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for pair in spec.split(","):
        if not pair.strip():
            continue
        name, weight = pair.split(":")
        out[name.strip()] = float(weight)
    return out


def _build_source_mixer(args: argparse.Namespace) -> SourceMixer | None:
    sources: list[SyntheticFaceSource] = []
    if args.digiface_root is not None:
        sources.append(DigiFaceSource(
            args.digiface_root,
            max_samples=args.max_digiface,
        ))
    if args.facesynthetics_root is not None:
        sources.append(FaceSyntheticsSource(
            args.facesynthetics_root,
            max_samples=args.max_facesynthetics,
        ))
    if args.synthetic_eyes_root is not None:
        sources.append(BlenderEyeSource(args.synthetic_eyes_root))
    if not sources:
        return None
    weights = _parse_mixer_weights(args.mixer_weights)
    active = {s.info.name: weights.get(s.info.name, 1.0) for s in sources}
    return SourceMixer(sources, weights=active, seed=args.seed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the GazeLock refiner UNet.")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("weights/debug"))
    parser.add_argument("--seed", type=int, default=0)
    # Real-data options (commercial-safe synthetic sources):
    parser.add_argument("--digiface-root", type=Path, default=None,
                        help="Microsoft DigiFace-1M root (MIT).")
    parser.add_argument("--facesynthetics-root", type=Path, default=None,
                        help="Microsoft FaceSynthetics root (MIT).")
    parser.add_argument("--synthetic-eyes-root", type=Path, default=None,
                        help="Blender-rendered eyes root (produced by make ml-render).")
    parser.add_argument(
        "--mixer-weights",
        type=str,
        default="digiface:0.45,blender_eye:0.35,facesynthetics:0.20",
        help="Comma-separated name:weight pairs.",
    )
    parser.add_argument(
        "--max-digiface",
        type=int,
        default=None,
        help="Cap on DigiFace samples to load. Useful when the full 1M is too many "
             "for the one-time landmark-cache pre-pass.",
    )
    parser.add_argument(
        "--max-facesynthetics",
        type=int,
        default=None,
        help="Cap on FaceSynthetics samples to load.",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"[train] device={device}")

    model = RefinerUNet().to(device)
    loss_fn = ComposedLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps)

    mixer = _build_source_mixer(args)
    if mixer is not None:
        from itertools import islice

        n_patches = args.steps * args.batch_size * 4

        def patch_source() -> Iterable[np.ndarray]:
            return islice(iter(mixer), n_patches)
    else:
        def patch_source() -> Iterable[np.ndarray]:
            return _fixture_source(args.steps * args.batch_size * 2)

    ds = EyePairDataset(patch_source, rng_seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    def _log_step(r):
        if r.step % max(1, args.steps // 5) == 0 or r.step == args.steps - 1:
            print(f"[train] step={r.step} loss_total={r.loss_total:.4f} lr={r.lr:.2e}")

    history = train_for_n_steps(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        loader=loader,
        device=device,
        n_steps=args.steps,
        lr_scheduler=scheduler,
        on_step=_log_step,
    )

    # Save final checkpoint
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "final.pt"
    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        metadata=CheckpointMetadata(
            step=history[-1].step,
            loss_total=history[-1].loss_total,
            notes=f"smoke/fixture run, {args.steps} steps, bs={args.batch_size}",
        ),
    )

    manifest = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_loss": history[-1].loss_total,
        "loss_components": history[-1].loss_components,
    }
    write_run_manifest(args.output_dir / "manifest.json", manifest)

    print(f"[train] wrote checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
