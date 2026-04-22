"""CLI: gazelock-export — Core ML export with L∞ validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from gazelock_ml.export.coreml import (
    DEFAULT_TOLERANCE,
    DEFAULT_VALIDATION_SAMPLES,
    export,
)
from gazelock_ml.models.refiner import RefinerUNet
from gazelock_ml.training.checkpoints import load_checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the refiner to Core ML.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("weights/refiner.mlpackage"))
    parser.add_argument("--samples", type=int, default=DEFAULT_VALIDATION_SAMPLES)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)

    model = RefinerUNet()
    load_checkpoint(args.checkpoint, model)

    result = export(
        model,
        args.output,
        num_validation_samples=args.samples,
        tolerance=args.tolerance,
    )
    print(f"[export] wrote {result['path']}")
    print(f"[export] L∞ = {result['linf']:.2e} (tol {result['tolerance']:.0e})")


if __name__ == "__main__":
    main()
