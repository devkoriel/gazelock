"""Synthetic training-pair generator.

Given any real eye patch, we apply the P2a analytic warp with a random
target-gaze offset to produce the ``warped`` view. The refiner's training
objective is to recover the original from ``(warped, original)``.

The randomised offset simulates the distribution of target gazes the
refiner will see at inference: small (5–15°) corrections that move the
iris toward the camera. The pair synthesiser does NOT itself generate
new identities or new photographic content — it only warps real data
that was sourced from UnityEyes / FFHQ / procedural fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.warp.apply import apply_flow
from gazelock_ml.warp.tps import fit_tps, flow_field_from_tps


@dataclass(frozen=True)
class TrainingPair:
    """A ``(warped, original)`` BGR uint8 pair plus the applied offset."""

    warped: np.ndarray  # (EYE_ROI_H, EYE_ROI_W, 3) uint8
    original: np.ndarray  # (EYE_ROI_H, EYE_ROI_W, 3) uint8
    offset_px: tuple[float, float]  # the iris displacement we forced


def synthesise_pair(
    eye_patch: np.ndarray,
    rng: np.random.Generator | None = None,
    max_offset_px: float = 8.0,
) -> TrainingPair:
    """Apply the analytic warp to ``eye_patch`` and return the pair.

    Args:
        eye_patch: (EYE_ROI_H, EYE_ROI_W, 3) uint8 — the real source.
        rng: optional numpy Generator for reproducibility.
        max_offset_px: bound on iris displacement magnitude.

    Returns:
        TrainingPair with ``warped`` = eye_patch with iris shifted and
        ``original`` = eye_patch unchanged.
    """
    assert eye_patch.shape == (EYE_ROI_H, EYE_ROI_W, 3)
    assert eye_patch.dtype == np.uint8

    rng = rng or np.random.default_rng()
    dx = float(rng.uniform(-max_offset_px, max_offset_px))
    dy = float(rng.uniform(-max_offset_px, max_offset_px))

    # Control points: 4 corners (anchored to identity) + center (displaced).
    source_points = np.array(
        [
            [0.0, 0.0],
            [EYE_ROI_W - 1.0, 0.0],
            [0.0, EYE_ROI_H - 1.0],
            [EYE_ROI_W - 1.0, EYE_ROI_H - 1.0],
            [EYE_ROI_W / 2.0, EYE_ROI_H / 2.0],
        ],
        dtype=np.float64,
    )
    target_points = source_points.copy()
    target_points[-1] += np.array([dx, dy])

    coefs = fit_tps(target_points, source_points)  # inverse mapping for sampling
    flow = flow_field_from_tps(coefs, target_points, EYE_ROI_H, EYE_ROI_W)
    warped = apply_flow(eye_patch, flow)

    return TrainingPair(warped=warped, original=eye_patch.copy(), offset_px=(dx, dy))


__all__ = ["TrainingPair", "synthesise_pair"]
