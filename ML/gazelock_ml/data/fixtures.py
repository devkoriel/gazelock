"""Procedural fixture generator for unit tests.

These patches do NOT look like real eyes — they're deterministic
synthetic gradients sized like eye ROIs (96×72 BGR). The point is to
exercise shape/dtype contracts in downstream code, not to train on.
Real training data comes from UnityEyes + FFHQ at runtime.
"""

from __future__ import annotations

import numpy as np

EYE_ROI_H = 72
EYE_ROI_W = 96


def make_fake_eye_patch(seed: int = 0) -> np.ndarray:
    """Return a deterministic (EYE_ROI_H, EYE_ROI_W, 3) uint8 BGR patch.

    The patch contains a circular "iris" placed at the center at a
    seed-deterministic position, plus a horizontal gradient background.
    Good enough to verify warp + refiner shape contracts.
    """
    rng = np.random.default_rng(seed)
    patch = np.zeros((EYE_ROI_H, EYE_ROI_W, 3), dtype=np.uint8)

    # Horizontal gradient background (sclera-ish white → skin-ish)
    for x in range(EYE_ROI_W):
        t = x / (EYE_ROI_W - 1)
        patch[:, x, :] = (int(200 + 40 * t), int(210 + 30 * t), int(220 + 20 * t))

    # Iris disc — random center offset, fixed radius
    cx = EYE_ROI_W // 2 + int(rng.integers(-4, 5))
    cy = EYE_ROI_H // 2 + int(rng.integers(-3, 4))
    r = 10

    yy, xx = np.ogrid[:EYE_ROI_H, :EYE_ROI_W]
    iris_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    patch[iris_mask] = (60, 40, 30)  # brownish-blue iris

    # Pupil — darker center
    pupil_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 4 * 4
    patch[pupil_mask] = (10, 10, 10)

    return patch


def make_fake_eye_pair(
    seed: int = 0,
    target_offset_px: tuple[int, int] = (3, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return (I_source, I_target) — same background, iris shifted.

    Used to verify warp pair construction: the target has the iris at a
    deliberately different position so a warp that correctly moves the
    iris should produce a patch close to target.
    """
    base = make_fake_eye_patch(seed=seed)
    shifted = np.zeros_like(base)
    dx, dy = target_offset_px

    # Shift iris region (center 20×20) by (dx, dy)
    h_start = EYE_ROI_H // 2 - 10
    w_start = EYE_ROI_W // 2 - 10
    iris_region = base[h_start : h_start + 20, w_start : w_start + 20].copy()

    shifted[:] = base
    shifted[h_start : h_start + 20, w_start : w_start + 20] = base[
        h_start : h_start + 20, w_start : w_start + 20
    ][:]
    # Draw new iris at shifted location
    cx = EYE_ROI_W // 2 + dx
    cy = EYE_ROI_H // 2 + dy
    r = 10
    yy, xx = np.ogrid[:EYE_ROI_H, :EYE_ROI_W]
    iris_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    pupil_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 4 * 4

    # Re-apply sclera first (to clear the old iris)
    for x in range(EYE_ROI_W):
        t = x / (EYE_ROI_W - 1)
        shifted[:, x, :] = (int(200 + 40 * t), int(210 + 30 * t), int(220 + 20 * t))
    shifted[iris_mask] = (60, 40, 30)
    shifted[pupil_mask] = (10, 10, 10)

    return base, shifted
