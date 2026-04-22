"""Gaze target generator for eye rendering.

Pure Python — safe to import outside Blender for testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GazeTarget:
    vec: tuple[float, float, float]
    pixel_target: tuple[float, float]


def random_gaze_targets(n: int, seed: int = 0) -> list[GazeTarget]:
    """Uniformly sample gaze directions within a 30-degree cone around -Z.

    30 degrees is the plausible range for 'looking at a screen' in
    video-call ergonomics.
    """
    rng = np.random.default_rng(seed)
    max_angle_rad = np.deg2rad(30)
    theta = rng.uniform(0, 2 * np.pi, n)
    cos_phi = rng.uniform(np.cos(max_angle_rad), 1.0, n)
    sin_phi = np.sqrt(1 - cos_phi**2)
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)
    z = -cos_phi

    targets: list[GazeTarget] = []
    for i in range(n):
        vec = (float(x[i]), float(y[i]), float(z[i]))
        px = (float(128 + x[i] * 40), float(128 + y[i] * 40))
        targets.append(GazeTarget(vec=vec, pixel_target=px))
    return targets


__all__ = ["GazeTarget", "random_gaze_targets"]
