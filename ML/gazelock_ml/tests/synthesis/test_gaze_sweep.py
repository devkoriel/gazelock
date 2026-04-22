"""Gaze sweep tests (pure Python, no bpy)."""

import numpy as np

from gazelock_ml.synthesis.blender.gaze_sweep import GazeTarget, random_gaze_targets


def test_random_gaze_targets_count() -> None:
    targets = random_gaze_targets(10, seed=1)
    assert len(targets) == 10


def test_random_gaze_targets_unit_vectors() -> None:
    for t in random_gaze_targets(50, seed=0):
        v = np.asarray(t.vec)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_random_gaze_targets_within_cone() -> None:
    max_angle_rad = np.deg2rad(30)
    min_cos = np.cos(max_angle_rad)
    for t in random_gaze_targets(50, seed=0):
        forward = np.array([0.0, 0.0, -1.0])
        cos_angle = float(np.dot(np.asarray(t.vec), forward))
        assert cos_angle >= min_cos - 1e-6


def test_random_gaze_targets_reproducible() -> None:
    a = random_gaze_targets(20, seed=42)
    b = random_gaze_targets(20, seed=42)
    for ga, gb in zip(a, b, strict=True):
        assert ga == gb
