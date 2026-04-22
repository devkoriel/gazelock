"""Loader for UnityEyes synthetic-pair data.

UnityEyes (Wood et al. 2016) exports each rendered sample as:
    <idx>.jpg  — 800x600 RGB rendering  # noqa: RUF002
    <idx>.json — gaze/head metadata

For the seam-hider refiner we consume PAIRS of renders sharing the
same face identity but with different gaze targets:
    pair_root/<identity_id>/
        0.jpg 0.json
        1.jpg 1.json
        ...

The wrapper exposes each (I_warped, I_target) pair; the warped view is
synthesised at runtime by applying our own analytic warp (rather than
by rendering a different gaze in Unity) so the training signal aligns
with what the Swift inference side will produce.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class UnityEyesSample:
    """One rendered UnityEyes frame + metadata."""

    image: np.ndarray  # (H, W, 3) uint8 RGB
    gaze_angle_deg: tuple[float, float]  # yaw, pitch
    head_pose_deg: tuple[float, float, float]  # yaw, pitch, roll
    eye_corners_px: tuple[tuple[float, float], tuple[float, float]]  # inner, outer
    iris_center_px: tuple[float, float]


class UnityEyesDataset:
    """Iterates over UnityEyes render identities and their samples.

    Not a torch.utils.data.Dataset — deliberately raw so it can be
    wrapped in whatever sampler strategy the training loop picks.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        if not self._root.exists():
            raise FileNotFoundError(f"UnityEyes root not found: {self._root}")
        self._identity_dirs: list[Path] = sorted(
            p for p in self._root.iterdir() if p.is_dir()
        )

    def __len__(self) -> int:
        return len(self._identity_dirs)

    def iter_identity(self, idx: int) -> Iterator[UnityEyesSample]:
        """Yield every sample belonging to identity ``idx``."""
        ident_dir = self._identity_dirs[idx]
        for jpg in sorted(ident_dir.glob("*.jpg")):
            meta_path = jpg.with_suffix(".json")
            if not meta_path.exists():
                continue
            yield _load_sample(jpg, meta_path)

    @property
    def identity_count(self) -> int:
        return len(self._identity_dirs)


def _load_sample(jpg_path: Path, meta_path: Path) -> UnityEyesSample:
    img = np.asarray(Image.open(jpg_path).convert("RGB"))
    meta = json.loads(meta_path.read_text())

    gaze = meta["eye_details"]["look_vec"]  # list like "(-0.121 0.083 -0.989)"
    gaze_v = _parse_vec(gaze)
    # Convert look-vector to yaw/pitch in degrees
    gaze_yaw = float(np.degrees(np.arctan2(gaze_v[0], -gaze_v[2])))
    gaze_pitch = float(np.degrees(np.arcsin(gaze_v[1])))

    head_pose_str = meta["head_pose"]  # "(yaw pitch roll)"
    hy, hp, hr = _parse_vec(head_pose_str)
    head_pose = (float(hy), float(hp), float(hr))

    interior = meta.get("interior_margin_2d", [])  # eyelid landmarks; 2 corners are first+mid
    if len(interior) >= 2:
        inner = _parse_2d(interior[0])
        outer = _parse_2d(interior[len(interior) // 2])
    else:
        inner = outer = (0.0, 0.0)

    iris = meta.get("iris_2d", [])
    if iris:
        ix = sum(_parse_2d(p)[0] for p in iris) / len(iris)
        iy = sum(_parse_2d(p)[1] for p in iris) / len(iris)
        iris_center = (float(ix), float(iy))
    else:
        iris_center = (0.0, 0.0)

    return UnityEyesSample(
        image=img,
        gaze_angle_deg=(gaze_yaw, gaze_pitch),
        head_pose_deg=head_pose,
        eye_corners_px=(inner, outer),
        iris_center_px=iris_center,
    )


def _parse_vec(raw: str) -> list[float]:
    clean = raw.strip().strip("()").split()
    return [float(x) for x in clean]


def _parse_2d(raw: str) -> tuple[float, float]:
    parts = _parse_vec(raw)
    return (parts[0], parts[1])


__all__ = ["UnityEyesDataset", "UnityEyesSample"]
