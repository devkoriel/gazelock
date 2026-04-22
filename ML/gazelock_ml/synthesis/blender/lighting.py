"""HDRI lighting preset rotation. Pure Python."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

HDRI_MANIFEST = Path(__file__).parent / "assets" / "hdri" / "MANIFEST.json"


@dataclass(frozen=True)
class LightingPreset:
    id: str
    path: Path
    rotation_z_rad: float


def load_presets() -> list[LightingPreset]:
    manifest = json.loads(HDRI_MANIFEST.read_text())
    presets: list[LightingPreset] = []
    base = HDRI_MANIFEST.parent
    for a in manifest["assets"]:
        presets.append(
            LightingPreset(
                id=a["id"],
                path=base / a["filename"],
                rotation_z_rad=0.0,
            )
        )
    return presets


def pick(seed: int, presets: list[LightingPreset] | None = None) -> LightingPreset:
    presets = presets or load_presets()
    if not presets:
        raise RuntimeError("No HDRI presets available")
    rng = random.Random(seed)
    preset = rng.choice(presets)
    rotation = rng.uniform(0, 2 * 3.14159265358979)
    return LightingPreset(id=preset.id, path=preset.path, rotation_z_rad=rotation)


__all__ = ["LightingPreset", "load_presets", "pick"]
