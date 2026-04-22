"""Builds a tiny DigiFace-like fixture tree for tests.

Layout produced:

    digiface_mini/
        0/ 0.png  1.png
        1/ 0.png  1.png
        2/ 0.png  1.png
        .landmarks_cache.jsonl   (6 entries, one per image)

Landmarks use realistic iBUG-68 coords: left-eye cluster at x~75-85,
right-eye cluster at x~166-178, both at y~95-105.

Run:
    uv run python ML/gazelock_ml/tests/synthesis/fixtures/_build_digiface_mini.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

FIXTURE_ROOT = Path(__file__).parent / "digiface_mini"


def _make_landmarks(rng: np.random.Generator, subject: int) -> np.ndarray:
    """Return a (68, 2) float32 landmark array with plausible eye clusters."""
    lm = rng.uniform(40, 216, (68, 2)).astype(np.float32)
    # Left-eye iBUG indices 36-41
    for j, idx in enumerate(range(36, 42)):
        lm[idx] = [75.0 + j * 2.0, 95.0 + (j % 2) * 4.0]
    # Right-eye iBUG indices 42-47
    for j, idx in enumerate(range(42, 48)):
        lm[idx] = [166.0 + j * 2.0, 95.0 + (j % 2) * 4.0]
    return lm


def build() -> None:
    # Wipe and recreate
    if FIXTURE_ROOT.exists():
        shutil.rmtree(FIXTURE_ROOT)
    FIXTURE_ROOT.mkdir(parents=True)

    rng = np.random.default_rng(0)
    cache_lines: list[str] = []

    for subject in range(3):
        subject_dir = FIXTURE_ROOT / str(subject)
        subject_dir.mkdir()
        for img_idx in range(2):
            img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
            img_path = subject_dir / f"{img_idx}.png"
            cv2.imwrite(str(img_path), img)

            lm = _make_landmarks(rng, subject)
            rel_path = f"{subject}/{img_idx}.png"
            cache_lines.append(json.dumps({"path": rel_path, "lm": lm.tolist()}))

    cache_path = FIXTURE_ROOT / ".landmarks_cache.jsonl"
    cache_path.write_text("\n".join(cache_lines) + "\n")


if __name__ == "__main__":
    build()
    fixture_count = sum(1 for _ in FIXTURE_ROOT.glob("*/*.png"))
    cache_lines = (FIXTURE_ROOT / ".landmarks_cache.jsonl").read_text().splitlines()
    print(f"Built fixture at {FIXTURE_ROOT}")
    print(f"  Images: {fixture_count}")
    print(f"  Cache entries: {len([l for l in cache_lines if l.strip()])}")
