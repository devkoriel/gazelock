"""Builds a tiny DigiFace-like fixture tree for tests."""

from pathlib import Path

import cv2
import numpy as np

FIXTURE_ROOT = Path(__file__).parent / "digiface_mini"


def build() -> None:
    (FIXTURE_ROOT / "images" / "id01").mkdir(parents=True, exist_ok=True)
    manifest_lines: list[str] = []

    rng = np.random.default_rng(0)
    for i in range(3):
        img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        img_rel = f"id01/{i:05d}.jpg"
        img_path = FIXTURE_ROOT / "images" / img_rel
        cv2.imwrite(str(img_path), img)

        lm = rng.uniform(40, 216, (68, 2)).astype(np.float32)
        for j, idx in enumerate(range(36, 42)):
            lm[idx] = [70 + j * 3, 95 + (j % 2) * 4]
        for j, idx in enumerate(range(42, 48)):
            lm[idx] = [166 + j * 3, 95 + (j % 2) * 4]

        line = img_rel + " " + " ".join(f"{v:.2f}" for v in lm.flatten())
        manifest_lines.append(line)

    (FIXTURE_ROOT / "landmarks2d.txt").write_text("\n".join(manifest_lines) + "\n")


if __name__ == "__main__":
    build()
    print(f"Built fixture at {FIXTURE_ROOT}")
