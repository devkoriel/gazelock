"""Builds a tiny FaceSynthetics-like fixture tree."""

from pathlib import Path

import cv2
import numpy as np

FIXTURE_ROOT = Path(__file__).parent / "facesynthetics_mini"


def build() -> None:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for i in range(3):
        img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(FIXTURE_ROOT / f"{i:06d}.png"), img)

        lm = rng.uniform(40, 216, (70, 2)).astype(np.float32)
        for j, idx in enumerate(range(36, 42)):
            lm[idx] = [70 + j * 3, 95 + (j % 2) * 4]
        for j, idx in enumerate(range(42, 48)):
            lm[idx] = [166 + j * 3, 95 + (j % 2) * 4]

        lines = [f"{x:.2f} {y:.2f}" for (x, y) in lm]
        (FIXTURE_ROOT / f"{i:06d}_ldmks.txt").write_text("\n".join(lines) + "\n")

        seg = np.zeros((256, 256), dtype=np.uint8)
        cv2.imwrite(str(FIXTURE_ROOT / f"{i:06d}_seg.png"), seg)


if __name__ == "__main__":
    build()
    print(f"Built fixture at {FIXTURE_ROOT}")
