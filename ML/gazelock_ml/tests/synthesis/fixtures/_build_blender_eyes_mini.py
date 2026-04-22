"""Builds a tiny Blender-eyes fixture tree matching render_eyes.py output."""

import json
import time
from pathlib import Path

import cv2
import numpy as np

FIXTURE_ROOT = Path(__file__).parent / "blender_eyes_mini"


def build() -> None:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    entries = []
    now = int(time.time())
    for i in range(3):
        img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        fname = f"{i:06d}.png"
        cv2.imwrite(str(FIXTURE_ROOT / fname), img)
        entries.append(
            {
                "frame_id": i,
                "eye_asset_id": "procedural",
                "iris_colour_hex": "#4a2e1a",
                "gaze_vec": [0.1, 0.0, -0.99],
                "gaze_pixel_target": [128.0, 128.0],
                "lighting_id": "fixture_hdri",
                "png_path": fname,
                "png_sha256": "",
                "render_seconds": 0.5,
                "rendered_at": now + i,
            }
        )

    with (FIXTURE_ROOT / "manifest.jsonl").open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


if __name__ == "__main__":
    build()
    print(f"Built fixture at {FIXTURE_ROOT}")
