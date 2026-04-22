"""Blender-side entry script for eye rendering.

Invoke via:
    blender --background --python render_eyes.py -- \\
        --count 500000 \\
        --output weights/synthetic-eyes \\
        --resolution 256 \\
        --samples 64 \\
        --seed 42 \\
        [--resume]

Generates synthetic eye renders using the procedural eye geometry
(scene.make_procedural_eye) — no external 3D assets needed. Appends
each frame's metadata to manifest.jsonl for crash-safe resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "ML"))

from gazelock_ml.synthesis.blender import gaze_sweep, lighting, scene  # noqa: E402

IRIS_COLOUR_POOL = [
    "#4a2e1a",  # dark brown
    "#8b6b4a",  # light brown
    "#5d7fa4",  # blue
    "#556b2f",  # hazel green
    "#808080",  # gray
    "#2c1810",  # very dark brown
    "#6b8e6b",  # green
    "#4682b4",  # steel blue
    "#9b7653",  # amber
    "#704214",  # amber-brown
    "#5f9ea0",  # cadet blue
    "#7d6b4e",  # olive-brown
    "#a0826d",  # tan
    "#465945",  # dark green
    "#3a5f5f",  # dark teal
]  # 15 variants for iris color diversity


@dataclass
class FrameSpec:
    frame_id: int
    iris_colour_hex: str
    gaze: gaze_sweep.GazeTarget
    lighting_preset: lighting.LightingPreset


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Render synthetic eye frames.")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("weights/synthetic-eyes"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _build_frame_specs(count: int, start_id: int, seed: int) -> list[FrameSpec]:
    import random

    presets = lighting.load_presets()
    if not presets:
        raise RuntimeError(
            "No HDRI presets in manifest. Run `make ml-fetch-assets` first."
        )

    gaze_pool = gaze_sweep.random_gaze_targets(count, seed=seed)
    rng = random.Random(seed)
    specs: list[FrameSpec] = []
    for i in range(count):
        frame_id = start_id + i
        iris = rng.choice(IRIS_COLOUR_POOL)
        preset = lighting.pick(seed=seed + frame_id, presets=presets)
        specs.append(
            FrameSpec(
                frame_id=frame_id,
                iris_colour_hex=iris,
                gaze=gaze_pool[i],
                lighting_preset=preset,
            )
        )
    return specs


def _read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_manifest(path: Path, entry: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_frame(spec: FrameSpec, out_dir: Path) -> dict:
    png_path = out_dir / f"{spec.frame_id:06d}.png"
    json_path = out_dir / f"{spec.frame_id:06d}.json"

    started = time.time()
    scene.clear_scene()
    obj = scene.make_procedural_eye(spec.iris_colour_hex)
    scene.apply_gaze(obj, spec.gaze.vec)
    scene.apply_lighting(spec.lighting_preset.path, spec.lighting_preset.rotation_z_rad)
    scene.render_frame(png_path)
    elapsed = time.time() - started

    entry = {
        "frame_id": spec.frame_id,
        "eye_asset_id": "procedural",
        "iris_colour_hex": spec.iris_colour_hex,
        "gaze_vec": list(spec.gaze.vec),
        "gaze_pixel_target": list(spec.gaze.pixel_target),
        "lighting_id": spec.lighting_preset.id,
        "png_path": str(png_path.name),
        "png_sha256": _sha256_file(png_path),
        "render_seconds": round(elapsed, 3),
        "rendered_at": int(time.time()),
    }
    json_path.write_text(json.dumps(entry, indent=2))
    return entry


def _install_sigterm_handler(on_shutdown) -> None:
    def handler(sig, frame):
        on_shutdown()
        sys.exit(0)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def main() -> int:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"

    existing = _read_manifest(manifest_path) if args.resume else []
    start_id = (existing[-1]["frame_id"] + 1) if existing else 0
    print(f"[render] starting at frame_id={start_id} (existing={len(existing)})")

    specs = _build_frame_specs(count=args.count, start_id=start_id, seed=args.seed)

    shutdown_msg = {"shutdown": False}
    _install_sigterm_handler(lambda: shutdown_msg.update(shutdown=True))

    scene.setup_scene(
        render_w=args.resolution,
        render_h=args.resolution,
        samples=args.samples,
    )

    for spec in specs:
        if shutdown_msg["shutdown"]:
            print("[render] received shutdown, exiting cleanly")
            break
        entry = _render_frame(spec, args.output)
        _append_manifest(manifest_path, entry)
        if spec.frame_id % 100 == 0:
            print(f"[render] frame {spec.frame_id} / {start_id + args.count}")

    print(f"[render] done. Wrote {len(specs)} frames to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
