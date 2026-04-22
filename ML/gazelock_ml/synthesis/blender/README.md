# Blender Eye Render Pipeline

Renders synthetic eye images using Blender's built-in Cycles engine
with procedurally-generated eye geometry (sphere + iris + cornea).
Output is consumed by `BlenderEyeSource` in the parent package.

## Why procedural?

Phase 2c auditing found zero CC0-licensed human eye 3D models in
Sketchfab, BlenderKit, or Poly Haven. Rather than ship a single
hand-sculpted asset that would limit variety, we generate eyes
from Blender primitives parameterised by iris color. Identity
diversity comes from iris color + procedural texture variation
+ lighting + gaze pose.

## Prerequisites

1. Blender 4.2+ installed (`brew install --cask blender`)
2. HDRI assets fetched: `uv run python ../fetch_assets.py`

## Running

Full 500k render:

    make ml-render COUNT=500000

Smoke (50 frames, ~1 minute):

    make ml-render-smoke

Resume after interruption:

    make ml-render COUNT=500000

## Live dashboard

While rendering, in another terminal:

    make ml-render-viz

Open `http://localhost:8765` (or the Mac mini's Tailscale IP from any
device on your tailnet).

## Output

    weights/synthetic-eyes/
    +-- manifest.jsonl
    +-- 000000.png
    +-- 000000.json
    +-- ...

## Cancellation

Ctrl-C is safe. The next frame's metadata is flushed before rendering
begins, so on resume we restart at the highest committed frame + 1.
