"""Full A-to-Z journey card: training + shipping + live correction.

Suitable for X / Twitter / Telegram posting.

Layout (1400x1000):
  Header:          GAZELOCK  +  tagline
  4-stage flow:    DATA -> TRAIN -> SHIP -> LIVE
  LIVE demo:       BEFORE (iris off-center) -> AFTER (iris centered via TPS warp)
  Footer:          licensing + status line

Careful with text bounds — every string is measured and positioned so
it stays inside its card's interior (padding kept >= 12 px).

Usage:
    uv run python -m gazelock_ml.cli.journey_card \\
        --blender-root weights/synthetic-eyes \\
        --output docs/gazelock_journey_card.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from gazelock_ml.warp.apply import apply_flow
from gazelock_ml.warp.tps import fit_tps, flow_field_from_tps

BG = (0, 0, 0)
FG = (243, 243, 243)
MUTED = (140, 140, 140)
DIM = (96, 96, 96)
ACCENT = (136, 255, 0)  # #00ff88 in BGR
BORDER = (45, 45, 42)
ELEVATED = (30, 30, 28)

FONT = cv2.FONT_HERSHEY_DUPLEX


def _put(img: np.ndarray, text: str, org: tuple[int, int], scale: float,
         color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _text_size(text: str, scale: float, thickness: int = 1) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


def _center_x(text: str, outer_x: int, outer_w: int, scale: float,
              thickness: int = 1) -> int:
    w, _ = _text_size(text, scale, thickness)
    return outer_x + (outer_w - w) // 2


def _draw_card(canvas: np.ndarray, x: int, y: int, w: int, h: int,
               border_color: tuple[int, int, int] = BORDER, fill: bool = True) -> None:
    if fill:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), ELEVATED, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1)


def _draw_h_arrow(canvas: np.ndarray, x0: int, x1: int, y: int,
                  color: tuple[int, int, int] = ACCENT) -> None:
    cv2.line(canvas, (x0, y), (x1 - 10, y), color, 2, cv2.LINE_AA)
    pts = np.array([[x1, y], [x1 - 10, y - 6], [x1 - 10, y + 6]], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], color)


def _load_manifest(blender_root: Path) -> list[dict]:
    mp = blender_root / "manifest.jsonl"
    if not mp.exists():
        return []
    return [
        json.loads(line)
        for line in mp.read_text().splitlines()
        if line.strip()
    ]


def _find_off_center_render(entries: list[dict], image_size: int = 256,
                             min_offset_px: float = 25.0) -> dict | None:
    """Pick an entry whose iris (gaze_pixel_target) is farthest from image center."""
    center = image_size / 2.0
    best, best_d = None, 0.0
    for e in entries:
        gpt = e.get("gaze_pixel_target")
        if not gpt:
            continue
        dx = gpt[0] - center
        dy = gpt[1] - center
        d = math.hypot(dx, dy)
        if d > best_d and d >= min_offset_px:
            best, best_d = e, d
    return best


def _correct_iris(img_bgr: np.ndarray, current_px: tuple[float, float],
                   target_px: tuple[float, float]) -> np.ndarray:
    """Warp img so iris moves from current_px to target_px via TPS."""
    h, w = img_bgr.shape[:2]
    source_points = np.array([
        [0.0, 0.0],
        [w - 1.0, 0.0],
        [0.0, h - 1.0],
        [w - 1.0, h - 1.0],
        [float(current_px[0]), float(current_px[1])],
    ], dtype=np.float64)
    target_points = source_points.copy()
    target_points[-1] = [float(target_px[0]), float(target_px[1])]
    coefs = fit_tps(target_points, source_points)
    flow = flow_field_from_tps(coefs, target_points, h, w)
    return apply_flow(img_bgr, flow)


# --------------------------------------------------------------------------
# Text wrapping util — keeps text inside a given pixel width.

def _wrap_lines(text: str, max_width: int, scale: float) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if _text_size(trial, scale)[0] <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _draw_stage_card(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                      number: str, title: str, body_lines: list[str],
                      accent_bottom: str | None = None) -> None:
    _draw_card(canvas, x, y, w, h, border_color=BORDER)

    pad = 14
    # Number + title
    _put(canvas, number, (x + pad, y + 28), 0.55, ACCENT)
    num_w, _ = _text_size(number, 0.55)
    _put(canvas, title, (x + pad + num_w + 8, y + 28), 0.55, FG)

    # Thin accent separator
    cv2.line(canvas, (x + pad, y + 42), (x + w - pad, y + 42), BORDER, 1)

    # Body lines
    line_y = y + 68
    for line in body_lines:
        wrapped = _wrap_lines(line, w - 2 * pad, 0.42)
        for wl in wrapped:
            _put(canvas, wl, (x + pad, line_y), 0.42, MUTED)
            line_y += 20

    # Accent bottom line (anchored above the card's bottom edge)
    if accent_bottom:
        _put(canvas, accent_bottom, (x + pad, y + h - 16), 0.42, ACCENT)


def build_journey_card(blender_root: Path, seed: int) -> np.ndarray:
    W, H = 1400, 1000
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)

    # ── Header ────────────────────────────────────────────────
    title = "G A Z E L O C K"
    tw, _ = _text_size(title, 1.7, 2)
    tx = (W - tw) // 2
    _put(canvas, title, (tx, 62), 1.7, FG, thickness=2)
    cv2.circle(canvas, (tx - 30, 52), 7, ACCENT, -1)
    cv2.circle(canvas, (tx - 30, 52), 14, ACCENT, 1)

    subtitle = "Natural eye-contact correction for macOS"
    sw, _ = _text_size(subtitle, 0.62)
    _put(canvas, subtitle, ((W - sw) // 2, 98), 0.62, MUTED)

    cv2.line(canvas, (80, 128), (W - 80, 128), BORDER, 1)

    # ── 4-stage flow ───────────────────────────────────────────
    section_label = "H O W   I T   W O R K S"
    sl_w, _ = _text_size(section_label, 0.5)
    _put(canvas, section_label, ((W - sl_w) // 2, 160), 0.5, ACCENT)

    stages_y = 180
    card_h = 220
    # 4 cards horizontally, with space for 3 arrows
    n = 4
    arrow_gap = 54
    total_arrow_w = arrow_gap * (n - 1)
    side_pad = 60
    card_w = (W - 2 * side_pad - total_arrow_w) // n

    stage_defs = [
        {
            "number": "1",
            "title": "DATA",
            "body": [
                "1M synthetic faces (MIT)",
                "100k segmented (MIT)",
                "500k procedural eyes",
            ],
            "accent": "all sources MIT / CC0",
        },
        {
            "number": "2",
            "title": "TRAIN",
            "body": [
                "Refiner U-Net",
                "50k steps on Apple MPS",
                "L1 + perceptual +",
                "identity + flicker loss",
            ],
            "accent": None,
        },
        {
            "number": "3",
            "title": "SHIP",
            "body": [
                "PyTorch to Core ML",
                "refiner.mlpackage",
                "zero-cloud inference",
            ],
            "accent": "L-inf gate < 1e-3",
        },
        {
            "number": "4",
            "title": "LIVE",
            "body": [
                "macOS Camera Extension",
                "60 fps on-device",
                "Zoom / Meet / FaceTime",
            ],
            "accent": "works in any app",
        },
    ]

    for i, s in enumerate(stage_defs):
        cx = side_pad + i * (card_w + arrow_gap)
        _draw_stage_card(
            canvas, cx, stages_y, card_w, card_h,
            number=f"{s['number']}.",
            title=s["title"],
            body_lines=s["body"],
            accent_bottom=s["accent"],
        )
        if i < n - 1:
            ax0 = cx + card_w + 6
            ax1 = ax0 + arrow_gap - 16
            _draw_h_arrow(canvas, ax0, ax1, stages_y + card_h // 2)

    # ── Live demo section ─────────────────────────────────────
    demo_y = stages_y + card_h + 50

    demo_label = "L I V E   C O R R E C T I O N"
    dl_w, _ = _text_size(demo_label, 0.5)
    _put(canvas, demo_label, ((W - dl_w) // 2, demo_y - 10), 0.5, ACCENT)

    # Pick a Blender render with clearly off-center iris
    entries = _load_manifest(blender_root)
    if not entries:
        raise RuntimeError(f"No manifest entries under {blender_root}")
    off_entry = _find_off_center_render(entries, image_size=256, min_offset_px=35.0)
    if off_entry is None:
        # Fall back to whatever the manifest's first entry is
        off_entry = entries[0]
    render_path = blender_root / off_entry["png_path"]
    render = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
    if render is None:
        raise RuntimeError(f"Couldn't load {render_path}")

    # If render isn't 256x256, resize so we know center
    if render.shape[:2] != (256, 256):
        render = cv2.resize(render, (256, 256), interpolation=cv2.INTER_AREA)

    current_px = tuple(off_entry["gaze_pixel_target"])
    target_px = (128.0, 128.0)  # image center
    corrected = _correct_iris(render, current_px, target_px)

    # Display panels (enlarged)
    panel_size = 360
    arrow_w = 100
    demo_total_w = panel_size * 2 + arrow_w
    demo_x0 = (W - demo_total_w) // 2

    panel_y = demo_y + 40
    panel_before = cv2.resize(render, (panel_size, panel_size), interpolation=cv2.INTER_AREA)
    panel_after = cv2.resize(corrected, (panel_size, panel_size), interpolation=cv2.INTER_AREA)

    # BEFORE panel (red-ish border for contrast — using FG as warm-white)
    before_border = (80, 80, 230)  # light red
    cv2.rectangle(canvas,
                  (demo_x0 - 2, panel_y - 2),
                  (demo_x0 + panel_size + 2, panel_y + panel_size + 2),
                  before_border, 2)
    canvas[panel_y:panel_y + panel_size, demo_x0:demo_x0 + panel_size] = panel_before

    # AFTER panel (accent border)
    after_x = demo_x0 + panel_size + arrow_w
    cv2.rectangle(canvas,
                  (after_x - 2, panel_y - 2),
                  (after_x + panel_size + 2, panel_y + panel_size + 2),
                  ACCENT, 2)
    canvas[panel_y:panel_y + panel_size, after_x:after_x + panel_size] = panel_after

    # Arrow between panels
    _draw_h_arrow(canvas,
                  demo_x0 + panel_size + 18,
                  after_x - 8,
                  panel_y + panel_size // 2)

    # Labels above panels
    lbl_before = "BEFORE   looking at screen"
    lbl_after = "AFTER   looking at camera"
    _put(canvas, lbl_before, (demo_x0, panel_y - 14), 0.52, FG)
    aw, _ = _text_size(lbl_after, 0.52)
    _put(canvas, lbl_after, (after_x + panel_size - aw, panel_y - 14), 0.52, ACCENT)

    # Caption under panels
    cap_y = panel_y + panel_size + 34
    cap = "Live on-device: landmark detect  ->  TPS warp (iris -> camera)  ->  refiner smooths seam"
    cw, _ = _text_size(cap, 0.56)
    _put(canvas, cap, ((W - cw) // 2, cap_y), 0.56, MUTED)

    # ── Footer ────────────────────────────────────────────────
    foot_y = H - 68
    cv2.line(canvas, (80, foot_y - 16), (W - 80, foot_y - 16), BORDER, 1)

    foot1 = "1M faces  +  100k seg faces  +  500k procedural eyes  =  1.6M training images"
    foot2 = "All MIT / CC0 licensed   Ship-ready commercial macOS app   No cloud, zero telemetry"

    f1w, _ = _text_size(foot1, 0.52)
    f2w, _ = _text_size(foot2, 0.48)
    _put(canvas, foot1, ((W - f1w) // 2, foot_y + 8), 0.52, FG)
    _put(canvas, foot2, ((W - f2w) // 2, foot_y + 36), 0.48, MUTED)

    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/gazelock_journey_card.png"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    print("[journey-card] composing...")
    img = build_journey_card(args.blender_root, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), img)
    print(f"[journey-card] wrote {args.output}  ({img.shape[1]}x{img.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
