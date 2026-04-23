"""Render a single packed PNG explaining the GazeLock training pipeline.

Horizontal flow: DATA SOURCES -> TRAINING PAIR -> REFINER -> CORE ML -> APP

Uses real images: samples a DigiFace face + crops the eye via face-alignment,
pulls a Blender render, synthesises an actual training pair via the P2b
analytic warp so the viewer sees exactly what the refiner learns from.

Usage:
    uv run python -m gazelock_ml.cli.pipeline_card \\
        --digiface-root ~/Datasets/DigiFace1M \\
        --blender-root weights/synthetic-eyes \\
        --output docs/gazelock_pipeline_card.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from gazelock_ml.data.pairs import synthesise_pair
from gazelock_ml.synthesis.eye_crop import Eye, crop_eye

BG = (0, 0, 0)
FG = (243, 243, 243)
MUTED = (140, 140, 140)
DIM = (96, 96, 96)
ACCENT = (136, 255, 0)
BORDER = (45, 45, 42)
ELEVATED = (30, 30, 28)


def _detect_lm_cpu(rgb: np.ndarray) -> np.ndarray | None:
    import face_alignment  # noqa: PLC0415
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False,
    )
    preds = fa.get_landmarks_from_image(rgb)
    if not preds:
        return None
    if len(preds) > 1:
        def area(lm: np.ndarray) -> float:
            return float(
                (lm[:, 0].max() - lm[:, 0].min()) * (lm[:, 1].max() - lm[:, 1].min())
            )
        return max(preds, key=area)
    return preds[0]


def _sample_digiface_eye(root: Path, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    rng.shuffle(dirs)
    for subject in dirs[:20]:
        imgs = list(subject.glob("*.png"))
        if not imgs:
            continue
        p = rng.choice(imgs)
        face = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if face is None:
            continue
        lm = _detect_lm_cpu(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        if lm is None:
            continue
        try:
            eye = crop_eye(face, lm, Eye.LEFT)
        except ValueError:
            continue
        return face, eye
    return None


def _sample_blender(root: Path, rng: random.Random) -> np.ndarray | None:
    imgs = sorted(root.glob("*.png"))
    if not imgs:
        return None
    p = rng.choice(imgs)
    return cv2.imread(str(p), cv2.IMREAD_COLOR)


def _fit_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    sq = img[y0:y0 + side, x0:x0 + side]
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def _put(img: np.ndarray, text: str, org: tuple[int, int], scale: float,
         color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_card(canvas: np.ndarray, x: int, y: int, w: int, h: int,
               border_color: tuple[int, int, int] = BORDER) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), ELEVATED, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1)


def _draw_arrow(canvas: np.ndarray, x0: int, x1: int, y: int,
                color: tuple[int, int, int] = ACCENT) -> None:
    cv2.line(canvas, (x0, y), (x1 - 10, y), color, 2, cv2.LINE_AA)
    pts = np.array([[x1, y], [x1 - 10, y - 5], [x1 - 10, y + 5]], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], color)


def _place_panel(canvas: np.ndarray, img: np.ndarray,
                 x: int, y: int, size: int) -> None:
    panel = _fit_square(img, size)
    canvas[y:y + size, x:x + size] = panel


def build_pipeline_card(digiface_root: Path, blender_root: Path, seed: int) -> np.ndarray:
    rng = random.Random(seed)

    W, H = 1400, 860
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)

    title = "G A Z E L O C K"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.6, 2)
    title_x = (W - tw) // 2
    _put(canvas, title, (title_x, 56), 1.6, FG, thickness=2)
    cv2.circle(canvas, (title_x - 28, 48), 6, ACCENT, -1)
    cv2.circle(canvas, (title_x - 28, 48), 12, ACCENT, 1)

    subtitle = "T R A I N I N G   P I P E L I N E"
    (sw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    _put(canvas, subtitle, ((W - sw) // 2, 92), 0.55, ACCENT)

    cv2.line(canvas, (60, 125), (W - 60, 125), BORDER, 1)

    row_y = 170
    stage_label_y = 150

    stage_x = 80
    stage_w = 220

    _put(canvas, "1. DATA SOURCES", (stage_x, stage_label_y), 0.5, MUTED)

    digi = _sample_digiface_eye(digiface_root, rng)
    if digi is None:
        raise RuntimeError("No DigiFace face with detectable landmarks")
    face_img, digi_eye = digi

    blender_img = _sample_blender(blender_root, rng)
    if blender_img is None:
        raise RuntimeError("No Blender renders available")

    data_items = [
        ("DigiFace-1M", "1M faces  |  MIT", face_img),
        ("FaceSynthetics", "100k + seg  |  MIT", digi_eye),
        ("Blender eye", "500k proc  |  CC0", blender_img),
    ]
    card_h = 140
    card_gap = 18
    for i, (name, sub, img) in enumerate(data_items):
        cy = row_y + i * (card_h + card_gap)
        _draw_card(canvas, stage_x, cy, stage_w, card_h)
        thumb_size = card_h - 20
        _place_panel(canvas, img, stage_x + 10, cy + 10, thumb_size)
        tx = stage_x + 10 + thumb_size + 14
        _put(canvas, name, (tx, cy + 40), 0.55, FG)
        _put(canvas, sub, (tx, cy + 64), 0.4, MUTED)

    arrow_y = row_y + (card_h * 3 + card_gap * 2) // 2
    _draw_arrow(canvas, stage_x + stage_w + 10, stage_x + stage_w + 60, arrow_y)

    s2_x = stage_x + stage_w + 70
    s2_w = 330
    _put(canvas, "2. TRAINING PAIR", (s2_x, stage_label_y), 0.5, MUTED)

    pair = synthesise_pair(digi_eye, rng=np.random.default_rng(seed))

    pair_card_h = 220
    _draw_card(canvas, s2_x, row_y, s2_w, pair_card_h, border_color=ACCENT)

    inner_pad = 16
    pair_panel = (s2_w - inner_pad * 3) // 2

    def _eye_panel(eye_bgr: np.ndarray, size: int) -> np.ndarray:
        target_w = size
        target_h = int(size * eye_bgr.shape[0] / eye_bgr.shape[1])
        resized = cv2.resize(eye_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        pad = (size - target_h) // 2
        return cv2.copyMakeBorder(resized, pad, size - target_h - pad, 0, 0,
                                   cv2.BORDER_CONSTANT, value=ELEVATED)

    orig_panel = _eye_panel(pair.original, pair_panel)
    warp_panel = _eye_panel(pair.warped, pair_panel)
    px1 = s2_x + inner_pad
    py1 = row_y + inner_pad + 20
    canvas[py1:py1 + pair_panel, px1:px1 + pair_panel] = orig_panel
    _put(canvas, "original", (px1 + 10, py1 - 8), 0.4, MUTED)
    px2 = s2_x + s2_w - inner_pad - pair_panel
    canvas[py1:py1 + pair_panel, px2:px2 + pair_panel] = warp_panel
    _put(canvas, "warped", (px2 + 10, py1 - 8), 0.4, MUTED)
    _draw_arrow(canvas, px1 + pair_panel + 4, px2 - 4, py1 + pair_panel // 2)

    cap_y = row_y + pair_card_h + 26
    _put(canvas, "Analytic TPS warp shifts iris by", (s2_x + 10, cap_y), 0.42, MUTED)
    _put(canvas, "+/- 8 px to synthesise training pairs.", (s2_x + 10, cap_y + 20), 0.42, MUTED)

    _draw_arrow(canvas, s2_x + s2_w + 10, s2_x + s2_w + 70, row_y + pair_card_h // 2)

    s3_x = s2_x + s2_w + 80
    s3_w = 260
    _put(canvas, "3. REFINER UNET", (s3_x, stage_label_y), 0.5, MUTED)

    _draw_card(canvas, s3_x, row_y, s3_w, pair_card_h, border_color=ACCENT)

    u_y = row_y + 30
    u_x = s3_x + 20
    u_w = s3_w - 40
    for i, depth in enumerate([60, 45, 30, 20, 30, 45, 60]):
        block_x = u_x + (i * (u_w - 20)) // 6
        block_y = u_y + (70 - depth) // 2
        color = ACCENT if i == 3 else FG
        cv2.rectangle(canvas, (block_x, block_y),
                      (block_x + 20, block_y + depth), color, 1)
        if i < 3:
            cv2.line(canvas, (block_x + 10, block_y - 4),
                     (u_x + ((6 - i) * (u_w - 20)) // 6 + 10, block_y - 4),
                     DIM, 1, cv2.LINE_AA)

    _put(canvas, "train 50k steps", (s3_x + 20, row_y + 130), 0.45, FG)
    _put(canvas, "MPS  |  Apple Silicon", (s3_x + 20, row_y + 152), 0.4, MUTED)
    _put(canvas, "loss: L1 + perceptual", (s3_x + 20, row_y + 172), 0.4, MUTED)
    _put(canvas, "      + identity + flicker", (s3_x + 20, row_y + 192), 0.4, MUTED)

    _draw_arrow(canvas, s3_x + s3_w + 10, s3_x + s3_w + 70, row_y + pair_card_h // 2)

    s4_x = s3_x + s3_w + 80
    s4_w = 240
    _put(canvas, "4. SHIP", (s4_x, stage_label_y), 0.5, MUTED)

    mlp_h = 100
    _draw_card(canvas, s4_x, row_y, s4_w, mlp_h, border_color=ACCENT)
    _put(canvas, "refiner.mlpackage", (s4_x + 14, row_y + 35), 0.55, FG)
    _put(canvas, "PyTorch -> CoreML", (s4_x + 14, row_y + 58), 0.4, MUTED)
    _put(canvas, "L-inf gate < 1e-3", (s4_x + 14, row_y + 78), 0.4, ACCENT)

    conn_x = s4_x + s4_w // 2
    cv2.line(canvas, (conn_x, row_y + mlp_h + 2),
             (conn_x, row_y + mlp_h + 26), ACCENT, 2, cv2.LINE_AA)
    pts = np.array([
        [conn_x, row_y + mlp_h + 32],
        [conn_x - 5, row_y + mlp_h + 22],
        [conn_x + 5, row_y + mlp_h + 22],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], ACCENT)

    app_y = row_y + mlp_h + 38
    app_h = pair_card_h - mlp_h - 38
    _draw_card(canvas, s4_x, app_y, s4_w, app_h)
    _put(canvas, "macOS Camera Extension", (s4_x + 14, app_y + 28), 0.5, FG)
    _put(canvas, "60 fps live inference", (s4_x + 14, app_y + 50), 0.4, MUTED)
    _put(canvas, "Zoom / Meet / FaceTime", (s4_x + 14, app_y + 70), 0.4, MUTED)

    footer_y = row_y + (card_h * 3 + card_gap * 2) + 40
    cv2.line(canvas, (60, footer_y - 20), (W - 60, footer_y - 20), BORDER, 1)
    foot_lines = [
        ("1M faces  +  100k seg faces  +  500k procedural eyes  =  1.6M training images", FG),
        ("All MIT / CC0 licensed.  Trained + shipped commercially.", MUTED),
    ]
    for i, (line, color) in enumerate(foot_lines):
        (fw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
        _put(canvas, line, ((W - fw) // 2, footer_y + i * 26), 0.52, color)

    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digiface-root", type=Path, required=True)
    parser.add_argument("--blender-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/gazelock_pipeline_card.png"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    print("[pipeline-card] composing...")
    img = build_pipeline_card(args.digiface_root, args.blender_root, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), img)
    print(f"[pipeline-card] wrote {args.output}  ({img.shape[1]}x{img.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
