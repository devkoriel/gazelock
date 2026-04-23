"""Render a single packed PNG suitable for social sharing.

Layout (1200x675, 16:9):
  +---------------------------------------------------------------+
  |                     GAZELOCK                                  |
  |         Natural eye-contact correction for macOS              |
  +---------------------------------------------------------------+
  |                                                               |
  |   [DigiFace face]  [eye crop]  [Blender render 1]  [Blender 2]|
  |                                                               |
  +---------------------------------------------------------------+
  |   Training pipeline: 1M synthetic faces + 500k procedural      |
  |   eye renders → Core ML refiner. All MIT / CC0 licensed.      |
  +---------------------------------------------------------------+

Usage:
    uv run python -m gazelock_ml.cli.share_card \\
        --digiface-root ~/Datasets/DigiFace1M \\
        --blender-root weights/synthetic-eyes \\
        --output docs/gazelock_share_card.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from gazelock_ml.synthesis.eye_crop import Eye, crop_eye


def _detect_landmarks_cpu(image_rgb: np.ndarray) -> np.ndarray | None:
    import face_alignment  # noqa: PLC0415
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device="cpu",
        flip_input=False,
    )
    preds = fa.get_landmarks_from_image(image_rgb)
    if not preds:
        return None
    if len(preds) > 1:
        def area(lm: np.ndarray) -> float:
            xs, ys = lm[:, 0], lm[:, 1]
            return float((xs.max() - xs.min()) * (ys.max() - ys.min()))
        return max(preds, key=area)
    return preds[0]


def _sample_digiface_eye(root: Path, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (full_face_bgr, eye_crop_bgr) or None on failure."""
    subject_dirs = [p for p in root.iterdir() if p.is_dir()]
    rng.shuffle(subject_dirs)
    for subject in subject_dirs[:20]:  # try up to 20 subjects
        imgs = list(subject.glob("*.png"))
        if not imgs:
            continue
        img_path = rng.choice(imgs)
        face = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if face is None:
            continue
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        lm = _detect_landmarks_cpu(rgb)
        if lm is None:
            continue
        try:
            eye = crop_eye(face, lm, Eye.LEFT)
        except ValueError:
            continue
        return face, eye
    return None


def _sample_blender(root: Path, count: int, rng: random.Random) -> list[np.ndarray]:
    imgs = sorted(root.glob("*.png"))
    if not imgs:
        return []
    sample = rng.sample(imgs, min(count, len(imgs)))
    out: list[np.ndarray] = []
    for p in sample:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            out.append(img)
    return out


# Precision Dark palette (matches the app UI)
BG = (0, 0, 0)
FG = (243, 243, 243)
MUTED = (140, 140, 140)
ACCENT = (136, 255, 0)  # BGR for #00ff88
ELEVATED = (30, 30, 28)
BORDER = (45, 45, 42)


def _fit_square(img: np.ndarray, size: int) -> np.ndarray:
    """Center-crop + resize to size x size."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    sq = img[y0:y0 + side, x0:x0 + side]
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def _put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_DUPLEX,
) -> None:
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def build_share_card(
    digiface_root: Path,
    blender_root: Path,
    seed: int,
) -> np.ndarray:
    rng = random.Random(seed)

    # Sample assets
    digi = _sample_digiface_eye(digiface_root, rng)
    if digi is None:
        raise RuntimeError("Couldn't sample a DigiFace face with detectable landmarks")
    face, eye_crop = digi
    blender_samples = _sample_blender(blender_root, count=2, rng=rng)
    if len(blender_samples) < 2:
        raise RuntimeError(f"Need 2+ Blender renders; found {len(blender_samples)}")

    # Canvas
    W, H = 1200, 675
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = BG

    # Header band
    header_h = 140
    canvas[:header_h] = BG

    # Title: GAZELOCK (letter-spaced, big)
    title = "G A Z E L O C K"
    title_scale = 2.1
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_scale, 2)
    tx = (W - tw) // 2
    _put_text(canvas, title, (tx, 62), title_scale, FG, thickness=2)

    # Accent dot next to title
    cv2.circle(canvas, (tx - 28, 52), 6, ACCENT, -1)
    cv2.circle(canvas, (tx - 28, 52), 12, ACCENT, 1)  # outer glow ring

    # Tagline
    tagline = "Natural eye-contact correction for macOS"
    (tw2, _), _ = cv2.getTextSize(tagline, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
    _put_text(canvas, tagline, ((W - tw2) // 2, 102), 0.7, MUTED)

    # Separator
    cv2.line(canvas, (60, header_h), (W - 60, header_h), BORDER, 1)

    # Content band
    content_top = header_h + 40
    content_h = 340
    panel_size = 260
    gap = 28
    total_w = panel_size * 4 + gap * 3
    start_x = (W - total_w) // 2

    # Panel labels under each image
    labels = ["Real face", "Eye ROI", "Procedural eye", "Procedural eye"]
    panels = [
        _fit_square(face, panel_size),
        _fit_square(eye_crop, panel_size),
        _fit_square(blender_samples[0], panel_size),
        _fit_square(blender_samples[1], panel_size),
    ]

    for i, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        x = start_x + i * (panel_size + gap)
        y = content_top

        # Panel border (accent color if it's a Blender render)
        is_blender = i >= 2
        border_color = ACCENT if is_blender else BORDER
        cv2.rectangle(
            canvas,
            (x - 2, y - 2),
            (x + panel_size + 2, y + panel_size + 2),
            border_color,
            1,
        )
        canvas[y:y + panel_size, x:x + panel_size] = panel

        # Label below
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        label_x = x + (panel_size - lw) // 2
        _put_text(canvas, label, (label_x, y + panel_size + 24), 0.5, MUTED)

    # Section caption between header and panels
    section_caption = "TRAINING SOURCES"
    (sw, _), _ = cv2.getTextSize(section_caption, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
    # Letter-space manually for tracking effect
    spaced = " ".join(section_caption)
    (sw2, _), _ = cv2.getTextSize(spaced, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
    _put_text(canvas, spaced, ((W - sw2) // 2, header_h + 22), 0.45, ACCENT)

    # Footer band
    footer_y = content_top + panel_size + 70
    cv2.line(canvas, (60, footer_y - 24), (W - 60, footer_y - 24), BORDER, 1)

    foot_lines = [
        "1M synthetic faces  +  500k procedural eye renders  -->  Core ML refiner",
        "All sources MIT / CC0 licensed.  Trained + shipping commercially.",
    ]
    for i, line in enumerate(foot_lines):
        (fw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        color = FG if i == 0 else MUTED
        _put_text(canvas, line, ((W - fw) // 2, footer_y + i * 26), 0.55, color)

    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digiface-root", type=Path, required=True)
    parser.add_argument("--blender-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/gazelock_share_card.png"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print("[share-card] composing...")
    img = build_share_card(
        digiface_root=args.digiface_root,
        blender_root=args.blender_root,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), img)
    print(f"[share-card] wrote {args.output}  ({img.shape[1]}x{img.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
