"""Generate a visual comparison of training-source pairs.

Produces a grid PNG showing, for each of N rows:
    [DigiFace full face] [DigiFace eye crop] [Blender eye render]

Not part of the training loop — purely a communication artifact. Run when
you want to show "here's what the refiner sees" to someone external.

Usage:
    uv run python -m gazelock_ml.cli.preview_pairs \\
        --digiface-root ~/Datasets/DigiFace1M \\
        --blender-root weights/synthetic-eyes \\
        --count 4 \\
        --output docs/training_pairs_preview.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.synthesis.eye_crop import Eye, crop_eye


def _detect_landmarks_cpu(image_rgb: np.ndarray) -> np.ndarray | None:
    """One-shot CPU landmark detection. Lazy import to keep heavy dep optional."""
    import face_alignment  # noqa: PLC0415
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device="cpu",
        flip_input=False,
    )
    preds = fa.get_landmarks_from_image(image_rgb)
    if not preds:
        return None
    # Pick largest face if multiple
    if len(preds) > 1:
        def area(lm: np.ndarray) -> float:
            xs, ys = lm[:, 0], lm[:, 1]
            return float((xs.max() - xs.min()) * (ys.max() - ys.min()))
        return max(preds, key=area)
    return preds[0]


def _sample_digiface(root: Path, rng: random.Random) -> Path | None:
    """Pick a random DigiFace image from the archive."""
    subject_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not subject_dirs:
        return None
    subject = rng.choice(subject_dirs)
    imgs = list(subject.glob("*.png"))
    if not imgs:
        return None
    return rng.choice(imgs)


def _sample_blender(root: Path, rng: random.Random) -> Path | None:
    """Pick a random Blender-rendered eye."""
    imgs = sorted(root.glob("*.png"))
    if not imgs:
        return None
    return rng.choice(imgs)


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    """Draw a small black bar with white label text at the bottom."""
    h, w = panel.shape[:2]
    strip_h = 18
    out = np.vstack([panel, np.zeros((strip_h, w, 3), dtype=np.uint8)])
    cv2.putText(
        out, text, (6, h + 13),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def build_grid(
    digiface_root: Path,
    blender_root: Path,
    count: int,
    seed: int,
    panel_size: int = 200,
) -> np.ndarray:
    rng = random.Random(seed)
    rows: list[np.ndarray] = []
    attempts = 0
    while len(rows) < count and attempts < count * 4:
        attempts += 1
        face_path = _sample_digiface(digiface_root, rng)
        if face_path is None:
            continue
        face_bgr = cv2.imread(str(face_path), cv2.IMREAD_COLOR)
        if face_bgr is None:
            continue
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        lm = _detect_landmarks_cpu(face_rgb)
        if lm is None:
            continue
        try:
            eye_patch = crop_eye(face_bgr, lm, Eye.LEFT)
        except ValueError:
            continue

        blender_path = _sample_blender(blender_root, rng)
        if blender_path is None:
            continue
        blender_bgr = cv2.imread(str(blender_path), cv2.IMREAD_COLOR)
        if blender_bgr is None:
            continue

        face_panel = _label(_resize(face_bgr, (panel_size, panel_size)), "DigiFace face")
        crop_display = _resize(eye_patch, (panel_size, panel_size * EYE_ROI_H // EYE_ROI_W))
        # Pad crop vertically to match panel_size height
        pad_top = (panel_size - crop_display.shape[0]) // 2
        pad_bot = panel_size - crop_display.shape[0] - pad_top
        if pad_top > 0 or pad_bot > 0:
            crop_display = cv2.copyMakeBorder(
                crop_display, pad_top, pad_bot, 0, 0,
                cv2.BORDER_CONSTANT, value=(32, 32, 32),
            )
        crop_panel = _label(crop_display, "DigiFace eye crop")

        blender_panel = _label(_resize(blender_bgr, (panel_size, panel_size)), "Blender render")

        rows.append(np.hstack([face_panel, crop_panel, blender_panel]))

    if not rows:
        raise RuntimeError("Failed to build any rows — check paths and landmark detection")

    grid = np.vstack(rows)
    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digiface-root", type=Path, required=True)
    parser.add_argument("--blender-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("training_pairs_preview.png"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-size", type=int, default=220)
    args = parser.parse_args()

    print(f"[preview] sampling {args.count} pairs…")
    grid = build_grid(
        digiface_root=args.digiface_root,
        blender_root=args.blender_root,
        count=args.count,
        seed=args.seed,
        panel_size=args.panel_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), grid)
    print(f"[preview] wrote {args.output}  ({grid.shape[1]}x{grid.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
