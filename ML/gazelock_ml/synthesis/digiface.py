"""DigiFace-1M loader.

Real archive layout:

    <root>/
        <id>/
            <n>.png
        .landmarks_cache.jsonl   # computed + cached at load time

68-point iBUG landmarks are computed by face-alignment on first use and
cached to .landmarks_cache.jsonl. Subsequent loads that find all paths
already cached skip the detection pass entirely.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from gazelock_ml.synthesis.base import SourceInfo, SyntheticFaceSource, _validate_patch
from gazelock_ml.synthesis.eye_crop import Eye, crop_eye

_CACHE_FILE = ".landmarks_cache.jsonl"


def _load_cache(root: Path) -> dict[str, np.ndarray]:
    """Return {rel_path: (68,2) float32 array} from the JSONL sidecar."""
    cache_path = root / _CACHE_FILE
    if not cache_path.exists():
        return {}
    result: dict[str, np.ndarray] = {}
    with cache_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                lm = np.asarray(obj["lm"], dtype=np.float32)
                if lm.shape == (68, 2):
                    result[obj["path"]] = lm
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    return result


def _pick_largest(preds: list[np.ndarray]) -> np.ndarray:
    def area(lm: np.ndarray) -> float:
        xs, ys = lm[:, 0], lm[:, 1]
        return float((xs.max() - xs.min()) * (ys.max() - ys.min()))

    return max(preds, key=area)


def _build_landmarks_cache(
    missing_paths: list[Path],
    root: Path,
    device: str,
) -> None:
    """Run face-alignment on missing images; append results to JSONL cache."""
    import face_alignment  # lazy — heavy PyTorch dep, only needed on first run

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=device,
        flip_input=False,
    )
    cache_path = root / _CACHE_FILE
    total = len(missing_paths)
    print(f"[digiface] building landmark cache for {total} images on {device}...")
    with cache_path.open("a") as cache_fh:
        for i, rel_path in enumerate(missing_paths):
            if i % 500 == 0:
                print(f"[digiface]   {i}/{total}")
            img_path = root / rel_path
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            preds = fa.get_landmarks_from_image(rgb)
            if not preds:
                continue
            lm = preds[0] if len(preds) == 1 else _pick_largest(preds)
            entry = {"path": str(rel_path), "lm": lm.tolist()}
            cache_fh.write(json.dumps(entry) + "\n")
            cache_fh.flush()
    print("[digiface] cache build complete")


class DigiFaceSource(SyntheticFaceSource):
    def __init__(
        self,
        root: Path,
        *,
        rng_seed: int = 0,
        max_samples: int | None = None,
        detector_device: str = "mps",
        build_cache_if_needed: bool = True,
    ) -> None:
        root = Path(root)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"DigiFace root does not exist: {root}")

        # Discover all <id>/<n>.png files
        discovered: list[Path] = sorted(
            Path(p).relative_to(root) for p in root.glob("*/*.png")
        )
        if not discovered:
            raise FileNotFoundError(
                f"No DigiFace-1M images in {root}. Expected <id>/<n>.png layout."
            )

        cache = _load_cache(root)

        if build_cache_if_needed:
            missing = [p for p in discovered if str(p) not in cache]
            if missing:
                _build_landmarks_cache(missing, root, detector_device)
                cache = _load_cache(root)

        # Keep only images with landmarks
        entries: list[tuple[Path, np.ndarray]] = [
            (p, cache[str(p)]) for p in discovered if str(p) in cache
        ]

        if max_samples is not None:
            entries = entries[:max_samples]

        self._root = root
        self._rng_seed = rng_seed
        self._entries = entries
        self._info = SourceInfo(
            name="digiface",
            root=root,
            sample_count=len(entries),
        )

    @property
    def info(self) -> SourceInfo:
        return self._info

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._rng_seed)
        for rel_path, landmarks in self._entries:
            img_path = self._root / rel_path
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            eye = Eye.LEFT if rng.random() < 0.5 else Eye.RIGHT
            try:
                patch = crop_eye(img, landmarks, eye)
            except ValueError:
                continue
            _validate_patch(patch)
            yield patch


__all__ = ["DigiFaceSource"]
