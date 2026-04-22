"""PyTorch Dataset wrapping the pair synthesiser.

Wraps any iterable of eye patches + the ``synthesise_pair`` function into
a ``torch.utils.data.IterableDataset``. The caller decides the source
(procedural fixtures for tests; UnityEyes + FFHQ for production).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import IterableDataset

from gazelock_ml.data.pairs import TrainingPair, synthesise_pair


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 BGR → CHW float32 in [0, 1]."""
    return torch.from_numpy(img).float().permute(2, 0, 1) / 255.0


class EyePairDataset(IterableDataset):
    """Streams TrainingPair → (warped, original, target) tensor triples.

    Target == original (the refiner learns to recover the unwarped eye
    given the warped eye + original-as-conditioning).
    """

    def __init__(
        self,
        patch_source: Callable[[], Iterable[np.ndarray]],
        rng_seed: int | None = None,
        max_offset_px: float = 8.0,
    ) -> None:
        super().__init__()
        self._patch_source = patch_source
        self._rng_seed = rng_seed
        self._max_offset_px = max_offset_px

    def __iter__(self):  # type: ignore[override]
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        seed = self._rng_seed + worker_id if self._rng_seed is not None else None
        rng = np.random.default_rng(seed)

        for patch in self._patch_source():
            pair: TrainingPair = synthesise_pair(patch, rng=rng, max_offset_px=self._max_offset_px)
            warped_t = _to_tensor(pair.warped)  # (3, H, W)
            original_t = _to_tensor(pair.original)
            # Input to refiner: 6-channel (warped ++ original)
            refiner_input = torch.cat([warped_t, original_t], dim=0)
            target = original_t
            yield refiner_input, target


__all__ = ["EyePairDataset"]
