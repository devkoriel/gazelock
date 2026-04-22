# GazeLock v2 — Phase 2b: ML Training, Export, and CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Phase 2a building blocks into a trained, exported, CI-tested training pipeline. End state: `make ml-train` runs 10 smoke steps on procedural fixtures without errors; `make ml-eval` prints PSNR / SSIM / identity / flicker metrics; `make ml-export` produces a Core ML `.mlpackage` that passes a PyTorch↔Core ML L∞ gate; the `ml-build.yml` workflow is green on every PR.

**Architecture:** A linear composable pipeline — `pairs` module synthesises `(I_warped, I_original, I_target)` triples from any eye patch by applying the Phase 2a analytic warp; `Dataset` wraps that synthesiser for PyTorch; `training` orchestrates the weighted loss (all four P2a loss modules) + optimizer + checkpoint IO; `metrics` computes objective quality numbers on held-out batches; `export` traces the PyTorch refiner and converts to Core ML with a tolerance-gated validation pass. Three CLI entry points (`train`, `eval`, `export`) wrap the modules for Makefile + CI use.

**Tech Stack:** Phase 2a stack + `torch.utils.data.DataLoader`, `torch.jit.trace`, `coremltools.convert`. No new third-party deps.

**Design spec:** `docs/superpowers/specs/2026-04-22-gazelock-design.md` §7.3–7.6 (training procedure, validation, Core ML export, versioning).

**Phase scope:** Phase 2b of 4. Phase 3 (Swift ML inference + UI) consumes the `.mlpackage` this phase produces. Phase 4 (release) adds blind-A/B human evaluation.

**Out of scope for this plan:**
- Mixed-precision training. Phase 2b trains in fp32; Core ML export quantises to fp16. AMP on MPS is still experimental and not worth the fragility for v1.
- TensorBoard / wandb logging. Stdout + a JSON metrics file are enough for now.
- Distributed training (single-device / single-process only).
- Dataset download scripts — the user provides `--unityeyes-root` / `--ffhq-root` paths. CI uses procedural fixtures only.
- Actually executing a full 50 k-step training run. The CI pipeline runs a 10-step smoke; the user runs a real training locally when the pipeline is ready.
- Blind human A/B (Phase 4).

**Prerequisites:**
- Phase 2a complete (`phase-2a-ml-foundations` tag exists, `.venv/` populated with `uv sync --extra dev`).
- Disk space for the VGG-16 torchvision cache (~500 MB, already on disk from P2a tests).

---

## File Structure

Files created by Phase 2b, relative to `/Users/koriel/Development/gazelock/`:

| Path | Responsibility |
|---|---|
| `ML/gazelock_ml/data/pairs.py` | Training-pair synthesiser (applies P2a warp to any real eye patch) |
| `ML/gazelock_ml/data/dataset.py` | PyTorch `Dataset` that wraps pair generators |
| `ML/gazelock_ml/training/__init__.py` | Training subpackage |
| `ML/gazelock_ml/training/composed_loss.py` | Weighted combination of the four P2a losses |
| `ML/gazelock_ml/training/loop.py` | `train_one_step`, `train_for_n_steps` |
| `ML/gazelock_ml/training/checkpoints.py` | Save / load checkpoints with metadata |
| `ML/gazelock_ml/metrics/__init__.py` | Metrics subpackage |
| `ML/gazelock_ml/metrics/psnr.py` | PSNR |
| `ML/gazelock_ml/metrics/ssim.py` | Simplified SSIM (11×11 Gaussian window) |
| `ML/gazelock_ml/metrics/identity.py` | Cosine identity metric using the P2a encoder |
| `ML/gazelock_ml/metrics/flicker.py` | Frame-to-frame L1 (flicker proxy) |
| `ML/gazelock_ml/export/__init__.py` | Export subpackage |
| `ML/gazelock_ml/export/coreml.py` | `torch.jit.trace` + `coremltools.convert` + validation gate |
| `ML/gazelock_ml/cli/__init__.py` | CLI subpackage |
| `ML/gazelock_ml/cli/train.py` | `gazelock-train` entry |
| `ML/gazelock_ml/cli/eval.py` | `gazelock-eval` entry |
| `ML/gazelock_ml/cli/export.py` | `gazelock-export` entry |
| `ML/gazelock_ml/tests/test_pairs.py` | Synthesiser tests |
| `ML/gazelock_ml/tests/test_dataset.py` | Dataset + DataLoader tests |
| `ML/gazelock_ml/tests/test_composed_loss.py` | Weighted-sum correctness |
| `ML/gazelock_ml/tests/test_training.py` | 10-step smoke run |
| `ML/gazelock_ml/tests/test_checkpoints.py` | Round-trip save/load |
| `ML/gazelock_ml/tests/test_metrics.py` | All four metrics |
| `ML/gazelock_ml/tests/test_export.py` | Core ML validation gate |
| `ML/README.md` | End-to-end instructions for training on real data |
| `Makefile` | Extended with `ml-test`, `ml-lint`, `ml-train`, `ml-eval`, `ml-export` targets |
| `.github/workflows/ml-build.yml` | Python lint + test + smoke-train + export workflow |
| `pyproject.toml` | Add `[project.scripts]` entries for the three CLIs |

No changes to the Swift side.

---

## Task 1: Training-pair synthesiser

**Files:**
- Create: `ML/gazelock_ml/data/pairs.py`
- Create: `ML/gazelock_ml/tests/test_pairs.py`

The synthesiser takes any real eye patch + an artificial target gaze and uses the Phase 2a warp to produce `(I_warped, I_original)`. The target for training is `I_original` — the refiner learns to invert the warp using the warped view + the original as conditioning.

- [ ] **Step 1: Write `ML/gazelock_ml/data/pairs.py`**

```python
"""Synthetic training-pair generator.

Given any real eye patch, we apply the P2a analytic warp with a random
target-gaze offset to produce the ``warped`` view. The refiner's training
objective is to recover the original from ``(warped, original)``.

The randomised offset simulates the distribution of target gazes the
refiner will see at inference: small (5–15°) corrections that move the
iris toward the camera. The pair synthesiser does NOT itself generate
new identities or new photographic content — it only warps real data
that was sourced from UnityEyes / FFHQ / procedural fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.warp.apply import apply_flow
from gazelock_ml.warp.tps import fit_tps, flow_field_from_tps


@dataclass(frozen=True)
class TrainingPair:
    """A ``(warped, original)`` BGR uint8 pair plus the applied offset."""

    warped: np.ndarray  # (EYE_ROI_H, EYE_ROI_W, 3) uint8
    original: np.ndarray  # (EYE_ROI_H, EYE_ROI_W, 3) uint8
    offset_px: tuple[float, float]  # the iris displacement we forced


def synthesise_pair(
    eye_patch: np.ndarray,
    rng: np.random.Generator | None = None,
    max_offset_px: float = 8.0,
) -> TrainingPair:
    """Apply the analytic warp to ``eye_patch`` and return the pair.

    Args:
        eye_patch: (EYE_ROI_H, EYE_ROI_W, 3) uint8 — the real source.
        rng: optional numpy Generator for reproducibility.
        max_offset_px: bound on iris displacement magnitude.

    Returns:
        TrainingPair with ``warped`` = eye_patch with iris shifted and
        ``original`` = eye_patch unchanged.
    """
    assert eye_patch.shape == (EYE_ROI_H, EYE_ROI_W, 3)
    assert eye_patch.dtype == np.uint8

    rng = rng or np.random.default_rng()
    dx = float(rng.uniform(-max_offset_px, max_offset_px))
    dy = float(rng.uniform(-max_offset_px, max_offset_px))

    # Control points: 4 corners (anchored to identity) + center (displaced).
    source_points = np.array(
        [
            [0.0, 0.0],
            [EYE_ROI_W - 1.0, 0.0],
            [0.0, EYE_ROI_H - 1.0],
            [EYE_ROI_W - 1.0, EYE_ROI_H - 1.0],
            [EYE_ROI_W / 2.0, EYE_ROI_H / 2.0],
        ],
        dtype=np.float64,
    )
    target_points = source_points.copy()
    target_points[-1] += np.array([dx, dy])

    coefs = fit_tps(target_points, source_points)  # inverse mapping for sampling
    flow = flow_field_from_tps(coefs, target_points, EYE_ROI_H, EYE_ROI_W)
    warped = apply_flow(eye_patch, flow)

    return TrainingPair(warped=warped, original=eye_patch.copy(), offset_px=(dx, dy))


__all__ = ["TrainingPair", "synthesise_pair"]
```

- [ ] **Step 2: Write `ML/gazelock_ml/tests/test_pairs.py`**

```python
"""Tests for training-pair synthesiser."""

from __future__ import annotations

import numpy as np

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W, make_fake_eye_patch
from gazelock_ml.data.pairs import synthesise_pair


def test_pair_shapes_match_input() -> None:
    patch = make_fake_eye_patch(seed=0)
    pair = synthesise_pair(patch, rng=np.random.default_rng(0))
    assert pair.warped.shape == (EYE_ROI_H, EYE_ROI_W, 3)
    assert pair.original.shape == (EYE_ROI_H, EYE_ROI_W, 3)
    assert pair.warped.dtype == np.uint8
    assert pair.original.dtype == np.uint8


def test_warped_differs_from_original_when_offset_nonzero() -> None:
    patch = make_fake_eye_patch(seed=1)
    rng = np.random.default_rng(42)  # produces a non-zero offset
    pair = synthesise_pair(patch, rng=rng, max_offset_px=8.0)
    assert pair.offset_px != (0.0, 0.0)
    assert not np.array_equal(pair.warped, pair.original)


def test_original_is_unchanged_from_input() -> None:
    patch = make_fake_eye_patch(seed=2)
    pair = synthesise_pair(patch, rng=np.random.default_rng(2))
    np.testing.assert_array_equal(pair.original, patch)


def test_determinism_with_same_rng_seed() -> None:
    patch = make_fake_eye_patch(seed=3)
    p1 = synthesise_pair(patch, rng=np.random.default_rng(7))
    p2 = synthesise_pair(patch, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(p1.warped, p2.warped)
    assert p1.offset_px == p2.offset_px
```

- [ ] **Step 3: Run tests**

```bash
cd /Users/koriel/Development/gazelock
source .venv/bin/activate
pytest ML/gazelock_ml/tests/test_pairs.py -v
deactivate
```

Expected: 4 passed.

- [ ] **Step 4: Stage and commit**

```bash
cd /Users/koriel/Development/gazelock
git add ML/gazelock_ml/data/pairs.py ML/gazelock_ml/tests/test_pairs.py
git commit -m "feat(ml/data): add training-pair synthesiser using analytic warp"
git push origin main
```

---

## Task 2: PyTorch Dataset wrapper

**Files:**
- Create: `ML/gazelock_ml/data/dataset.py`
- Create: `ML/gazelock_ml/tests/test_dataset.py`

A small `IterableDataset` that calls `synthesise_pair` on the fly from any iterable of eye patches. Deliberately `IterableDataset` (not map-style) because training-pair offsets are sampled per-call, so indexing doesn't make sense.

- [ ] **Step 1: Write `ML/gazelock_ml/data/dataset.py`**

```python
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
```

- [ ] **Step 2: Write `ML/gazelock_ml/tests/test_dataset.py`**

```python
"""Tests for EyePairDataset + DataLoader integration."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from gazelock_ml.data.dataset import EyePairDataset
from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W, make_fake_eye_patch


def _fixture_source(n: int = 5) -> Iterable[np.ndarray]:
    return (make_fake_eye_patch(seed=i) for i in range(n))


def test_iterdataset_yields_expected_shapes() -> None:
    ds = EyePairDataset(lambda: _fixture_source(3), rng_seed=0)
    batches = list(ds)
    assert len(batches) == 3
    for inp, tgt in batches:
        assert inp.shape == (6, EYE_ROI_H, EYE_ROI_W)
        assert tgt.shape == (3, EYE_ROI_H, EYE_ROI_W)
        assert inp.dtype == torch.float32
        assert tgt.dtype == torch.float32
        assert 0.0 <= inp.min().item() and inp.max().item() <= 1.0


def test_dataloader_batches_correctly() -> None:
    ds = EyePairDataset(lambda: _fixture_source(8), rng_seed=1)
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    batches = list(loader)
    # 8 samples, batch_size=4 → 2 batches of 4
    assert len(batches) == 2
    for inp, tgt in batches:
        assert inp.shape == (4, 6, EYE_ROI_H, EYE_ROI_W)
        assert tgt.shape == (4, 3, EYE_ROI_H, EYE_ROI_W)
```

- [ ] **Step 3: Run tests + commit**

```bash
cd /Users/koriel/Development/gazelock
source .venv/bin/activate
pytest ML/gazelock_ml/tests/test_dataset.py -v
deactivate
git add ML/gazelock_ml/data/dataset.py ML/gazelock_ml/tests/test_dataset.py
git commit -m "feat(ml/data): add EyePairDataset IterableDataset + DataLoader tests"
git push origin main
```

Expected: 2 passed.

---

## Task 3: Composed loss module

**Files:**
- Create: `ML/gazelock_ml/training/__init__.py`
- Create: `ML/gazelock_ml/training/composed_loss.py`
- Create: `ML/gazelock_ml/tests/test_composed_loss.py`

Weighted sum of the four Phase 2a loss modules.

- [ ] **Step 1: Write `ML/gazelock_ml/training/__init__.py`**

```python
"""See ``gazelock_ml`` top-level package for the full map."""
```

- [ ] **Step 2: Write `ML/gazelock_ml/training/composed_loss.py`**

```python
"""Weighted composition of L1 + VGG + identity + temporal.

Weights per design spec §7.2:
    L1 reconstruction : 1.0
    VGG perceptual    : 0.2
    Identity          : 0.5
    Temporal          : 0.1
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gazelock_ml.losses.identity import IdentityLoss
from gazelock_ml.losses.perceptual import VGGPerceptualLoss
from gazelock_ml.losses.reconstruction import L1ReconstructionLoss
from gazelock_ml.losses.temporal import TemporalConsistencyLoss


@dataclass(frozen=True)
class LossWeights:
    l1: float = 1.0
    perceptual: float = 0.2
    identity: float = 0.5
    temporal: float = 0.1


class ComposedLoss(nn.Module):
    """Weighted sum of the four loss components.

    Usage:
        loss = composed(
            pred=refiner_output,
            target=ground_truth,
            pred_neighbour=refiner_output_on_shifted_input,  # optional
        )
    """

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.l1 = L1ReconstructionLoss()
        self.perceptual = VGGPerceptualLoss()
        self.identity = IdentityLoss()
        self.temporal = TemporalConsistencyLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_neighbour: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a dict with per-component + total scalars."""
        l_l1 = self.l1(pred, target)
        l_perc = self.perceptual(pred, target)
        l_id = self.identity(pred, target)

        if pred_neighbour is not None:
            l_temp = self.temporal(pred, pred_neighbour)
        else:
            l_temp = torch.zeros((), device=pred.device, dtype=pred.dtype)

        total = (
            self.weights.l1 * l_l1
            + self.weights.perceptual * l_perc
            + self.weights.identity * l_id
            + self.weights.temporal * l_temp
        )

        return {
            "l1": l_l1.detach(),
            "perceptual": l_perc.detach(),
            "identity": l_id.detach(),
            "temporal": l_temp.detach(),
            "total": total,  # the gradient source
        }


__all__ = ["ComposedLoss", "LossWeights"]
```

- [ ] **Step 3: Write `ML/gazelock_ml/tests/test_composed_loss.py`**

```python
"""Tests for ComposedLoss weighted-sum behavior."""

from __future__ import annotations

import torch

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.training.composed_loss import ComposedLoss, LossWeights


def _rand_pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    p = torch.rand((2, 3, EYE_ROI_H, EYE_ROI_W))
    t = torch.rand((2, 3, EYE_ROI_H, EYE_ROI_W))
    return p, t


def test_composed_returns_dict_with_all_keys() -> None:
    loss = ComposedLoss()
    p, t = _rand_pair()
    out = loss(p, t)
    assert set(out.keys()) == {"l1", "perceptual", "identity", "temporal", "total"}
    for v in out.values():
        assert v.shape == torch.Size([])


def test_temporal_is_zero_when_no_neighbour_provided() -> None:
    loss = ComposedLoss()
    p, t = _rand_pair()
    out = loss(p, t)
    assert out["temporal"].item() == 0.0


def test_total_matches_weighted_sum() -> None:
    weights = LossWeights(l1=1.0, perceptual=0.2, identity=0.5, temporal=0.1)
    loss = ComposedLoss(weights=weights)
    p, t = _rand_pair()
    neighbour = p + 0.01 * torch.randn_like(p)
    out = loss(p, t, pred_neighbour=neighbour)

    expected = (
        weights.l1 * out["l1"]
        + weights.perceptual * out["perceptual"]
        + weights.identity * out["identity"]
        + weights.temporal * out["temporal"]
    )
    torch.testing.assert_close(out["total"], expected, rtol=1e-5, atol=1e-5)


def test_total_has_gradient() -> None:
    loss = ComposedLoss()
    p, t = _rand_pair()
    p.requires_grad_(True)
    out = loss(p, t)
    out["total"].backward()
    assert p.grad is not None
    assert not torch.isnan(p.grad).any()
```

- [ ] **Step 4: Run tests + commit**

```bash
cd /Users/koriel/Development/gazelock
source .venv/bin/activate
pytest ML/gazelock_ml/tests/test_composed_loss.py -v
deactivate
git add ML/gazelock_ml/training ML/gazelock_ml/tests/test_composed_loss.py
git commit -m "feat(ml/training): add ComposedLoss weighted combination"
git push origin main
```

Expected: 4 passed.

---

## Task 4: Training loop + checkpointing

**Files:**
- Create: `ML/gazelock_ml/training/loop.py`
- Create: `ML/gazelock_ml/training/checkpoints.py`
- Create: `ML/gazelock_ml/tests/test_training.py`
- Create: `ML/gazelock_ml/tests/test_checkpoints.py`

- [ ] **Step 1: Write `ML/gazelock_ml/training/loop.py`**

```python
"""Training loop core.

Single-GPU / single-MPS, fp32 (no mixed precision for v1). Caller owns
the model, loss, optimizer, and data loader; this module wires them
into a loop that runs ``n_steps`` gradient updates.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from gazelock_ml.training.composed_loss import ComposedLoss


@dataclass
class TrainStepResult:
    step: int
    loss_total: float
    loss_components: dict[str, float]
    lr: float


def _infinite_iter(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Wrap a DataLoader to yield forever (wrap-around on StopIteration)."""
    while True:
        for batch in loader:
            yield batch


def train_one_step(
    model: nn.Module,
    loss_fn: ComposedLoss,
    optimizer: Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> TrainStepResult:
    """Run exactly one forward + backward + optimizer step."""
    model.train()
    inp, target = batch
    inp = inp.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    pred = model(inp)
    out = loss_fn(pred, target)
    total: torch.Tensor = out["total"]
    total.backward()
    optimizer.step()

    return TrainStepResult(
        step=0,  # caller fills in
        loss_total=float(total.detach().item()),
        loss_components={k: float(v.item()) for k, v in out.items() if k != "total"},
        lr=float(optimizer.param_groups[0]["lr"]),
    )


def train_for_n_steps(
    *,
    model: nn.Module,
    loss_fn: ComposedLoss,
    optimizer: Optimizer,
    loader: DataLoader,
    device: torch.device,
    n_steps: int,
    lr_scheduler: LRScheduler | None = None,
    on_step: callable | None = None,
) -> list[TrainStepResult]:
    """Run ``n_steps`` gradient updates. Returns per-step results."""
    history: list[TrainStepResult] = []
    iterator = _infinite_iter(loader)
    for step in range(n_steps):
        batch = next(iterator)
        result = train_one_step(model, loss_fn, optimizer, batch, device)
        result = TrainStepResult(
            step=step,
            loss_total=result.loss_total,
            loss_components=result.loss_components,
            lr=result.lr,
        )
        history.append(result)
        if lr_scheduler is not None:
            lr_scheduler.step()
        if on_step is not None:
            on_step(result)
    return history


def pick_device() -> torch.device:
    """Prefer MPS (Apple Silicon GPU); fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = [
    "TrainStepResult",
    "train_one_step",
    "train_for_n_steps",
    "pick_device",
]
```

- [ ] **Step 2: Write `ML/gazelock_ml/training/checkpoints.py`**

```python
"""Checkpoint save/load — model + optimizer state + step + metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class CheckpointMetadata:
    step: int
    loss_total: float
    git_sha: str | None = None
    notes: str = ""


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    metadata: CheckpointMetadata,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": asdict(metadata),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> CheckpointMetadata:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    md = payload["metadata"]
    return CheckpointMetadata(**md)


def write_run_manifest(path: Path, data: dict) -> None:
    """Persist a JSON manifest next to the checkpoint (step counts,
    losses, git SHA, etc.)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


__all__ = [
    "CheckpointMetadata",
    "save_checkpoint",
    "load_checkpoint",
    "write_run_manifest",
]
```

- [ ] **Step 3: Write `ML/gazelock_ml/tests/test_training.py`**

```python
"""End-to-end smoke test: 10 steps on procedural fixtures."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from gazelock_ml.data.dataset import EyePairDataset
from gazelock_ml.data.fixtures import make_fake_eye_patch
from gazelock_ml.models.refiner import RefinerUNet
from gazelock_ml.training.composed_loss import ComposedLoss
from gazelock_ml.training.loop import pick_device, train_for_n_steps


def _source(n: int = 64) -> Iterable[np.ndarray]:
    return (make_fake_eye_patch(seed=i) for i in range(n))


def test_10_step_training_runs_without_nan() -> None:
    device = pick_device()
    torch.manual_seed(0)
    np.random.seed(0)

    model = RefinerUNet().to(device)
    loss_fn = ComposedLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    ds = EyePairDataset(lambda: _source(32), rng_seed=0)
    loader = DataLoader(ds, batch_size=4, num_workers=0)

    history = train_for_n_steps(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        loader=loader,
        device=device,
        n_steps=10,
    )

    assert len(history) == 10
    for r in history:
        assert r.loss_total == r.loss_total  # not NaN (NaN != NaN)
        assert r.loss_total > 0
        assert r.lr == 1e-4
```

- [ ] **Step 4: Write `ML/gazelock_ml/tests/test_checkpoints.py`**

```python
"""Round-trip checkpoint save/load tests."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW

from gazelock_ml.models.refiner import RefinerUNet
from gazelock_ml.training.checkpoints import (
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
    write_run_manifest,
)


def test_roundtrip_preserves_weights(tmp_path: Path) -> None:
    model = RefinerUNet()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    torch.manual_seed(0)
    state_before = {k: v.clone() for k, v in model.state_dict().items()}

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        metadata=CheckpointMetadata(step=42, loss_total=1.5, git_sha="abc", notes="x"),
    )

    new_model = RefinerUNet()
    new_opt = AdamW(new_model.parameters(), lr=1e-4)
    md = load_checkpoint(ckpt_path, new_model, new_opt)

    assert md.step == 42
    assert md.loss_total == 1.5
    assert md.git_sha == "abc"
    for k, v in state_before.items():
        torch.testing.assert_close(new_model.state_dict()[k], v)


def test_manifest_writes_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_run_manifest(manifest_path, {"step": 1, "loss": 1.2})
    import json
    data = json.loads(manifest_path.read_text())
    assert data == {"step": 1, "loss": 1.2}
```

- [ ] **Step 5: Run tests + commit**

```bash
cd /Users/koriel/Development/gazelock
source .venv/bin/activate
pytest ML/gazelock_ml/tests/test_training.py ML/gazelock_ml/tests/test_checkpoints.py -v
deactivate
git add ML/gazelock_ml/training ML/gazelock_ml/tests/test_training.py ML/gazelock_ml/tests/test_checkpoints.py
git commit -m "feat(ml/training): add training loop + checkpoint IO"
git push origin main
```

Expected: 3 passed.

---

## Task 5: Quality metrics

**Files:**
- Create: `ML/gazelock_ml/metrics/__init__.py`
- Create: `ML/gazelock_ml/metrics/psnr.py`
- Create: `ML/gazelock_ml/metrics/ssim.py`
- Create: `ML/gazelock_ml/metrics/identity.py`
- Create: `ML/gazelock_ml/metrics/flicker.py`
- Create: `ML/gazelock_ml/tests/test_metrics.py`

- [ ] **Step 1: Write `ML/gazelock_ml/metrics/__init__.py`**

```python
"""See ``gazelock_ml`` top-level package for the full map."""
```

- [ ] **Step 2: Write `ML/gazelock_ml/metrics/psnr.py`**

```python
"""Peak Signal-to-Noise Ratio."""

from __future__ import annotations

import math

import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 100.0  # effectively infinite
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


__all__ = ["psnr"]
```

- [ ] **Step 3: Write `ML/gazelock_ml/metrics/ssim.py`**

```python
"""Simplified SSIM — 11×11 Gaussian window, per-channel mean."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_K1 = 0.01
_K2 = 0.03


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)  # (size, size)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    window_size: int = 11,
) -> float:
    """Compute mean SSIM over batch + channels. Inputs in [0, max_val]."""
    assert pred.shape == target.shape
    channels = pred.shape[1]
    window = _gaussian_window(window_size).to(pred.device).to(pred.dtype)
    window = window.expand(channels, 1, window_size, window_size).contiguous()

    c1 = (_K1 * max_val) ** 2
    c2 = (_K2 * max_val) ** 2

    mu_x = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(torch.mean(num / den).item())


__all__ = ["ssim"]
```

- [ ] **Step 4: Write `ML/gazelock_ml/metrics/identity.py`**

```python
"""Identity cosine similarity metric (using the P2a eye encoder)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gazelock_ml.models.eye_encoder import EyeIdentityEncoder


def identity_cosine(
    pred: torch.Tensor,
    target: torch.Tensor,
    encoder: EyeIdentityEncoder | None = None,
) -> float:
    """Mean cosine similarity of (pred, target) eye embeddings."""
    enc = encoder or EyeIdentityEncoder()
    enc.eval()
    with torch.no_grad():
        ep = enc(pred)
        et = enc(target)
    cos = F.cosine_similarity(ep, et, dim=1)
    return float(cos.mean().item())


__all__ = ["identity_cosine"]
```

- [ ] **Step 5: Write `ML/gazelock_ml/metrics/flicker.py`**

```python
"""Frame-to-frame L1 — a flicker proxy for video sequences."""

from __future__ import annotations

import torch


def frame_to_frame_l1(sequence: torch.Tensor) -> float:
    """Given (N, 3, H, W), compute mean |frame_i - frame_{i-1}| over N-1 pairs."""
    assert sequence.dim() == 4, "expected (N, 3, H, W)"
    if sequence.shape[0] < 2:
        return 0.0
    diffs = torch.abs(sequence[1:] - sequence[:-1])
    return float(diffs.mean().item())


__all__ = ["frame_to_frame_l1"]
```

- [ ] **Step 6: Write `ML/gazelock_ml/tests/test_metrics.py`**

```python
"""Tests for PSNR / SSIM / identity / flicker."""

from __future__ import annotations

import torch

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W
from gazelock_ml.metrics.flicker import frame_to_frame_l1
from gazelock_ml.metrics.identity import identity_cosine
from gazelock_ml.metrics.psnr import psnr
from gazelock_ml.metrics.ssim import ssim


def test_psnr_infinite_on_identical_inputs() -> None:
    x = torch.rand((1, 3, 8, 8))
    assert psnr(x, x) >= 99.0  # "effectively infinite" sentinel


def test_psnr_finite_and_positive_on_different_inputs() -> None:
    torch.manual_seed(0)
    x = torch.rand((1, 3, 8, 8))
    y = torch.rand((1, 3, 8, 8))
    val = psnr(x, y)
    assert 0.0 < val < 99.0


def test_ssim_one_on_identical_inputs() -> None:
    x = torch.rand((1, 3, 16, 16))
    assert abs(ssim(x, x) - 1.0) < 1e-3


def test_ssim_lower_on_different_inputs() -> None:
    torch.manual_seed(1)
    x = torch.rand((1, 3, 16, 16))
    y = torch.rand((1, 3, 16, 16))
    assert ssim(x, y) < 0.95


def test_identity_cosine_close_to_one_on_identical_inputs() -> None:
    x = torch.rand((2, 3, EYE_ROI_H, EYE_ROI_W))
    assert identity_cosine(x, x) > 0.999


def test_flicker_zero_on_constant_sequence() -> None:
    frame = torch.rand((3, 8, 8))
    seq = frame.unsqueeze(0).expand(5, -1, -1, -1)
    assert frame_to_frame_l1(seq) == 0.0


def test_flicker_positive_on_varying_sequence() -> None:
    torch.manual_seed(2)
    seq = torch.rand((5, 3, 8, 8))
    assert frame_to_frame_l1(seq) > 0.0
```

- [ ] **Step 7: Run tests + commit**

```bash
cd /Users/koriel/Development/gazelock
source .venv/bin/activate
pytest ML/gazelock_ml/tests/test_metrics.py -v
deactivate
git add ML/gazelock_ml/metrics ML/gazelock_ml/tests/test_metrics.py
git commit -m "feat(ml/metrics): add PSNR, SSIM, identity cosine, flicker metrics"
git push origin main
```

Expected: 7 passed.

---

## Task 6: Core ML export + L∞ validation gate

**Files:**
- Create: `ML/gazelock_ml/export/__init__.py`
- Create: `ML/gazelock_ml/export/coreml.py`
- Create: `ML/gazelock_ml/tests/test_export.py`

- [ ] **Step 1: Write `ML/gazelock_ml/export/__init__.py`**

```python
"""See ``gazelock_ml`` top-level package for the full map."""
```

- [ ] **Step 2: Write `ML/gazelock_ml/export/coreml.py`**

```python
"""PyTorch → Core ML conversion with a tolerance-gated validation pass.

Process:
    1. Set the refiner to eval mode.
    2. Trace it with ``torch.jit.trace`` on a deterministic sample input.
    3. Convert the traced graph via ``coremltools.convert`` targeting
       float16 weights and macOS 15.
    4. Run both the PyTorch model and the Core ML model on N random
       inputs and compute L∞ on the outputs. If it exceeds the tolerance,
       raise.
"""

from __future__ import annotations

from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W

MACOS_MIN_VERSION = "15"
DEFAULT_TOLERANCE = 1e-3
DEFAULT_VALIDATION_SAMPLES = 1000


def trace_refiner(model: nn.Module, device: torch.device) -> torch.jit.ScriptModule:
    model = model.to(device).eval()
    sample = torch.zeros((1, 6, EYE_ROI_H, EYE_ROI_W), device=device)
    with torch.no_grad():
        traced = torch.jit.trace(model, sample, strict=False)
    return traced


def convert_to_coreml(
    traced: torch.jit.ScriptModule,
    output_path: Path,
) -> ct.models.MLModel:
    input_shape = (1, 6, EYE_ROI_H, EYE_ROI_W)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=input_shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    return mlmodel


def validate_linf(
    pytorch_model: nn.Module,
    coreml_model: ct.models.MLModel,
    num_samples: int = DEFAULT_VALIDATION_SAMPLES,
    tolerance: float = DEFAULT_TOLERANCE,
    device: torch.device | None = None,
) -> float:
    """Return the L∞ over ``num_samples`` random inputs. Raises if over tol."""
    device = device or torch.device("cpu")
    pytorch_model = pytorch_model.to(device).eval()

    rng = np.random.default_rng(0)
    max_err = 0.0
    with torch.no_grad():
        for _ in range(num_samples):
            x_np = rng.random((1, 6, EYE_ROI_H, EYE_ROI_W), dtype=np.float32)
            x_t = torch.from_numpy(x_np).to(device)
            y_pt = pytorch_model(x_t).cpu().numpy()

            y_cm_dict = coreml_model.predict({"input": x_np})
            y_cm = next(iter(y_cm_dict.values()))

            err = float(np.abs(y_pt - y_cm).max())
            if err > max_err:
                max_err = err

    if max_err > tolerance:
        raise AssertionError(
            f"PyTorch↔CoreML L∞ {max_err:.2e} exceeds tolerance {tolerance:.0e}"
        )
    return max_err


def export(
    pytorch_model: nn.Module,
    output_path: Path,
    num_validation_samples: int = DEFAULT_VALIDATION_SAMPLES,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """End-to-end: trace, convert, validate. Returns summary metrics."""
    device = torch.device("cpu")  # Core ML conversion prefers CPU input
    traced = trace_refiner(pytorch_model, device=device)
    mlmodel = convert_to_coreml(traced, output_path)
    linf = validate_linf(
        pytorch_model,
        mlmodel,
        num_samples=num_validation_samples,
        tolerance=tolerance,
        device=device,
    )
    return {
        "path": str(output_path),
        "linf": linf,
        "tolerance": tolerance,
        "samples": num_validation_samples,
    }


__all__ = [
    "DEFAULT_TOLERANCE",
    "DEFAULT_VALIDATION_SAMPLES",
    "convert_to_coreml",
    "export",
    "trace_refiner",
    "validate_linf",
]
```

- [ ] **Step 3: Write `ML/gazelock_ml/tests/test_export.py`**

```python
"""Smoke test for Core ML export — 16-sample validation gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from gazelock_ml.export.coreml import export
from gazelock_ml.models.refiner import RefinerUNet


@pytest.mark.slow
def test_export_runs_and_passes_linf_gate(tmp_path: Path) -> None:
    model = RefinerUNet()
    model.eval()
    out_path = tmp_path / "refiner.mlpackage"
    result = export(
        model,
        out_path,
        num_validation_samples=16,  # Fast smoke — full export uses 1000
        tolerance=1e-3,
    )
    assert out_path.exists()
    assert result["linf"] <= 1e-3
```

The `@pytest.mark.slow` marker lets fast CI skip this test if needed (Core ML conversion takes 20–60 s even for a tiny model). Phase 2b's CI runs it; local dev can `pytest -m "not slow"` to skip.

- [ ] **Step 4: Add `slow` marker to pyproject.toml**

Edit `pyproject.toml`, inside `[tool.pytest.ini_options]`, add:
```toml
markers = ["slow: marks tests as slow (select with -m slow)"]
```

Replace the existing `[tool.pytest.ini_options]` block with:
```toml
[tool.pytest.ini_options]
testpaths = ["ML/gazelock_ml/tests"]
python_files = ["test_*.py"]
addopts = "-ra --strict-markers --strict-config"
markers = ["slow: marks tests as slow (select with -m slow)"]
```

- [ ] **Step 5: Run the slow test manually + commit**

```bash
cd /Users/koriel/Development/gazelock
source .venv/bin/activate
pytest ML/gazelock_ml/tests/test_export.py -v
deactivate
git add ML/gazelock_ml/export ML/gazelock_ml/tests/test_export.py pyproject.toml
git commit -m "feat(ml/export): add Core ML conversion + L∞ validation gate"
git push origin main
```

Expected: 1 passed (runtime ~30–60 s for the Core ML conversion + 16 validation samples).

---

## Task 7: CLI entry points (train, eval, export)

**Files:**
- Create: `ML/gazelock_ml/cli/__init__.py`
- Create: `ML/gazelock_ml/cli/train.py`
- Create: `ML/gazelock_ml/cli/eval.py`
- Create: `ML/gazelock_ml/cli/export.py`
- Edit: `pyproject.toml` — add `[project.scripts]` entries

- [ ] **Step 1: Write `ML/gazelock_ml/cli/__init__.py`**

```python
"""See ``gazelock_ml`` top-level package for the full map."""
```

- [ ] **Step 2: Write `ML/gazelock_ml/cli/train.py`**

```python
"""CLI: gazelock-train — kick off a training run."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from gazelock_ml.data.dataset import EyePairDataset
from gazelock_ml.data.fixtures import make_fake_eye_patch
from gazelock_ml.models.refiner import RefinerUNet
from gazelock_ml.training.checkpoints import (
    CheckpointMetadata,
    save_checkpoint,
    write_run_manifest,
)
from gazelock_ml.training.composed_loss import ComposedLoss
from gazelock_ml.training.loop import pick_device, train_for_n_steps


def _fixture_source(n: int) -> Iterable[np.ndarray]:
    return (make_fake_eye_patch(seed=i) for i in range(n))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the GazeLock refiner UNet.")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("weights/debug"))
    parser.add_argument("--seed", type=int, default=0)
    # Real-data options (not used for fixtures):
    parser.add_argument("--unityeyes-root", type=Path, default=None)
    parser.add_argument("--ffhq-root", type=Path, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"[train] device={device}")

    model = RefinerUNet().to(device)
    loss_fn = ComposedLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps)

    if args.unityeyes_root or args.ffhq_root:
        raise NotImplementedError(
            "Real-dataset loaders (UnityEyes/FFHQ) are plumbed; hook them into "
            "the Dataset factory below. Phase 2b ships the fixture path only."
        )

    ds = EyePairDataset(lambda: _fixture_source(args.steps * args.batch_size * 2), rng_seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    def _log_step(r):
        if r.step % max(1, args.steps // 5) == 0 or r.step == args.steps - 1:
            print(f"[train] step={r.step} loss_total={r.loss_total:.4f} lr={r.lr:.2e}")

    history = train_for_n_steps(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        loader=loader,
        device=device,
        n_steps=args.steps,
        lr_scheduler=scheduler,
        on_step=_log_step,
    )

    # Save final checkpoint
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "final.pt"
    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        metadata=CheckpointMetadata(
            step=history[-1].step,
            loss_total=history[-1].loss_total,
            notes=f"smoke/fixture run, {args.steps} steps, bs={args.batch_size}",
        ),
    )

    manifest = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_loss": history[-1].loss_total,
        "loss_components": history[-1].loss_components,
    }
    write_run_manifest(args.output_dir / "manifest.json", manifest)

    print(f"[train] wrote checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `ML/gazelock_ml/cli/eval.py`**

```python
"""CLI: gazelock-eval — compute objective metrics on a held-out set."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gazelock_ml.data.dataset import EyePairDataset
from gazelock_ml.data.fixtures import make_fake_eye_patch
from gazelock_ml.metrics.flicker import frame_to_frame_l1
from gazelock_ml.metrics.identity import identity_cosine
from gazelock_ml.metrics.psnr import psnr
from gazelock_ml.metrics.ssim import ssim
from gazelock_ml.models.refiner import RefinerUNet
from gazelock_ml.training.checkpoints import load_checkpoint


def _fixture_source(n: int, start_seed: int) -> Iterable[np.ndarray]:
    return (make_fake_eye_patch(seed=start_seed + i) for i in range(n))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the GazeLock refiner.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    model = RefinerUNet()
    load_checkpoint(args.checkpoint, model)
    model.eval()

    ds = EyePairDataset(
        lambda: _fixture_source(args.batches * args.batch_size, start_seed=10_000),
        rng_seed=100,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    psnrs: list[float] = []
    ssims: list[float] = []
    ids: list[float] = []
    last_preds: list[torch.Tensor] = []
    with torch.no_grad():
        for inp, tgt in loader:
            pred = model(inp)
            psnrs.append(psnr(pred, tgt))
            ssims.append(ssim(pred, tgt))
            ids.append(identity_cosine(pred, tgt))
            last_preds.append(pred[0])
    preds_stack = torch.stack(last_preds, dim=0) if last_preds else torch.zeros((0, 3, 1, 1))
    flicker = frame_to_frame_l1(preds_stack) if preds_stack.shape[0] >= 2 else 0.0

    print(f"[eval] PSNR={sum(psnrs) / len(psnrs):.2f}")
    print(f"[eval] SSIM={sum(ssims) / len(ssims):.4f}")
    print(f"[eval] identity_cos={sum(ids) / len(ids):.4f}")
    print(f"[eval] flicker_l1={flicker:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `ML/gazelock_ml/cli/export.py`**

```python
"""CLI: gazelock-export — Core ML export with L∞ validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from gazelock_ml.export.coreml import (
    DEFAULT_TOLERANCE,
    DEFAULT_VALIDATION_SAMPLES,
    export,
)
from gazelock_ml.models.refiner import RefinerUNet
from gazelock_ml.training.checkpoints import load_checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the refiner to Core ML.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("weights/refiner.mlpackage"))
    parser.add_argument("--samples", type=int, default=DEFAULT_VALIDATION_SAMPLES)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)

    model = RefinerUNet()
    load_checkpoint(args.checkpoint, model)

    result = export(
        model,
        args.output,
        num_validation_samples=args.samples,
        tolerance=args.tolerance,
    )
    print(f"[export] wrote {result['path']}")
    print(f"[export] L∞ = {result['linf']:.2e} (tol {result['tolerance']:.0e})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add scripts to pyproject.toml**

Edit `pyproject.toml`. Inside the `[project]` block (after the `dependencies = [...]`, before `[project.optional-dependencies]`), add:

```toml
[project.scripts]
gazelock-train = "gazelock_ml.cli.train:main"
gazelock-eval = "gazelock_ml.cli.eval:main"
gazelock-export = "gazelock_ml.cli.export:main"
```

- [ ] **Step 6: Reinstall editable package + smoke-test one CLI**

```bash
cd /Users/koriel/Development/gazelock
source .venv/bin/activate
uv pip install -e .
gazelock-train --steps 2 --batch-size 2 --output-dir /tmp/gazelock-cli-smoke
ls /tmp/gazelock-cli-smoke
deactivate
```

Expected: `final.pt` and `manifest.json` in `/tmp/gazelock-cli-smoke`.

- [ ] **Step 7: Stage and commit**

```bash
cd /Users/koriel/Development/gazelock
git add ML/gazelock_ml/cli pyproject.toml uv.lock
git commit -m "feat(ml/cli): add train / eval / export entry points"
git push origin main
```

Note: `uv.lock` may change when adding `[project.scripts]`; include it if so. If `git status` shows no change to `uv.lock`, drop it from `git add`.

---

## Task 8: ML-side Makefile targets + ML README

**Files:**
- Edit: `Makefile` — add ML targets
- Create: `ML/README.md`

- [ ] **Step 1: Append ML targets to Makefile**

Read current `Makefile`, then append at the end:

```makefile
# -------- ML / Python pipeline --------

.PHONY: ml-setup ml-test ml-test-slow ml-lint ml-train ml-eval ml-export ml-verify

ml-setup:
	@echo "==> Installing Python dev dependencies via uv"
	@uv sync --extra dev
	@uv pip install -e .

ml-test:
	@echo "==> Running ML test suite (fast tests only)"
	@uv run pytest ML/gazelock_ml/tests -m "not slow" -v

ml-test-slow:
	@echo "==> Running ML test suite (including slow tests — Core ML export)"
	@uv run pytest ML/gazelock_ml/tests -v

ml-lint:
	@echo "==> Linting ML package with ruff"
	@uv run ruff check ML/gazelock_ml

ml-train:
	@echo "==> Smoke training run (10 steps on fixtures)"
	@uv run gazelock-train --steps 10 --batch-size 4 --output-dir weights/debug

ml-eval:
	@echo "==> Evaluating weights/debug/final.pt"
	@uv run gazelock-eval --checkpoint weights/debug/final.pt

ml-export:
	@echo "==> Exporting to Core ML"
	@uv run gazelock-export --checkpoint weights/debug/final.pt --output weights/refiner.mlpackage

ml-verify: ml-lint ml-test
	@echo "==> ml-verify: lint + test passed"
```

- [ ] **Step 2: Write `ML/README.md`**

```markdown
# gazelock_ml — Training & Export Pipeline

Internal Python package that produces the refiner `.mlpackage` consumed
by the Swift app.

## One-shot setup

```bash
make ml-setup
```

That runs `uv sync --extra dev` + `uv pip install -e .`. After it, the
CLIs (`gazelock-train`, `gazelock-eval`, `gazelock-export`) are on PATH.

## Smoke run (end-to-end on procedural fixtures)

```bash
make ml-train   # 10 steps; writes weights/debug/final.pt
make ml-eval    # reads that checkpoint, prints PSNR/SSIM/identity/flicker
make ml-export  # writes weights/refiner.mlpackage with L∞ gate
```

No datasets required — everything runs on synthetic fixtures. The
resulting weights are garbage (the model only saw 10 gradient steps on
trivial patterns) but the pipeline is validated end-to-end.

## Real training

1. Download UnityEyes renders. Organise as:
   ```
   <unityeyes_root>/
       identity_0001/0.jpg 0.json 1.jpg 1.json ...
       identity_0002/...
   ```
2. Download FFHQ images (any resolution; 512×512 or 1024×1024 recommended).
3. *(Plumbing for real-data paths is stubbed in `gazelock-train`;
   Phase 2c adds the `--unityeyes-root` + `--ffhq-root` wiring. For
   now train on fixtures; when you're ready, wire up the loaders by
   replacing the `_fixture_source` lambda in `ML/gazelock_ml/cli/train.py`
   with `UnityEyesDataset(...).iter_identity(...)` compositions.)*

## Tests

```bash
make ml-test       # fast tests only
make ml-test-slow  # includes Core ML export test (~30–60 s)
```

## License

MIT (see repository `LICENSE`). Note the UnityEyes-data licensing
constraint documented in the top-level README; it applies when
shipping trained weights.
```

- [ ] **Step 3: Test the new Makefile targets**

```bash
cd /Users/koriel/Development/gazelock
make ml-verify
```

Expected: ruff clean + all non-slow tests pass.

- [ ] **Step 4: Stage and commit**

```bash
cd /Users/koriel/Development/gazelock
git add Makefile ML/README.md
git commit -m "chore(ml): add ml-* Makefile targets and ML README"
git push origin main
```

---

## Task 9: Python CI workflow

**Files:**
- Create: `.github/workflows/ml-build.yml`

- [ ] **Step 1: Write `.github/workflows/ml-build.yml`**

```yaml
name: ML Build

on:
  pull_request:
    branches: [main]
    paths:
      - "ML/**"
      - "pyproject.toml"
      - "uv.lock"
      - "Makefile"
      - ".github/workflows/ml-build.yml"
  push:
    branches: [main]
    paths:
      - "ML/**"
      - "pyproject.toml"
      - "uv.lock"
      - "Makefile"
      - ".github/workflows/ml-build.yml"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  ml-test:
    name: ML Test (macOS 15, Apple Silicon)
    runs-on: macos-15
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: latest

      - name: Cache uv env
        uses: actions/cache@v4
        with:
          path: |
            ~/.cache/uv
            .venv
          key: uv-${{ runner.os }}-${{ hashFiles('uv.lock') }}

      - name: Cache torch hub (VGG-16 weights)
        uses: actions/cache@v4
        with:
          path: ~/.cache/torch
          key: torch-${{ runner.os }}

      - name: Sync deps + install package
        run: |
          uv venv --python 3.12
          uv sync --extra dev --no-install-project
          uv pip install -e .

      - name: Lint
        run: make ml-lint

      - name: Test (fast)
        run: make ml-test

      - name: Smoke train
        run: make ml-train

      - name: Smoke export
        run: uv run gazelock-export --checkpoint weights/debug/final.pt --output weights/refiner.mlpackage --samples 16

      - name: Upload .mlpackage as build artifact
        uses: actions/upload-artifact@v4
        with:
          name: refiner-mlpackage
          path: weights/refiner.mlpackage
          retention-days: 7
```

- [ ] **Step 2: Stage and commit**

```bash
cd /Users/koriel/Development/gazelock
git add .github/workflows/ml-build.yml
git commit -m "ci: add ML build workflow (lint + test + smoke-train + export)"
git push origin main
```

- [ ] **Step 3: Verify locally (CI parity)**

```bash
cd /Users/koriel/Development/gazelock
make ml-verify
make ml-train
uv run gazelock-export --checkpoint weights/debug/final.pt --output /tmp/ci-smoke.mlpackage --samples 16
ls /tmp/ci-smoke.mlpackage
```

Expected: all steps succeed; `.mlpackage` bundle exists.

- [ ] **Step 4: Wait for CI run on origin/main, verify green**

Run:
```bash
gh run list --workflow ml-build.yml --limit 1
```

Expected: latest run shows `completed success`. If it fails, diagnose from the log.

---

## Task 10: Phase 2b final verification + tag

**Files:** none created; end-state audit.

- [ ] **Step 1: Full ML test suite (including slow)**

```bash
cd /Users/koriel/Development/gazelock
make ml-test-slow
```

Expected: every test passes (~27 tests total).

- [ ] **Step 2: Full local smoke pipeline**

```bash
cd /Users/koriel/Development/gazelock
make ml-verify
make ml-train
make ml-eval
make ml-export
```

Expected: no errors at any stage.

- [ ] **Step 3: Verify clean working tree**

```bash
cd /Users/koriel/Development/gazelock
git status --short
```

Expected: empty OR only `?? .superpowers/` and `?? weights/` (weights generated by smoke run; should be gitignored via Git LFS rules — actually they should NOT be committed; confirm `weights/debug/` and `weights/*.mlpackage` don't show up as staged).

**If `weights/` shows up in `git status`:** add the following to `.gitignore` and commit:
```gitignore
weights/debug/
```
(Real release weights go to `weights/refiner.mlpackage` and are committed via Git LFS — that's a Phase 4 concern; for now, debug artefacts stay untracked.)

- [ ] **Step 4: Tag phase-2b end**

```bash
cd /Users/koriel/Development/gazelock
git tag -a phase-2b-ml-training -m "Phase 2b: ML training + export + CI complete

Training pipeline (pair synth + Dataset + composed loss + loop +
checkpoints), quality metrics (PSNR, SSIM, identity, flicker), Core ML
export with L∞ validation gate, three CLIs (train, eval, export),
Makefile ml-* targets, and the ml-build GitHub Actions workflow.

End state: make ml-verify passes locally; ml-build runs green on
origin; a fp16 Core ML .mlpackage passes the L∞ tolerance check on
16 random inputs.

Ready for Phase 3 (Swift ML inference + UI)."
git push origin phase-2b-ml-training
```

- [ ] **Step 5: Confirm tag visible on origin**

```bash
cd /Users/koriel/Development/gazelock
git ls-remote --tags origin | grep phase-2b-ml-training
```

Expected: the tag line is printed.

- [ ] **Step 6: Print handoff summary**

Manually confirm for the report:
- `make ml-test-slow` all green
- `make ml-verify` clean
- `make ml-train && make ml-eval && make ml-export` all succeed
- `.mlpackage` written and passes L∞
- CI green on origin
- Tag `phase-2b-ml-training` pushed
- HEAD SHA == tag SHA

---

## Notes for the executing engineer

- **Mixed precision is deliberately omitted.** Adding AMP on MPS is a small task that can land in Phase 2c if real training shows fp32 is memory-pinched. Don't add it here.
- **Real-data loaders are stubbed in the CLI.** `gazelock-train --unityeyes-root` raises `NotImplementedError` on purpose. Phase 2c wires `UnityEyesDataset` + `FFHQEyeExtractor` into the `Dataset` factory. The `_fixture_source` lambda in `cli/train.py` is the replace-point.
- **The `@pytest.mark.slow` marker keeps local dev fast.** `make ml-test` skips the Core ML export test; CI runs everything but the test itself is ~30–60 s.
- **Core ML export quirks.**
  - `torch.jit.trace` with `strict=False` allows Python-native control flow in the model.
  - `ct.ComputeUnit.ALL` lets the Neural Engine run the refiner when available, with CPU fallback.
  - `ct.precision.FLOAT16` quantises weights; the L∞ gate catches catastrophic deviations but subtle quality loss is possible. In Phase 4 we may tighten the tolerance as quality data comes in.
- **weights/debug/ is ephemeral.** Debug checkpoints from `make ml-train` should be gitignored; they're 10-step smoke runs, not release artefacts. Real release weights live in `weights/refiner.mlpackage` tracked via Git LFS — populated during the Phase 4 release process, not Phase 2b.
- **CI cache keys.** The `uv.lock` hash and a generic torch cache key let subsequent runs skip the ~2 GB torch download. If torch version changes, invalidate by bumping the `uv.lock` file.

---

*End of Phase 2b plan.*
