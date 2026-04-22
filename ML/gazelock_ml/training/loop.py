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
        yield from loader


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
    "pick_device",
    "train_for_n_steps",
    "train_one_step",
]
