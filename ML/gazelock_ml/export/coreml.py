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
