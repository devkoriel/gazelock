"""Thin-plate spline interpolation for dense 2D flow fields.

Given a set of source and target control points, solve the TPS system
and evaluate on a dense grid to produce a flow field suitable for
bilinear sampling.

References:
    Bookstein 1989, "Principal Warps: Thin-Plate Splines and the
    Decomposition of Deformations." IEEE PAMI 11(6).
"""

from __future__ import annotations

import numpy as np


def _phi(r2: np.ndarray) -> np.ndarray:
    """TPS radial basis: r^2 log r, safe at r=0."""
    out = np.zeros_like(r2)
    mask = r2 > 1e-12
    out[mask] = r2[mask] * np.log(np.sqrt(r2[mask]))
    return out


def fit_tps(
    source_points: np.ndarray,  # (N, 2)
    target_points: np.ndarray,  # (N, 2)
    regularization: float = 1e-4,
) -> np.ndarray:
    """Solve the TPS system; return coefficients of shape (N+3, 2).

    Coefficient layout: [w_1..w_N, a_0, a_x, a_y] per output coordinate.
    """
    n = source_points.shape[0]
    assert source_points.shape == (n, 2)
    assert target_points.shape == (n, 2)

    # Build K matrix (N, N): K[i,j] = phi(|p_i - p_j|^2)
    diff = source_points[:, None, :] - source_points[None, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    k_mat = _phi(r2)
    k_mat += regularization * np.eye(n)  # ridge for stability

    # Build P matrix (N, 3): [1, x, y]
    p_mat = np.concatenate([np.ones((n, 1)), source_points], axis=1)

    # Full block matrix (N+3, N+3):
    # [K   P]
    # [P^T 0]
    l_mat = np.zeros((n + 3, n + 3), dtype=np.float64)
    l_mat[:n, :n] = k_mat
    l_mat[:n, n:] = p_mat
    l_mat[n:, :n] = p_mat.T
    # Bottom-right 3x3 stays zero

    # RHS: target coords, with 3 zero rows
    rhs = np.zeros((n + 3, 2), dtype=np.float64)
    rhs[:n] = target_points

    coefs = np.linalg.solve(l_mat, rhs)
    return coefs


def eval_tps(
    coefs: np.ndarray,  # (N+3, 2)
    source_points: np.ndarray,  # (N, 2)
    query_points: np.ndarray,  # (M, 2)
) -> np.ndarray:
    """Evaluate fitted TPS at arbitrary query points. Returns (M, 2)."""
    n = source_points.shape[0]
    m = query_points.shape[0]

    diff = query_points[:, None, :] - source_points[None, :, :]  # (M, N, 2)
    r2 = np.sum(diff * diff, axis=-1)  # (M, N)
    phi_mat = _phi(r2)  # (M, N)

    w = coefs[:n]  # (N, 2)
    a = coefs[n:]  # (3, 2)  -- a_0, a_x, a_y

    result = phi_mat @ w  # (M, 2)
    result += a[0]
    result += query_points @ a[1:].T if a[1:].shape == (2, 2) else query_points @ a[1:]
    return result


def flow_field_from_tps(
    coefs: np.ndarray,
    source_points: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """Sample a dense (H, W, 2) flow field: output[y, x] = (tx, ty)."""
    xs = np.arange(grid_w, dtype=np.float64)
    ys = np.arange(grid_h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    query = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (H*W, 2)
    out = eval_tps(coefs, source_points, query)
    return out.reshape(grid_h, grid_w, 2)


__all__ = ["fit_tps", "eval_tps", "flow_field_from_tps"]
