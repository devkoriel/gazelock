"""Apply a 2D flow field to an image via bilinear sampling."""

from __future__ import annotations

import numpy as np


def apply_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp ``image`` so output[y, x] = image[flow[y, x]] (bilinear).

    Args:
        image: (H, W, 3) uint8 or float32 array.
        flow:  (H, W, 2) float array of source coordinates (x, y).

    Returns:
        Warped image with same shape and dtype as input.
    """
    h, w = image.shape[:2]
    assert flow.shape == (h, w, 2)

    src_x = flow[..., 0]
    src_y = flow[..., 1]

    # Clip to valid range
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)

    x0 = np.floor(src_x).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(src_y).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = (src_x - x0)[..., None]  # (H, W, 1)
    wy = (src_y - y0)[..., None]

    img_f = image.astype(np.float32)

    top = img_f[y0, x0] * (1 - wx) + img_f[y0, x1] * wx
    bot = img_f[y1, x0] * (1 - wx) + img_f[y1, x1] * wx
    out = top * (1 - wy) + bot * wy

    if image.dtype == np.uint8:
        return np.clip(out, 0, 255).astype(np.uint8)
    return out.astype(image.dtype)


__all__ = ["apply_flow"]
