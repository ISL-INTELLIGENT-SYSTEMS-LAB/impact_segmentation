"""
Crop a detection region and zero pixels outside the instance mask (background removal).

Used after SAM3: full-resolution image + per-instance mask + bbox → RGB crop with
background set to black, matching the Catherine pipeline pattern.
"""

from __future__ import annotations

import numpy as np


def _to_2d_mask(mask: np.ndarray) -> np.ndarray:
    """Return H×W boolean mask."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = np.squeeze(m, axis=0)
    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D or 1×H×W; got shape {m.shape}")
    if m.dtype == bool:
        return m
    return m > 0.5


def crop_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy,
) -> np.ndarray:
    """
    Crop to bbox and set pixels outside the mask to zero (RGB 0).

    Args:
        image: H×W×3 uint8 or float RGB, same spatial size as mask.
        mask: Instance mask aligned to ``image`` (H×W or 1×H×W), bool or float in [0,1].
        bbox_xyxy: (x0, y0, x1, y1) in pixel coordinates (same space as image).

    Returns:
        Cropped array (y1-y0)×(x1-x0)×3, background pixels zeroed.
    """
    x0, y0, x1, y1 = (int(round(c)) for c in bbox_xyxy)
    h, w = image.shape[:2]
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bbox after clamp: ({x0}, {y0}, {x1}, {y1})")

    crop = image[y0:y1, x0:x1].copy()
    mask_2d = _to_2d_mask(mask)
    if mask_2d.shape[:2] != (h, w):
        raise ValueError(
            f"Mask shape {mask_2d.shape} does not match image spatial size {(h, w)}"
        )
    mask_crop = mask_2d[y0:y1, x0:x1]
    crop[~mask_crop] = 0
    return crop
