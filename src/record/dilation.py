"""Morphological dilation of the argmax mask as a monotone mask family.

The region-CRC family of the article is the probability-threshold family
``{p : prob[p] >= 1 - lam}``. The natural geometric alternative keeps the
argmax mask and dilates it by a radius ``r``. Both families are nested
(a larger radius never removes a pixel), so Lemma 1 applies unchanged and the
same CRC rule selects a radius instead of a threshold.

This module maps the dilation family onto the canonical lambda grid so that
stage 4 and stage 5 run on it without modification. The mapping uses a
pseudo-probability ``prob = 1 - score`` whose score is the grid value at the
index that the pixel's distance rounds up to:

    score[p] = LAMBDA_GRID[ceil(d(p) * (N_POINTS - 1) / R_MAX_PX)],

where ``d(p)`` is the Euclidean distance from pixel ``p`` to the nearest
argmax pixel of the class, and indices past the end saturate at the last
grid point. A pixel is inside the mask at grid index ``k`` iff
``score <= LAMBDA_GRID[k]``, that is iff ``d <= k * R_MAX_PX / (N_POINTS - 1)``.
Grid index ``k`` therefore *is* the dilation by radius
``k * R_MAX_PX / (N_POINTS - 1)`` pixels whatever the spacing of the grid
values (the log-tail grid included), and the last grid point marks the whole
image, which keeps the ``L(lambda_max) = 0`` endpoint that the CRC
feasibility argument needs. Pixels farther than ``R_MAX_PX`` from any argmax
pixel, and every pixel when the class is absent from the argmax, are reached
only at that last point.

Radius 0 is the argmax mask itself, so the ``argmax`` baseline row and the
dilation row at ``lam = 0`` coincide by construction; that identity is a
useful check on the tables.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from record.grid import LAMBDA_GRID

# Largest dilation radius the grid can express, in pixels. Cityscapes frames
# are 1024 x 2048; a 100-pixel dilation of a missed pedestrian already covers
# a large part of the frame, so the interesting radii (a few pixels up to a
# few tens) are resolved at 0.1 px per grid step.
R_MAX_PX: float = 100.0


def dilation_pseudo_prob(argmax_mask: np.ndarray, r_max_px: float = R_MAX_PX) -> np.ndarray:
    """Return the pseudo-probability map whose threshold family is dilation.

    Parameters
    ----------
    argmax_mask : np.ndarray
        2-D boolean array, pixels argmax-assigned to the class.
    r_max_px : float
        Radius mapped to ``lam = 1``; larger distances saturate there.

    Returns
    -------
    np.ndarray
        2-D float32 array in [0, 1]; 1 on the argmax mask, decreasing linearly
        with Euclidean distance from it, 0 at distance >= ``r_max_px`` and
        everywhere when the mask is empty.
    """
    mask = np.asarray(argmax_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"argmax_mask must be 2-D, got shape {mask.shape}")
    if r_max_px <= 0:
        raise ValueError("r_max_px must be positive")
    last = LAMBDA_GRID.size - 1
    if not mask.any():
        # No argmax pixel of the class: no finite dilation reaches anything;
        # score 1 puts every pixel at the last grid point only.
        return np.full(mask.shape, np.float32(1.0 - LAMBDA_GRID[last]), dtype=np.float32)
    # distance_transform_edt measures the distance of each *nonzero* pixel to
    # the nearest zero, so it is applied to the complement of the mask.
    dist = ndimage.distance_transform_edt(~mask)
    index = np.ceil(dist * (last / float(r_max_px)) - 1e-9).astype(np.int64)
    index = np.clip(index, 0, last)
    score = LAMBDA_GRID[index]
    return (1.0 - score).astype(np.float32)


def radius_of_index(lam_index: int, r_max_px: float = R_MAX_PX) -> float:
    """Dilation radius in pixels of grid index ``lam_index``."""
    return float(lam_index) * float(r_max_px) / (LAMBDA_GRID.size - 1)


def radius_px(lam: float, r_max_px: float = R_MAX_PX) -> float:
    """Dilation radius in pixels of the grid value ``lam`` (looked up by index)."""
    idx = int(np.searchsorted(LAMBDA_GRID, float(lam), side="left"))
    return radius_of_index(min(idx, LAMBDA_GRID.size - 1), r_max_px)
