"""Morphological dilation of the argmax mask as a monotone mask family.

The region-CRC family of the article is the probability-threshold family
``{p : prob[p] >= 1 - lam}``. A referee asked for the natural alternative:
keep the argmax mask and dilate it by a radius ``r``. Both families are nested
(a larger radius never removes a pixel), so Lemma 1 applies unchanged and the
same CRC rule selects a radius instead of a threshold.

This module maps the dilation family onto the canonical lambda grid so that
stage 4 and stage 5 run on it without modification. The trick is a
pseudo-probability

    prob[p] = 1 - min(d(p) / R_MAX_PX, 1),

where ``d(p)`` is the Euclidean distance from pixel ``p`` to the nearest
argmax pixel of the class. Stage 4 turns it into ``score = 1 - prob =
min(d / R_MAX_PX, 1)``, and a pixel is inside the mask at grid point ``lam``
iff ``score <= lam``, that is iff ``d <= lam * R_MAX_PX``. Grid point ``k``
therefore *is* the dilation by radius ``k * R_MAX_PX / (N_POINTS - 1)``
pixels, and the last grid point (``lam = 1``, ``score <= 1`` for every pixel)
marks the whole image, which keeps the ``L(lambda_max) = 0`` endpoint that the
CRC feasibility argument needs. Pixels farther than ``R_MAX_PX`` from any
argmax pixel, and every pixel when the class is absent from the argmax, are
reached only at that last point.

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
    if not mask.any():
        # No argmax pixel of the class: no finite dilation reaches anything.
        return np.zeros(mask.shape, dtype=np.float32)
    # distance_transform_edt measures the distance of each *nonzero* pixel to
    # the nearest zero, so it is applied to the complement of the mask.
    dist = ndimage.distance_transform_edt(~mask)
    score = np.minimum(dist / float(r_max_px), 1.0)
    return (1.0 - score).astype(np.float32)


def radius_px(lam: float, r_max_px: float = R_MAX_PX) -> float:
    """Dilation radius in pixels that grid value ``lam`` stands for."""
    return float(lam) * float(r_max_px)


def radius_of_index(lam_index: int, r_max_px: float = R_MAX_PX) -> float:
    """Dilation radius in pixels of grid index ``lam_index``."""
    return radius_px(LAMBDA_GRID[int(lam_index)], r_max_px)
