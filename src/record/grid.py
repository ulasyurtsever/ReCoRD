"""Canonical threshold grid shared by all cached statistics.

Every coverage curve, marked-area curve, and loss curve is tabulated on this
fixed grid. Threshold selection returns grid indices, so downstream analyses
remain exact re-tabulations of cached statistics rather than recomputations.
"""

from __future__ import annotations

import numpy as np

N_POINTS = 1001

LAMBDA_GRID: np.ndarray = np.linspace(0.0, 1.0, N_POINTS)

# Coarser subgrid for statistics that require per-threshold connected-component
# labeling (e.g. false-positive component counts), which is too expensive to
# evaluate at every grid point. Indices into LAMBDA_GRID.
#
# The grid is deliberately non-uniform. Calibrated thresholds concentrate in the
# top twenty grid points, so a uniform 25-step subgrid resolves the operating
# region not at all and forces every selected threshold onto lambda_max, where
# the mask covers the whole image and the component count degenerates. The tail
# is therefore sampled at every point from index 960 upward.
FP_SUBGRID_INDICES: np.ndarray = np.unique(np.concatenate([
    np.arange(0, 960, 25),      # coarse body
    np.arange(960, N_POINTS),   # dense tail, where lambda-hat lives
]))


def fp_subgrid_slot(lam_index: int) -> int:
    """Position in ``FP_SUBGRID_INDICES`` of the largest index <= ``lam_index``.

    Never rounds upward. Rounding up crosses into a more permissive mask than
    the one that was calibrated, and at ``lam_index = N_POINTS - 1`` it reaches
    the mask that covers the whole image, where the count is degenerate.
    """
    pos = int(np.searchsorted(FP_SUBGRID_INDICES, lam_index, side="right")) - 1
    return max(pos, 0)


def curve_on_grid(scores: np.ndarray) -> np.ndarray:
    """Return the empirical CDF of ``scores`` evaluated on the lambda grid.

    Parameters
    ----------
    scores : np.ndarray
        Per-pixel scores in ``[0, 1]`` (typically ``1 - p`` for a class
        probability ``p``). A pixel is inside the mask at threshold ``lam``
        iff its score is ``<= lam``.

    Returns
    -------
    np.ndarray
        Array of shape ``(N_POINTS,)`` with values in ``[0, 1]``, the
        fraction of scores ``<= lam`` for each grid value; nondecreasing.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("empty score array")
    sorted_scores = np.sort(scores)
    counts = np.searchsorted(sorted_scores, LAMBDA_GRID, side="right")
    return counts / scores.size
