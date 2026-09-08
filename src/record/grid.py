"""Canonical threshold grid shared by all cached statistics.

Every coverage curve, marked-area curve, and loss curve is tabulated on this
fixed grid. Threshold selection returns grid indices, so downstream analyses
remain exact re-tabulations of cached statistics rather than recomputations.
"""

from __future__ import annotations

import os

import numpy as np

N_POINTS = 1001


def _build_grid(kind: str) -> np.ndarray:
    """Return the threshold grid named by ``kind``.

    ``logtail`` (default since 2026-09-08) spaces the cutoff 1 - lambda
    logarithmically over six decades: lambda_k = 1 - 10^(-6 k / (N-1)) for
    k < N-1, and lambda_{N-1} = 1 exactly so the last point still marks the
    whole image and the loss still vanishes there (Lemma 1 needs that
    endpoint). Calibrated thresholds of confident models sit within 1e-3 of
    lambda = 1, where the previous uniform grid had a single point; on that
    grid Mask2Former could only be calibrated to "mark everything", which the
    fourth referee panel showed to be a resolution artifact rather than a
    property of the posterior. ``uniform`` is the earlier grid, 1001 equally
    spaced points, kept selectable for reproducing the pre-2026-09-08 numbers.
    The grid is selected with the RECORD_GRID environment variable at import
    time; every cached statistic is tabulated on it, so tables and experiments
    built under one grid must never be read under another.
    """
    if kind == "uniform":
        return np.linspace(0.0, 1.0, N_POINTS)
    if kind == "logtail":
        k = np.arange(N_POINTS, dtype=np.float64)
        grid = 1.0 - 10.0 ** (-6.0 * k / (N_POINTS - 1))
        grid[-1] = 1.0
        return grid
    raise ValueError(f"unknown RECORD_GRID={kind!r}; use 'uniform' or 'logtail'")


GRID_KIND: str = os.environ.get("RECORD_GRID", "logtail")
LAMBDA_GRID: np.ndarray = _build_grid(GRID_KIND)

# Coarser subgrid for statistics that require per-threshold connected-component
# labeling (e.g. false-positive component counts), which is too expensive to
# evaluate at every grid point. Indices into LAMBDA_GRID.
#
# The subgrid is index-based and deliberately non-uniform: every point from
# index 960 upward plus every 25th index below. On the uniform grid that was
# where calibrated thresholds concentrated; on the log-tail grid the same
# indices cover 1 - lambda below 10^-5.76 densely and the rest at 0.15 decades
# per step, which resolves the operating region of every model here.
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
