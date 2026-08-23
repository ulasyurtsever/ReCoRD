"""Conformal risk control threshold selection.

Implements the CRC rule of Angelopoulos et al. (2024) for monotone, bounded
losses tabulated on a fixed threshold grid, together with an importance-
weighted variant for covariate shift in the spirit of Tibshirani et al.
(2019). The weighted variant is exact when the importance weights are exact;
with estimated weights the guarantee degrades gracefully with the weight
estimation error, which is reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdSelection:
    """Result of a CRC threshold search."""

    lam_index: int
    lam: float
    calibration_risk: float
    n_calibration: int
    feasible: bool


def min_alpha(n_calibration: int, loss_bound: float = 1.0) -> float:
    """Return the smallest controllable risk level for a calibration size."""
    return loss_bound / (n_calibration + 1)


def crc_threshold(
    calibration_losses: np.ndarray,
    alpha: float,
    lambda_grid: np.ndarray,
    loss_bound: float = 1.0,
) -> ThresholdSelection:
    """Select the smallest threshold controlling expected loss at level alpha.

    Parameters
    ----------
    calibration_losses : np.ndarray
        Array of shape ``(n, n_grid)``; rows are per-unit loss curves,
        nonincreasing in the grid, bounded by ``loss_bound``. NaN rows are
        rejected (filter them out before calling).
    alpha : float
        Target risk level.
    lambda_grid : np.ndarray
        The grid the losses are tabulated on.
    loss_bound : float
        Upper bound B of the loss.

    Returns
    -------
    ThresholdSelection
        The selected grid index and threshold. ``feasible`` is False when
        ``alpha < B / (n + 1)``, in which case the most conservative
        threshold (last grid point) is returned.
    """
    losses = np.asarray(calibration_losses, dtype=np.float64)
    if losses.ndim != 2 or losses.shape[1] != lambda_grid.size:
        raise ValueError(f"losses shape {losses.shape} incompatible with grid {lambda_grid.size}")
    if np.isnan(losses).any():
        raise ValueError("calibration losses contain NaN; filter undefined rows first")
    n = losses.shape[0]
    if n == 0:
        raise ValueError("empty calibration set")

    risk = (losses.sum(axis=0) + loss_bound) / (n + 1)
    feasible_mask = risk <= alpha
    if not feasible_mask.any():
        idx = lambda_grid.size - 1
        return ThresholdSelection(idx, float(lambda_grid[idx]), float(risk[idx]), n, False)
    idx = int(np.argmax(feasible_mask))
    return ThresholdSelection(idx, float(lambda_grid[idx]), float(risk[idx]), n, True)


def weighted_crc_threshold(
    calibration_losses: np.ndarray,
    calibration_weights: np.ndarray,
    test_weight: float,
    alpha: float,
    lambda_grid: np.ndarray,
    loss_bound: float = 1.0,
) -> ThresholdSelection:
    """Importance-weighted CRC threshold for covariate shift.

    The risk estimate at each grid point is the weight-normalized calibration
    loss plus the (conservative) bound ``loss_bound`` at the test point's
    normalized weight:

    ``R(lam) = sum_i p_i L_i(lam) + p_test * B`` with
    ``p_i = w_i / (sum_j w_j + w_test)``.

    Parameters
    ----------
    calibration_weights : np.ndarray
        Nonnegative importance weights ``w(x_i)`` of calibration units.
    test_weight : float
        Weight assigned to the hypothetical test unit; a conservative common
        choice is the maximum anticipated weight (e.g. the clip ceiling).
    """
    losses = np.asarray(calibration_losses, dtype=np.float64)
    w = np.asarray(calibration_weights, dtype=np.float64)
    if w.shape[0] != losses.shape[0]:
        raise ValueError("weights length mismatch")
    if (w < 0).any() or test_weight < 0:
        raise ValueError("weights must be nonnegative")
    total = w.sum() + test_weight
    if total <= 0:
        raise ValueError("all weights are zero")

    risk = (losses * w[:, None]).sum(axis=0) / total + (test_weight / total) * loss_bound
    feasible_mask = risk <= alpha
    n = losses.shape[0]
    if not feasible_mask.any():
        idx = lambda_grid.size - 1
        return ThresholdSelection(idx, float(lambda_grid[idx]), float(risk[idx]), n, False)
    idx = int(np.argmax(feasible_mask))
    return ThresholdSelection(idx, float(lambda_grid[idx]), float(risk[idx]), n, True)


def heuristic_threshold(
    calibration_losses: np.ndarray,
    alpha: float,
    lambda_grid: np.ndarray,
) -> ThresholdSelection:
    """Uncorrected empirical-risk threshold (baseline, no guarantee).

    Selects the smallest threshold whose plain calibration mean loss is at
    most alpha, omitting the ``+B/(n+1)`` conformal correction.
    """
    losses = np.asarray(calibration_losses, dtype=np.float64)
    n = losses.shape[0]
    risk = losses.mean(axis=0)
    feasible_mask = risk <= alpha
    if not feasible_mask.any():
        idx = lambda_grid.size - 1
        return ThresholdSelection(idx, float(lambda_grid[idx]), float(risk[idx]), n, False)
    idx = int(np.argmax(feasible_mask))
    return ThresholdSelection(idx, float(lambda_grid[idx]), float(risk[idx]), n, True)
