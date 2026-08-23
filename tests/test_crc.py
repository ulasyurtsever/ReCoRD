"""Tests for CRC threshold selection, including a synthetic validation of the
finite-sample risk guarantee."""

import numpy as np
import pytest

from record.crc import (
    crc_threshold,
    heuristic_threshold,
    min_alpha,
    weighted_crc_threshold,
)
from record.grid import LAMBDA_GRID
from record.losses import enforce_nonincreasing

RNG = np.random.default_rng(11)


def _synthetic_loss_curves(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random nonincreasing loss curves in [0, 1] with heterogeneous difficulty."""
    difficulty = rng.uniform(0.1, 0.9, size=n)
    drop_points = rng.uniform(0.05, 0.95, size=(n, 4)) * difficulty[:, None]
    curves = np.empty((n, LAMBDA_GRID.size))
    for i in range(n):
        thresholds = np.sort(drop_points[i])
        curves[i] = 1.0 - (LAMBDA_GRID[None, :] >= thresholds[:, None]).mean(axis=0)
    return enforce_nonincreasing(curves)


def test_synthetic_curves_are_not_degenerate():
    # Guard against vacuous guarantee tests: the synthetic losses must be
    # genuinely nonzero at small thresholds, otherwise every selection rule
    # trivially "controls" the risk.
    pool = _synthetic_loss_curves(100, np.random.default_rng(0))
    assert pool[:, 0].mean() > 0.5
    assert pool[:, -1].max() == 0.0


def test_threshold_controls_risk_on_exchangeable_data():
    """Empirical validation of E[L(lam_hat)] <= alpha over repeated splits."""
    alpha = 0.15
    n_cal, n_test, n_trials = 60, 200, 300
    test_losses, lam_indices = [], []
    for _ in range(n_trials):
        pool = _synthetic_loss_curves(n_cal + n_test, RNG)
        cal, test = pool[:n_cal], pool[n_cal:]
        sel = crc_threshold(cal, alpha, LAMBDA_GRID)
        test_losses.append(test[:, sel.lam_index].mean())
        lam_indices.append(sel.lam_index)
    mean_risk = float(np.mean(test_losses))
    sem = float(np.std(test_losses) / np.sqrt(n_trials))
    assert mean_risk <= alpha + 3 * sem, f"risk {mean_risk:.4f} exceeds alpha={alpha}"
    # The guarantee must be earned, not trivial: a degenerate threshold at the
    # grid origin would indicate collapsed loss curves.
    assert min(lam_indices) > 0
    assert mean_risk > 0.01


def test_threshold_is_smallest_feasible():
    losses = _synthetic_loss_curves(50, RNG)
    sel = crc_threshold(losses, 0.2, LAMBDA_GRID)
    risk = (losses.sum(axis=0) + 1.0) / (losses.shape[0] + 1)
    assert risk[sel.lam_index] <= 0.2
    if sel.lam_index > 0:
        assert risk[sel.lam_index - 1] > 0.2


def test_infeasible_alpha_flagged():
    # With n=3 the smallest controllable level is B/(n+1) = 0.25; alpha below
    # that must be reported infeasible and fall back to the last grid point.
    assert min_alpha(3) == 0.25
    losses = np.zeros((3, LAMBDA_GRID.size))
    sel = crc_threshold(losses, alpha=0.1, lambda_grid=LAMBDA_GRID)
    assert not sel.feasible
    assert sel.lam == 1.0
    # At or above the minimum level the same losses are feasible.
    sel2 = crc_threshold(losses, alpha=0.25, lambda_grid=LAMBDA_GRID)
    assert sel2.feasible and sel2.lam_index == 0


def test_nan_losses_rejected():
    losses = np.full((5, LAMBDA_GRID.size), np.nan)
    with pytest.raises(ValueError):
        crc_threshold(losses, 0.5, LAMBDA_GRID)


def test_heuristic_smaller_or_equal_threshold():
    losses = _synthetic_loss_curves(40, RNG)
    crc_sel = crc_threshold(losses, 0.2, LAMBDA_GRID)
    heur_sel = heuristic_threshold(losses, 0.2, LAMBDA_GRID)
    assert heur_sel.lam <= crc_sel.lam  # missing correction => less conservative


def test_weighted_reduces_to_unweighted():
    losses = _synthetic_loss_curves(30, RNG)
    unw = crc_threshold(losses, 0.25, LAMBDA_GRID)
    w = weighted_crc_threshold(
        losses, np.ones(30), test_weight=1.0, alpha=0.25, lambda_grid=LAMBDA_GRID)
    assert w.lam_index == unw.lam_index


def test_weighted_upweights_hard_units():
    losses = _synthetic_loss_curves(40, RNG)
    hardness = losses[:, 500]  # loss at mid-threshold as difficulty proxy
    weights = 1.0 + 5.0 * hardness
    w_sel = weighted_crc_threshold(
        losses, weights, test_weight=weights.max(), alpha=0.2, lambda_grid=LAMBDA_GRID)
    u_sel = crc_threshold(losses, 0.2, LAMBDA_GRID)
    assert w_sel.lam >= u_sel.lam  # emphasizing hard units cannot loosen the threshold


def test_weighted_guarantee_under_known_shift():
    """Weighted CRC with exact weights controls risk under covariate shift."""
    alpha = 0.2
    n_pool, n_trials = 4000, 200
    pool = _synthetic_loss_curves(n_pool, RNG)
    hardness = pool[:, 400]
    # Target distribution tilted toward hard units; exact importance weights known.
    tilt = 0.5 + 2.0 * hardness
    p_target = tilt / tilt.sum()

    test_losses = []
    for _ in range(n_trials):
        cal_idx = RNG.choice(n_pool, size=80, replace=False)  # source: uniform
        test_idx = RNG.choice(n_pool, size=1, p=p_target)
        w_cal = tilt[cal_idx]
        sel = weighted_crc_threshold(
            pool[cal_idx], w_cal, test_weight=tilt.max(), alpha=alpha,
            lambda_grid=LAMBDA_GRID)
        test_losses.append(pool[test_idx, sel.lam_index].mean())
    mean_risk = float(np.mean(test_losses))
    sem = float(np.std(test_losses) / np.sqrt(n_trials))
    assert mean_risk <= alpha + 3 * sem, f"shifted risk {mean_risk:.4f} exceeds {alpha}"
