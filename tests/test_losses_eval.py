"""Tests for loss aggregation, evaluation, weights, and the drift monitor."""

import numpy as np
import pytest

from record.crc import ThresholdSelection
from record.evaluation import (
    evaluate_at_threshold,
    index_rows,
    stratified_region_fnr,
    triage_curve,
)
from record.grid import LAMBDA_GRID
from record.losses import component_miss_matrix, enforce_nonincreasing, image_loss_curves
from record.monitor import ks_drift_check
from record.weights import knn_density_ratio, logistic_density_ratio

RNG = np.random.default_rng(23)


def test_image_loss_aggregation():
    curves = np.vstack([
        np.linspace(0, 1, LAMBDA_GRID.size),   # img 0
        np.ones(LAMBDA_GRID.size),             # img 0 (always captured for rho<=1)
        np.zeros(LAMBDA_GRID.size),            # img 2 (never captured)
    ])
    idx = np.array([0, 0, 2])
    losses, has = image_loss_curves(curves, idx, n_images=3, rho=0.5)
    assert has.tolist() == [True, False, True]
    assert np.isnan(losses[1]).all()
    assert losses[2, 0] == 1.0 and losses[2, -1] == 1.0
    # img 0: first component captured once linspace >= 0.5, second always.
    assert losses[0, 0] == 0.5
    assert losses[0, -1] == 0.0


def test_size_weighted_loss():
    curves = np.vstack([np.zeros(LAMBDA_GRID.size), np.ones(LAMBDA_GRID.size)])
    idx = np.array([0, 0])
    unweighted, _ = image_loss_curves(curves, idx, 1, rho=0.5)
    weighted, _ = image_loss_curves(curves, idx, 1, rho=0.5,
                                    size_weights=np.array([3.0, 1.0]))
    assert unweighted[0, 0] == 0.5
    assert weighted[0, 0] == 0.75  # large missed component dominates


def test_miss_matrix_rho_validation():
    with pytest.raises(ValueError):
        component_miss_matrix(np.zeros((1, 10)), rho=0.0)


def test_enforce_nonincreasing_is_identity_on_valid_curves():
    # Regression test: a faulty implementation once collapsed every
    # nonincreasing curve to its final value, silently zeroing all losses.
    curve = np.array([[1.0, 0.8, 0.5, 0.2, 0.0]])
    np.testing.assert_array_equal(enforce_nonincreasing(curve), curve)


def test_enforce_nonincreasing_lifts_small_violations():
    curve = np.array([[1.0, 0.5, 0.6, 0.2, 0.0]])  # bump at index 2
    out = enforce_nonincreasing(curve)[0]
    assert (np.diff(out) <= 0).all()
    assert out[1] == 0.6  # violation resolved upward (conservative)
    assert (out >= curve[0]).all()  # loss never underestimated


def test_enforce_nonincreasing_preserves_nan_rows():
    curve = np.vstack([np.full(5, np.nan), [1.0, 0.6, 0.3, 0.1, 0.0]])
    out = enforce_nonincreasing(curve)
    assert np.isnan(out[0]).all()
    np.testing.assert_array_equal(out[1], curve[1])


def test_index_rows_strict():
    ids = ["a", "b", "c"]
    assert index_rows(ids, ["c", "a"]).tolist() == [2, 0]
    with pytest.raises(KeyError):
        index_rows(ids, ["z"])


def test_evaluate_at_threshold():
    sel = ThresholdSelection(lam_index=500, lam=0.5, calibration_risk=0.1,
                             n_calibration=50, feasible=True)
    losses = np.full((4, LAMBDA_GRID.size), 0.25)
    losses[3] = np.nan
    area = np.full((4, LAMBDA_GRID.size), 0.1)
    counts = np.array([2, 1, 1, 0])
    m = evaluate_at_threshold(sel, losses, area, counts)
    assert m.region_fnr == 0.25
    assert m.n_test_images == 3
    assert m.marked_area_fraction == pytest.approx(0.1)


def test_stratified_fnr():
    curves = np.vstack([np.zeros(LAMBDA_GRID.size), np.ones(LAMBDA_GRID.size)])
    strata = np.array([0, 2])
    vals = stratified_region_fnr(curves, strata, lam_index=500, rho=0.5)
    assert vals[0] == 1.0 and np.isnan(vals[1]) and vals[2] == 0.0


def test_triage_curve_monotone():
    scores = RNG.uniform(size=100)
    losses = (scores > 0.5).astype(float)  # losses concentrated on high scores
    budgets = np.linspace(0, 1, 11)
    residual = triage_curve(scores, losses, budgets)
    assert residual[0] >= residual[-1]
    assert (np.diff(residual) <= 1e-9).all()  # perfect ranking: nonincreasing
    assert residual[-1] == 0.0


def test_logistic_weights_detect_shift():
    src = RNG.normal(0, 1, size=(300, 8))
    tgt = RNG.normal(1.0, 1, size=(300, 8))
    w = logistic_density_ratio(src, tgt)
    shifted_side = src[:, 0] > 0.5
    assert w[shifted_side].mean() > w[~shifted_side].mean()


def test_knn_weights_detect_shift():
    src = RNG.normal(0, 1, size=(400, 4))
    tgt = RNG.normal(1.2, 1, size=(400, 4))
    w = knn_density_ratio(src, tgt, k=30)
    shifted_side = src[:, 0] > 0.5
    assert w[shifted_side].mean() > w[~shifted_side].mean()


def test_identical_distributions_give_flat_weights():
    pool = RNG.normal(0, 1, size=(600, 6))
    w = logistic_density_ratio(pool[:300], pool[300:])
    assert 0.5 < np.median(w) < 2.0


def test_monitor_flags_shift_and_not_null():
    cal = RNG.normal(0, 1, size=400)
    same = RNG.normal(0, 1, size=400)
    shifted = RNG.normal(1.5, 1, size=400)
    assert not ks_drift_check(cal, same).flagged
    assert ks_drift_check(cal, shifted).flagged
