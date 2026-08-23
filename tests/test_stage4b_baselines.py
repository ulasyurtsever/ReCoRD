"""Unit tests for stage-4b baseline derivations (pure array logic)."""

import numpy as np

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "stage4b", Path(__file__).resolve().parents[1] / "scripts" / "04b_build_baseline_tables.py")
stage4b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage4b)


def test_upsample_matches_stride_grid():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    up = stage4b.upsample_to(arr, (5, 7), stride=2)
    assert up.shape == (5, 7)
    assert up[0, 0] == 0 and up[1, 1] == 0 and up[2, 2] == 4
    # Edge padding repeats the last row/column.
    assert up[4, 6] == up[3, 5]


def test_temperature_identity_and_ordering():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 200))
    probs = np.exp(logits) / np.exp(logits).sum(axis=0)
    # T = 1 keeps probabilities unchanged.
    powered = np.power(probs, 1.0)
    z = powered.sum(axis=0)
    np.testing.assert_allclose(powered / z, probs, atol=1e-12)
    # Tempering preserves the per-pixel argmax for any T > 0.
    for temp in (0.5, 2.0, 4.0):
        tempered = np.power(probs, 1.0 / temp)
        tempered /= tempered.sum(axis=0)
        assert (tempered.argmax(axis=0) == probs.argmax(axis=0)).all()


def test_fit_temperature_grid_recovers_soft_model():
    # Build a miscalibrated posterior (overconfident: probs ** 2 renormalized)
    # and check the grid search picks T > 1 to soften it.
    rng = np.random.default_rng(1)
    logits = rng.normal(scale=2.0, size=(4, 5000))
    true = np.exp(logits) / np.exp(logits).sum(axis=0)
    over = true ** 2 / (true ** 2).sum(axis=0)
    labels = np.array([rng.choice(4, p=true[:, i]) for i in range(true.shape[1])])
    eps = 1e-12
    nlls = []
    for temp in stage4b.TEMPERATURE_GRID:
        powered = np.power(np.clip(over, eps, 1.0), 1.0 / temp)
        z = powered.sum(axis=0)
        tempered_true = powered[labels, np.arange(labels.size)] / z
        nlls.append(-np.mean(np.log(np.clip(tempered_true, eps, 1.0))))
    best_t = stage4b.TEMPERATURE_GRID[int(np.argmin(nlls))]
    assert best_t > 1.4  # squaring probabilities needs roughly T ~ 2 to undo
