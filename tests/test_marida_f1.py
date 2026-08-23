"""Unit tests for the MARIDA test-split debris-F1 script."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "marida_f1", Path(__file__).resolve().parents[1] / "scripts" / "10_marida_test_f1.py")
marida_f1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(marida_f1)


def test_binary_counts_excludes_invalid_pixels():
    pred = np.array([[True, True], [False, True]])
    ref = np.array([[True, False], [True, True]])
    valid = np.array([[True, True], [True, False]])
    tp, fp, fn = marida_f1.binary_counts(pred, ref, valid)
    assert (tp, fp, fn) == (1, 1, 1)


def test_prf_perfect_and_empty():
    assert marida_f1.prf(10, 0, 0) == (1.0, 1.0, 1.0)
    assert marida_f1.prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_prf_matches_direct_formula():
    tp, fp, fn = 30, 20, 10
    precision, recall, f1 = marida_f1.prf(tp, fp, fn)
    assert precision == tp / (tp + fp)
    assert recall == tp / (tp + fn)
    expected = 2 * precision * recall / (precision + recall)
    assert abs(f1 - expected) < 1e-12


def test_counts_aggregate_to_known_f1():
    rng = np.random.default_rng(0)
    tp = fp = fn = 0
    for _ in range(5):
        pred = rng.random((16, 16)) > 0.5
        ref = rng.random((16, 16)) > 0.5
        valid = rng.random((16, 16)) > 0.2
        t, p, n = marida_f1.binary_counts(pred, ref, valid)
        tp, fp, fn = tp + t, fp + p, fn + n
    _, _, f1 = marida_f1.prf(tp, fp, fn)
    assert 0.0 < f1 < 1.0
    assert 2 * tp / (2 * tp + fp + fn) == f1
