"""Scalar temperature scaling on the cached strided posterior.

Both the baseline builder (stage 4b) and the seed-spread check (stage 4c) fit
the same scalar by the same grid search, so the grid and the pixel extraction
live here rather than in either script.
"""

from __future__ import annotations

import numpy as np

TEMPERATURE_GRID = np.round(np.arange(0.50, 5.01, 0.05), 2)


def strided_true_probs(cached, label, value_to_channel, stride):
    """Per-pixel true-class probabilities on the strided grid.

    Returns ``(true_p, probs, valid)``: probabilities of the ground-truth class
    for strided pixels whose ground-truth value maps to a model channel, the
    cropped full posterior, and the validity mask.
    """
    probs = cached["strided_probs"].astype(np.float32)
    lab = label[::stride, ::stride]
    h = min(lab.shape[0], probs.shape[1])
    w = min(lab.shape[1], probs.shape[2])
    lab = lab[:h, :w]
    channel = np.full(lab.shape, -1, dtype=np.int32)
    for value, ch in value_to_channel.items():
        channel[lab == value] = ch
    valid = channel >= 0
    rr, cc = np.nonzero(valid)
    true_p = probs[channel[valid], rr, cc]
    return true_p, probs[:, :h, :w], valid


def nll_grid_search(true_p, full_p):
    """Temperature minimizing the NLL of the true class over pooled pixels."""
    eps = 1e-12
    true_p = np.asarray(true_p, dtype=np.float64)
    full_p = np.asarray(full_p, dtype=np.float64)
    nlls = []
    for temp in TEMPERATURE_GRID:
        powered = np.power(np.clip(full_p, eps, 1.0), 1.0 / temp)
        z = powered.sum(axis=0)
        tempered_true = np.power(np.clip(true_p, eps, 1.0), 1.0 / temp) / z
        nlls.append(float(-np.mean(np.log(np.clip(tempered_true, eps, 1.0)))))
    best = int(np.argmin(nlls))
    return float(TEMPERATURE_GRID[best]), nlls[best]
