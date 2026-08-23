"""Cross-fitted density-ratio estimation.

The tier-B negative result rests on the estimated weights collapsing to the
lower clip.  Two very different things produce that collapse: a genuine loss
of source/target support overlap, and a domain classifier that separates its
own training set because the embedding dimension dwarfs the sample size.
These tests pin down the difference, so the distinction is checked by the
suite rather than argued in prose.
"""

from __future__ import annotations

import numpy as np
import pytest

from record.weights import logistic_cv_density_ratio, logistic_density_ratio

CLIP = (0.05, 20.0)


def _at_floor(w: np.ndarray) -> float:
    return float((w <= CLIP[0] * 1.001).mean())


def test_crossfit_matches_insample_under_a_real_shift():
    """When the domains are genuinely disjoint both estimators collapse.

    Cross-fitting is not expected to rescue tier B. Its role is to make a
    collapse interpretable as a statement about support overlap.
    """
    rng = np.random.default_rng(0)
    source = rng.normal(0.0, 1.0, (60, 16))
    target = rng.normal(2.5, 1.0, (80, 16))

    w_in = logistic_density_ratio(source, target, clip=CLIP)
    w_cv = logistic_cv_density_ratio(source, target, clip=CLIP)

    assert _at_floor(w_in) > 0.9
    assert _at_floor(w_cv) > 0.9


def test_insample_estimator_collapses_with_no_shift_at_all():
    """No shift, high dimension, small sample: the in-sample estimator still
    collapses, and cross-fitting does not.

    This is the confound.  The embeddings are 768-dimensional and the
    calibration sets hold on the order of 167 images, which is this regime.  A collapse observed with the in-sample estimator therefore cannot
    on its own be read as evidence about support overlap.
    """
    rng = np.random.default_rng(1)
    source = rng.normal(0.0, 1.0, (60, 400))
    target = rng.normal(0.0, 1.0, (60, 400))   # identical law: true w == 1

    w_in = logistic_density_ratio(source, target, clip=CLIP)
    w_cv = logistic_cv_density_ratio(source, target, clip=CLIP)

    assert _at_floor(w_in) > 0.9, "in-sample estimator expected to collapse"
    assert _at_floor(w_cv) < 0.5, "cross-fitted estimator should not collapse"
    # Truth is sum(w) == n_source; cross-fitting stays the right side of it.
    assert w_cv.sum() > 10 * w_in.sum()


def test_crossfit_is_deterministic():
    rng = np.random.default_rng(2)
    source = rng.normal(0.0, 1.0, (40, 32))
    target = rng.normal(0.7, 1.0, (40, 32))
    a = logistic_cv_density_ratio(source, target, clip=CLIP, random_state=7)
    b = logistic_cv_density_ratio(source, target, clip=CLIP, random_state=7)
    assert np.array_equal(a, b)


def test_crossfit_respects_the_clip():
    rng = np.random.default_rng(3)
    source = rng.normal(0.0, 1.0, (50, 8))
    target = rng.normal(1.0, 1.0, (50, 8))
    w = logistic_cv_density_ratio(source, target, clip=(0.1, 3.0))
    assert w.min() >= 0.1 - 1e-12
    assert w.max() <= 3.0 + 1e-12


def test_crossfit_rejects_degenerate_input():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError):
        logistic_cv_density_ratio(rng.normal(size=(1, 4)),
                                  rng.normal(size=(5, 4)), clip=CLIP)
