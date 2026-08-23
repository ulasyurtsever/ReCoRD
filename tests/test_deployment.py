"""Tests for the Algorithm 2 decision rule."""

from __future__ import annotations

import numpy as np
import pytest

from record.crc import weighted_crc_threshold
from record.deployment import (
    DeploymentDecision,
    conservative_test_mass,
    decide_tier,
    min_target_images,
)
from record.grid import LAMBDA_GRID


def test_conservative_test_mass_matches_article_arithmetic():
    # Remark 3: weights collapsed to the floor c = 0.05 over n = 167 images.
    weight_sum = 0.05 * 167
    assert conservative_test_mass(weight_sum, 2.0) == pytest.approx(0.193, abs=5e-4)
    assert conservative_test_mass(weight_sum, 20.0) == pytest.approx(0.7055, abs=5e-4)


def test_conservative_test_mass_increases_with_ceiling():
    # Widening the clip raises the test mass; there is no kappa that escapes.
    masses = [conservative_test_mass(8.35, k) for k in (2.0, 5.0, 10.0, 20.0)]
    assert masses == sorted(masses)


def test_min_target_images_floor():
    assert min_target_images(0.05) == 19
    assert min_target_images(0.1) == 9
    assert min_target_images(0.2) == 4
    with pytest.raises(ValueError):
        min_target_images(0.0)


def test_tier_b_when_certificate_does_not_fire():
    # Good overlap: weights near one, so the test mass is small.
    d = decide_tier(weight_sum=150.0, clip_ceiling=2.0, alpha=0.1)
    assert isinstance(d, DeploymentDecision)
    assert d.tier == "tier_b"
    assert not d.vacuous and not d.guarantee_at_risk


def test_tier_a_escalation_when_labels_suffice():
    d = decide_tier(weight_sum=8.35, clip_ceiling=2.0, alpha=0.1,
                    n_target_with_class=25)
    assert d.tier == "tier_a"
    assert d.vacuous and d.guarantee_at_risk
    assert d.required_target_images == 9


def test_fail_safe_when_labels_are_too_few():
    d = decide_tier(weight_sum=8.35, clip_ceiling=2.0, alpha=0.05,
                    n_target_with_class=5)
    assert d.tier == "fail_safe"
    assert d.guarantee_at_risk


def test_fail_safe_when_no_labels_at_all():
    d = decide_tier(weight_sum=8.35, clip_ceiling=20.0, alpha=0.2,
                    n_target_with_class=None)
    assert d.tier == "fail_safe"


def test_borderline_cell_is_not_certified_vacuous():
    """kappa = 2 at alpha = 0.2 sits at the level, so the certificate does not
    fire even though most draws are still uninformative.  The certificate is
    sufficient for vacuity, never necessary."""
    d = decide_tier(weight_sum=8.35, clip_ceiling=2.0, alpha=0.2)
    assert d.p_test == pytest.approx(0.193, abs=5e-4)
    assert d.tier == "tier_b"
    assert not d.vacuous


def test_certificate_agrees_with_the_weighted_selection_rule():
    """When the certificate fires, weighted CRC must be infeasible in fact."""
    rng = np.random.default_rng(0)
    n = 167
    weights = np.full(n, 0.05)
    # Monotone decreasing loss curves, zero at lambda = 1.
    base = rng.uniform(0.2, 0.9, size=n)
    curves = np.clip(base[:, None] * (1.0 - LAMBDA_GRID)[None, :], 0.0, 1.0)
    for alpha in (0.05, 0.1):
        d = decide_tier(weight_sum=float(weights.sum()), clip_ceiling=2.0,
                        alpha=alpha)
        assert d.vacuous, "certificate should fire under collapsed weights"
        sel = weighted_crc_threshold(curves, weights, 2.0, alpha, LAMBDA_GRID)
        assert not sel.feasible, "certificate fired but selection was feasible"
