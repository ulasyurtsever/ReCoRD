"""Two-tier deployment decision rule.

Tier B returns a threshold only when the conservative test mass leaves room
for it.  The check is a-priori: it uses the estimated weights alone and never
touches a calibration loss, so it can be evaluated before any label is spent.

The three outcomes mirror Algorithm 2 exactly:

``tier_b``
    the certificate does not fire; the weighted threshold is used.
``tier_a``
    the certificate fires and a labelled target set large enough for the
    requested level is available; recalibrate on it.
``fail_safe``
    the certificate fires and no adequate labelled target set exists; return
    ``lambda_max`` (mark everything) and request labels.

The ``fail_safe`` threshold is trivially valid -- the loss vanishes at
``lambda_max`` -- and carries no information, which is the point of raising
the flag alongside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["DeploymentDecision", "conservative_test_mass", "min_target_images",
           "decide_tier"]


@dataclass(frozen=True)
class DeploymentDecision:
    """Outcome of the Algorithm 2 decision rule for one class."""

    tier: str                    # "tier_b" | "tier_a" | "fail_safe"
    p_test: float                # conservative test mass p_hat_{n+1}
    vacuous: bool                # certificate fired
    guarantee_at_risk: bool      # flag raised for the operator
    required_target_images: int  # ceil(1/alpha) - 1
    reason: str


def conservative_test_mass(weight_sum: float, clip_ceiling: float) -> float:
    """Conservative test mass ``p_hat_{n+1} = w_max / (sum_j w_j + w_max)``.

    Parameters
    ----------
    weight_sum:
        ``sum_j w_hat(X_j)`` over the ``n`` calibration images.
    clip_ceiling:
        ``kappa``, the upper clip, which doubles as the test-point weight.
    """
    if clip_ceiling <= 0:
        raise ValueError(f"clip ceiling must be positive, got {clip_ceiling}")
    if weight_sum < 0:
        raise ValueError(f"weight sum must be nonnegative, got {weight_sum}")
    return clip_ceiling / (weight_sum + clip_ceiling)


def min_target_images(alpha: float) -> int:
    """``ceil(1/alpha) - 1``: the smallest tier-A sample admitting level alpha.

    This is the sample-size floor of eq. (4) with ``B = 1``.  It counts images
    *containing the class*, not images drawn.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
    return math.ceil(1.0 / alpha) - 1


def decide_tier(
    weight_sum: float,
    clip_ceiling: float,
    alpha: float,
    n_target_with_class: int | None = None,
    loss_bound: float = 1.0,
) -> DeploymentDecision:
    """Run the Algorithm 2 decision rule for one class.

    Parameters
    ----------
    weight_sum, clip_ceiling:
        As in :func:`conservative_test_mass`.
    alpha:
        Requested risk level for this class.
    n_target_with_class:
        Number of labelled target images containing the class, or ``None``
        when no labelled target set is available.
    loss_bound:
        ``B``; one for the region-miss loss (Lemma 1).

    Notes
    -----
    The certificate is *sufficient* for vacuity, not necessary: a decision of
    ``tier_b`` means the a-priori check leaves room for an informative
    threshold, not that one is guaranteed to be found.
    """
    p_test = conservative_test_mass(weight_sum, clip_ceiling)
    required = min_target_images(alpha)
    vacuous = p_test * loss_bound > alpha

    if not vacuous:
        return DeploymentDecision(
            tier="tier_b", p_test=p_test, vacuous=False,
            guarantee_at_risk=False, required_target_images=required,
            reason=(f"p_test*B = {p_test * loss_bound:.4f} <= alpha = {alpha}; "
                    "weighted calibration may be informative"),
        )

    if n_target_with_class is not None and n_target_with_class >= required:
        return DeploymentDecision(
            tier="tier_a", p_test=p_test, vacuous=True,
            guarantee_at_risk=True, required_target_images=required,
            reason=(f"p_test*B = {p_test * loss_bound:.4f} > alpha = {alpha}; "
                    f"escalating to tier A on {n_target_with_class} labelled "
                    f"images (>= {required} required)"),
        )

    have = 0 if n_target_with_class is None else n_target_with_class
    return DeploymentDecision(
        tier="fail_safe", p_test=p_test, vacuous=True,
        guarantee_at_risk=True, required_target_images=required,
        reason=(f"p_test*B = {p_test * loss_bound:.4f} > alpha = {alpha} and "
                f"only {have} labelled target images contain the class "
                f"(>= {required} required); returning lambda_max"),
    )
