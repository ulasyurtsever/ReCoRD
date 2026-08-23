"""Guarantee health monitor: score-distribution drift detection.

Compares per-image summary scores between the calibration set and a stream of
target images with a two-sample Kolmogorov-Smirnov test. A raised flag means
the exchangeability assumption underlying the calibration is suspect and the
nominal guarantee should not be trusted until recalibration; it does not by
itself repair the guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DriftReport:
    """Outcome of a drift check."""

    ks_statistic: float
    p_value: float
    flagged: bool
    n_calibration: int
    n_target: int


def ks_drift_check(
    calibration_scores: np.ndarray,
    target_scores: np.ndarray,
    significance: float = 0.01,
) -> DriftReport:
    """Two-sample KS test between calibration and target score samples.

    Parameters
    ----------
    calibration_scores, target_scores : np.ndarray
        One-dimensional samples of a per-image summary score (the same
        functional must be used on both sides).
    significance : float
        Flagging level for the KS p-value.
    """
    cal = np.asarray(calibration_scores, dtype=np.float64).ravel()
    tgt = np.asarray(target_scores, dtype=np.float64).ravel()
    if cal.size < 2 or tgt.size < 2:
        raise ValueError("need at least two scores on each side")
    result = stats.ks_2samp(cal, tgt, method="auto")
    return DriftReport(
        ks_statistic=float(result.statistic),
        p_value=float(result.pvalue),
        flagged=bool(result.pvalue < significance),
        n_calibration=cal.size,
        n_target=tgt.size,
    )


def image_summary_scores(marked_area_curves: np.ndarray, lam_index: int) -> np.ndarray:
    """Default per-image summary score: marked-area fraction at a threshold.

    Cheap, label-free, and computable from cached statistics for both
    calibration and incoming target images.
    """
    curves = np.asarray(marked_area_curves)
    return curves[:, lam_index].astype(np.float64)
