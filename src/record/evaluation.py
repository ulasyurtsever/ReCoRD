"""Experiment evaluation: threshold selection on calibration units, metric
computation on test units.

All functions operate on precomputed arrays (loss curves, marked-area curves,
coverage curves) aligned with an image-id list, so full experiment sweeps are
pure array arithmetic over cached statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from record.crc import ThresholdSelection


@dataclass(frozen=True)
class RegionMetrics:
    """Test-set metrics at a selected threshold."""

    lam: float
    lam_index: int
    feasible: bool
    region_fnr: float
    marked_area_fraction: float
    n_test_images: int
    n_test_components: int


def index_rows(ids: list[str], subset: list[str]) -> np.ndarray:
    """Return row indices of ``subset`` identifiers within ``ids``.

    Raises if any identifier is missing, so silent misalignment between split
    files and cached tables is impossible.
    """
    lookup = {img_id: row for row, img_id in enumerate(ids)}
    missing = [s for s in subset if s not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} split ids missing from table, e.g. {missing[:3]}")
    return np.array([lookup[s] for s in subset], dtype=np.int64)


def evaluate_at_threshold(
    selection: ThresholdSelection,
    test_losses: np.ndarray,
    test_marked_area: np.ndarray,
    test_component_counts: np.ndarray,
) -> RegionMetrics:
    """Compute test metrics at a selected threshold.

    Parameters
    ----------
    test_losses : np.ndarray
        ``(n_test, n_grid)`` per-image loss curves; NaN rows (images without
        components) are excluded from the FNR average.
    test_marked_area : np.ndarray
        ``(n_test, n_grid)`` marked-area fraction curves (all images count).
    test_component_counts : np.ndarray
        ``(n_test,)`` number of ground-truth components per image.
    """
    col = selection.lam_index
    loss_col = test_losses[:, col]
    defined = ~np.isnan(loss_col)
    if not defined.any():
        raise ValueError("no test image contains ground-truth components")
    return RegionMetrics(
        lam=selection.lam,
        lam_index=selection.lam_index,
        feasible=selection.feasible,
        region_fnr=float(loss_col[defined].mean()),
        marked_area_fraction=float(test_marked_area[:, col].mean()),
        n_test_images=int(defined.sum()),
        n_test_components=int(test_component_counts.sum()),
    )


def stratified_region_fnr(
    coverage_curves: np.ndarray,
    component_strata: np.ndarray,
    lam_index: int,
    rho: float,
    n_strata: int = 3,
) -> list[float]:
    """Per-stratum component miss rates at a threshold.

    Component-level (not image-averaged) miss rates, reported per size
    stratum; NaN for empty strata.
    """
    # The curves must arrive in their stored dtype; see component_miss_matrix
    # for why upcasting them before this comparison changes the rho = 0.1 result.
    miss = coverage_curves[:, lam_index] < rho
    out: list[float] = []
    for s in range(n_strata):
        sel = component_strata == s
        out.append(float(miss[sel].mean()) if sel.any() else float("nan"))
    return out


def triage_curve(
    review_scores: np.ndarray,
    test_losses_at_lam: np.ndarray,
    budgets: np.ndarray,
) -> np.ndarray:
    """Residual miss rate as a function of human-review budget.

    Images are ranked by ``review_scores`` (descending: highest priority
    first). For each budget fraction b, the top-b fraction of images is
    assumed reviewed (their misses resolved by the human) and the total
    unresolved loss divided by the full test-set size is reported, i.e. the
    system-level miss rate after review. This quantity is nonincreasing in
    the budget by construction.

    Parameters
    ----------
    review_scores : np.ndarray
        ``(n_test,)`` priority scores (higher = reviewed first).
    test_losses_at_lam : np.ndarray
        ``(n_test,)`` per-image losses at the operating threshold; NaN rows
        (no components) are treated as zero loss and never prioritized out.
    budgets : np.ndarray
        Budget fractions in ``[0, 1]``.
    """
    losses = np.nan_to_num(test_losses_at_lam, nan=0.0)
    order = np.argsort(-review_scores)
    n = losses.size
    residual = np.empty(budgets.size)
    for i, b in enumerate(budgets):
        k = int(np.floor(b * n))
        remaining = order[k:]
        residual[i] = losses[remaining].sum() / n if remaining.size else 0.0
    return residual
