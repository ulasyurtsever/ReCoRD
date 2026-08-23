"""Sufficient statistics for region-level analysis.

For each image and critical class, this module reduces a per-pixel probability
map and the ground-truth mask to small per-component and per-image curves on
the canonical lambda grid. All downstream calibration and evaluation is pure
arithmetic over these statistics; the probability map is never needed again.

Definitions
-----------
The prediction mask at threshold ``lam`` is ``{p : prob[p] >= 1 - lam}``,
equivalently ``{p : score[p] <= lam}`` with ``score = 1 - prob``. The mask is
nondecreasing in ``lam``, hence every curve produced here is nondecreasing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from record.components import STRUCTURE_8, Component, extract_components
from record.grid import FP_SUBGRID_INDICES, LAMBDA_GRID, curve_on_grid


@dataclass
class ImageClassStats:
    """Region and image statistics for one (image, class) pair."""

    components: list[Component]
    coverage_curves: np.ndarray        # (n_components, N_POINTS) in [0, 1]
    argmax_coverage: np.ndarray        # (n_components,) coverage under argmax mask
    marked_area_curve: np.ndarray      # (N_POINTS,) fraction of image pixels in mask
    argmax_marked_area: float
    fp_component_counts: np.ndarray    # (len(FP_SUBGRID_INDICES),) predicted components
    #                                    disjoint from all GT components of the class
    gt_pixel_fraction: float = field(default=0.0)


def compute_image_class_stats(
    prob: np.ndarray,
    gt_mask: np.ndarray,
    argmax_mask: np.ndarray | None = None,
    min_component_px: int = 1,
) -> ImageClassStats:
    """Compute all statistics for one image and one critical class.

    Parameters
    ----------
    prob : np.ndarray
        2-D float array, per-pixel probability of the class.
    gt_mask : np.ndarray
        2-D boolean array, ground-truth pixels of the class.
    argmax_mask : np.ndarray or None
        2-D boolean array, pixels argmax-assigned to the class. When None,
        argmax-based statistics are reported as NaN.
    min_component_px : int
        Minimum ground-truth component size to retain.
    """
    if prob.shape != gt_mask.shape:
        raise ValueError(f"shape mismatch: prob {prob.shape} vs gt {gt_mask.shape}")
    score = 1.0 - np.asarray(prob, dtype=np.float32)

    labels, components = extract_components(gt_mask, min_size_px=min_component_px)

    n = len(components)
    curves = np.zeros((n, LAMBDA_GRID.size), dtype=np.float32)
    argmax_cov = np.full(n, np.nan, dtype=np.float32)
    for row, comp in enumerate(components):
        r0, c0, r1, c1 = comp.bbox
        window_labels = labels[r0:r1, c0:c1]
        inside = window_labels == comp.component_id
        curves[row] = curve_on_grid(score[r0:r1, c0:c1][inside])
        if argmax_mask is not None:
            argmax_cov[row] = argmax_mask[r0:r1, c0:c1][inside].mean()

    marked_area = curve_on_grid(score)
    argmax_area = float(argmax_mask.mean()) if argmax_mask is not None else float("nan")

    fp_counts = _fp_component_counts(score, gt_mask)

    return ImageClassStats(
        components=components,
        coverage_curves=curves,
        argmax_coverage=argmax_cov,
        marked_area_curve=marked_area,
        argmax_marked_area=argmax_area,
        fp_component_counts=fp_counts,
        gt_pixel_fraction=float(gt_mask.mean()),
    )


def _fp_component_counts(score: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Count predicted components disjoint from ground truth on the subgrid.

    Evaluated on ``FP_SUBGRID_INDICES`` only, since each threshold requires a
    connected-component labeling of the prediction mask.
    """
    counts = np.zeros(FP_SUBGRID_INDICES.size, dtype=np.int32)
    for i, gi in enumerate(FP_SUBGRID_INDICES):
        mask = score <= LAMBDA_GRID[gi]
        if not mask.any():
            continue
        pred_labels, n_pred = ndimage.label(mask, structure=STRUCTURE_8)
        if n_pred == 0:
            continue
        hit = np.unique(pred_labels[gt_mask])
        counts[i] = n_pred - np.count_nonzero(hit)
    return counts


def pixel_fnr_curve(stats: ImageClassStats) -> np.ndarray:
    """Return the image's pixel-level false-negative-rate curve for the class.

    Derived exactly from component statistics: the covered fraction of class
    pixels is the size-weighted mean of component coverages.
    """
    if not stats.components:
        raise ValueError("image has no ground-truth components for this class")
    sizes = np.array([c.size_px for c in stats.components], dtype=np.float64)
    covered = (stats.coverage_curves * sizes[:, None]).sum(axis=0) / sizes.sum()
    return 1.0 - covered
