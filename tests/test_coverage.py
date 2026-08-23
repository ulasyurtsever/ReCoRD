"""Unit tests for coverage statistics against brute-force references."""

import numpy as np

from record.coverage import compute_image_class_stats, pixel_fnr_curve
from record.grid import FP_SUBGRID_INDICES, LAMBDA_GRID, curve_on_grid

RNG = np.random.default_rng(7)


def _brute_force_coverage(prob, gt_labels, cid, lam):
    inside = gt_labels == cid
    return ((1.0 - prob[inside]) <= lam).mean()


def test_curve_on_grid_matches_definition():
    scores = RNG.uniform(size=257)
    curve = curve_on_grid(scores)
    for k in (0, 250, 500, 750, 1000):
        assert curve[k] == (scores <= LAMBDA_GRID[k]).mean()


def test_curves_nondecreasing_and_bounded():
    prob = RNG.uniform(size=(64, 64))
    gt = RNG.uniform(size=(64, 64)) > 0.6
    stats = compute_image_class_stats(prob, gt)
    for curve in stats.coverage_curves:
        assert (np.diff(curve) >= 0).all()
        assert curve[0] >= 0 and curve[-1] == 1.0
    assert (np.diff(stats.marked_area_curve) >= 0).all()


def test_component_coverage_matches_brute_force():
    prob = RNG.uniform(size=(48, 48))
    gt = np.zeros((48, 48), dtype=bool)
    gt[5:15, 5:15] = True
    gt[30:40, 30:44] = True
    stats = compute_image_class_stats(prob, gt)
    from record.components import extract_components

    labels, comps = extract_components(gt)
    assert len(stats.components) == len(comps) == 2
    for row, comp in enumerate(stats.components):
        for k in (100, 500, 900):
            expected = _brute_force_coverage(prob, labels, comp.component_id, LAMBDA_GRID[k])
            assert abs(stats.coverage_curves[row, k] - expected) < 1e-6


def test_argmax_coverage_and_area():
    prob = np.full((20, 20), 0.4)
    argmax = np.zeros((20, 20), dtype=bool)
    argmax[:10] = True
    gt = np.zeros((20, 20), dtype=bool)
    gt[8:12, :] = True  # half inside the argmax region
    stats = compute_image_class_stats(prob, gt, argmax_mask=argmax)
    assert abs(stats.argmax_coverage[0] - 0.5) < 1e-6
    assert abs(stats.argmax_marked_area - 0.5) < 1e-6


def test_fp_component_counts():
    # Probability 0.9 on two blobs: one overlapping GT, one disjoint.
    prob = np.zeros((30, 30))
    prob[2:6, 2:6] = 0.9    # overlaps GT
    prob[20:24, 20:24] = 0.9  # disjoint: false-positive component
    gt = np.zeros((30, 30), dtype=bool)
    gt[2:6, 2:6] = True
    stats = compute_image_class_stats(prob, gt)
    # For lam in [0.2, 1) the mask includes both blobs but not the background;
    # exactly one blob is disjoint from GT. At lam = 1 the background joins and
    # the whole image collapses into a single GT-overlapping component.
    sub = LAMBDA_GRID[FP_SUBGRID_INDICES]
    mid = (sub >= 0.2) & (sub < 1.0)
    assert (stats.fp_component_counts[mid] == 1).all()
    assert stats.fp_component_counts[-1] == 0


def test_pixel_fnr_matches_direct_computation():
    prob = RNG.uniform(size=(40, 40))
    gt = np.zeros((40, 40), dtype=bool)
    gt[3:17, 4:18] = True
    gt[25:38, 22:39] = True
    stats = compute_image_class_stats(prob, gt)
    fnr = pixel_fnr_curve(stats)
    for k in (200, 600):
        covered = ((1.0 - prob[gt]) <= LAMBDA_GRID[k]).mean()
        assert abs(fnr[k] - (1.0 - covered)) < 1e-6


def test_no_components_case():
    prob = RNG.uniform(size=(16, 16))
    gt = np.zeros((16, 16), dtype=bool)
    stats = compute_image_class_stats(prob, gt)
    assert stats.coverage_curves.shape == (0, LAMBDA_GRID.size)
