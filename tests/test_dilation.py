"""The dilation family mapped onto the lambda grid (record.dilation)."""

import numpy as np
import pytest

from record.coverage import compute_image_class_stats
from record.dilation import R_MAX_PX, dilation_pseudo_prob, radius_of_index, radius_px
from record.grid import LAMBDA_GRID


def _scene():
    """A 40x60 frame: argmax blob at the left, one GT component near it and
    one far from it."""
    argmax = np.zeros((40, 60), dtype=bool)
    argmax[10:20, 5:15] = True
    gt = np.zeros((40, 60), dtype=bool)
    gt[12:18, 12:22] = True   # overlaps the blob partly, rest within ~7 px
    gt[30:36, 50:58] = True   # far component: >= 35 px away
    return argmax, gt


def test_pseudo_prob_is_one_on_mask_and_decreases_with_distance():
    argmax, _ = _scene()
    prob = dilation_pseudo_prob(argmax)
    assert prob.dtype == np.float32
    assert np.all(prob[argmax] == 1.0)
    # One pixel to the right of the blob: distance 1 -> grid index 10 on a
    # 1001-point grid with R_MAX_PX = 100, whatever the grid's spacing.
    last = LAMBDA_GRID.size - 1
    assert prob[15, 15] == pytest.approx(1.0 - LAMBDA_GRID[int(round(1.0 * last / R_MAX_PX))], abs=1e-6)
    # Ten pixels to the right: distance 10 -> index 100.
    assert prob[15, 24] == pytest.approx(1.0 - LAMBDA_GRID[int(round(10.0 * last / R_MAX_PX))], abs=1e-6)
    assert prob.min() >= 0.0 and prob.max() <= 1.0


def test_empty_argmax_gives_zero_everywhere():
    prob = dilation_pseudo_prob(np.zeros((8, 8), dtype=bool))
    assert prob.shape == (8, 8) and np.all(prob == np.float32(1.0 - LAMBDA_GRID[-1]))


def test_grid_point_is_a_dilation_radius():
    assert radius_px(0.0) == 0.0
    assert radius_px(1.0) == R_MAX_PX
    assert radius_of_index(len(LAMBDA_GRID) - 1) == R_MAX_PX
    # 0.1 px per grid index on a 1001-point grid, independent of the spacing.
    assert radius_of_index(1) == pytest.approx(0.1)
    assert radius_px(LAMBDA_GRID[500]) == pytest.approx(50.0)


def test_stage4_statistics_read_the_family_as_dilation():
    """Through compute_image_class_stats the family behaves as advertised:
    lam = 0 is the argmax mask, coverage is nondecreasing in lam, and lam = 1
    covers every component."""
    argmax, gt = _scene()
    prob = dilation_pseudo_prob(argmax)
    stats = compute_image_class_stats(prob, gt, argmax_mask=argmax)
    assert len(stats.components) == 2
    curves = stats.coverage_curves
    # Radius 0 == argmax coverage, component by component.
    np.testing.assert_allclose(curves[:, 0], stats.argmax_coverage, atol=1e-6)
    # Nested masks -> nondecreasing coverage.
    assert np.all(np.diff(curves, axis=1) >= -1e-7)
    # lam = 1 marks the whole image: every component fully covered, area 1.
    assert np.all(curves[:, -1] == 1.0)
    assert stats.marked_area_curve[-1] == 1.0
    # The near component is fully covered well before the far one: at 10 px
    # (index 100) the near one is complete, the far one untouched.
    near, far = (0, 1) if stats.components[0].bbox[1] < 30 else (1, 0)
    assert curves[near, 100] == 1.0
    assert curves[far, 100] == 0.0
    # Marked area grows with the radius.
    assert stats.marked_area_curve[0] == pytest.approx(argmax.mean())
    assert stats.marked_area_curve[100] > stats.marked_area_curve[0]


def test_far_component_is_reached_only_at_lambda_max_when_beyond_r_max():
    argmax = np.zeros((20, 300), dtype=bool)
    argmax[5:10, 0:5] = True
    gt = np.zeros((20, 300), dtype=bool)
    gt[5:10, 250:260] = True        # 245 px away, beyond R_MAX_PX = 100
    prob = dilation_pseudo_prob(argmax)
    stats = compute_image_class_stats(prob, gt, argmax_mask=argmax)
    curve = stats.coverage_curves[0]
    assert np.all(curve[:-1] == 0.0)
    assert curve[-1] == 1.0
