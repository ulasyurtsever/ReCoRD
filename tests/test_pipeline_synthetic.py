"""End-to-end synthetic pipeline test.

Simulates a segmentation model on images with planted ground-truth components,
runs the full analysis chain (statistics -> losses -> CRC -> evaluation), and
validates the empirical region-level risk guarantee. This exercises every
interface an experiment script uses, with no real data or model involved.
"""

import numpy as np

from record.coverage import compute_image_class_stats
from record.crc import crc_threshold
from record.evaluation import evaluate_at_threshold
from record.grid import LAMBDA_GRID
from record.losses import image_loss_curves

RNG = np.random.default_rng(42)

IMG = 40  # image side length


def _make_image(rng: np.random.Generator):
    """Plant 1-4 rectangular GT components; simulate a noisy prob map."""
    gt = np.zeros((IMG, IMG), dtype=bool)
    for _ in range(rng.integers(1, 5)):
        r, c = rng.integers(0, IMG - 8, size=2)
        h, w = rng.integers(3, 8, size=2)
        gt[r:r + h, c:c + w] = True
    # Model: informative but imperfect; per-image difficulty varies.
    quality = rng.uniform(0.4, 0.95)
    prob = np.where(gt, rng.beta(5 * quality, 2, size=gt.shape),
                    rng.beta(1, 8, size=gt.shape))
    return prob, gt


def _dataset_tables(n_images: int, rng: np.random.Generator, rho: float):
    """Build the per-image loss and marked-area tables an experiment consumes."""
    all_curves, comp_img_idx, comp_counts = [], [], []
    area = np.zeros((n_images, LAMBDA_GRID.size), dtype=np.float32)
    for i in range(n_images):
        prob, gt = _make_image(rng)
        stats = compute_image_class_stats(prob, gt)
        all_curves.append(stats.coverage_curves)
        comp_img_idx.extend([i] * len(stats.components))
        comp_counts.append(len(stats.components))
        area[i] = stats.marked_area_curve
    curves = np.vstack(all_curves)
    losses, _ = image_loss_curves(
        curves, np.array(comp_img_idx), n_images, rho=rho)
    return losses, area, np.array(comp_counts)


def test_end_to_end_guarantee_and_efficiency():
    rho, alpha = 0.5, 0.2
    n_images, n_trials = 160, 40
    losses, area, counts = _dataset_tables(n_images, RNG, rho)
    defined = ~np.isnan(losses[:, 0])
    assert defined.all()  # every synthetic image has at least one component

    fnrs, areas = [], []
    for _ in range(n_trials):
        perm = RNG.permutation(n_images)
        cal, test = perm[:80], perm[80:]
        sel = crc_threshold(losses[cal], alpha, LAMBDA_GRID)
        assert sel.feasible
        metrics = evaluate_at_threshold(sel, losses[test], area[test], counts[test])
        fnrs.append(metrics.region_fnr)
        areas.append(metrics.marked_area_fraction)

    mean_fnr = float(np.mean(fnrs))
    sem = float(np.std(fnrs) / np.sqrt(n_trials))
    assert mean_fnr <= alpha + 3 * sem, f"empirical FNR {mean_fnr:.3f} > alpha={alpha}"
    # The guarantee must not be achieved by trivially marking everything.
    assert float(np.mean(areas)) < 0.9


def test_stricter_alpha_gives_larger_threshold_and_area():
    rho = 0.5
    losses, area, counts = _dataset_tables(120, RNG, rho)
    sel_loose = crc_threshold(losses, 0.3, LAMBDA_GRID)
    sel_strict = crc_threshold(losses, 0.05, LAMBDA_GRID)
    assert sel_strict.lam >= sel_loose.lam
    m_loose = evaluate_at_threshold(sel_loose, losses, area, counts)
    m_strict = evaluate_at_threshold(sel_strict, losses, area, counts)
    assert m_strict.marked_area_fraction >= m_loose.marked_area_fraction


def test_detection_regime_easier_than_containment():
    losses_det, _, _ = _dataset_tables(100, np.random.default_rng(5), rho=0.1)
    losses_cont, _, _ = _dataset_tables(100, np.random.default_rng(5), rho=0.5)
    # Same images and probabilities: detection (rho=0.1) losses cannot exceed
    # containment (rho=0.5) losses anywhere.
    assert (losses_det <= losses_cont + 1e-9).all()
