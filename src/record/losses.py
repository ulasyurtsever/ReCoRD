"""Region-level loss curves derived from cached coverage statistics.

The image-level loss at threshold ``lam`` is the fraction of the image's
ground-truth components whose coverage is below the capture threshold ``rho``.
Losses are bounded in ``[0, 1]`` and nonincreasing in ``lam`` (coverage curves
are nondecreasing), which is the monotonicity required by conformal risk
control.

Images without any ground-truth component of the class carry no region-level
information; they are excluded from loss aggregation, and the guarantee is
stated over images containing at least one component.
"""

from __future__ import annotations

import numpy as np


def component_miss_matrix(coverage_curves: np.ndarray, rho: float) -> np.ndarray:
    """Return the per-component miss indicator matrix at capture level ``rho``.

    Parameters
    ----------
    coverage_curves : np.ndarray
        Array of shape ``(n_components, n_grid)``.
    rho : float
        Capture threshold in ``(0, 1]``: a component is captured at ``lam``
        iff its coverage is ``>= rho``.
    """
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    return (np.asarray(coverage_curves) < rho).astype(np.float32)


def image_loss_curves(
    coverage_curves: np.ndarray,
    component_image_index: np.ndarray,
    n_images: int,
    rho: float,
    size_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate component misses into per-image loss curves.

    Parameters
    ----------
    coverage_curves : np.ndarray
        Array of shape ``(n_components, n_grid)`` for one class, possibly
        spanning many images.
    component_image_index : np.ndarray
        Array of shape ``(n_components,)`` mapping each component to an image
        row in ``[0, n_images)``.
    n_images : int
        Total number of images considered (with or without components).
    rho : float
        Capture threshold.
    size_weights : np.ndarray or None
        Optional per-component weights (e.g. pixel sizes) for the
        size-weighted loss ablation. Unweighted by default.

    Returns
    -------
    losses : np.ndarray
        Array of shape ``(n_images, n_grid)``; rows for images without
        components are NaN.
    has_components : np.ndarray
        Boolean array of shape ``(n_images,)``.
    """
    miss = component_miss_matrix(coverage_curves, rho)
    n_grid = miss.shape[1]

    weights = np.ones(miss.shape[0]) if size_weights is None else np.asarray(size_weights, float)
    if weights.shape[0] != miss.shape[0]:
        raise ValueError("size_weights length mismatch")

    weighted_miss = miss * weights[:, None]
    num = np.zeros((n_images, n_grid))
    den = np.zeros(n_images)
    np.add.at(num, component_image_index, weighted_miss)
    np.add.at(den, component_image_index, weights)

    has_components = den > 0
    losses = np.full((n_images, n_grid), np.nan)
    losses[has_components] = num[has_components] / den[has_components, None]
    return losses, has_components


def enforce_nonincreasing(losses: np.ndarray) -> np.ndarray:
    """Clamp tiny numerical violations so each row is nonincreasing.

    Each entry is replaced by the running maximum over its suffix (the upper
    envelope): the identity for genuinely nonincreasing rows, and a
    conservative correction otherwise (the loss is never underestimated,
    which preserves the direction of the risk-control guarantee).
    """
    return np.maximum.accumulate(losses[..., ::-1], axis=-1)[..., ::-1]
