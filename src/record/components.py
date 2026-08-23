"""Connected-component extraction from ground-truth label masks.

Components are always extracted from ground truth, never from predictions:
the region-level loss is defined over ground-truth components, so merge/split
behavior on the prediction side cannot affect the loss definition.

Connectivity is 8-neighborhood, fixed project-wide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

STRUCTURE_8 = np.ones((3, 3), dtype=bool)

# Size strata (pixel-count intervals) for stratified reporting: boundaries at
# 32^2 and 96^2 pixels.
SIZE_STRATA: tuple[tuple[int, float], ...] = (
    (0, 32**2),
    (32**2, 96**2),
    (96**2, float("inf")),
)


@dataclass(frozen=True)
class Component:
    """A single ground-truth connected component."""

    component_id: int
    size_px: int
    bbox: tuple[int, int, int, int]  # (row_min, col_min, row_max_excl, col_max_excl)


def size_stratum(size_px: int) -> int:
    """Return the index of the size stratum containing ``size_px``."""
    for idx, (lo, hi) in enumerate(SIZE_STRATA):
        if lo <= size_px < hi:
            return idx
    raise ValueError(f"size_px={size_px} not covered by strata")


def extract_components(binary_mask: np.ndarray, min_size_px: int = 1) -> tuple[np.ndarray, list[Component]]:
    """Label connected components of a binary mask.

    Parameters
    ----------
    binary_mask : np.ndarray
        2-D boolean array marking pixels of one class.
    min_size_px : int
        Components smaller than this are dropped (default keeps all).

    Returns
    -------
    labels : np.ndarray
        2-D int array; 0 is background, components are numbered from 1.
        Dropped components are zeroed out.
    components : list[Component]
        Metadata for each retained component, ordered by component id.
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"expected 2-D mask, got shape {binary_mask.shape}")
    labels, n = ndimage.label(binary_mask, structure=STRUCTURE_8)
    components: list[Component] = []
    if n == 0:
        return labels, components

    slices = ndimage.find_objects(labels)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    for cid in range(1, n + 1):
        size = int(sizes[cid])
        if size < min_size_px:
            labels[labels == cid] = 0
            continue
        sl = slices[cid - 1]
        components.append(
            Component(
                component_id=cid,
                size_px=size,
                bbox=(sl[0].start, sl[1].start, sl[0].stop, sl[1].stop),
            )
        )
    return labels, components
