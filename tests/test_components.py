"""Unit tests for connected-component extraction."""

import numpy as np
import pytest

from record.components import extract_components, size_stratum


def test_simple_components():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True          # 4 px
    mask[6:9, 6:9] = True          # 9 px
    labels, comps = extract_components(mask)
    assert len(comps) == 2
    assert sorted(c.size_px for c in comps) == [4, 9]
    assert labels.max() == 2


def test_eight_connectivity_joins_diagonal():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    _, comps = extract_components(mask)
    assert len(comps) == 1  # diagonal neighbors merge under 8-connectivity


def test_min_size_filter():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    mask[5:8, 5:8] = True
    labels, comps = extract_components(mask, min_size_px=2)
    assert len(comps) == 1
    assert comps[0].size_px == 9
    assert not labels[0, 0]  # dropped component zeroed out


def test_empty_mask():
    labels, comps = extract_components(np.zeros((5, 5), dtype=bool))
    assert comps == []
    assert labels.sum() == 0


def test_bbox_covers_component():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True
    _, comps = extract_components(mask)
    assert comps[0].bbox == (2, 3, 5, 7)


def test_size_strata():
    assert size_stratum(1) == 0
    assert size_stratum(32**2 - 1) == 0
    assert size_stratum(32**2) == 1
    assert size_stratum(96**2 - 1) == 1
    assert size_stratum(96**2) == 2
    assert size_stratum(10**6) == 2


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        extract_components(np.zeros((2, 2, 2), dtype=bool))
