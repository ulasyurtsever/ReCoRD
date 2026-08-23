"""Unit tests for deterministic split generation."""

import numpy as np
import pytest

from record.splits import make_fixed_scheme, make_two_way_scheme

IDS = [f"img_{i:04d}" for i in range(500)]


def test_partition_sizes_and_disjointness():
    scheme = make_two_way_scheme("t_scheme", IDS, 250, base_seed=1, n_seeds=5)
    for entry in scheme["seeds"].values():
        cal, test = entry["calibration"], entry["test"]
        assert len(cal) == 250
        assert len(test) == 250
        assert not set(cal) & set(test)
        assert sorted(cal + test) == sorted(IDS)


def test_determinism_same_inputs():
    a = make_two_way_scheme("t_scheme", IDS, 250, base_seed=1, n_seeds=3)
    b = make_two_way_scheme("t_scheme", list(reversed(IDS)), 250, base_seed=1, n_seeds=3)
    assert a == b  # input order must not matter


def test_seeds_differ():
    scheme = make_two_way_scheme("t_scheme", IDS, 250, base_seed=1, n_seeds=2)
    assert scheme["seeds"]["0"] != scheme["seeds"]["1"]


def test_schemes_independent():
    a = make_two_way_scheme("scheme_a", IDS, 250, base_seed=1, n_seeds=1)
    b = make_two_way_scheme("scheme_b", IDS, 250, base_seed=1, n_seeds=1)
    assert a["seeds"]["0"] != b["seeds"]["0"]


def test_base_seed_changes_output():
    a = make_two_way_scheme("t_scheme", IDS, 250, base_seed=1, n_seeds=1)
    b = make_two_way_scheme("t_scheme", IDS, 250, base_seed=2, n_seeds=1)
    assert a["seeds"]["0"] != b["seeds"]["0"]


def test_custom_calibration_key():
    scheme = make_two_way_scheme(
        "t_scheme", IDS, 25, base_seed=1, n_seeds=1, cal_key="target_calibration")
    entry = scheme["seeds"]["0"]
    assert len(entry["target_calibration"]) == 25
    assert len(entry["test"]) == len(IDS) - 25


def test_invalid_cal_size_raises():
    with pytest.raises(ValueError):
        make_two_way_scheme("t_scheme", IDS, len(IDS), base_seed=1, n_seeds=1)
    with pytest.raises(ValueError):
        make_two_way_scheme("t_scheme", IDS, 0, base_seed=1, n_seeds=1)


def test_fixed_scheme_sorts_and_counts():
    groups = {"train": ["b", "a"], "test": ["c"]}
    scheme = make_fixed_scheme("t_fixed", groups)
    assert scheme["seeds"]["0"]["train"] == ["a", "b"]
    assert scheme["n_items"] == 3


def test_statistical_sanity_of_sampling():
    # Each item should appear in calibration roughly half the time across seeds.
    scheme = make_two_way_scheme("t_scheme", IDS, 250, base_seed=1, n_seeds=200)
    counts = {i: 0 for i in IDS}
    for entry in scheme["seeds"].values():
        for i in entry["calibration"]:
            counts[i] += 1
    freqs = np.array(list(counts.values())) / 200
    assert 0.35 < freqs.mean() < 0.65
    assert freqs.min() > 0.2 and freqs.max() < 0.8
