"""Unit tests for the schematic and qualitative figure scripts."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schematic = _load("schematic", "11_make_schematic.py")
qualitative = _load("qualitative", "12_make_qualitative.py")


def test_schematic_mask_grows_with_threshold():
    score = schematic._score_field()
    areas = [(score >= 1.0 - lam).sum() for lam in (0.1, 0.3, 0.5, 0.9)]
    assert areas == sorted(areas)


def test_schematic_illustrates_both_capture_outcomes():
    """The three panels must show a miss and a capture, or they teach nothing."""
    score = schematic._score_field()
    small = schematic._disc(44, 46, 4.5)
    coverages = [((score >= 1.0 - lam) & small).sum() / small.sum()
                 for lam in (0.30, 0.55, 0.80)]
    assert coverages[0] < 0.5 < coverages[-1]


def test_schematic_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORD_RESULTS_ROOT", str(tmp_path))
    out = Path(schematic.build())
    assert out.exists() and out.stat().st_size > 0


def test_rgb_band_indices_are_descending_visible():
    """True color is (B4, B3, B2): red before green before blue."""
    assert qualitative.RGB_BANDS == (3, 2, 1)


def test_rgb_composite_stretches_to_unit_range(monkeypatch):
    rng = np.random.default_rng(0)
    bands = rng.normal(1000, 200, size=(11, 16, 16)).astype(np.float32)
    monkeypatch.setattr(qualitative, "load_bands", lambda _pid: bands)
    out = qualitative.rgb_composite("marida_test", "dummy")
    assert out.shape == (16, 16, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.max() > out.min()


def test_rgb_composite_handles_constant_band(monkeypatch):
    bands = np.ones((11, 8, 8), dtype=np.float32)
    monkeypatch.setattr(qualitative, "load_bands", lambda _pid: bands)
    out = qualitative.rgb_composite("marida_test", "dummy")
    assert np.all(np.isfinite(out))


def _write_experiment(tmp_path, rows):
    import pandas as pd

    exp = tmp_path / "experiments"
    exp.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(exp / "exp.csv", index=False)


def test_calibrated_threshold_reads_median_of_feasible_rows(tmp_path, monkeypatch):
    _write_experiment(tmp_path, {
        "method": ["region_crc"] * 4 + ["argmax"],
        "alpha": [0.2] * 5,
        "rho": [0.5] * 5,
        "class_name": ["building"] * 5,
        "feasible": [True, True, True, False, True],
        "lam": [0.90, 0.94, 0.98, 0.10, 0.50],
    })
    monkeypatch.setenv("RECORD_RESULTS_ROOT", str(tmp_path))
    assert qualitative.calibrated_threshold(
        "exp", 0.2, 0.5, "building") == pytest.approx(0.94)


def test_calibrated_threshold_is_class_specific(tmp_path, monkeypatch):
    """Averaging across classes would show a threshold no table reports."""
    _write_experiment(tmp_path, {
        "method": ["region_crc"] * 4,
        "alpha": [0.2] * 4,
        "rho": [0.5] * 4,
        "class_name": ["building", "building", "water", "water"],
        "feasible": [True] * 4,
        "lam": [0.80, 0.82, 0.20, 0.22],
    })
    monkeypatch.setenv("RECORD_RESULTS_ROOT", str(tmp_path))
    assert qualitative.calibrated_threshold(
        "exp", 0.2, 0.5, "building") == pytest.approx(0.81)


def test_calibrated_threshold_rejects_empty_selection(tmp_path, monkeypatch):
    _write_experiment(tmp_path, {"method": ["region_crc"], "alpha": [0.1],
                                 "rho": [0.5], "class_name": ["building"],
                                 "feasible": [True], "lam": [0.9]})
    monkeypatch.setenv("RECORD_RESULTS_ROOT", str(tmp_path))
    with pytest.raises(ValueError):
        qualitative.calibrated_threshold("exp", 0.2, 0.5, "building")


def test_benchmarks_declare_consistent_keys():
    for name, spec in qualitative.BENCHMARKS.items():
        assert {"dataset_key", "model", "experiment", "family", "class_name",
                "label", "min_visible_px"} <= set(spec), name
        assert spec["min_visible_px"] >= 1


def test_ranking_prefers_recovered_regions(monkeypatch):
    """The selection rule must rank by regions recovered, not by argmax failure."""
    def fake_stats(model, dataset_key, patch_id, gt_value, channel, lam, rho):
        table = {
            # all regions tiny and missed by everything: must be filtered out
            "tiny": {"n_regions": 4, "regions": [
                {"size_px": 1, "cov_argmax": 0.0, "cov_crc": 0.0} for _ in range(4)]},
            # visible, argmax misses two, the threshold recovers both
            "good": {"n_regions": 3, "regions": [
                {"size_px": 300, "cov_argmax": 0.0, "cov_crc": 0.9},
                {"size_px": 120, "cov_argmax": 0.1, "cov_crc": 0.8},
                {"size_px": 90, "cov_argmax": 0.9, "cov_crc": 0.9}]},
            # visible but nothing recovered
            "flat": {"n_regions": 3, "regions": [
                {"size_px": 300, "cov_argmax": 0.9, "cov_crc": 0.9},
                {"size_px": 120, "cov_argmax": 0.9, "cov_crc": 0.9},
                {"size_px": 90, "cov_argmax": 0.0, "cov_crc": 0.0}]},
        }
        stats = dict(table[patch_id])
        stats["patch_id"] = patch_id
        stats["missed_argmax"] = sum(r["cov_argmax"] < rho for r in stats["regions"])
        stats["missed_crc"] = sum(r["cov_crc"] < rho for r in stats["regions"])
        return stats

    monkeypatch.setattr(qualitative, "list_ids", lambda _k: ["tiny", "good", "flat"])
    monkeypatch.setattr(qualitative, "load_label_array",
                        lambda _k, _p: np.ones((4, 4), dtype=int))
    monkeypatch.setattr(qualitative, "patch_statistics", fake_stats)
    ranked = qualitative.rank_patches("m", "loveda_Val_Urban", 1, 0, 0.7, 0.5, 2, 25)
    assert [s["patch_id"] for s in ranked] == ["good", "flat"], "tiny must be filtered"
    assert ranked[0]["recovered"] == 2


def test_ranking_rejects_all_tiny_regions(monkeypatch):
    def only_tiny(model, dataset_key, patch_id, gt_value, channel, lam, rho):
        return {"patch_id": patch_id, "n_regions": 2,
                "regions": [{"size_px": 2, "cov_argmax": 0.0, "cov_crc": 0.0}] * 2,
                "missed_argmax": 2, "missed_crc": 2}

    monkeypatch.setattr(qualitative, "list_ids", lambda _k: ["a", "b"])
    monkeypatch.setattr(qualitative, "load_label_array",
                        lambda _k, _p: np.ones((4, 4), dtype=int))
    monkeypatch.setattr(qualitative, "patch_statistics", only_tiny)
    with pytest.raises(ValueError):
        qualitative.select_patch("m", "loveda_Val_Urban", 1, 0, 0.7, 0.5, 2, 25)
