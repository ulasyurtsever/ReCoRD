"""Integration test: stage-5 experiment script over synthetic stage-4 tables.

Fabricates a complete stage-4 table set and a split scheme in temporary
directories, invokes ``scripts/05_run_experiments.py`` as a subprocess, and
validates the produced CSV. This exercises the exact file formats and CLI
used in the full pipeline, with no model or dataset involved.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(3)

N_IMAGES = 60
MODEL = "synthetic_model"
DATASET = "synthetic_val"
CLASS_NAMES = ["person"]


def _make_stage4_tables(results_root: Path) -> None:
    from record.grid import FP_SUBGRID_INDICES, LAMBDA_GRID, curve_on_grid

    out_dir = results_root / "raw" / MODEL / DATASET
    out_dir.mkdir(parents=True)

    rows, curves = [], []
    marked_area = np.zeros((N_IMAGES, 1, LAMBDA_GRID.size), dtype=np.float16)
    argmax_area = RNG.uniform(0.01, 0.05, size=(N_IMAGES, 1)).astype(np.float32)
    fp_counts = np.zeros((N_IMAGES, 1, FP_SUBGRID_INDICES.size), dtype=np.int32)
    component_counts = np.zeros((N_IMAGES, 1), dtype=np.int32)

    for img_row in range(N_IMAGES):
        marked_area[img_row, 0] = curve_on_grid(RNG.uniform(0.2, 1.0, 500)).astype(np.float16)
        # First five images carry no components: they exercise the
        # degenerate-calibration path (rare class absent from a small draw).
        n_comp = 0 if img_row < 5 else int(RNG.integers(1, 4))
        component_counts[img_row, 0] = n_comp
        for cid in range(1, n_comp + 1):
            quality = RNG.uniform(0.2, 0.8)
            scores = RNG.uniform(0, 1, size=200) * quality
            size = int(RNG.integers(20, 5000))
            rows.append({
                "image_id": f"img_{img_row:03d}", "image_row": img_row,
                "class_name": "person", "component_id": cid,
                "size_px": size, "size_stratum": 0 if size < 1024 else 1,
                "bbox_r0": 0, "bbox_c0": 0, "bbox_r1": 10, "bbox_c1": 10,
                "argmax_coverage": float(RNG.uniform(0, 1)),
                "curve_row": len(curves),
            })
            curves.append(curve_on_grid(scores))

    pd.DataFrame(rows).to_parquet(out_dir / "components.parquet", index=False)
    np.save(out_dir / "coverage_curves.npy", np.array(curves, dtype=np.float16))
    np.savez_compressed(
        out_dir / "image_stats.npz",
        image_ids=np.array([f"img_{i:03d}" for i in range(N_IMAGES)]),
        class_names=np.array(CLASS_NAMES),
        marked_area=marked_area,
        argmax_marked_area=argmax_area,
        fp_component_counts=fp_counts,
        component_counts=component_counts,
    )


def _make_scheme(splits_root: Path) -> None:
    from record.splits import make_fixed_scheme, make_two_way_scheme, write_scheme

    ids = [f"img_{i:03d}" for i in range(N_IMAGES)]
    scheme = make_two_way_scheme("synthetic_half", ids, N_IMAGES // 2,
                                 base_seed=99, n_seeds=3)
    write_scheme(scheme, splits_root)
    # Degenerate target-calibration draw: only component-free images.
    degenerate = make_fixed_scheme(
        "synthetic_degenerate",
        {"target_calibration": ids[:5], "test": ids[5:]})
    write_scheme(degenerate, splits_root)
    # Official-style scheme with custom entry keys (train/val/test).
    official = make_fixed_scheme(
        "synthetic_official",
        {"train": ids[:20], "val": ids[20:40], "test": ids[40:]})
    write_scheme(official, splits_root)


def _make_lac_curves(results_root: Path) -> None:
    from record.grid import LAMBDA_GRID

    # Monotone nonincreasing miscoverage curves, one per image.
    start = RNG.uniform(0.2, 0.9, size=N_IMAGES)
    curves = start[:, None] * (1.0 - LAMBDA_GRID[None, :])
    np.save(results_root / "raw" / MODEL / DATASET / "lac_miscoverage.npy",
            curves.astype(np.float16))


def _make_embeddings(cache_root: Path) -> None:
    out = cache_root / "synthetic_emb" / DATASET
    out.mkdir(parents=True)
    np.savez_compressed(
        out / "embeddings.npz",
        image_ids=np.array([f"img_{i:03d}" for i in range(N_IMAGES)]),
        embeddings=RNG.normal(size=(N_IMAGES, 16)).astype(np.float32),
    )


@pytest.fixture
def synthetic_env(tmp_path, monkeypatch):
    # Reseed the module RNG so every test sees identical synthetic data
    # regardless of execution order.
    global RNG
    RNG = np.random.default_rng(3)
    results_root = tmp_path / "results"
    splits_root = tmp_path / "splits"
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("RECORD_RESULTS_ROOT", str(results_root))
    monkeypatch.setenv("RECORD_SPLITS_ROOT", str(splits_root))
    monkeypatch.setenv("RECORD_CACHE_ROOT", str(cache_root))
    _make_stage4_tables(results_root)
    _make_lac_curves(results_root)
    _make_scheme(splits_root)
    _make_embeddings(cache_root)
    return results_root


def test_stage5_end_to_end(synthetic_env):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    cmd = [
        sys.executable, str(REPO / "scripts" / "05_run_experiments.py"),
        "--name", "synthetic_run",
        "--model", MODEL,
        "--scheme", "synthetic_half",
        "--cal-datasets", DATASET,
        "--test-datasets", DATASET,
        "--methods", "region_crc", "pixel_crc", "heuristic", "argmax",
        "--alphas", "0.2",
        "--rhos", "0.5",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RESULT: PASS" in proc.stdout

    csv_path = synthetic_env / "experiments" / "synthetic_run.csv"
    frame = pd.read_csv(csv_path)
    assert set(frame["method"]) == {"region_crc", "pixel_crc", "heuristic", "argmax"}
    assert len(frame) == 3 * 4  # seeds x methods

    crc_rows = frame[frame["method"] == "region_crc"]
    assert crc_rows["feasible"].all()
    # Region CRC controls the region FNR on average (generous tolerance for 3 seeds).
    assert crc_rows["region_fnr"].mean() <= 0.2 + 0.1
    assert (crc_rows["marked_area_fraction"] < 1.0).all()
    # Non-degeneracy: synthetic components are never captured at lam = 0, so a
    # zero threshold would indicate collapsed loss curves (regression guard).
    assert (crc_rows["lam"] > 0).all()
    assert crc_rows["region_fnr"].mean() > 0.005
    # Heuristic threshold is never larger than the CRC threshold.
    heur = frame[frame["method"] == "heuristic"].set_index("seed")["lam"]
    crc = crc_rows.set_index("seed")["lam"]
    assert (heur <= crc + 1e-9).all()

    meta = json.loads((synthetic_env / "experiments" / "synthetic_run.meta.json").read_text())
    assert meta["n_records"] == len(frame)


def test_stage5_degenerate_calibration_is_recorded_not_fatal(synthetic_env):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    cmd = [
        sys.executable, str(REPO / "scripts" / "05_run_experiments.py"),
        "--name", "degenerate_run",
        "--model", MODEL,
        "--scheme", "synthetic_degenerate",
        "--cal-datasets", DATASET,
        "--test-datasets", DATASET,
        "--methods", "region_crc", "heuristic",
        "--alphas", "0.2",
        "--rhos", "0.5",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    frame = pd.read_csv(synthetic_env / "experiments" / "degenerate_run.csv")
    assert len(frame) == 2  # region_crc + heuristic, single fixed split
    assert (~frame["feasible"]).all()
    assert frame["region_fnr"].isna().all()
    assert (frame["n_cal_images"] == 0).all()


def _run_stage5(synthetic_env, name, extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    cmd = [sys.executable, str(REPO / "scripts" / "05_run_experiments.py"),
           "--name", name, "--model", MODEL,
           "--cal-datasets", DATASET, "--test-datasets", DATASET,
           "--alphas", "0.2", "--rhos", "0.5", *extra]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return pd.read_csv(synthetic_env / "experiments" / f"{name}.csv")


def test_stage5_custom_split_keys(synthetic_env):
    frame = _run_stage5(synthetic_env, "official_run", [
        "--scheme", "synthetic_official",
        "--cal-split", "val", "--test-split", "test",
        "--methods", "region_crc"])
    assert len(frame) == 1  # single fixed entry
    row = frame.iloc[0]
    assert row["n_test_images"] == 20
    assert 0 < row["n_cal_images"] <= 20


def test_stage5_weight_clip_and_diagnostics(synthetic_env):
    base = ["--scheme", "synthetic_half", "--methods", "weighted_crc",
            "--embedding", "synthetic_emb", "--max-seeds", "2"]
    default = _run_stage5(synthetic_env, "wclip_default", base)
    tight = _run_stage5(synthetic_env, "wclip_tight", base + ["--weight-clip-max", "2"])

    for frame in (default, tight):
        assert frame["weight_ess"].notna().all()
        assert frame["weight_p_test"].between(0, 1).all()
        assert (frame["weight_ess"] <= frame["n_cal_images"] + 1e-9).all()
    # A tighter clip lowers the conservative test-point mass.
    assert (tight["weight_p_test"] < default["weight_p_test"]).all()


def test_stage5_shared_lambda_and_triage(synthetic_env):
    frame = _run_stage5(synthetic_env, "shared_run", [
        "--scheme", "synthetic_half", "--max-seeds", "2",
        "--methods", "region_crc", "shared_crc", "--triage"])
    shared = frame[frame["method"] == "shared_crc"]
    assert "max_over_classes" in set(shared["class_name"])
    # Joint control: per-class FNR never exceeds the max-over-classes FNR.
    for seed in shared["seed"].unique():
        sub = shared[shared["seed"] == seed]
        joint = sub[sub["class_name"] == "max_over_classes"]["region_fnr"].iloc[0]
        per = sub[sub["class_name"] != "max_over_classes"]["region_fnr"]
        assert (per <= joint + 1e-9).all()
    # Triage sidecar: exists, residual risk nonincreasing in budget, and
    # budget zero matches the reported FNR.
    triage = pd.read_csv(synthetic_env / "experiments" / "shared_run_triage.csv")
    assert set(triage["method"]) == {"region_crc", "shared_crc"}
    one = triage[(triage["method"] == "region_crc")
                 & (triage["seed"] == triage["seed"].iloc[0])]
    curve = one.sort_values("budget")["residual_region_fnr"].to_numpy()
    assert (np.diff(curve) <= 1e-9).all()
    rc = frame[(frame["method"] == "region_crc")
               & (frame["seed"] == one["seed"].iloc[0])].iloc[0]
    # budget=0 residual uses NaN-as-zero convention over ALL test images,
    # so it is a scaled version of region_fnr (defined-image mean).
    b0 = curve[0]
    assert b0 <= rc["region_fnr"] + 1e-9


def test_stage5_size_weighted_loss(synthetic_env):
    frame = _run_stage5(synthetic_env, "sw_run", [
        "--scheme", "synthetic_half", "--max-seeds", "2",
        "--methods", "region_crc", "--loss", "size_weighted"])
    assert frame["feasible"].all()
    # The calibrated (size-weighted) risk respects the level on average.
    assert frame["controlled_risk"].mean() <= 0.2 + 0.1


def test_stage5_lac_global(synthetic_env):
    frame = _run_stage5(synthetic_env, "lac_run", [
        "--scheme", "synthetic_half", "--max-seeds", "2",
        "--methods", "lac_global", "region_crc"])
    lac = frame[frame["method"] == "lac_global"]
    assert len(lac) == 2  # 2 seeds x 1 class x 1 alpha x 1 rho
    assert lac["feasible"].all()
    # Global pixel-coverage threshold and region CRC share the mask family
    # but calibrate different losses; both produce valid records.
    assert lac["region_fnr"].notna().all()
    assert lac["marked_area_fraction"].between(0, 1).all()
