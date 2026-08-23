#!/usr/bin/env python
"""Stage 5: calibration experiments over cached region statistics.

Consumes stage-4 tables and committed split schemes; produces one CSV of
metrics per invocation plus a ``.meta.json`` sidecar. No model or image is
touched: a full sweep over seeds, risk levels, and capture thresholds is
array arithmetic.

Methods
-------
- ``region_crc``   : CRC on region-miss loss (this work; also tier A when the
                     scheme's calibration key is ``target_calibration``)
- ``weighted_crc`` : importance-weighted CRC with embedding-based weights
                     (tier B; requires ``--embedding`` and a source scheme)
- ``pixel_crc``    : CRC controlling pixel FNR, evaluated at region level
- ``heuristic``    : uncorrected empirical threshold (no guarantee)
- ``argmax``       : plain argmax prediction (reference operating point)

Examples
--------
In-distribution (Cityscapes val halves):
    python scripts/05_run_experiments.py --name e1_cityscapes_indist \
        --model segformer_b2_cityscapes --scheme cityscapes_val_half \
        --cal-datasets cityscapes_val --test-datasets cityscapes_val \
        --methods region_crc pixel_crc heuristic argmax

Shift, tier A (25 labeled target images per condition):
    python scripts/05_run_experiments.py --name e2_acdc_fog_tierA25 \
        --model segformer_b2_cityscapes --scheme acdc_fog_targetcal25 \
        --cal-datasets acdc_fog_train acdc_fog_val \
        --test-datasets acdc_fog_train acdc_fog_val \
        --methods region_crc heuristic argmax

Shift, tier B (source calibration, embedding weights):
    python scripts/05_run_experiments.py --name e2_acdc_fog_tierB \
        --model segformer_b2_cityscapes --scheme cityscapes_val_half \
        --cal-datasets cityscapes_val \
        --test-datasets acdc_fog_train acdc_fog_val \
        --methods region_crc weighted_crc --embedding dinov2_vitb14
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from record.cache import embeddings_path
from record.crc import ThresholdSelection, crc_threshold, heuristic_threshold, weighted_crc_threshold
from record.evaluation import evaluate_at_threshold, index_rows, stratified_region_fnr
from record.grid import FP_SUBGRID_INDICES, LAMBDA_GRID
from record.losses import enforce_nonincreasing, image_loss_curves
from record.monitor import ks_drift_check
from record.paths import results_dir, splits_dir
from record.splits import load_scheme
from record.weights import (knn_density_ratio, logistic_cv_density_ratio,
                            logistic_density_ratio)

WEIGHT_CLIP_MIN = 0.05
WEIGHT_CLIP_MAX_DEFAULT = 20.0


class TableStore:
    """Concatenated stage-4 tables for one model over one or more datasets."""

    def __init__(self, model_key: str, dataset_keys: list[str]):
        frames, curve_blocks = [], []
        ids: list[str] = []
        area_blocks, count_blocks, argmax_area_blocks, fp_blocks = [], [], [], []
        class_names: list[str] | None = None
        row_offset, curve_offset = 0, 0

        for key in dataset_keys:
            table_dir = results_dir(f"raw/{model_key}/{key}")
            frame = pd.read_parquet(table_dir / "components.parquet")
            curves = np.load(table_dir / "coverage_curves.npy").astype(np.float32)
            stats = np.load(table_dir / "image_stats.npz", allow_pickle=False)

            names = [str(c) for c in stats["class_names"]]
            if class_names is None:
                class_names = names
            elif class_names != names:
                raise ValueError(f"class name mismatch across datasets: {class_names} vs {names}")

            frame["image_row"] += row_offset
            frame["curve_row"] += curve_offset
            frames.append(frame)
            curve_blocks.append(curves)
            ids.extend(str(i) for i in stats["image_ids"])
            area_blocks.append(stats["marked_area"].astype(np.float32))
            argmax_area_blocks.append(stats["argmax_marked_area"].astype(np.float32))
            count_blocks.append(stats["component_counts"])
            fp_blocks.append(stats["fp_component_counts"])
            row_offset = len(ids)
            curve_offset += curves.shape[0]

        lac_blocks = []
        for key in dataset_keys:
            lac_path = results_dir(f"raw/{model_key}/{key}") / "lac_miscoverage.npy"
            lac_blocks.append(np.load(lac_path) if lac_path.exists() else None)
        self.lac = (np.vstack([b.astype(np.float32) for b in lac_blocks])
                    if all(b is not None for b in lac_blocks) else None)

        assert class_names is not None
        self.class_names = class_names
        self.components = pd.concat(frames, ignore_index=True)
        self.curves = np.vstack(curve_blocks)
        self.image_ids = ids
        self.marked_area = np.concatenate(area_blocks)          # (N, C, G)
        self.argmax_marked_area = np.concatenate(argmax_area_blocks)
        self.component_counts = np.concatenate(count_blocks)    # (N, C)
        self.fp_counts = np.concatenate(fp_blocks)              # (N, C, subgrid)
        self._view_cache: dict = {}

    def class_index(self, class_name: str) -> int:
        return self.class_names.index(class_name)

    def class_view(self, class_name: str, rho: float):
        """Per-image loss curves and per-component data for one class.

        Views are seed-independent and cached: full sweeps evaluate the same
        (class, rho) pair for every seed replicate.
        """
        key = (class_name, rho)
        if key in self._view_cache:
            return self._view_cache[key]
        sel = self.components[self.components["class_name"] == class_name]
        curves = self.curves[sel["curve_row"].to_numpy()]
        img_idx = sel["image_row"].to_numpy()
        n = len(self.image_ids)
        raw_losses, _ = image_loss_curves(curves, img_idx, n, rho=rho)
        losses = enforce_nonincreasing(raw_losses)
        # Monotonicity is exact by construction; enforcement must be a no-op.
        # A materially different result indicates a defect upstream, so fail
        # loudly instead of calibrating on corrupted curves.
        drift = float(np.nanmax(np.abs(losses - raw_losses))) if losses.size else 0.0
        if drift > 1e-3:
            raise RuntimeError(
                f"loss-curve monotonicity enforcement changed values by {drift:.4f}; "
                "refusing to calibrate on inconsistent curves")

        # Pixel-FNR curves (size-weighted coverage complement), for pixel_crc.
        sizes = sel["size_px"].to_numpy().astype(np.float64)
        num = np.zeros((n, LAMBDA_GRID.size))
        den = np.zeros(n)
        np.add.at(num, img_idx, curves * sizes[:, None])
        np.add.at(den, img_idx, sizes)
        pixel_fnr = np.full((n, LAMBDA_GRID.size), np.nan)
        defined = den > 0
        pixel_fnr[defined] = 1.0 - num[defined] / den[defined, None]

        # Size-weighted region-miss curves: components weighted by pixel area.
        ind = (curves < rho).astype(np.float64)
        num_sw = np.zeros((n, LAMBDA_GRID.size))
        np.add.at(num_sw, img_idx, ind * sizes[:, None])
        losses_sw = np.full((n, LAMBDA_GRID.size), np.nan)
        losses_sw[defined] = num_sw[defined] / den[defined, None]

        c = self.class_index(class_name)
        view = {
            "losses": losses,
            "losses_sw": losses_sw,
            "pixel_fnr": pixel_fnr,
            "curves": curves,
            "strata": sel["size_stratum"].to_numpy(),
            "component_image_rows": img_idx,
            "argmax_component_coverage": sel["argmax_coverage"].to_numpy(),
            "marked_area": self.marked_area[:, c, :],
            "argmax_marked_area": self.argmax_marked_area[:, c],
            "component_counts": self.component_counts[:, c],
            "fp_counts": self.fp_counts[:, c, :],
        }
        self._view_cache[key] = view
        return view


def load_embeddings(embedding_key: str, dataset_keys: list[str]) -> tuple[list[str], np.ndarray]:
    ids, blocks = [], []
    for key in dataset_keys:
        data = np.load(embeddings_path(embedding_key, key), allow_pickle=False)
        ids.extend(str(i) for i in data["image_ids"])
        blocks.append(data["embeddings"])
    return ids, np.vstack(blocks)


def defined_rows(matrix: np.ndarray, rows: np.ndarray) -> np.ndarray:
    sub = matrix[rows]
    return rows[~np.isnan(sub[:, 0])]


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_cluster_map(path: str | None) -> dict[str, str]:
    """Read an image-to-cluster map for the clustered bootstrap.

    The file is a CSV whose first column identifies the image and whose
    ``scene`` (or ``cluster``) column names the group it belongs to. Images
    absent from the map fall back to being their own cluster, so a partial map
    degrades to the image-level bootstrap rather than failing.
    """
    if not path:
        return {}
    frame = pd.read_csv(path)
    id_col = next((c for c in ("patch_id", "image_id", "id") if c in frame.columns),
                  frame.columns[0])
    cl_col = next((c for c in ("scene", "cluster", "group") if c in frame.columns), None)
    if cl_col is None:
        raise ValueError(f"{path}: no 'scene'/'cluster'/'group' column; "
                         f"found {list(frame.columns)}")
    mapping = {str(k): str(v) for k, v in zip(frame[id_col], frame[cl_col])}
    print(f"bootstrap clusters: {len(mapping)} images -> "
          f"{frame[cl_col].nunique()} clusters from {path}")
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="output file stem")
    parser.add_argument("--model", required=True)
    parser.add_argument("--scheme", required=True, help="split scheme name (without .json)")
    parser.add_argument("--cal-datasets", nargs="+", required=True)
    parser.add_argument("--test-datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True,
                        choices=["region_crc", "shared_crc", "lac_global",
                                 "weighted_crc", "pixel_crc", "heuristic",
                                 "argmax"])
    parser.add_argument("--loss", default="region", choices=["region", "size_weighted"],
                        help="calibration loss for region_crc: per-region miss "
                             "fraction or size-weighted miss fraction")
    parser.add_argument("--triage", action="store_true",
                        help="write per-budget triage curves for region_crc and "
                             "shared_crc records to <name>_triage.csv")
    parser.add_argument("--triage-rankings", nargs="+", default=["area"],
                        choices=["area", "random", "oracle"],
                        help="priority scores to compare in the triage curves: "
                             "marked area (the deployable score), a random "
                             "permutation (the floor any ranking must beat), "
                             "and the loss itself (the unreachable ceiling)")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.1, 0.5])
    parser.add_argument("--embedding", default=None, help="embedding key for weighted_crc")
    parser.add_argument("--weight-source-holdout", type=float, default=0.0,
                        help="fraction of the calibration images reserved for "
                             "fitting the density ratio and excluded from the "
                             "CRC calibration set (weighted_crc only). Together "
                             "with --weight-holdout this makes the weight "
                             "function independent of both the calibration "
                             "points and the test point, which is what "
                             "Proposition 1 assumes")
    parser.add_argument("--weight-holdout", type=float, default=0.0,
                        help="fraction of the target images reserved for "
                             "fitting the density ratio and excluded from the "
                             "test set (weighted_crc only). 0 keeps the "
                             "transductive protocol, in which the weights are "
                             "fitted on the images the threshold is applied to")
    parser.add_argument("--weight-estimator", default="logistic",
                        choices=["logistic", "logistic_cv", "knn"],
                        help="density-ratio estimator for weighted_crc")
    parser.add_argument("--weight-clip-min", type=float, default=WEIGHT_CLIP_MIN,
                        help="lower clip for importance weights; the vacuity "
                             "condition depends on the ratio of the two clips, "
                             "not on the ceiling alone")
    parser.add_argument("--weight-clip-max", type=float, default=WEIGHT_CLIP_MAX_DEFAULT,
                        help="upper clip for importance weights; also the "
                             "conservative test weight in weighted_crc")
    parser.add_argument("--cal-split", default=None,
                        help="scheme entry key providing calibration ids "
                             "(default: target_calibration if present, else calibration)")
    parser.add_argument("--test-split", default="test",
                        help="scheme entry key providing test ids")
    parser.add_argument("--bootstrap-clusters", default=None, metavar="CSV",
                        help="CSV mapping each image identifier to a cluster "
                             "label; the bootstrap then resamples clusters "
                             "instead of images. On MARIDA pass "
                             "splits/marida_meta.csv, whose 'scene' column "
                             "groups the patches of one acquisition. Without "
                             "it every image is its own cluster.")
    parser.add_argument("--bootstrap-ci", type=int, default=0,
                        help="number of image-level bootstrap resamples for a "
                             "95%% interval on the region FNR; the sampling "
                             "unit is the test image, which is the unit the "
                             "reported risk averages over. 0 disables it")
    parser.add_argument("--max-seeds", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheme = load_scheme(splits_dir() / f"{args.scheme}.json")
    bootstrap_clusters = load_cluster_map(args.bootstrap_clusters)
    cal_store = TableStore(args.model, args.cal_datasets)
    test_store = (cal_store if args.cal_datasets == args.test_datasets
                  else TableStore(args.model, args.test_datasets))

    seeds = list(scheme["seeds"].items())
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]

    emb_ids: list[str] | None = None
    emb: np.ndarray | None = None
    if "weighted_crc" in args.methods:
        if not args.embedding:
            print("ERROR: weighted_crc requires --embedding")
            return 1
        keys = list(dict.fromkeys(args.cal_datasets + args.test_datasets))
        emb_ids, emb = load_embeddings(args.embedding, keys)

    loss_key = "losses" if args.loss == "region" else "losses_sw"
    budgets = np.round(np.arange(0.0, 0.51, 0.05), 2)
    records: list[dict] = []
    triage_records: list[dict] = []
    weights_cache: dict = {}
    clip = (args.weight_clip_min, args.weight_clip_max)
    for seed_key, entry in seeds:
        cal_key = args.cal_split or (
            "target_calibration" if "target_calibration" in entry else "calibration")
        if cal_key not in entry or (
                args.cal_datasets == args.test_datasets and args.test_split not in entry):
            print(f"ERROR: scheme '{args.scheme}' entry has keys {sorted(entry)}; "
                  f"requested calibration '{cal_key}' / test '{args.test_split}'")
            return 1
        cal_rows = index_rows(cal_store.image_ids, entry[cal_key])
        test_rows = index_rows(test_store.image_ids, entry[args.test_split]) \
            if args.cal_datasets == args.test_datasets \
            else np.arange(len(test_store.image_ids))

        # Optional non-transductive protocol: hold out part of the target set
        # to fit the density ratio, and evaluate on the rest, so that the
        # weight function does not depend on the images it is applied to.
        weight_pool_rows = test_rows
        if args.weight_holdout > 0:
            if not 0 < args.weight_holdout < 1:
                print("ERROR: --weight-holdout must be in (0, 1)")
                return 1
            rng = np.random.default_rng(
                int.from_bytes(hashlib.sha256(
                    f"weight-holdout|{seed_key}".encode()).digest()[:8], "big"))
            perm = rng.permutation(len(test_rows))
            k = int(round(args.weight_holdout * len(test_rows)))
            if k < 2 or len(test_rows) - k < 2:
                print(f"ERROR: holdout leaves too few images "
                      f"({k} / {len(test_rows) - k})")
                return 1
            weight_pool_rows = test_rows[perm[:k]]
            test_rows = test_rows[perm[k:]]

        # Source-side counterpart of the holdout above: part of the
        # calibration set fits the density ratio and is then excluded from the
        # risk calibration, so the weight function is a function of neither the
        # calibration points nor the test point.
        source_pool_ids = None
        if args.weight_source_holdout > 0:
            if not 0 < args.weight_source_holdout < 1:
                print("ERROR: --weight-source-holdout must be in (0, 1)")
                return 1
            rng = np.random.default_rng(
                int.from_bytes(hashlib.sha256(
                    f"source-holdout|{seed_key}".encode()).digest()[:8], "big"))
            perm = rng.permutation(len(cal_rows))
            k = int(round(args.weight_source_holdout * len(cal_rows)))
            if k < 2 or len(cal_rows) - k < 2:
                print("ERROR: source holdout leaves too few images")
                return 1
            source_pool_ids = [cal_store.image_ids[r] for r in cal_rows[perm[:k]]]
            cal_rows = cal_rows[perm[k:]]

        for class_name in cal_store.class_names:
            for rho in args.rhos:
                cal_view = cal_store.class_view(class_name, rho)
                test_view = (cal_view if test_store is cal_store
                             else test_store.class_view(class_name, rho))
                cal_def = defined_rows(cal_view["losses"], cal_rows)

                for alpha in args.alphas:
                    for method in args.methods:
                        if method in ("shared_crc", "lac_global"):
                            continue  # handled jointly across classes below
                        rec = run_method(
                            method, alpha, rho, cal_view, test_view,
                            cal_def, test_rows, cal_store, test_store,
                            emb_ids, emb, cal_rows, args.weight_estimator,
                            weights_cache, (seed_key, class_name), clip,
                            loss_key, weight_pool_rows, source_pool_ids,
                            args.bootstrap_ci, bootstrap_clusters)
                        if rec is None:
                            continue
                        rec.update({
                            "seed": seed_key, "class_name": class_name,
                            "alpha": alpha, "rho": rho, "method": method,
                            "model": args.model, "scheme": args.scheme,
                            "n_cal_images": len(cal_def),
                        })
                        records.append(rec)
                        if args.triage and method == "region_crc" \
                                and rec["feasible"] and rec["lam_index"] >= 0:
                            collect_triage(
                                triage_records, test_view, test_rows,
                                rec, budgets, tuple(args.triage_rankings))

        if "shared_crc" in args.methods:
            run_shared(records, triage_records, cal_store, test_store,
                       cal_rows, test_rows, seed_key, args, loss_key, budgets)
        if "lac_global" in args.methods:
            run_lac(records, cal_store, test_store, cal_rows, test_rows,
                    seed_key, args)

    frame = pd.DataFrame(records)
    out_dir = results_dir("experiments")
    csv_path = out_dir / f"{args.name}.csv"
    frame.to_csv(csv_path, index=False)
    if triage_records:
        triage_path = out_dir / f"{args.name}_triage.csv"
        pd.DataFrame(triage_records).to_csv(triage_path, index=False)
        print(f"wrote {triage_path} ({len(triage_records)} records)")
    meta = {
        "name": args.name,
        "argv": vars(args),
        "scheme_file": f"{args.scheme}.json",
        "n_records": len(frame),
        "git_revision": git_revision(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / f"{args.name}.meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {csv_path} ({len(frame)} records)")
    print("RESULT: PASS")
    return 0


def collect_triage(triage_records, test_view, test_rows, rec, budgets,
                   rankings=("area",)):
    """Append residual-risk-vs-budget rows for one calibrated record.

    ``rankings`` selects the priority score. ``area`` is the deployable one
    (a by-product of thresholding). ``random`` is the floor: under the oracle
    review model any ranking already removes a fraction beta of the risk, so a
    useful score has to beat this curve. ``oracle`` ranks by the realized loss
    and is the unreachable ceiling. Reporting all three is what turns "the
    score is effective" from an assertion into a measurement.
    """
    from record.evaluation import triage_curve

    col = rec["lam_index"]
    losses_at_lam = test_view["losses"][test_rows][:, col]
    area = test_view["marked_area"][test_rows][:, col]
    for ranking in rankings:
        if ranking == "area":
            scores = area
        elif ranking == "oracle":
            scores = np.nan_to_num(losses_at_lam, nan=-1.0)
        else:
            # A deterministic seed: Python's hash() is salted per process,
            # so a hash-derived seed would not reproduce across runs.
            key = f'{rec["seed"]}|{rec["class_name"]}|{ranking}'.encode()
            rng = np.random.default_rng(
                int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
            scores = rng.permutation(len(area)).astype(float)
        residual = triage_curve(scores, losses_at_lam, budgets)
        for b, r in zip(budgets, residual):
            triage_records.append({
                "seed": rec["seed"], "class_name": rec["class_name"],
                "alpha": rec["alpha"], "rho": rec["rho"],
                "method": rec["method"], "ranking": ranking,
                "budget": float(b),
                "residual_region_fnr": float(r), "lam": rec["lam"],
            })


def run_shared(records, triage_records, cal_store, test_store, cal_rows,
               test_rows, seed_key, args, loss_key, budgets):
    """Single shared threshold controlling the max per-class region loss."""
    for rho in args.rhos:
        views = {c: cal_store.class_view(c, rho) for c in cal_store.class_names}
        test_views = views if test_store is cal_store else {
            c: test_store.class_view(c, rho) for c in test_store.class_names}
        stacked = np.stack([views[c][loss_key] for c in cal_store.class_names])
        with np.errstate(all="ignore"):
            combined = np.nanmax(stacked, axis=0)
        cal_combined = combined[cal_rows]
        cal_def = cal_rows[~np.isnan(cal_combined[:, 0])]
        test_stacked = np.stack(
            [test_views[c]["losses"] for c in test_store.class_names])
        with np.errstate(all="ignore"):
            test_combined = np.nanmax(test_stacked, axis=0)

        for alpha in args.alphas:
            base = {"seed": seed_key, "alpha": alpha, "rho": rho,
                    "method": "shared_crc", "model": args.model,
                    "scheme": args.scheme, "n_cal_images": len(cal_def)}
            if len(cal_def) == 0:
                records.append(dict(UNDEFINED_RECORD, n_test_images=int(len(test_rows)),
                                    class_name="max_over_classes", **base))
                continue
            selection = crc_threshold(combined[cal_def], alpha, LAMBDA_GRID)
            col = selection.lam_index
            joint = test_combined[test_rows][:, col]
            records.append(dict(
                UNDEFINED_RECORD, lam=selection.lam, lam_index=int(col),
                feasible=selection.feasible,
                controlled_risk=float(np.nanmean(joint)),
                region_fnr=float(np.nanmean(joint)),
                n_test_images=int(np.sum(~np.isnan(joint))),
                class_name="max_over_classes", **base))
            for class_name in test_store.class_names:
                tv = test_views[class_name]
                per = tv["losses"][test_rows][:, col]
                area = tv["marked_area"][test_rows][:, col]
                rec = dict(
                    UNDEFINED_RECORD, lam=selection.lam, lam_index=int(col),
                    feasible=selection.feasible,
                    controlled_risk=float(np.nanmean(per)),
                    region_fnr=float(np.nanmean(per)),
                    marked_area_fraction=float(np.mean(area)),
                    n_test_images=int(np.sum(~np.isnan(per))),
                    class_name=class_name, **base)
                records.append(rec)
                if args.triage and selection.feasible:
                    collect_triage(triage_records, tv, test_rows, rec, budgets,
                                   tuple(args.triage_rankings))


def run_lac(records, cal_store, test_store, cal_rows, test_rows, seed_key,
            args):
    """LAC baseline: one global threshold calibrated on all-class pixel
    miscoverage (stage-4b curves), evaluated on region-level metrics."""
    if cal_store.lac is None:
        raise FileNotFoundError(
            "lac_miscoverage.npy missing; run scripts/04b_build_baseline_tables.py")
    cal_curves = cal_store.lac[cal_rows]
    cal_def_mask = ~np.isnan(cal_curves[:, 0])
    for alpha in args.alphas:
        selection = crc_threshold(
            np.nan_to_num(cal_curves[cal_def_mask], nan=0.0), alpha, LAMBDA_GRID)
        col = selection.lam_index
        for rho in args.rhos:
            for class_name in test_store.class_names:
                tv = test_store.class_view(class_name, rho)
                per = tv["losses"][test_rows][:, col]
                area = tv["marked_area"][test_rows][:, col]
                records.append(dict(
                    UNDEFINED_RECORD, lam=selection.lam, lam_index=int(col),
                    feasible=selection.feasible,
                    region_fnr=float(np.nanmean(per)),
                    marked_area_fraction=float(np.mean(area)),
                    n_test_images=int(np.sum(~np.isnan(per))),
                    seed=seed_key, class_name=class_name, alpha=alpha,
                    rho=rho, method="lac_global", model=args.model,
                    scheme=args.scheme,
                    n_cal_images=int(cal_def_mask.sum())))


UNDEFINED_RECORD = {
    "lam": np.nan, "lam_index": -1, "feasible": False,
    "controlled_risk": np.nan, "region_fnr": np.nan,
    "marked_area_fraction": np.nan, "fp_components_per_image": np.nan,
    "monitor_ks": np.nan, "monitor_flag": False,
    "fnr_stratum0": np.nan, "fnr_stratum1": np.nan, "fnr_stratum2": np.nan,
    "weight_ess": np.nan, "weight_p_test": np.nan,
}


def run_method(method, alpha, rho, cal_view, test_view, cal_def, test_rows,
               cal_store, test_store, emb_ids, emb, cal_rows,
               weight_estimator="logistic", weights_cache=None,
               weight_key=(), clip=(WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX_DEFAULT),
               loss_key="losses", weight_pool_rows=None,
               source_pool_ids=None, bootstrap_ci=0, clusters=None) -> dict | None:
    test_losses = test_view["losses"][test_rows]
    test_area = test_view["marked_area"][test_rows]
    test_counts = test_view["component_counts"][test_rows]

    # Degenerate draws are recorded, not skipped: with small target-calibration
    # sets a rare class may be absent from every calibration image, and the
    # frequency of such draws is itself a reportable finding.
    if method != "argmax" and len(cal_def) == 0:
        return dict(UNDEFINED_RECORD, n_test_images=int(len(test_rows)))
    if np.all(np.isnan(test_losses[:, 0])):
        return dict(UNDEFINED_RECORD, n_test_images=0)

    if method == "argmax":
        comp_rows = test_view["component_image_rows"]
        comp_in_test = np.isin(comp_rows, test_rows)
        cov = test_view["argmax_component_coverage"][comp_in_test]
        if cov.size == 0:
            return None
        # Every CRC row of the baseline table reports the mean over images of
        # the per-image missed-region fraction (eq. 3).  The argmax row must
        # use the same estimand or the comparison is not like-for-like; the
        # rate is therefore image-averaged here, with the component-weighted
        # rate kept alongside as a diagnostic.
        miss = cov < rho
        rows_of_comp = comp_rows[comp_in_test]
        per_image = [
            miss[rows_of_comp == r].mean()
            for r in np.unique(rows_of_comp)
        ]
        return {
            "lam": np.nan, "lam_index": -1, "feasible": True,
            "controlled_risk": np.nan,
            "region_fnr": float(np.mean(per_image)),
            "region_fnr_component_avg": float(miss.mean()),
            "marked_area_fraction": float(test_view["argmax_marked_area"][test_rows].mean()),
            "fp_components_per_image": np.nan,
            "n_test_images": int(len(test_rows)),
            "monitor_ks": np.nan, "monitor_flag": False,
            "fnr_stratum0": np.nan, "fnr_stratum1": np.nan, "fnr_stratum2": np.nan,
            "weight_ess": np.nan, "weight_p_test": np.nan,
        }

    if method == "pixel_crc":
        cal_curves = np.nan_to_num(cal_view["pixel_fnr"][cal_def], nan=0.0)
        selection = crc_threshold(cal_curves, alpha, LAMBDA_GRID)
    elif method == "heuristic":
        selection = heuristic_threshold(cal_view["losses"][cal_def], alpha, LAMBDA_GRID)
    elif method == "region_crc":
        selection = crc_threshold(cal_view[loss_key][cal_def], alpha, LAMBDA_GRID)
    weight_diag = {"weight_ess": np.nan, "weight_p_test": np.nan}
    if method == "weighted_crc":
        assert emb is not None and emb_ids is not None
        # Importance weights depend only on (seed, class, estimator, clip)
        # through the defined calibration rows; cache them across the
        # alpha x rho sweep.
        pool_rows = test_rows if weight_pool_rows is None else weight_pool_rows
        cache_key = (*weight_key, weight_estimator, clip, int(len(pool_rows)),
                     0 if source_pool_ids is None else len(source_pool_ids))
        weights = None if weights_cache is None else weights_cache.get(cache_key)
        if weights is None:
            cal_ids = ([cal_store.image_ids[r] for r in cal_def]
                       if source_pool_ids is None else source_pool_ids)
            tgt_ids = [test_store.image_ids[r] for r in pool_rows]
            cal_emb = emb[index_rows(emb_ids, cal_ids)]
            tgt_emb = emb[index_rows(emb_ids, tgt_ids)]
            estimator = {
                "logistic": logistic_density_ratio,
                "logistic_cv": logistic_cv_density_ratio,
                "knn": knn_density_ratio,
            }[weight_estimator]
            if source_pool_ids is None:
                weights = estimator(cal_emb, tgt_emb, clip=clip)
            elif weight_estimator != "logistic":
                raise ValueError(
                    "--weight-source-holdout is implemented for the logistic "
                    f"estimator only, got '{weight_estimator}'")
            else:
                # Fit on the disjoint source pool, then evaluate the fitted
                # ratio at the calibration points the risk is computed on.
                eval_ids = [cal_store.image_ids[r] for r in cal_def]
                eval_emb = emb[index_rows(emb_ids, eval_ids)]
                weights = estimator(cal_emb, tgt_emb, clip=clip,
                                    eval_emb=eval_emb)
            if weights_cache is not None:
                weights_cache[cache_key] = weights
        w_sum = float(weights.sum())
        weight_diag = {
            "weight_ess": w_sum ** 2 / float((weights ** 2).sum()),
            "weight_p_test": clip[1] / (w_sum + clip[1]),
        }
        selection = weighted_crc_threshold(
            cal_view["losses"][cal_def], weights, test_weight=clip[1],
            alpha=alpha, lambda_grid=LAMBDA_GRID)
    elif method not in ("pixel_crc", "heuristic", "region_crc"):
        raise ValueError(method)

    metrics = evaluate_at_threshold(selection, test_losses, test_area, test_counts)

    # Bootstrap interval on the reported risk. The estimand is the mean over
    # test images of the within-image miss fraction, so the image is the unit
    # the risk averages over. Resampling images keeps the components of an
    # image together but still treats images as independent. On MARIDA they are
    # not: the patches of one acquisition scene share illumination, sea state
    # and annotator, and a held-out tile can contain as few as three scenes. A
    # cluster map therefore promotes the sampling unit from the image to the
    # scene, which widens the interval to the precision the design actually
    # carries. Without a map every image is its own cluster and the two agree.
    boot_lo = boot_hi = np.nan
    n_clusters = np.nan
    col = np.asarray(test_losses[:, selection.lam_index], dtype=float)
    defined = ~np.isnan(col)
    col = col[defined]
    if bootstrap_ci > 0 and col.size >= 2:
        brng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(
                f"boot|{weight_key}|{method}|{alpha}|{rho}".encode()
            ).digest()[:8], "big"))
        if clusters:
            ids = [test_store.image_ids[r] for r in np.asarray(test_rows)[defined]]
            labels = [clusters.get(i, i) for i in ids]
            groups, order = {}, []
            for pos, lab in enumerate(labels):
                if lab not in groups:
                    groups[lab] = []
                    order.append(lab)
                groups[lab].append(pos)
            members = [np.asarray(groups[lab]) for lab in order]
            n_clusters = len(members)
        else:
            members = None
            n_clusters = int(col.size)
        if members is not None and n_clusters >= 2:
            draws = brng.integers(0, n_clusters, size=(int(bootstrap_ci), n_clusters))
            means = np.empty(int(bootstrap_ci))
            for b in range(int(bootstrap_ci)):
                pooled = np.concatenate([members[j] for j in draws[b]])
                means[b] = col[pooled].mean()
        else:
            idx = brng.integers(0, col.size, size=(int(bootstrap_ci), col.size))
            means = col[idx].mean(axis=1)
        boot_lo = float(np.quantile(means, 0.025))
        boot_hi = float(np.quantile(means, 0.975))

    # False-positive component count at the nearest subgrid threshold.
    sub_pos = int(np.argmin(np.abs(FP_SUBGRID_INDICES - selection.lam_index)))
    fp_per_image = float(test_view["fp_counts"][test_rows][:, sub_pos].mean())

    comp_in_test = np.isin(test_view["component_image_rows"], test_rows)
    strata_fnr = stratified_region_fnr(
        test_view["curves"][comp_in_test],
        test_view["strata"][comp_in_test],
        selection.lam_index, rho)

    # Component-level tally at the selected threshold. The controlled risk is
    # the mean over images of the within-image missed fraction, which weights
    # images equally; the tally below weights components equally and is the
    # count a manual audit of the test set would produce. The two differ
    # whenever images carry unequal component counts, so the count is recorded
    # rather than inferred from the rate.
    comp_cov = test_view["curves"][comp_in_test][:, selection.lam_index]
    comp_missed = int(np.count_nonzero(comp_cov < rho))
    comp_total = int(comp_cov.size)

    drift = ks_drift_check(
        cal_view["marked_area"][cal_rows][:, selection.lam_index],
        test_area[:, selection.lam_index],
    ) if len(cal_rows) >= 2 and len(test_rows) >= 2 else None

    controlled = test_view[loss_key][test_rows][:, selection.lam_index] \
        if method == "region_crc" else test_losses[:, selection.lam_index]
    return {
        "fnr_boot_lo": boot_lo, "fnr_boot_hi": boot_hi,
        "n_test_clusters": n_clusters,
        "lam": metrics.lam, "lam_index": int(selection.lam_index),
        "feasible": metrics.feasible,
        "controlled_risk": float(np.nanmean(controlled)),
        "region_fnr": metrics.region_fnr,
        "marked_area_fraction": metrics.marked_area_fraction,
        "fp_components_per_image": fp_per_image,
        "n_test_images": metrics.n_test_images,
        "n_test_components": metrics.n_test_components,
        "n_missed_components": comp_missed,
        "region_fnr_component_avg": (comp_missed / comp_total
                                     if comp_total else np.nan),
        "monitor_ks": drift.ks_statistic if drift else np.nan,
        "monitor_flag": drift.flagged if drift else False,
        "fnr_stratum0": strata_fnr[0], "fnr_stratum1": strata_fnr[1],
        "fnr_stratum2": strata_fnr[2],
        **weight_diag,
    }


if __name__ == "__main__":
    raise SystemExit(main())
