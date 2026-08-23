#!/usr/bin/env python
"""Stage 21: temperature scaling refitted on each calibration draw.

Stage 4b fits one scalar temperature on the seed-0 calibration list and reuses
it for every seeded draw of the same scheme. The draws repartition a single
pool, so on later draws the tempered score has already seen part of its own
test half. Stage 20 bounds how far the scalar travels; this stage removes the
dependence instead, by refitting the temperature on each draw's own
calibration list and recalibrating against tables built from that draw's
temperature.

Three properties keep the cost bounded.

The negative log-likelihood is additive over pixels, so the fits do not need
one cache pass per draw. A single pass records, for every image and every grid
temperature, the summed log-probability of the true class and the number of
contributing pixels. The NLL of any calibration list is then a ratio of sums
over that matrix, which is the pooled-pixel NLL stage 4b minimizes.

The temperature grid has a step of 0.05, so the refits collapse onto a handful
of distinct values. Region tables are rebuilt once per distinct value rather
than once per draw.

The tempered score is not a monotone reparametrization of the raw score, since
the normalizer varies from pixel to pixel, so the region tables do have to be
rebuilt for each distinct temperature. That rebuild is the expensive step and
the reason the first two properties matter.

Pixels are sampled per image from a generator seeded by the image identifier,
so the sample is a property of the image rather than of the traversal order and
is identical across draws. Only the set of images entering a fit varies.

Writes ``results/experiments/x6_temp_leakfree__<model>.csv`` in the schema of
the other experiment files, with a sidecar recording each draw's temperature.

Example
-------
    python scripts/21_temperature_leakfree.py \
        --model segformer_b2_cityscapes --dataset cityscapes_val \
        --scheme cityscapes_val_half
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from record.cache import image_cache_path, is_complete
from record.crc import crc_threshold, heuristic_threshold
from record.evaluation import evaluate_at_threshold, index_rows
from record.grid import LAMBDA_GRID
from record.gt import list_ids, load_label_array, parse_dataset_key
from record.labelmaps import trainid_map_for
from record.models import load_model_registry
from record.paths import results_dir, splits_dir
from record.splits import load_scheme
from record.temperature import TEMPERATURE_GRID, strided_true_probs

SCRIPTS = Path(__file__).resolve().parent
ALPHAS = (0.05, 0.10, 0.20)
RHOS = (0.1, 0.5)
MAX_PIXELS_PER_IMAGE = 4000


def load_stage(filename: str, alias: str):
    """Import a numbered stage script for reuse.

    Stage module names begin with a digit and cannot be imported by name. The
    alternative is to copy the table builder and the table loader into this
    file, which is the code most likely to drift away from the tables the
    reported results are actually built from.
    """
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_rng(image_id: str) -> np.random.Generator:
    """Generator keyed by the image identifier, independent of traversal order."""
    digest = hashlib.sha256(image_id.encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "big"))


def nll_statistics(model_key, dataset_key, ids, value_to_channel, stride):
    """Per-image, per-temperature sufficient statistics for the pooled NLL.

    Returns ``log_sums`` of shape ``(n_images, n_temperatures)`` and ``counts``
    of shape ``(n_images,)``. For a subset ``S`` of image rows the pooled NLL at
    grid position ``t`` is ``-log_sums[S, t].sum() / counts[S].sum()``.
    """
    eps = 1e-12
    log_sums = np.zeros((len(ids), TEMPERATURE_GRID.size), dtype=np.float64)
    counts = np.zeros(len(ids), dtype=np.int64)

    for row, image_id in enumerate(tqdm(ids, desc="nll statistics", unit="img")):
        cached = np.load(image_cache_path(model_key, dataset_key, image_id))
        true_p, probs, valid = strided_true_probs(
            cached, load_label_array(dataset_key, image_id),
            value_to_channel, stride)
        if true_p.size == 0:
            continue
        take = min(MAX_PIXELS_PER_IMAGE, true_p.size)
        idx = image_rng(image_id).choice(true_p.size, take, replace=False)
        rr, cc = np.nonzero(valid)
        true_s = np.clip(true_p[idx].astype(np.float64), eps, 1.0)
        full_s = np.clip(probs[:, rr[idx], cc[idx]].astype(np.float64), eps, 1.0)
        counts[row] = take
        for col, temp in enumerate(TEMPERATURE_GRID):
            z = np.power(full_s, 1.0 / temp).sum(axis=0)
            tempered = np.power(true_s, 1.0 / temp) / np.maximum(z, eps)
            log_sums[row, col] = float(np.log(np.clip(tempered, eps, 1.0)).sum())
    return log_sums, counts


def fit_from_statistics(log_sums, counts, rows):
    """Grid temperature minimizing the pooled NLL over the given image rows."""
    total = int(counts[rows].sum())
    if total == 0:
        raise ValueError("no valid pixels in the calibration list")
    nlls = -log_sums[rows].sum(axis=0) / total
    best = int(np.argmin(nlls))
    return float(TEMPERATURE_GRID[best]), float(nlls[best])


def temp_model_key(model_key: str, temperature: float) -> str:
    """Model key under which the tables for one temperature are stored."""
    return f"{model_key}_tempscaled_T{temperature:.2f}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scheme", required=True,
                        help="split scheme whose seeded draws are recalibrated")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--min-component-px", type=int, default=1)
    parser.add_argument("--stats-only", action="store_true",
                        help="write the per-draw temperatures and stop before "
                             "the table rebuild")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_model_registry()
    stride = registry["cache"]["strided_probs_stride"]
    family, _ = parse_dataset_key(args.dataset)
    value_to_channel = trainid_map_for(family)

    ids = list_ids(args.dataset)
    missing = [i for i in ids
               if not is_complete(image_cache_path(args.model, args.dataset, i))]
    if missing:
        print(f"ERROR: {len(missing)} images missing from the cache")
        return 1

    stats_path = results_dir("experiments") / f"x6_temp_nll_stats__{args.model}.npz"
    if stats_path.exists() and not args.force:
        blob = np.load(stats_path, allow_pickle=True)
        log_sums, counts, stat_ids = blob["log_sums"], blob["counts"], list(blob["ids"])
        print(f"reusing NLL statistics for {len(stat_ids)} images")
    else:
        log_sums, counts = nll_statistics(
            args.model, args.dataset, ids, value_to_channel, stride)
        stat_ids = ids
        np.savez_compressed(stats_path, log_sums=log_sums, counts=counts,
                            ids=np.array(ids, dtype=object), grid=TEMPERATURE_GRID)
        print(f"wrote {stats_path}")

    row_of = {image_id: row for row, image_id in enumerate(stat_ids)}
    scheme = load_scheme(splits_dir() / f"{args.scheme}.json")

    per_seed = {}
    for seed in range(args.seeds):
        key = str(seed)
        if key not in scheme["seeds"]:
            break
        cal_ids = scheme["seeds"][key]["calibration"]
        rows = np.array([row_of[i] for i in cal_ids], dtype=np.int64)
        temperature, nll = fit_from_statistics(log_sums, counts, rows)
        per_seed[key] = {"temperature": temperature, "nll": nll}

    temps = sorted({v["temperature"] for v in per_seed.values()})
    print(f"{len(per_seed)} draws -> {len(temps)} distinct temperatures: {temps}")

    meta = {
        "model_key": args.model, "dataset_key": args.dataset,
        "scheme": args.scheme, "n_seeds": len(per_seed),
        "distinct_temperatures": temps,
        "max_pixels_per_image": MAX_PIXELS_PER_IMAGE,
        "per_seed": per_seed,
        "temperature_grid": [float(TEMPERATURE_GRID[0]),
                             float(TEMPERATURE_GRID[-1]), 0.05],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = results_dir("experiments") / f"x6_temp_leakfree__{args.model}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"wrote {meta_path}")

    if args.stats_only:
        print("stats-only: stopping before the table rebuild")
        return 0

    stage_4b = load_stage("04b_build_baseline_tables.py", "stage_4b")
    for temperature in temps:
        out_key = temp_model_key(args.model, temperature)
        out_dir = results_dir(f"raw/{out_key}/{args.dataset}")
        if (out_dir / "components.parquet").exists() and not args.force:
            print(f"T={temperature}: tables exist, skipping")
            continue
        print(f"T={temperature}: building tempered region tables as {out_key}")
        stage_4b.build_tempered_tables(
            args.model, args.dataset, temperature, stride,
            args.min_component_px, out_key=out_key)

    stage_5 = load_stage("05_run_experiments.py", "stage_5")
    stores = {t: stage_5.TableStore(temp_model_key(args.model, t), [args.dataset])
              for t in temps}

    records = []
    for seed_key, info in tqdm(sorted(per_seed.items(), key=lambda kv: int(kv[0])),
                               desc="recalibrate", unit="draw"):
        store = stores[info["temperature"]]
        cal_ids = scheme["seeds"][seed_key]["calibration"]
        test_ids = scheme["seeds"][seed_key]["test"]
        cal_rows = index_rows(store.image_ids, cal_ids)
        test_rows = index_rows(store.image_ids, test_ids)
        for class_name in store.class_names:
            for rho in RHOS:
                view = store.class_view(class_name, rho)
                cal_def = stage_5.defined_rows(view["losses"], cal_rows)
                if cal_def.size == 0:
                    continue
                for alpha in ALPHAS:
                    for method, select in (
                            ("region_crc", crc_threshold),
                            ("heuristic", heuristic_threshold)):
                        selection = select(view["losses"][cal_def], alpha,
                                           LAMBDA_GRID)
                        metrics = evaluate_at_threshold(
                            selection, view["losses"][test_rows],
                            view["marked_area"][test_rows],
                            view["component_counts"][test_rows])
                        records.append({
                            "lam": metrics.lam,
                            "lam_index": int(selection.lam_index),
                            "feasible": metrics.feasible,
                            "region_fnr": metrics.region_fnr,
                            "marked_area_fraction": metrics.marked_area_fraction,
                            "n_test_images": metrics.n_test_images,
                            "n_test_components": metrics.n_test_components,
                            "n_cal_images": int(cal_def.size),
                            "temperature": info["temperature"],
                            "seed": int(seed_key), "class_name": class_name,
                            "alpha": alpha, "rho": rho, "method": method,
                            "model": args.model, "scheme": args.scheme,
                        })

    frame = pd.DataFrame(records)
    out_csv = results_dir("experiments") / f"x6_temp_leakfree__{args.model}.csv"
    frame.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(frame)} rows)")

    crc = frame[frame["method"] == "region_crc"]
    for (alpha, rho), cell in crc.groupby(["alpha", "rho"]):
        mean = cell["region_fnr"].mean()
        flag = "" if mean <= alpha else "   <-- ABOVE"
        print(f"alpha={alpha} rho={rho}: mean FNR {mean:.4f}"
              f"  area {cell['marked_area_fraction'].mean():.4f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
