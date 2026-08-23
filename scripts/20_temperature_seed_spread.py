#!/usr/bin/env python
"""Stage 20: how far the fitted temperature moves when the calibration draw moves.

Stage 4b fits one scalar temperature on the seed-0 calibration list and then
reuses it for every seeded draw of the same scheme. For seed 0 the fitting set
and the test set are disjoint, but the other draws repartition the same pool,
so images that entered the fit reappear on the test side. The tempered
baseline therefore rests on a score function that is not independent of its
own test set.

The size of that dependence is bounded by how much the temperature itself
moves with the draw: a temperature that is effectively constant across draws
carries almost no information about which images it saw. This script measures
that directly. It refits the temperature on each seed's own calibration list
and reports the spread, without rebuilding any region table.

Pixels are sampled once per image with a fixed generator and reused for every
seed, so the only thing that varies between fits is which images the fit sees.

Writes ``results/experiments/temperature_seed_spread__<model>.json`` and prints
a summary line per model.

Example
-------
    python scripts/20_temperature_seed_spread.py
    python scripts/20_temperature_seed_spread.py --models segformer_b5_cityscapes
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

from record.cache import image_cache_path, is_complete
from record.gt import load_label_array, parse_dataset_key
from record.labelmaps import trainid_map_for
from record.models import load_model_registry
from record.paths import results_dir, splits_dir
from record.splits import load_scheme
from record.temperature import (TEMPERATURE_GRID, nll_grid_search,
                                strided_true_probs)

DEFAULT_MODELS = ["segformer_b2_cityscapes", "segformer_b5_cityscapes"]


def sample_pixels(model_key, dataset_key, image_ids, value_to_channel, stride,
                  max_pixels_per_image=4000, seed=0):
    """One fixed pixel sample per image, shared by every temperature fit."""
    rng = np.random.default_rng(seed)
    per_image = {}
    for image_id in tqdm(sorted(image_ids), desc="sampling", unit="img"):
        cached = np.load(image_cache_path(model_key, dataset_key, image_id))
        label = load_label_array(dataset_key, image_id)
        true_p, probs, valid = strided_true_probs(
            cached, label, value_to_channel, stride)
        if true_p.size == 0:
            continue
        idx = rng.choice(true_p.size, min(max_pixels_per_image, true_p.size),
                         replace=False)
        rr, cc = np.nonzero(valid)
        per_image[image_id] = (true_p[idx].astype(np.float32),
                               probs[:, rr[idx], cc[idx]].astype(np.float32))
    return per_image


def fit_on(per_image, image_ids):
    """Grid-search the temperature on the pooled pixels of these images."""
    present = [i for i in image_ids if i in per_image]
    if not present:
        return None, None
    return nll_grid_search(
        np.concatenate([per_image[i][0] for i in present]),
        np.concatenate([per_image[i][1] for i in present], axis=1))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--dataset", default="cityscapes_val")
    parser.add_argument("--scheme", default="cityscapes_val_half")
    parser.add_argument("--max-pixels-per-image", type=int, default=4000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_model_registry()
    stride = registry["cache"]["strided_probs_stride"]
    scheme = load_scheme(splits_dir() / f"{args.scheme}.json")
    seeds = sorted(scheme["seeds"], key=int)
    pool = sorted({i for e in scheme["seeds"].values() for i in e["calibration"]}
                  | {i for e in scheme["seeds"].values() for i in e["test"]})

    family, _ = parse_dataset_key(args.dataset)
    value_to_channel = trainid_map_for(family)
    out_dir = results_dir("experiments")

    for model_key in args.models:
        missing = [i for i in pool
                   if not is_complete(image_cache_path(model_key, args.dataset, i))]
        if missing:
            print(f"ERROR: {model_key}: {len(missing)} pooled images missing "
                  f"from the stage-3 cache")
            return 1

        print(f"\n=== {model_key}: {len(pool)} pooled images, "
              f"{len(seeds)} seeded draws ===")
        per_image = sample_pixels(model_key, args.dataset, pool,
                                  value_to_channel, stride,
                                  args.max_pixels_per_image)

        temps = {}
        for s in tqdm(seeds, desc="refitting", unit="seed"):
            t, _ = fit_on(per_image, scheme["seeds"][s]["calibration"])
            if t is not None:
                temps[s] = t
        values = np.array(list(temps.values()), dtype=float)

        payload = {
            "model_key": model_key,
            "dataset_key": args.dataset,
            "scheme": args.scheme,
            "n_pool_images": len(pool),
            "n_seeds": len(temps),
            "max_pixels_per_image": args.max_pixels_per_image,
            "grid_step": float(TEMPERATURE_GRID[1] - TEMPERATURE_GRID[0]),
            "temperature_seed0": temps.get("0"),
            "temperature_min": float(values.min()),
            "temperature_max": float(values.max()),
            "temperature_median": float(np.median(values)),
            "temperature_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "temperature_by_seed": temps,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        path = out_dir / f"temperature_seed_spread__{model_key}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"{model_key}: seed-0 T={payload['temperature_seed0']}, "
              f"across {payload['n_seeds']} draws T in "
              f"[{payload['temperature_min']}, {payload['temperature_max']}], "
              f"sd={payload['temperature_sd']:.4f} "
              f"(grid step {payload['grid_step']})")
        print(f"wrote {path}")

    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
