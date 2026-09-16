#!/usr/bin/env python
"""Stage 22: marked area of the union of the per-class masks.

Marked area is reported per class and then averaged over the critical classes.
An operator who monitors several classes at once pays for the union of their
masks, not for the mean, and the union is not recoverable from the per-class
numbers: it depends on how far the class masks overlap. Two bounds are, and
they are computed here alongside the exact figure so the gap between them can
be read directly.

    max_c area_c  <=  area of the union  <=  min(1, sum_c area_c)

The exact union needs the class probability maps at the selected thresholds, so
it costs one cache read per test image per configuration. The thresholds
themselves are not recomputed: they are read from the experiment CSV that
already records the selected lambda for every class, level, capture level and
draw, so this stage measures the deployed operating point rather than a
reconstruction of it.

Writes ``results/experiments/x7_union_area__<model>.csv`` with one row per
(seed, level, capture level), and prints a summary against the class mean.

Example
-------
    python scripts/22_union_marked_area.py \
        --model segformer_b2_cityscapes --dataset cityscapes_val \
        --scheme cityscapes_val_half --experiment e1_indist__segformer_b2_cityscapes \
        --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from record.cache import image_cache_path, is_complete
from record.gt import parse_dataset_key
from record.labelmaps import critical_classes_for
from record.paths import results_dir, splits_dir
from record.splits import load_scheme
from record.provenance import git_revision


def union_area_for_image(model_key, dataset_key, image_id, thresholds,
                         channel_of):
    """Fraction of pixels marked by at least one class mask.

    ``thresholds`` maps class name to the selected lambda; a pixel is marked
    for class c when its probability reaches ``1 - lambda_c``, which is the
    mask of Equation (2).
    """
    cached = np.load(image_cache_path(model_key, dataset_key, image_id))
    channels = list(cached["critical_channels"])
    union = None
    per_class = {}
    for class_name, lam in thresholds.items():
        pos = channels.index(channel_of[class_name])
        prob = cached["critical_probs"][pos].astype(np.float32)
        mask = prob >= (1.0 - lam)
        per_class[class_name] = float(mask.mean())
        union = mask if union is None else (union | mask)
    return float(union.mean()) if union is not None else 0.0, per_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--experiment", required=True,
                        help="experiment name whose selected thresholds are used")
    parser.add_argument("--method", default="region_crc")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="draws to evaluate; the exact union costs one "
                             "cache read per test image per draw")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.05, 0.10, 0.20])
    parser.add_argument("--rho", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    family, _ = parse_dataset_key(args.dataset)
    # {class_name: (ground-truth value, model channel)}; the channel is what
    # indexes the cached critical-probability stack.
    critical_map = critical_classes_for(family)
    critical = list(critical_map)
    channel_of = {name: channel for name, (_, channel) in critical_map.items()}

    csv_path = results_dir("experiments") / f"{args.experiment}.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return 1
    frame = pd.read_csv(csv_path)
    frame = frame[(frame["method"] == args.method)
                  & np.isclose(frame["rho"], args.rho)]

    scheme = load_scheme(splits_dir() / f"{args.scheme}.json")
    if not any(str(seed) in scheme["seeds"] for seed in args.seeds):
        print("ERROR: none of the requested seeds exist in the scheme")
        return 1

    records = []
    for seed in args.seeds:
        key = str(seed)
        if key not in scheme["seeds"]:
            print(f"seed {seed}: absent from the scheme, skipped")
            continue
        test_ids = scheme["seeds"][key]["test"]
        incomplete = [i for i in test_ids
                      if not is_complete(image_cache_path(args.model,
                                                          args.dataset, i))]
        if incomplete:
            print(f"ERROR: {len(incomplete)} test images missing from the cache")
            return 1

        for alpha in args.alphas:
            cell = frame[(frame["seed"] == seed) & np.isclose(frame["alpha"], alpha)]
            thresholds = {}
            for class_name in critical:
                row = cell[cell["class_name"] == class_name]
                # ``feasible`` reads back as bool from a clean file and as a
                # string where the column carries undefined rows, so it is
                # normalized rather than trusted.
                ok = (not row.empty
                      and str(row["feasible"].iloc[0]).lower() == "true")
                if not ok:
                    thresholds = {}
                    break
                thresholds[class_name] = float(row["lam"].iloc[0])
            if not thresholds:
                print(f"seed {seed}, alpha {alpha}: no feasible threshold "
                      "for every class, skipped")
                continue

            unions, per_class_areas = [], {c: [] for c in critical}
            for image_id in tqdm(test_ids, desc=f"seed {seed} a={alpha}",
                                 unit="img", leave=False):
                union, per_class = union_area_for_image(
                    args.model, args.dataset, image_id, thresholds, channel_of)
                unions.append(union)
                for c, v in per_class.items():
                    per_class_areas[c].append(v)

            means = {c: float(np.mean(v)) for c, v in per_class_areas.items()}
            records.append({
                "model": args.model, "scheme": args.scheme, "seed": seed,
                "alpha": alpha, "rho": args.rho,
                "union_area": float(np.mean(unions)),
                "class_mean_area": float(np.mean(list(means.values()))),
                "max_class_area": float(max(means.values())),
                "sum_class_area": float(min(1.0, sum(means.values()))),
                **{f"area__{c}": means[c] for c in critical},
                **{f"lam__{c}": thresholds[c] for c in critical},
            })

    if not records:
        print("no configuration produced a union area")
        return 1

    out = pd.DataFrame(records)
    out_csv = results_dir("experiments") / f"x7_union_area__{args.model}.csv"
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(out)} rows)")

    meta = results_dir("experiments") / f"x7_union_area__{args.model}.meta.json"
    meta.write_text(json.dumps({
        "git_revision": git_revision(),
        "model_key": args.model, "dataset_key": args.dataset,
        "scheme": args.scheme, "threshold_source": args.experiment,
        "method": args.method, "seeds": args.seeds, "rho": args.rho,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=1))

    # A draw whose union covers essentially the whole image is a degenerate
    # operating point, not an operator cost, and one such draw dominates any
    # mean it enters. They are counted and excluded rather than averaged in,
    # which is the convention the infeasibility column already follows.
    out["degenerate"] = out["union_area"] > 0.99
    print(f"{'alpha':>6}{'draws':>7}{'degen':>7}{'class mean':>12}"
          f"{'max class':>11}{'UNION':>9}{'sum (cap 1)':>13}{'U/mean':>8}")
    for alpha, cell in out.groupby("alpha"):
        good = cell[~cell["degenerate"]]
        if good.empty:
            print(f"{alpha:>6}{len(cell):>7}{int(cell['degenerate'].sum()):>7}"
                  f"{'':>12}{'':>11}{'all degenerate':>9}")
            continue
        ratio = good["union_area"].mean() / good["class_mean_area"].mean()
        print(f"{alpha:>6}{len(cell):>7}{int(cell['degenerate'].sum()):>7}"
              f"{good['class_mean_area'].mean():>12.4f}"
              f"{good['max_class_area'].mean():>11.4f}"
              f"{good['union_area'].mean():>9.4f}"
              f"{good['sum_class_area'].mean():>13.4f}{ratio:>8.2f}")
    out.to_csv(out_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
