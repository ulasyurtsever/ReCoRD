#!/usr/bin/env python
"""Stage 23: what the tier-B target pool does to the estimated weights.

Stage 5 fits the density ratio between two populations that are conditioned
differently. The source side is the calibration images that *contain* the
critical class, because those are the points the weighted risk sums over. The
target side is the whole target set, class-bearing or not. The ratio that
comes back is therefore dQ/dP(.|c present), while the weighted-CRC statement
is written for dQ(.|c present)/dP(.|c present): the test point is a target
image that contains the class, since the region loss is undefined otherwise.

On the class-bearing support the two differ by the target class frequency,
which is below one, so the mismatch pushes the fitted weights down --- the
same direction as the collapse to the clipping floor that stage 5 reports.
That makes the collapse ambiguous: part of it may be an artifact of the pool
rather than a property of the shift. This stage measures the difference by
refitting the same estimator with the target side restricted to target images
that contain the class, and reporting both fits side by side.

Weights do not depend on alpha or on the capture level, and the clip enters
only after the fit, so one logistic fit per (condition, class, seed, pool)
serves the whole grid: the unclipped ratio is computed once and each clip
interval is applied to it afterwards.

Writes ``results/experiments/x8_tierb_pool__<model>__<embedding>.csv`` with one
row per (condition, class, seed, pool, clip, alpha), carrying the effective
sample size, the fraction of weights sitting on the floor, the conservative
test mass, the selected threshold and the realized risk.

Example
-------
    python scripts/23_tierb_target_pool.py \
        --model segformer_b2_cityscapes --scheme cityscapes_val_half \
        --cal-datasets cityscapes_val --embedding dinov2_vitb14 \
        --conditions fog night rain snow --max-seeds 25
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from record.crc import weighted_crc_threshold
from record.evaluation import evaluate_at_threshold, index_rows
from record.grid import LAMBDA_GRID
from record.paths import results_dir, splits_dir
from record.splits import load_scheme
from record.weights import clip_weights

SCRIPTS = Path(__file__).resolve().parent
ALPHAS = (0.05, 0.10, 0.20)
RHO = 0.5
CLIPS = ((0.05, 2.0), (0.05, 5.0), (0.05, 20.0))
UNCLIPPED = (1e-12, 1e12)
CRITICAL = ("person", "rider", "bicycle")


def load_stage(filename: str, alias: str):
    """Import a numbered stage script for reuse.

    Stage module names begin with a digit and cannot be imported by name. The
    alternative is to copy the table loader into this file, which is the code
    most likely to drift away from the tables the reported results are built
    from.
    """
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_logistic_ratio(source_emb, target_emb, c_reg=1.0, random_state=0):
    """The unclipped logistic density ratio at the source points.

    This is ``record.weights.logistic_density_ratio`` with the clip deferred,
    so that one fit serves every clip interval. Keeping the body here rather
    than calling the library function with a wide clip would let the two drift
    apart, so the library function is called and the wide clip is inverted by
    construction: ``UNCLIPPED`` is far outside any weight the classifier can
    produce, so ``clip_weights`` is the identity on its output.
    """
    from record.weights import logistic_density_ratio

    return logistic_density_ratio(source_emb, target_emb, clip=UNCLIPPED,
                                  c_reg=c_reg, random_state=random_state)


def summarize(weights, floor):
    w_sum = float(weights.sum())
    return {
        "weight_sum": w_sum,
        "weight_ess": w_sum ** 2 / float((weights ** 2).sum()),
        "frac_at_floor": float(np.mean(np.isclose(weights, floor))),
        "weight_median": float(np.median(weights)),
        "weight_max": float(weights.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--cal-datasets", nargs="+", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--conditions", nargs="+",
                        default=["fog", "night", "rain", "snow"])
    parser.add_argument(
        "--dataset-template", default="acdc_{cond}_train,acdc_{cond}_val",
        help="comma-separated target dataset keys, with {cond} substituted")
    parser.add_argument("--classes", nargs="+", default=list(CRITICAL))
    parser.add_argument("--max-seeds", type=int, default=25)
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    stage5 = load_stage("05_run_experiments.py", "stage5")
    scheme = load_scheme(splits_dir() / f"{args.scheme}.json")
    seeds = list(scheme["seeds"].items())[: args.max_seeds]

    cal_store = stage5.TableStore(args.model, args.cal_datasets)
    rows: list[dict] = []

    for cond in args.conditions:
        test_datasets = [k.format(cond=cond)
                         for k in args.dataset_template.split(",")]
        test_store = stage5.TableStore(args.model, test_datasets)
        keys = list(dict.fromkeys(args.cal_datasets + test_datasets))
        emb_ids, emb = stage5.load_embeddings(args.embedding, keys)
        test_rows = np.arange(len(test_store.image_ids))

        for class_name in args.classes:
            cal_view = cal_store.class_view(class_name, RHO)
            test_view = test_store.class_view(class_name, RHO)

            # The two candidate target pools. "class" is the one the weighted
            # statement is written for; "all" is what stage 5 uses.
            pools = {
                "all": test_rows,
                "class": stage5.defined_rows(test_view["losses"], test_rows),
            }
            test_losses = test_view["losses"][test_rows]
            test_area = test_view["marked_area"][test_rows]
            test_counts = test_view["component_counts"][test_rows]

            for seed_key, entry in seeds:
                cal_rows = index_rows(cal_store.image_ids, entry["calibration"])
                cal_def = stage5.defined_rows(cal_view["losses"], cal_rows)
                if cal_def.size == 0:
                    continue
                cal_ids = [cal_store.image_ids[r] for r in cal_def]
                cal_emb = emb[index_rows(emb_ids, cal_ids)]

                for pool_name, pool in pools.items():
                    tgt_ids = [test_store.image_ids[r] for r in pool]
                    tgt_emb = emb[index_rows(emb_ids, tgt_ids)]
                    raw = raw_logistic_ratio(cal_emb, tgt_emb)

                    for clip in CLIPS:
                        w = clip_weights(raw, clip)
                        stats = summarize(w, clip[0])
                        p_test = clip[1] / (float(w.sum()) + clip[1])
                        for alpha in ALPHAS:
                            sel = weighted_crc_threshold(
                                cal_view["losses"][cal_def], w,
                                test_weight=clip[1], alpha=alpha,
                                lambda_grid=LAMBDA_GRID)
                            metrics = evaluate_at_threshold(
                                sel, test_losses, test_area, test_counts)
                            rows.append(dict(
                                condition=cond, class_name=class_name,
                                seed=int(seed_key), pool=pool_name,
                                n_cal=int(cal_def.size), n_pool=int(pool.size),
                                n_target_total=int(test_rows.size),
                                c_low=clip[0], kappa=clip[1],
                                ratio=clip[1] / clip[0], alpha=alpha, rho=RHO,
                                raw_median=float(np.median(raw)),
                                raw_max=float(raw.max()),
                                weight_p_test=p_test, lam=float(sel.lam),
                                feasible=bool(sel.feasible),
                                region_fnr=metrics.region_fnr,
                                marked_area_fraction=(
                                    metrics.marked_area_fraction),
                                **stats))
        print(f"{cond}: {len(rows)} rows so far")

    frame = pd.DataFrame(rows)
    name = args.out_name or f"x8_tierb_pool__{args.model}__{args.embedding}"
    out_dir = results_dir("experiments")
    path = out_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    (out_dir / f"{name}.meta.json").write_text(json.dumps({
        "name": name, "argv": vars(args), "n_records": len(frame),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    print(f"\nwrote {path}  ({len(frame)} rows)")

    # A compact console summary: the question is whether restricting the
    # target pool to class-bearing images lifts the weights off the floor.
    key = ["condition", "class_name", "kappa", "pool"]
    summary = frame.groupby(key).agg(
        ess_over_n=("weight_ess", "mean"),
        n_cal=("n_cal", "mean"),
        floor_frac=("frac_at_floor", "mean"),
        raw_median=("raw_median", "mean"),
        p_test=("weight_p_test", "mean"),
        fnr=("region_fnr", "mean")).reset_index()
    summary["ess_over_n"] = summary["ess_over_n"] / summary["n_cal"]
    pd.set_option("display.width", 160)
    print("\n" + summary.round(4).to_string(index=False))

    piv = frame.pivot_table(index=["condition", "class_name"], columns="pool",
                            values=["frac_at_floor", "raw_median"])
    print("\nfraction of weights on the floor, and median unclipped ratio:")
    print(piv.round(4).to_string())
    print("\npool sizes: all vs class-bearing")
    print(frame.groupby(["condition", "class_name", "pool"]).n_pool.max()
          .unstack().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
