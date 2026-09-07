#!/usr/bin/env python
"""Stage 30: region tables for the dilation family (referee baseline).

Same output as stage 4 (``components.parquet``, ``coverage_curves.npy``,
``image_stats.npz``), written under ``results/raw/<model>__dilation/<dataset>/``,
but the per-pixel "probability" is the pseudo-probability of
:mod:`record.dilation`: grid point ``k`` is the argmax mask dilated by
``k * R_MAX_PX / 1000`` pixels, and the last grid point marks the whole image.
Stage 5 then runs on ``--model <model>__dilation`` exactly as it runs on the
probability family, and its ``region_crc`` rows are "dilation CRC": the CRC
rule selecting a dilation radius instead of a probability threshold. Its
``argmax`` rows coincide with dilation CRC at ``lam = 0`` by construction.

Only the cached ``argmax`` channel is read; the probabilities are not used,
which is the point of the baseline: it asks how much of the region guarantee
a purely geometric relaxation of the hard prediction buys.

Example
-------
    python scripts/30_build_dilation_tables.py \
        --model segformer_b2_cityscapes --datasets cityscapes_val
    python scripts/05_run_experiments.py --name x16_dilation__segformer_b2_cityscapes \
        --model segformer_b2_cityscapes__dilation --scheme cityscapes_val_half \
        --cal-datasets cityscapes_val --test-datasets cityscapes_val \
        --methods region_crc argmax
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from record.cache import image_cache_path, is_complete, write_manifest
from record.components import size_stratum
from record.coverage import compute_image_class_stats
from record.dilation import R_MAX_PX, dilation_pseudo_prob
from record.gt import list_ids, load_label_array, parse_dataset_key
from record.grid import FP_SUBGRID_INDICES, GRID_KIND, LAMBDA_GRID
from record.labelmaps import critical_classes_for
from record.paths import results_dir

DILATION_SUFFIX = "__dilation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model key (cache must exist)")
    parser.add_argument("--datasets", nargs="+", required=True, help="dataset keys")
    parser.add_argument("--min-component-px", type=int, default=1)
    parser.add_argument("--r-max-px", type=float, default=R_MAX_PX,
                        help="dilation radius mapped to lam = 1 (default %(default)s)")
    parser.add_argument("--force", action="store_true", help="rebuild existing tables")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_model = args.model + DILATION_SUFFIX

    for dataset_key in args.datasets:
        family, _ = parse_dataset_key(dataset_key)
        critical = critical_classes_for(family)
        out_dir = results_dir(f"raw/{out_model}/{dataset_key}")
        if (out_dir / "components.parquet").exists() and not args.force:
            print(f"{dataset_key}: dilation tables exist, skipping (use --force to rebuild)")
            continue

        ids = list_ids(dataset_key)
        missing = [i for i in ids if not is_complete(image_cache_path(args.model, dataset_key, i))]
        if missing:
            print(f"ERROR: {dataset_key}: {len(missing)} images missing from cache "
                  f"(run stage 3 first), e.g. {missing[:3]}")
            return 1

        rows: list[dict] = []
        curves: list[np.ndarray] = []
        n_img = len(ids)
        class_names = list(critical)
        marked_area = np.zeros((n_img, len(class_names), LAMBDA_GRID.size), dtype=np.float16)
        argmax_area = np.zeros((n_img, len(class_names)), dtype=np.float32)
        fp_counts = np.zeros((n_img, len(class_names), FP_SUBGRID_INDICES.size), dtype=np.int32)
        component_counts = np.zeros((n_img, len(class_names)), dtype=np.int32)

        for img_row, image_id in enumerate(tqdm(ids, desc=f"{dataset_key} (dilation)", unit="img")):
            cached = np.load(image_cache_path(args.model, dataset_key, image_id))
            label = load_label_array(dataset_key, image_id)
            argmax = cached["argmax"]

            for c_idx, (class_name, (gt_value, model_channel)) in enumerate(critical.items()):
                argmax_mask = argmax == model_channel
                prob = dilation_pseudo_prob(argmax_mask, args.r_max_px)
                gt_mask = label == gt_value
                stats = compute_image_class_stats(
                    prob, gt_mask, argmax_mask=argmax_mask,
                    min_component_px=args.min_component_px)
                marked_area[img_row, c_idx] = stats.marked_area_curve.astype(np.float16)
                argmax_area[img_row, c_idx] = stats.argmax_marked_area
                fp_counts[img_row, c_idx] = stats.fp_component_counts
                component_counts[img_row, c_idx] = len(stats.components)
                for comp_row, comp in enumerate(stats.components):
                    rows.append({
                        "image_id": image_id,
                        "image_row": img_row,
                        "class_name": class_name,
                        "component_id": comp.component_id,
                        "size_px": comp.size_px,
                        "size_stratum": size_stratum(comp.size_px),
                        "bbox_r0": comp.bbox[0], "bbox_c0": comp.bbox[1],
                        "bbox_r1": comp.bbox[2], "bbox_c1": comp.bbox[3],
                        "argmax_coverage": float(stats.argmax_coverage[comp_row]),
                        "curve_row": len(curves) + comp_row,
                    })
                curves.extend(stats.coverage_curves)

        out_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_parquet(out_dir / "components.parquet", index=False)
        np.save(out_dir / "coverage_curves.npy",
                np.array(curves, dtype=np.float16).reshape(len(curves), LAMBDA_GRID.size))
        np.savez_compressed(
            out_dir / "image_stats.npz",
            image_ids=np.array(ids),
            class_names=np.array(class_names),
            marked_area=marked_area,
            argmax_marked_area=argmax_area,
            fp_component_counts=fp_counts,
            component_counts=component_counts,
        )
        write_manifest(out_dir, {
            "model_key": out_model,
            "source_model_key": args.model,
            "family": "dilation",
            "r_max_px": args.r_max_px,
            "grid": GRID_KIND,
            "dataset_key": dataset_key,
            "n_images": n_img,
            "n_components": len(frame),
            "classes": class_names,
            "min_component_px": args.min_component_px,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"{dataset_key}: {n_img} images, {len(frame)} components -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
