#!/usr/bin/env python
"""Stage 4: reduce the inference cache to region-level sufficient statistics.

For each (model, dataset, critical class), produces:

- ``components.parquet``: one row per ground-truth component with image id,
  class, size, size stratum, bounding box, argmax coverage, and the row index
  of its coverage curve;
- ``coverage_curves.npy``: ``(n_components, 1001)`` float16 coverage curves;
- ``image_stats.npz``: per-image marked-area curves, argmax marked area,
  false-positive component counts on the subgrid, and component counts.

All calibration experiments run from these tables alone; the cache is not
read again. Resumable at (model, dataset) granularity.

Example
-------
    python scripts/04_build_region_tables.py \
        --model segformer_b2_cityscapes --datasets cityscapes_val acdc_fog_val
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
from record.gt import list_ids, load_label_array, parse_dataset_key
from record.grid import FP_SUBGRID_INDICES, LAMBDA_GRID
from record.labelmaps import critical_classes_for
from record.paths import results_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model key (cache must exist)")
    parser.add_argument("--datasets", nargs="+", required=True, help="dataset keys")
    parser.add_argument("--min-component-px", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="rebuild existing tables")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for dataset_key in args.datasets:
        family, _ = parse_dataset_key(dataset_key)
        critical = critical_classes_for(family)
        out_dir = results_dir(f"raw/{args.model}/{dataset_key}")
        if (out_dir / "components.parquet").exists() and not args.force:
            print(f"{dataset_key}: tables exist, skipping (use --force to rebuild)")
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

        for img_row, image_id in enumerate(tqdm(ids, desc=f"{dataset_key}", unit="img")):
            cached = np.load(image_cache_path(args.model, dataset_key, image_id))
            label = load_label_array(dataset_key, image_id)
            argmax = cached["argmax"]
            channels = cached["critical_channels"]

            for c_idx, (class_name, (gt_value, model_channel)) in enumerate(critical.items()):
                channel_pos = int(np.where(channels == model_channel)[0][0])
                prob = cached["critical_probs"][channel_pos].astype(np.float32)
                gt_mask = label == gt_value
                stats = compute_image_class_stats(
                    prob, gt_mask,
                    argmax_mask=(argmax == model_channel),
                    min_component_px=args.min_component_px,
                )
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
            "model_key": args.model,
            "dataset_key": dataset_key,
            "n_images": n_img,
            "n_components": len(frame),
            "classes": class_names,
            "min_component_px": args.min_component_px,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"{dataset_key}: {len(frame)} components across {n_img} images")

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
