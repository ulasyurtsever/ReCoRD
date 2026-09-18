#!/usr/bin/env python
"""Stage 4b: auxiliary tables for the temperature-scaling and LAC baselines.

Derives two baseline inputs from the stage-3 cache, without touching any
model:

1. Temperature scaling: a scalar temperature is fitted by NLL on the strided
   full posterior of a fixed calibration image list, tempered per-class
   probabilities are reconstructed at full resolution (exact numerator from
   the cached critical-class probabilities; normalizer upsampled from the
   strided posterior), and standard region tables are written under the
   model key ``<model>_tempscaled``.
2. LAC miscoverage curves: per-image all-class pixel miscoverage as a
   function of the threshold, computed on the strided posterior and written
   as ``lac_miscoverage.npy`` next to the existing region tables. A single
   global threshold calibrated on these curves reproduces the LAC-style
   multi-label baseline at the pixel-coverage level.
3. Class-conditional LAC miscoverage curves: the same quantity restricted to
   one critical class at a time, written as
   ``lac_miscoverage_by_class.npz``. WHY a second table: the marginal curve
   of (2) is dominated by road, building and vegetation, so a threshold
   calibrated on it is set by the easy classes and then charged with the
   critical classes' region loss. Per-class curves let stage 5 calibrate one
   threshold per critical class, which is the fair form of the baseline.

All derivations are deterministic CPU passes over the cache.

Example
-------
    python scripts/04b_build_baseline_tables.py \
        --model segformer_b2_cityscapes --datasets cityscapes_val \
        --temperature-cal-scheme cityscapes_val_half
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from record.cache import image_cache_path, is_complete, write_manifest
from record.components import size_stratum
from record.coverage import compute_image_class_stats
from record.gt import list_ids, load_label_array, parse_dataset_key
from record.grid import FP_SUBGRID_INDICES, LAMBDA_GRID, curve_on_grid
from record.labelmaps import critical_classes_for, trainid_map_for
from record.models import load_model_registry
from record.paths import results_dir, splits_dir
from record.splits import load_scheme
from record.temperature import (TEMPERATURE_GRID, nll_grid_search,
                                strided_true_probs)


def fit_temperature(model_key, dataset_key, cal_ids, value_to_channel,
                    stride, max_pixels_per_image=4000, seed=0):
    """Fit a scalar temperature by NLL over strided calibration pixels."""
    rng = np.random.default_rng(seed)
    true_list, full_list = [], []
    for image_id in tqdm(cal_ids, desc="temperature fit", unit="img"):
        cached = np.load(image_cache_path(model_key, dataset_key, image_id))
        label = load_label_array(dataset_key, image_id)
        true_p, probs, valid = strided_true_probs(
            cached, label, value_to_channel, stride)
        if true_p.size == 0:
            continue
        idx = rng.choice(true_p.size, min(max_pixels_per_image, true_p.size),
                         replace=False)
        rr, cc = np.nonzero(valid)
        true_list.append(true_p[idx])
        full_list.append(probs[:, rr[idx], cc[idx]])
    return nll_grid_search(np.concatenate(true_list),
                           np.concatenate(full_list, axis=1))


def upsample_to(arr, shape, stride):
    """Nearest-neighbor upsample of a strided array to a full-resolution shape."""
    up = np.repeat(np.repeat(arr, stride, axis=0), stride, axis=1)
    out = np.zeros(shape, dtype=arr.dtype)
    h = min(shape[0], up.shape[0])
    w = min(shape[1], up.shape[1])
    out[:h, :w] = up[:h, :w]
    if h < shape[0]:
        out[h:, :] = out[h - 1 : h, :]
    if w < shape[1]:
        out[:, w:] = out[:, w - 1 : w]
    return out


def build_tempered_tables(model_key, dataset_key, temperature, stride,
                          min_component_px, out_key=None):
    """Region tables for the tempered posterior.

    Written under ``<model>_tempscaled`` unless ``out_key`` names a different
    destination, which stage 21 uses to keep one table set per temperature.
    """
    out_key = out_key or f"{model_key}_tempscaled"
    critical = critical_classes_for(parse_dataset_key(dataset_key)[0])
    out_dir = results_dir(f"raw/{out_key}/{dataset_key}")
    ids = list_ids(dataset_key)
    eps = 1e-12

    rows, curves = [], []
    n_img = len(ids)
    class_names = list(critical)
    marked_area = np.zeros((n_img, len(class_names), LAMBDA_GRID.size), dtype=np.float16)
    argmax_area = np.zeros((n_img, len(class_names)), dtype=np.float32)
    fp_counts = np.zeros((n_img, len(class_names), FP_SUBGRID_INDICES.size), dtype=np.int32)
    component_counts = np.zeros((n_img, len(class_names)), dtype=np.int32)

    for img_row, image_id in enumerate(tqdm(ids, desc=f"tempered {dataset_key}", unit="img")):
        cached = np.load(image_cache_path(model_key, dataset_key, image_id))
        label = load_label_array(dataset_key, image_id)
        argmax = cached["argmax"]
        channels = cached["critical_channels"]
        strided = np.clip(cached["strided_probs"].astype(np.float32), eps, 1.0)
        z_strided = np.power(strided, 1.0 / temperature).sum(axis=0)

        for c_idx, (class_name, (gt_value, model_channel)) in enumerate(critical.items()):
            channel_pos = int(np.where(channels == model_channel)[0][0])
            prob = np.clip(cached["critical_probs"][channel_pos].astype(np.float32), eps, 1.0)
            z_full = upsample_to(z_strided, prob.shape, stride)
            tempered = np.power(prob, 1.0 / temperature) / np.maximum(z_full, eps)
            tempered = np.clip(tempered, 0.0, 1.0)
            gt_mask = label == gt_value
            stats = compute_image_class_stats(
                tempered, gt_mask,
                argmax_mask=(argmax == model_channel),
                min_component_px=min_component_px,
            )
            marked_area[img_row, c_idx] = stats.marked_area_curve.astype(np.float16)
            argmax_area[img_row, c_idx] = stats.argmax_marked_area
            fp_counts[img_row, c_idx] = stats.fp_component_counts
            component_counts[img_row, c_idx] = len(stats.components)
            for comp_row, comp in enumerate(stats.components):
                rows.append({
                    "image_id": image_id, "image_row": img_row,
                    "class_name": class_name, "component_id": comp.component_id,
                    "size_px": comp.size_px,
                    "size_stratum": size_stratum(comp.size_px),
                    "bbox_r0": comp.bbox[0], "bbox_c0": comp.bbox[1],
                    "bbox_r1": comp.bbox[2], "bbox_c1": comp.bbox[3],
                    "argmax_coverage": float(stats.argmax_coverage[comp_row]),
                    "curve_row": len(curves) + comp_row,
                })
            curves.extend(stats.coverage_curves)

    pd.DataFrame(rows).to_parquet(out_dir / "components.parquet", index=False)
    np.save(out_dir / "coverage_curves.npy",
            np.array(curves, dtype=np.float16).reshape(len(curves), LAMBDA_GRID.size))
    np.savez_compressed(
        out_dir / "image_stats.npz",
        image_ids=np.array(ids), class_names=np.array(class_names),
        marked_area=marked_area, argmax_marked_area=argmax_area,
        fp_component_counts=fp_counts, component_counts=component_counts,
    )
    write_manifest(out_dir, {
        "model_key": out_key, "source_model_key": model_key,
        "dataset_key": dataset_key, "temperature": temperature,
        "normalizer": "strided posterior, nearest-neighbor upsampled",
        "n_images": n_img, "n_components": len(rows),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"{dataset_key}: tempered tables, {len(rows)} components, T={temperature}")


def build_lac_curves(model_key, dataset_key, value_to_channel, stride):
    """Per-image all-class pixel miscoverage curves on the strided posterior."""
    out_dir = results_dir(f"raw/{model_key}/{dataset_key}")
    ids = list_ids(dataset_key)
    curves = np.zeros((len(ids), LAMBDA_GRID.size), dtype=np.float16)
    for img_row, image_id in enumerate(tqdm(ids, desc=f"lac {dataset_key}", unit="img")):
        cached = np.load(image_cache_path(model_key, dataset_key, image_id))
        label = load_label_array(dataset_key, image_id)
        true_p, _, _ = strided_true_probs(cached, label, value_to_channel, stride)
        if true_p.size == 0:
            curves[img_row] = np.nan
            continue
        coverage = curve_on_grid(1.0 - true_p)
        curves[img_row] = (1.0 - coverage).astype(np.float16)
    np.save(out_dir / "lac_miscoverage.npy", curves)
    (out_dir / "lac_meta.json").write_text(json.dumps({
        "model_key": model_key, "dataset_key": dataset_key,
        "stride": stride, "n_images": len(ids),
        "definition": "per-image fraction of strided pixels whose true-class "
                      "probability is below 1 - lambda",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    print(f"{dataset_key}: LAC miscoverage curves for {len(ids)} images")


def build_lac_class_curves(model_key, dataset_key, stride):
    """Per-image pixel miscoverage curves restricted to each critical class.

    Identical in definition to :func:`build_lac_curves` -- the fraction of
    strided pixels whose true-class probability falls below ``1 - lambda`` --
    except that the pixel population is one critical class at a time instead
    of every labeled pixel.

    WHY this exists. The marginal curve pools all classes, and the pool is
    dominated by road, building and vegetation, which the model covers almost
    everywhere. A threshold calibrated on it is therefore chosen by the easy
    classes and then scored on the critical classes' region loss, which is an
    unfair comparison. With one curve per
    critical class, stage 5 can calibrate one threshold per class against that
    class's own pixel-coverage target and score it exactly as before, so what
    is left in the comparison is the genuine gap between controlling pixel
    coverage and capturing regions.

    Images with no ground-truth pixels of a class carry a NaN row for that
    class; stage 5 drops those rows from the calibration set, as it already
    does for images without ground-truth components.
    """
    critical = critical_classes_for(parse_dataset_key(dataset_key)[0])
    out_dir = results_dir(f"raw/{model_key}/{dataset_key}")
    ids = list_ids(dataset_key)
    class_names = list(critical)
    curves = np.full((len(ids), len(class_names), LAMBDA_GRID.size), np.nan,
                     dtype=np.float16)
    for img_row, image_id in enumerate(
            tqdm(ids, desc=f"lac/class {dataset_key}", unit="img")):
        cached = np.load(image_cache_path(model_key, dataset_key, image_id))
        label = load_label_array(dataset_key, image_id)
        probs = cached["strided_probs"].astype(np.float32)
        lab = label[::stride, ::stride]
        # The cached posterior and the strided label can differ by a pixel at
        # the border; crop both to the common window, as strided_true_probs
        # does, so the two are indexed consistently.
        h = min(lab.shape[0], probs.shape[1])
        w = min(lab.shape[1], probs.shape[2])
        lab = lab[:h, :w]
        for c_idx, (gt_value, model_channel) in enumerate(critical.values()):
            sel = lab == gt_value
            if not sel.any():
                continue  # class absent from this image: row stays NaN
            true_p = probs[model_channel, :h, :w][sel]
            coverage = curve_on_grid(1.0 - true_p)
            curves[img_row, c_idx] = (1.0 - coverage).astype(np.float16)
    np.savez_compressed(out_dir / "lac_miscoverage_by_class.npz",
                        curves=curves, class_names=np.array(class_names))
    (out_dir / "lac_class_meta.json").write_text(json.dumps({
        "model_key": model_key, "dataset_key": dataset_key,
        "stride": stride, "n_images": len(ids), "classes": class_names,
        "definition": "per-image fraction of strided pixels of the class "
                      "whose true-class probability is below 1 - lambda; "
                      "NaN where the class is absent from the image",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    n_defined = int(np.sum(~np.isnan(curves[:, :, 0])))
    print(f"{dataset_key}: class-conditional LAC curves for {len(ids)} images "
          f"x {len(class_names)} classes ({n_defined} defined rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--temperature-cal-scheme", required=True,
                        help="split scheme whose seed-0 calibration list is "
                             "used to fit the temperature")
    parser.add_argument("--min-component-px", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_model_registry()
    stride = registry["cache"]["strided_probs_stride"]

    scheme = load_scheme(splits_dir() / f"{args.temperature_cal_scheme}.json")
    cal_ids = scheme["seeds"]["0"]["calibration"]

    for dataset_key in args.datasets:
        family, _ = parse_dataset_key(dataset_key)
        value_to_channel = trainid_map_for(family)
        ids = list_ids(dataset_key)
        missing = [i for i in ids if not is_complete(image_cache_path(args.model, dataset_key, i))]
        if missing:
            print(f"ERROR: {dataset_key}: {len(missing)} images missing from cache")
            return 1

        temp_dir = results_dir(f"raw/{args.model}_tempscaled/{dataset_key}")
        if (temp_dir / "components.parquet").exists() and not args.force:
            print(f"{dataset_key}: tempered tables exist, skipping")
        else:
            temperature, nll = fit_temperature(
                args.model, dataset_key, cal_ids, value_to_channel, stride)
            print(f"fitted temperature T={temperature} (NLL {nll:.4f})")
            # Record the fit next to the experiment CSVs so it travels with
            # them (results/raw stays on the compute machine).
            fit_path = results_dir("experiments") / \
                f"x5_temperature__{args.model}.json"
            fit_path.write_text(json.dumps({
                "model_key": args.model, "dataset_key": dataset_key,
                "temperature": temperature, "nll": nll,
                "cal_scheme": args.temperature_cal_scheme,
                "cal_seed": "0", "n_cal_images": len(cal_ids),
                "temperature_grid": [float(TEMPERATURE_GRID[0]),
                                     float(TEMPERATURE_GRID[-1]), 0.05],
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }, indent=1))
            build_tempered_tables(args.model, dataset_key, temperature,
                                  stride, args.min_component_px)

        lac_path = results_dir(f"raw/{args.model}/{dataset_key}") / "lac_miscoverage.npy"
        if lac_path.exists() and not args.force:
            print(f"{dataset_key}: LAC curves exist, skipping")
        else:
            build_lac_curves(args.model, dataset_key, value_to_channel, stride)

        # Additive: the per-class table is a new file next to the marginal
        # one, so an existing lac_miscoverage.npy (and every table built from
        # it) is left exactly as it was.
        lac_class_path = results_dir(f"raw/{args.model}/{dataset_key}") \
            / "lac_miscoverage_by_class.npz"
        if lac_class_path.exists() and not args.force:
            print(f"{dataset_key}: class-conditional LAC curves exist, skipping")
        else:
            build_lac_class_curves(args.model, dataset_key, stride)

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
