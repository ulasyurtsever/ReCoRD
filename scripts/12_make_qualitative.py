#!/usr/bin/env python
"""Qualitative comparison on one test scene: argmax versus region CRC.

Reads the cached posterior of a trained model and the ground-truth class
mask, and renders three panels: the image with ground-truth regions of a
critical class outlined, the argmax prediction with missed regions marked,
and the region-CRC mask at the calibrated threshold with the marked area
reported. The threshold is not recomputed here; it is read from the
experiment CSV so that the figure shows the same threshold the tables do.

Two benchmarks are supported. LoveDA is the default because its critical
regions are large enough to be legible in print; MARIDA is available but
its debris components span one to eight pixels on the official test split,
which no rendering makes visible at scene scale.

Patch selection is deterministic and stated in the output manifest. The
figure has to make the mechanism visible, so candidates are restricted to the
patches of the benchmark's evaluation pool -- the official test split on
MARIDA, the labeled Val/Urban split on LoveDA, since LoveDA's own test split
carries no labels -- that contain at least two regions of the critical class
and at least one region large enough to be legible in print; among those, the
patch chosen is the one where the calibrated threshold recovers the most
regions that argmax misses, with ties broken by region count and then by
patch identifier. The full ranking is printed with ``--report``.
The panel illustrates the mechanism; aggregate performance is reported in the
tables.

Writes ``results/figures/qualitative_<benchmark>.pdf``.

Example
-------
    python scripts/12_make_qualitative.py
    python scripts/12_make_qualitative.py --patch S2_22-12-20_18QYF_0 --alpha 0.2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from record.cache import image_cache_path, is_complete
from record.components import extract_components
from record.gt import (list_ids, load_image_array, load_label_array,
                       parse_dataset_key)
from record.grid import LAMBDA_GRID
from record.labelmaps import critical_classes_for
from record.marida import load_bands
from record.paths import results_dir

# MARIDA ships 11 Sentinel-2 bands in the order B1, B2, B3, B4, B5, B6, B7,
# B8, B8A, B11, B12; the true-color composite is therefore (B4, B3, B2).
RGB_BANDS = (3, 2, 1)

BENCHMARKS = {
    "loveda": {
        "dataset_key": "loveda_Val_Urban",
        "model": "segformer_b2_loveda_urban",
        "experiment": "l1_indist__urban",
        "family": "loveda",
        "class_name": "building",
        "label": "LoveDA urban",
        "min_visible_px": 2000,
    },
    "marida": {
        "dataset_key": "marida_test",
        "model": "marida_unet_official_holdout_ens5",
        "experiment": "h1_official__marida_unet_official_holdout_ens5",
        "family": "marida",
        "class_name": "marine_debris",
        "label": "MARIDA",
        "min_visible_px": 4,
    },
}
PALETTE = {"gt": "#000000", "missed": "#D55E00", "captured": "#009E73",
           "marked": "#0072B2"}
PAGE_W = 7.16  # inches, IEEE double column

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def rgb_composite(dataset_key: str, image_id: str,
                  percentile: float = 2.0) -> np.ndarray:
    """Displayable RGB image, percentile-stretched per channel."""
    family, _ = parse_dataset_key(dataset_key)
    if family == "marida":
        bands = load_bands(image_id)
        rgb = np.stack([bands[i] for i in RGB_BANDS], axis=-1)
    else:
        rgb = load_image_array(dataset_key, image_id).astype(np.float32)
    out = np.zeros(rgb.shape, dtype=np.float32)
    for i in range(3):
        channel = rgb[..., i].astype(np.float32)
        lo, hi = np.percentile(channel, [percentile, 100.0 - percentile])
        if hi <= lo:
            hi = lo + 1.0
        out[..., i] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return out


def calibrated_threshold(experiment: str, alpha: float, rho: float,
                         class_name: str) -> float:
    """Median selected threshold for region CRC from the experiment CSV."""
    path = results_dir("experiments") / f"{experiment}.csv"
    frame = pd.read_csv(path)
    sub = frame[(frame.method == "region_crc") & (frame.alpha == alpha)
                & (frame.rho == rho) & (frame.class_name == class_name)
                & frame.feasible.astype(bool)]
    if sub.empty:
        raise ValueError(
            f"{path.name}: no feasible region_crc rows at alpha={alpha}, rho={rho}")
    return float(np.median(sub["lam"]))


def patch_statistics(model: str, dataset_key: str, patch_id: str,
                     gt_value: int, channel: int, lam: float,
                     rho: float) -> dict:
    """Per-region capture under argmax and under the calibrated threshold."""
    path = image_cache_path(model, dataset_key, patch_id)
    if not is_complete(path):
        raise FileNotFoundError(f"missing cached posterior for {patch_id}")
    with np.load(path) as arrays:
        critical = arrays["critical_probs"]
        channels = list(arrays["critical_channels"])
        argmax = arrays["argmax"]
    prob = critical[channels.index(channel)].astype(np.float32)
    gt = load_label_array(dataset_key, patch_id)
    labels, components = extract_components(gt == gt_value)

    mask_crc = prob >= 1.0 - lam
    mask_argmax = argmax == channel
    rows = []
    for comp in components:
        region = labels == comp.component_id
        area = region.sum()
        rows.append({
            "component_id": comp.component_id,
            "size_px": int(area),
            "cov_argmax": float((mask_argmax & region).sum() / area),
            "cov_crc": float((mask_crc & region).sum() / area),
        })
    return {
        "patch_id": patch_id,
        "n_regions": len(rows),
        "regions": rows,
        "missed_argmax": sum(r["cov_argmax"] < rho for r in rows),
        "missed_crc": sum(r["cov_crc"] < rho for r in rows),
        "area_argmax": float(mask_argmax.mean()),
        "area_crc": float(mask_crc.mean()),
        "labels": labels,
        "mask_argmax": mask_argmax,
        "mask_crc": mask_crc,
    }


def rank_patches(model: str, dataset_key: str, gt_value: int, channel: int,
                 lam: float, rho: float, min_regions: int,
                 min_visible_px: int) -> list[dict]:
    """All eligible scenes, ordered by the documented selection rule."""
    ranked = []
    for patch_id in list_ids(dataset_key):
        gt = load_label_array(dataset_key, patch_id)
        if not np.any(gt == gt_value):
            continue
        stats = patch_statistics(model, dataset_key, patch_id, gt_value,
                                 channel, lam, rho)
        if stats["n_regions"] < min_regions:
            continue
        if max(r["size_px"] for r in stats["regions"]) < min_visible_px:
            continue
        recovered = sum(
            1 for r in stats["regions"]
            if r["cov_argmax"] < rho <= r["cov_crc"])
        stats["recovered"] = recovered
        ranked.append(stats)
    ranked.sort(key=lambda s: (-s["recovered"], -s["n_regions"], s["patch_id"]))
    return ranked


def select_patch(model: str, dataset_key: str, gt_value: int, channel: int,
                 lam: float, rho: float, min_regions: int,
                 min_visible_px: int) -> list[dict]:
    """Deterministic scene choice; see the module docstring for the rule."""
    ranked = rank_patches(model, dataset_key, gt_value, channel, lam, rho,
                          min_regions, min_visible_px)
    if not ranked:
        raise ValueError(
            f"no test scene with >= {min_regions} regions and a region of "
            f">= {min_visible_px} px; lower --min-visible-px or --min-regions")
    if ranked[0]["recovered"] == 0:
        print("WARNING: no eligible patch shows a region recovered by the "
              "calibrated threshold that argmax misses; the figure will not "
              "illustrate the mechanism. Inspect the ranking with --report.")
    return ranked


def _outline(ax, labels: np.ndarray, component_id: int, color: str) -> None:
    ax.contour(labels == component_id, levels=[0.5], colors=color, linewidths=1.1)


def render(stats: dict, spec: dict, rho: float, out_name: str) -> str:
    rgb = rgb_composite(spec["dataset_key"], stats["patch_id"])
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, PAGE_W / 3.0 + 0.35))

    label = spec["class_name"].replace("_", " ")
    axes[0].imshow(rgb)
    for row in stats["regions"]:
        _outline(axes[0], stats["labels"], row["component_id"], PALETTE["gt"])
    axes[0].set_title(f"(a) {spec['label']} scene\n"
                      f"{stats['n_regions']} {label} regions")

    for ax, key, cov_key, title in (
        (axes[1], "mask_argmax", "cov_argmax", "(b) Argmax"),
        (axes[2], "mask_crc", "cov_crc", "(c) Region CRC"),
    ):
        ax.imshow(rgb)
        ax.imshow(np.ma.masked_where(~stats[key], stats[key].astype(float)),
                  cmap=matplotlib.colors.ListedColormap([PALETTE["marked"]]),
                  alpha=0.55, vmin=0, vmax=1)
        missed = 0
        for row in stats["regions"]:
            captured = row[cov_key] >= rho
            missed += not captured
            _outline(ax, stats["labels"], row["component_id"],
                     PALETTE["captured"] if captured else PALETTE["missed"])
        area = stats["area_argmax"] if key == "mask_argmax" else stats["area_crc"]
        ax.set_title(f"{title}\n{missed} missed, {area:.1%} of image marked")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    out = results_dir("figures") / f"{out_name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="loveda", choices=sorted(BENCHMARKS),
                        help="which benchmark to illustrate")
    parser.add_argument("--model", default=None,
                        help="override the benchmark's default model key")
    parser.add_argument("--patch", default=None,
                        help="override the automatic scene selection")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--min-regions", type=int, default=2)
    parser.add_argument("--min-visible-px", type=int, default=None,
                        help="smallest largest-region size for a legible panel")
    parser.add_argument("--out-name", default=None,
                        help="output stem under results/figures; "
                             "defaults to qualitative_<benchmark>")
    parser.add_argument("--report", type=int, default=10,
                        help="print this many ranked candidates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = BENCHMARKS[args.benchmark]
    model = args.model or spec["model"]
    min_visible = (args.min_visible_px if args.min_visible_px is not None
                   else spec["min_visible_px"])
    gt_value, channel = critical_classes_for(spec["family"])[spec["class_name"]]
    lam = calibrated_threshold(spec["experiment"], args.alpha, args.rho,
                               spec["class_name"])
    lam = float(LAMBDA_GRID[int(np.argmin(np.abs(LAMBDA_GRID - lam)))])
    ranking = []
    if args.patch:
        stats = patch_statistics(model, spec["dataset_key"], args.patch,
                                 gt_value, channel, lam, args.rho)
        stats["recovered"] = sum(
            1 for r in stats["regions"]
            if r["cov_argmax"] < args.rho <= r["cov_crc"])
    else:
        ranked = select_patch(model, spec["dataset_key"], gt_value, channel,
                              lam, args.rho, args.min_regions, min_visible)
        stats = ranked[0]
        ranking = [
            {"patch_id": s["patch_id"], "n_regions": s["n_regions"],
             "recovered": s["recovered"], "missed_argmax": s["missed_argmax"],
             "missed_crc": s["missed_crc"],
             "largest_region_px": max(r["size_px"] for r in s["regions"])}
            for s in ranked[:max(args.report, 1)]
        ]
        if args.report:
            print(f"{'scene':<26}{'regions':>8}{'recovered':>11}"
                  f"{'miss@argmax':>13}{'miss@crc':>10}{'largest px':>12}")
            for row in ranking:
                print(f"{row['patch_id']:<26}{row['n_regions']:>8}"
                      f"{row['recovered']:>11}{row['missed_argmax']:>13}"
                      f"{row['missed_crc']:>10}{row['largest_region_px']:>12}")
    out_name = args.out_name or f"qualitative_{args.benchmark}"
    path = render(stats, spec, args.rho, out_name)

    manifest = {
        "figure": out_name,
        "path": path,
        "benchmark": args.benchmark,
        "model_key": model,
        "dataset_key": spec["dataset_key"],
        "class_name": spec["class_name"],
        "patch_id": stats["patch_id"],
        "selection_rule": ("most regions recovered by the calibrated threshold "
                           f"among {spec['dataset_key']} scenes with at least "
                           f"{args.min_regions} regions and a region of "
                           f"at least {min_visible} px; ties by region "
                           "count then patch id")
                          if not args.patch else "explicit --patch",
        "regions_recovered_vs_argmax": stats["recovered"],
        "ranking_top": ranking,
        "alpha": args.alpha,
        "rho": args.rho,
        "lambda": lam,
        "lambda_source": ("median feasible region_crc lam in "
                          f"{spec['experiment']}.csv"),
        "n_regions": stats["n_regions"],
        "missed_argmax": stats["missed_argmax"],
        "missed_crc": stats["missed_crc"],
        "marked_area_argmax": round(stats["area_argmax"], 4),
        "marked_area_crc": round(stats["area_crc"], 4),
        "region_sizes_px": [r["size_px"] for r in stats["regions"]],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(results_dir("figures") / f"{out_name}.meta.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"scene {stats['patch_id']}: {stats['n_regions']} regions, "
          f"argmax misses {stats['missed_argmax']}, "
          f"region CRC misses {stats['missed_crc']} "
          f"(lambda={lam:.3f}, area {stats['area_argmax']:.1%} -> "
          f"{stats['area_crc']:.1%})")
    print(f"wrote {path}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
