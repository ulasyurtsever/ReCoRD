#!/usr/bin/env python
"""Stage 13: area-matched comparison of pixel-level and region-level CRC.

Comparing the two methods at a single requested level leaves them at
different operating points: on SegFormer-B5 at rho=0.5, pixel CRC reaches a
region FNR of 0.257 while marking 1.8 per cent of the image, and region CRC
reaches 0.180 while marking 9.0 per cent.  The two are spending different
marked-area budgets and are not comparable as they stand.

This script sweeps the requested level for both methods, traces each one's
(marked area, region FNR) curve, and interpolates both onto a common area grid,
which answers the like-for-like question: at equal marked area, which method
misses fewer regions.

Input: a dense-alpha run written by 05_run_experiments.py, e.g.

    python scripts/05_run_experiments.py --name p1_pareto__segformer_b5_cityscapes \\
        --model segformer_b5_cityscapes --scheme cityscapes_val_half \\
        --cal-datasets cityscapes_val --test-datasets cityscapes_val \\
        --methods region_crc pixel_crc \\
        --alphas 0.02 0.03 0.05 0.07 0.10 0.15 0.20 0.30 0.40 0.50

Output: results/figures/pareto_area_matched.pdf and
        results/tables/pareto_area_matched.tex
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

from record.paths import results_dir
from record.reporting import (MODEL_LABELS, cell_means, fmt, latex_table,
                              load_experiments)

PALETTE = {"region_crc": "#0072B2", "pixel_crc": "#D55E00"}
LABELS = {"region_crc": "Region CRC (ours)", "pixel_crc": "Pixel CRC"}
COLUMN_W = 3.5

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def curves(frame: pd.DataFrame, rho: float) -> dict[str, pd.DataFrame]:
    """Mean (area, FNR) per requested level, one frame per method."""
    sub = frame[np.isclose(frame["rho"], rho)]
    means = cell_means(sub, ["method", "alpha"])
    out = {}
    for method in ("region_crc", "pixel_crc"):
        m = means[means["method"] == method].sort_values("marked_area")
        if not m.empty:
            out[method] = m
    return out


def interpolate_at(curve: pd.DataFrame, areas: np.ndarray) -> np.ndarray:
    """FNR at given marked areas; NaN outside the measured range.

    No extrapolation: a method is only compared where it was actually run.
    """
    x = curve["marked_area"].to_numpy(float)
    y = curve["region_fnr"].to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2:
        return np.full_like(areas, np.nan, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    out = np.interp(areas, x, y, left=np.nan, right=np.nan)
    out[(areas < x[0]) | (areas > x[-1])] = np.nan
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="p1_pareto*",
                        help="experiment-name glob for the dense-alpha run")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--n-grid", type=int, default=6,
                        help="number of common marked-area points to tabulate")
    args = parser.parse_args()

    frame = load_experiments(args.pattern)
    models = sorted(frame["model"].unique())
    rows: list[list[str]] = []

    fig, axes = plt.subplots(1, len(models), figsize=(COLUMN_W * len(models), 2.6),
                             squeeze=False)
    for ax, model in zip(axes[0], models):
        per_model = curves(frame[frame["model"] == model], args.rho)
        if len(per_model) < 2:
            print(f"WARNING: {model} has fewer than two methods; skipping")
            continue

        for method, curve in per_model.items():
            ax.plot(curve["marked_area"], curve["region_fnr"], "o-",
                    color=PALETTE[method], label=LABELS[method], markersize=3,
                    linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("marked area (fraction of pixels)")
        ax.set_ylabel(f"region FNR ($\\rho={args.rho}$)")
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.legend(frameon=False)

        # Common area range where BOTH methods were measured.
        lo = max(c["marked_area"].min() for c in per_model.values())
        hi = min(c["marked_area"].max() for c in per_model.values())
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            print(f"WARNING: {model} curves do not overlap in marked area; "
                  "no area-matched comparison is possible")
            continue
        grid = np.geomspace(lo, hi, args.n_grid)
        region = interpolate_at(per_model["region_crc"], grid)
        pixel = interpolate_at(per_model["pixel_crc"], grid)
        ax.axvspan(lo, hi, color="0.9", zorder=0)

        for area, r, p in zip(grid, region, pixel):
            better = "region" if r < p else ("pixel" if p < r else "tie")
            rows.append([MODEL_LABELS.get(model, model),
                         fmt(area, 4), fmt(r, 3), fmt(p, 3),
                         fmt(p - r, 3), better])
        n_region_wins = int(np.sum(region < pixel))
        print(f"{model}: region CRC lower FNR at {n_region_wins}/{len(grid)} "
              f"matched areas in [{lo:.4f}, {hi:.4f}]")

    fig_dir = results_dir("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "pareto_area_matched.pdf"
    fig.savefig(fig_path)
    plt.close(fig)

    header = "Model & Area & Region CRC & Pixel CRC & $\\Delta$ & Lower FNR"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    caption = (f"Area-matched comparison at $\\rho={args.rho}$. Both methods are "
               "swept over the requested level; the reported FNRs are linearly "
               "interpolated onto a common marked-area grid over the range in "
               "which both were measured, with no extrapolation. $\\Delta$ is "
               "pixel minus region, so positive favors region CRC.")
    table = latex_table(body, caption, "tab:pareto", "lccccc", header)
    tab_dir = results_dir("tables")
    tab_dir.mkdir(parents=True, exist_ok=True)
    tab_path = tab_dir / "pareto_area_matched.tex"
    tab_path.write_text(table)

    manifest = tab_dir / "PARETO_MANIFEST.json"
    manifest.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pattern": args.pattern, "rho": args.rho,
        "models": models, "n_rows": len(rows),
    }, indent=1))
    print(f"wrote {fig_path}")
    print(f"wrote {tab_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
