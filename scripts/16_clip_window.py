#!/usr/bin/env python
"""Stage 16: the tier-B operating window as a function of the clip ratio.

The conservative test mass depends on the clip interval only through the ratio
of its two ends:

    p_hat_{n+1} = kappa / (ell*n + kappa) = 1 / (n*ell/kappa + 1),

so (ell, kappa) = (0.05, 2) and (0.50, 20) are the same experiment.  Sweeping
the ceiling at a fixed floor therefore pins the ratio high, where the
certificate fires for every setting; sweeping the ratio exposes the whole
frontier instead.

Widen the clip and the certificate fires: nothing informative comes back.
Narrow it too far and the conservative charge no longer covers the shift, so
informative thresholds come back but stop respecting the level.  Between the
two the procedure returns an informative and valid threshold in half to three
quarters of draws.

Output: results/figures/clip_window.pdf
"""

from __future__ import annotations

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from record.paths import results_dir
from record.reporting import load_experiments

COLUMN_W = 3.5
PALETTE = {0.05: "#0072B2", 0.10: "#E69F00", 0.20: "#009E73"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def _clip_frame(rho: float, cond: str) -> pd.DataFrame:
    """Every tier-B clip run for one condition, with c_low and kappa columns."""
    frames = []
    for pattern, fixed_c in (("p3_lo*", None), ("c1_clip*", 0.05)):
        try:
            f = load_experiments(pattern)
        except FileNotFoundError:
            continue
        f = f[np.isclose(f["rho"], rho)].copy()
        f = f[f["experiment"].str.contains(f"__{cond}__")]
        if f.empty:
            continue
        if fixed_c is None:
            f["c_low"] = f["experiment"].str.extract(r"lo0(\d+)_")[0].map(
                lambda s: float(f"0.{s}"))
            f["kappa"] = pd.to_numeric(
                f["experiment"].str.extract(r"_clip(\d+)__")[0])
        else:
            f["c_low"] = fixed_c
            f["kappa"] = pd.to_numeric(
                f["experiment"].str.extract(r"clip(\d+)__")[0])
        frames.append(f)
    if not frames:
        raise FileNotFoundError("no tier-B clip runs found")
    grid = pd.concat(frames, ignore_index=True)
    # The c1_clip* runs repeat the lo=0.05 arm of p3_lo005_clip* verbatim;
    # keyed on the clip interval rather than the file name, the repeat drops
    # out instead of being averaged with itself.
    grid = grid.drop_duplicates(
        subset=["c_low", "kappa", "seed", "class_name", "alpha", "rho",
                "method"])
    grid["ratio"] = grid["kappa"] / grid["c_low"]
    return grid


def collect(rho: float, cond: str) -> pd.DataFrame:
    """One row per (clip ratio, alpha): test mass, informativeness, risk."""
    grid = _clip_frame(rho, cond)
    rows = []
    for (ratio, alpha), sub in grid.groupby(["ratio", "alpha"]):
        # The theorem bounds the expectation over ALL draws; an infeasible
        # draw returns lambda_max and contributes zero loss. Conditioning on
        # the informative draws would delete exactly those zeros.
        loss = sub["region_fnr"].fillna(0.0)
        mean = float(loss.mean())
        se = float(loss.std(ddof=1) / np.sqrt(len(loss))) if len(loss) > 1 else 0.0
        rows.append(dict(
            ratio=float(ratio), alpha=float(alpha),
            p_test=float(sub["weight_p_test"].mean()),
            informative=float((sub["lam"] < 1.0).mean()),
            risk=mean, lo=mean - 1.96 * se, hi=mean + 1.96 * se,
            area=float(sub["marked_area_fraction"].mean()),
            n=len(sub)))
    return pd.DataFrame(rows).sort_values(["alpha", "ratio"])


def collect_pairs(rho: float, cond: str) -> pd.DataFrame:
    """One row per (lower clip, upper clip): the conservative test mass.

    ``collect`` keys on the ratio, which is what panels (b) and (c) sweep. The
    invariance claim of panel (a) is that two clip intervals sharing a ratio
    give the same test mass, so that panel has to keep the pairs apart: three
    ratios in the released grid are reached by two distinct intervals each
    (40, 20 and 200), and collapsing them first would hide exactly the
    agreement the panel is there to show.
    """
    grid = _clip_frame(rho, cond)
    rows = []
    for (c_low, kappa), sub in grid.groupby(["c_low", "kappa"]):
        rows.append(dict(c_low=float(c_low), kappa=float(kappa),
                         ratio=float(kappa) / float(c_low),
                         p_test=float(sub["weight_p_test"].mean()),
                         n=len(sub)))
    return pd.DataFrame(rows).sort_values(["ratio", "c_low"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--condition", default="fog")
    args = parser.parse_args()

    t = collect(args.rho, args.condition)
    fig, axes = plt.subplots(1, 3, figsize=(COLUMN_W * 2.05, 2.2))

    ax = axes[0]
    pairs = collect_pairs(args.rho, args.condition)
    ax.plot(pairs["ratio"], pairs["p_test"], "-", color="0.25", lw=1.2,
            zorder=1)
    ax.plot(pairs["ratio"], pairs["p_test"], "o", color="0.25", ms=3.4,
            mfc="none", zorder=2)
    for a, col in PALETTE.items():
        ax.axhline(a, ls=":", lw=0.9, color=col)
        # Anchored inside the axes: at the right-hand data edge the label lands
        # in the gap between panels, on top of the next panel's y-axis label.
        ax.text(0.02, a, f"{a}", color=col, va="bottom", ha="left", fontsize=6,
                transform=ax.get_yaxis_transform())
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"clip ratio $\kappa/\ell$")
    ax.set_ylabel(r"$\hat{p}_{n+1}$")
    ax.set_title("(a) conservative test mass")

    ax = axes[1]
    for a, sub in t.groupby("alpha"):
        ax.plot(sub["ratio"], sub["informative"], "o-", ms=3, lw=1.2,
                color=PALETTE.get(a, "0.4"), label=rf"$\alpha={a:g}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"clip ratio $\kappa/\ell$")
    ax.set_ylabel(r"fraction with $\hat{\lambda}<1$")
    ax.set_title("(b) informative draws")
    ax.legend(frameon=False)

    ax = axes[2]
    for a, sub in t.groupby("alpha"):
        col = PALETTE.get(a, "0.4")
        ax.plot(sub["ratio"], sub["risk"], "o-", ms=3, lw=1.2, color=col,
                label=rf"$\alpha={a:g}$")
        ax.fill_between(sub["ratio"], sub["lo"], sub["hi"], color=col, alpha=0.2)
        ax.axhline(a, ls=":", lw=0.9, color=col)
    ax.set_xscale("log")
    ax.set_xlabel(r"clip ratio $\kappa/\ell$")
    ax.set_ylabel(r"$\mathbb{E}[L]$ over all draws")
    ax.set_title("(c) risk against the level")

    fig.tight_layout()

    fig_dir = results_dir("figures")
    path = fig_dir / "clip_window.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")

    print("\nratio  alpha  p_test  inform.   E[L]            95% CI    area  verdict")
    for _, r in t.iterrows():
        v = ("violates" if r.lo > r.alpha
             else ("valid" if r.hi <= r.alpha else "borderline"))
        print(f"{r.ratio:>5.0f}{r.alpha:>7.2f}{r.p_test:>8.3f}{r.informative:>9.2f}"
              f"{r.risk:>7.3f}  [{r.lo:.3f},{r.hi:.3f}]{r.area:>8.3f}  {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
