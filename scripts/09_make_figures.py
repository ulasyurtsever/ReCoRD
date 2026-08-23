#!/usr/bin/env python
"""Stage 9: figures from stage-5 experiment CSVs.

Pure arithmetic over ``results/experiments/``; writes one PDF per figure
under ``results/figures/`` plus a manifest. Colors follow the Okabe--Ito
colorblind-safe palette; sizes target a single IEEE column.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from record.paths import results_dir
from record.reporting import (MODEL_LABELS, cell_means, infeasibility,
                              load_experiments, parse_axis_fields)

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
           "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
ALPHAS = [0.05, 0.10, 0.20]
CITYSCAPES_MODELS = [
    "segformer_b2_cityscapes", "segformer_b5_cityscapes",
    "mask2former_swinb_cityscapes", "segformer_b2_cityscapes_mcdrop8",
]
COLUMN_W = 3.5  # inches, IEEE single column

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def fig_validity(df: pd.DataFrame, path) -> None:
    """Empirical FNR vs nominal level across all in-distribution axes."""
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.6))
    axes_spec = [
        ("e1", "Cityscapes", "o"),
        ("l1", "LoveDA", "s"),
        ("h1", "MARIDA (official)", "^"),
    ]
    for (block, label, marker), color in zip(axes_spec, PALETTE):
        sub = df[(df.block == block) & (df.method == "region_crc")]
        cells = cell_means(sub, ["model", "class_name", "alpha", "rho"])
        # A deterministic offset: Python's hash() is salted per process, so a
        # hash-derived jitter would move the points between runs.
        digest = hashlib.sha256(block.encode()).digest()
        jitter = (digest[0] / 255.0 - 0.5) * 0.004
        ax.scatter(cells.alpha + jitter, cells.region_fnr, s=12,
                   marker=marker, color=color, label=label, alpha=0.8,
                   linewidths=0)
    lim = 0.30
    ax.plot([0, lim], [0, lim], color="black", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(0.02, lim); ax.set_ylim(-0.005, lim)
    ax.set_xticks(ALPHAS)
    ax.set_xlabel(r"nominal level $\alpha$")
    ax.set_ylabel("empirical region FNR")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path); plt.close(fig)


def fig_area_tradeoff(df: pd.DataFrame, path) -> None:
    """Marked-area cost of the guarantee per model (Cityscapes, in-dist)."""
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.6))
    sub = df[(df.block == "e1") & (df.method == "region_crc")]
    cells = cell_means(sub, ["model", "alpha"])
    for model, color in zip(CITYSCAPES_MODELS, PALETTE):
        rows = cells[cells.model == model].sort_values("alpha")
        ax.plot(rows.alpha, rows.marked_area, marker="o", ms=3.5,
                color=color, label=MODEL_LABELS[model])
    ax.set_xticks(ALPHAS)
    ax.set_yscale("log")
    ax.set_xlabel(r"nominal level $\alpha$")
    ax.set_ylabel("marked-area fraction (log)")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(path); plt.close(fig)


def fig_breakdown_heatmap(df: pd.DataFrame, path) -> None:
    """FNR / alpha violation ratio per model x condition at alpha = 0.2."""
    sub = df[(df.block == "e2") & (df.method == "region_crc")
             & (df.alpha == 0.20)]
    cells = cell_means(sub, ["model", "condition"])
    conds = ["fog", "night", "rain", "snow"]
    mat = np.full((len(CITYSCAPES_MODELS), len(conds)), np.nan)
    for i, model in enumerate(CITYSCAPES_MODELS):
        for j, cond in enumerate(conds):
            row = cells[(cells.model == model) & (cells.condition == cond)]
            if not row.empty:
                mat[i, j] = row.region_fnr.iloc[0] / 0.20
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.2))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0.0, vmax=2.2, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if mat[i, j] > 1.6 else "black")
    ax.set_xticks(range(len(conds)), [c.capitalize() for c in conds])
    ax.set_yticks(range(len(CITYSCAPES_MODELS)),
                  [MODEL_LABELS[m] for m in CITYSCAPES_MODELS])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label(r"FNR / $\alpha$ at $\alpha=0.2$")
    fig.savefig(path); plt.close(fig)


def fig_tier_a(df: pd.DataFrame, path) -> None:
    """Tier-A recovery and per-class applicability vs n_t."""
    sub = df[(df.block == "e3") & (df.method == "region_crc")]
    fig, axes = plt.subplots(1, 2, figsize=(2 * COLUMN_W, 2.4))

    feas = sub[sub.feasible.astype(bool)]
    cells = cell_means(feas, ["n_target", "alpha"])
    for alpha, color in zip(ALPHAS, PALETTE):
        rows = cells[cells.alpha == alpha].sort_values("n_target")
        axes[0].plot(rows.n_target, rows.region_fnr, marker="o", ms=3.5,
                     color=color, label=rf"$\alpha={alpha:g}$")
        axes[0].axhline(alpha, color=color, lw=0.7, ls="--", alpha=0.6)
    axes[0].set_xticks([25, 50, 100])
    axes[0].set_xlabel(r"labeled target images $n_t$")
    axes[0].set_ylabel("region FNR (feasible draws)")
    axes[0].legend(frameon=False)

    inf = infeasibility(sub, ["n_target", "class_name", "alpha"])
    inf = inf[inf.alpha == 0.10]
    classes = ["person", "rider", "bicycle"]
    x = np.arange(3)
    width = 0.25
    for k, (cls, color) in enumerate(zip(classes, PALETTE)):
        rows = inf[inf.class_name == cls].sort_values("n_target")
        axes[1].bar(x + (k - 1) * width, rows.infeasible_rate, width,
                    color=color, label=cls)
    axes[1].set_xticks(x, ["25", "50", "100"])
    axes[1].set_xlabel(r"labeled target images $n_t$")
    axes[1].set_ylabel(r"infeasible-draw rate ($\alpha=0.1$)")
    axes[1].legend(frameon=False)
    fig.savefig(path); plt.close(fig)


def fig_loveda_asymmetry(df: pd.DataFrame, path) -> None:
    """Directional breakdown on LoveDA."""
    sub = df[(df.block == "l2") & (df.method == "region_crc")]
    cells = cell_means(sub, ["condition", "alpha"])
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.4))
    x = np.arange(len(ALPHAS))
    width = 0.35
    for k, (cond, label, color) in enumerate((
            ("urban2rural", r"Urban$\to$Rural", PALETTE[3]),
            ("rural2urban", r"Rural$\to$Urban", PALETTE[0]))):
        rows = cells[cells.condition == cond].sort_values("alpha")
        ax.bar(x + (k - 0.5) * width, rows.region_fnr, width, color=color,
               label=label)
    for i, alpha in enumerate(ALPHAS):
        ax.hlines(alpha, i - 0.5 * width - 0.18, i + 0.5 * width + 0.18,
                  color="black", lw=1.0, ls="--")
    ax.set_xticks(x, [rf"$\alpha={a:g}$" for a in ALPHAS])
    ax.set_ylabel("region FNR (source-calibrated)")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path); plt.close(fig)


def fig_marida_axes(df: pd.DataFrame, path) -> None:
    """MARIDA held-out tile and season FNR at alpha = 0.1, with intervals.

    One model per axis, each fitted without its own region or season. The bars
    carry scene-level bootstrap intervals: a held-out tile can hold as few as
    three acquisition scenes, and without the intervals the bar heights would
    invite comparisons the partitions cannot support.
    """
    def _axis(block, key):
        sub = df[(df.block == block) & (df.method == "region_crc")
                 & (df.alpha == 0.10)]
        agg = cell_means(sub, [key])
        boot = sub.groupby(key, dropna=False)[["fnr_boot_lo", "fnr_boot_hi"]].mean()
        return agg.merge(boot.reset_index(), on=key, how="left")

    h2, h3 = _axis("h2", "region"), _axis("h3", "season")
    labels = list(h2.region) + [s.capitalize() for s in h3.season]
    values = np.array(list(h2.region_fnr) + list(h3.region_fnr))
    lo = np.array(list(h2.fnr_boot_lo) + list(h3.fnr_boot_lo))
    hi = np.array(list(h2.fnr_boot_hi) + list(h3.fnr_boot_hi))
    colors = [PALETTE[0]] * len(h2) + [PALETTE[2]] * len(h3)

    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.6))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors)
    ax.errorbar(x, values, yerr=[np.maximum(values - lo, 0), np.maximum(hi - values, 0)],
                fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    ax.axhline(0.10, color="black", lw=1.0, ls="--")
    ax.text(len(labels) - 0.4, 0.104, r"$\alpha=0.1$", fontsize=6, ha="right")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("region FNR")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[0]),
               plt.Rectangle((0, 0), 1, 1, color=PALETTE[2])]
    ax.legend(handles, ["held-out tile", "held-out season"], frameon=False)
    fig.savefig(path); plt.close(fig)


# The deployment settings of the triage study, in one place so that a setting
# keeps the same colour in both triage figures.
TRIAGE_SETTINGS = {
    "official": "MARIDA official",
    "region_16PCC": "Held-out 16PCC",
    "region_16PDC": "Held-out 16PDC",
    "region_16PEC": "Held-out 16PEC",
    "season_spring": "Held-out spring",
    "night_tierA50": r"ACDC night (tier A, $n_t{=}50$)",
}


def _triage_frame():
    """Triage curves at alpha=0.2, rho=0.5, with a setting column."""
    from record.reporting import load_triage

    tr = load_triage("x3_*_triage")
    tr = tr[(tr.alpha == 0.20) & (tr.rho == 0.5)
            & (tr.method == "region_crc")].copy()
    tr["setting"] = tr.experiment.str.extract(r"x3_triage__(\w+?)__")
    dupes = tr.groupby("setting").experiment.nunique()
    if (dupes > 1).any():
        raise ValueError(
            "more than one experiment per triage setting: "
            + str(dupes[dupes > 1].to_dict())
            + "; a curve would average distinct runs")
    return tr


def _permutation_band(alpha: float = 0.20, rho: float = 0.5):
    """Stage-27 permutation quantiles, or ``None`` if they were never computed.

    Returns ``{setting: (budgets, q05, q95)}`` in units of the no-review rate,
    read from the setting-level aggregate rows only: the per-class rows of the
    same file belong to the individual classes, not to the curve drawn here,
    and averaging the two together would mix two different aggregations.
    """
    path = results_dir("experiments") / "x12_triage_permutation_band.csv"
    if not path.exists():
        print(f"WARNING: {path.name} not found; drawing the triage baselines "
              "against the closed-form line alone, with no permutation band. "
              "Run scripts/27_triage_permutation_band.py to add it.")
        return None
    band = pd.read_csv(path)
    band = band[(band["class_name"] == "all_classes")
                & np.isclose(band["alpha"].astype(float), alpha)
                & np.isclose(band["rho"].astype(float), rho)]
    if band.empty:
        print(f"WARNING: {path.name} carries no aggregate rows at "
              f"alpha={alpha:g}, rho={rho:g}; drawing without the band")
        return None
    out = {}
    for setting, sub in band.groupby("setting"):
        sub = sub.sort_values("budget")
        out[str(setting)] = (sub["budget"].to_numpy(dtype=float),
                             sub["q05_rel"].to_numpy(dtype=float),
                             sub["q95_rel"].to_numpy(dtype=float))
    return out


def fig_triage_baselines(df: pd.DataFrame, path) -> None:
    """Marked-area ranking against the random floor and its sampling band.

    Under the oracle review model any ordering removes part of the risk, so an
    absolute reduction is not by itself evidence that a priority score works.
    What the score has to beat is a random ranking -- and the centre of that
    reference needs no simulation. With ``triage_curve`` reporting the
    unresolved loss of the unreviewed images divided by the full test-set
    size, a uniformly random order includes each image with probability beta,
    so

        E[residual(beta)] = (1 - beta) * residual(0),

    up to the integer rounding of the reviewed count (the review model takes
    floor(beta n) images, so the two agree whenever beta n is an integer and
    differ by less than 1/n otherwise; measured over the six settings the gap
    is at most 0.7% of the no-review rate). The single sampled permutation
    stored in the CSVs is one draw around that line and deviates from it by up
    to 29% of the no-review rate on the fixed MARIDA partitions, which is why
    the closed form is used here instead. Plotting the difference rather than
    the two rates puts every deployment setting on one axis: negative means
    the score orders images better than chance, and the zero line is the
    closed form.

    An expectation is not a scale, though. Five of the six settings are single
    MARIDA partitions with one realized permutation, and a curve sitting below
    zero there may be nothing but the spread of the random ranking itself. The
    shaded envelopes are that spread: the 5th-to-95th percentile of the
    residual over 1000 redrawn uniform permutations per setting and per class,
    from ``results/experiments/x12_triage_permutation_band.csv`` (stage 27),
    plotted in the same units and shaded in each setting's colour. A curve
    inside its own band is indistinguishable from chance at that budget; only
    where it leaves the band below is the advantage larger than permutation
    noise. When the stage-27 file is absent the figure falls back to the
    closed-form line alone and says so.
    """
    tr = _triage_frame()
    if "ranking" in tr.columns:
        tr = tr[tr["ranking"] == "area"]
    bands = _permutation_band()

    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.6))
    drawn = 0
    for (setting, label), color in zip(TRIAGE_SETTINGS.items(), PALETTE):
        area = tr[tr.setting == setting].groupby(
            "budget")["residual_region_fnr"].mean()
        if area.empty or 0.0 not in area.index:
            continue
        base = float(area.loc[0.0])
        if base <= 0:
            continue
        beta = area.index.values.astype(float)
        gap = area.values / base - (1.0 - beta)
        if bands and setting in bands:
            # Same subtraction as the curve, so the band is read on the same
            # axis: both are residuals in units of the no-review rate with the
            # closed-form line removed.
            b_beta, q05, q95 = bands[setting]
            ax.fill_between(b_beta, q05 - (1.0 - b_beta), q95 - (1.0 - b_beta),
                            color=color, alpha=0.13, lw=0, zorder=0)
        ax.plot(beta, gap, marker="o", ms=2.2, lw=1.1,
                color=color, label=label)
        drawn += 1
    if not drawn:
        print("no triage settings carry the marked-area ranking")
        plt.close(fig)
        return

    ax.axhline(0.0, color="black", lw=0.9)
    ax.annotate("random floor", xy=(ax.get_xlim()[1], 0.0),
                xytext=(-2, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=6)
    # Open a strip of empty axes below the curves so the legend does not sit
    # on top of the deepest one.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.30 * (hi - lo), hi)
    ax.set_xlabel(r"review budget $\beta$ (fraction of images)")
    ax.set_ylabel("marked area minus random\n(fraction of the no-review rate)")
    handles, labels = ax.get_legend_handles_labels()
    if bands:
        handles.append(plt.Rectangle((0, 0), 1, 1, color="0.45", alpha=0.30))
        labels.append("5-95% of random orders")
    ax.legend(handles, labels, frameon=False, fontsize=6, ncol=2,
              loc="lower left")
    fig.savefig(path); plt.close(fig)


def fig_triage(df: pd.DataFrame, path) -> None:
    """Residual system-level miss rate versus review budget (alpha=0.2)."""
    tr = _triage_frame()
    if "ranking" in tr.columns:
        # The deployable score. The random and oracle rankings are the subject
        # of the companion figure, not of these means.
        tr = tr[tr["ranking"] == "area"]
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.6))
    for (setting, label), color in zip(TRIAGE_SETTINGS.items(), PALETTE):
        sub = tr[tr.setting == setting].groupby("budget")["residual_region_fnr"].mean()
        if sub.empty:
            continue
        ax.plot(sub.index, sub.values, marker="o", ms=2.5, lw=1.1,
                color=color, label=label)
    ax.set_xlabel(r"review budget $\beta$ (fraction of images)")
    ax.set_ylabel("residual region miss rate")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.savefig(path); plt.close(fig)


FIGURES = {
    "validity_scatter": fig_validity,
    "area_tradeoff": fig_area_tradeoff,
    "breakdown_heatmap": fig_breakdown_heatmap,
    "tier_a": fig_tier_a,
    "triage_baselines": fig_triage_baselines,
    "loveda_asymmetry": fig_loveda_asymmetry,
    "marida_axes": fig_marida_axes,
    "triage_curves": fig_triage,
}


# fig_validity spans every capture level on purpose: the point of that
# scatter is that the whole matrix sits under the diagonal. Everything else
# reports one capture level, matching the tables.
ALL_RHO_FIGURES = {"validity_scatter"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", type=float, default=0.5,
                        help="capture level for the single-rho figures")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    df = parse_axis_fields(load_experiments())
    df = df[df.experiment != "e1_cs_indist_b2"]
    single = df[np.isclose(df.rho, args.rho)]
    if single.empty:
        print(f"ERROR: no rows at rho={args.rho}")
        return 1
    out_dir = results_dir("figures")
    written = []
    for name, builder in FIGURES.items():
        path = out_dir / f"{name}{args.suffix}.pdf"
        builder(df if name in ALL_RHO_FIGURES else single, path)
        written.append(path.name)
        print(f"wrote {path}")
    manifest = {
        "figures": written,
        "rho": args.rho,
        "n_experiment_rows": int(len(df)),
        "n_rows_at_rho": int(len(single)),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / f"MANIFEST{args.suffix}.json").write_text(
        json.dumps(manifest, indent=1))
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
