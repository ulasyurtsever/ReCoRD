#!/usr/bin/env python
"""Stage 8: LaTeX tables from stage-5 experiment CSVs.

Pure arithmetic over ``results/experiments/``; writes one ``.tex`` file per
table under ``results/tables/`` plus a manifest. The ``.tex`` output is the
reported form of each table, so a change here changes every reported value.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from record.paths import results_dir
from record.reporting import (MODEL_LABELS, cell_means, fmt, infeasibility,
                              latex_table, load_experiments, parse_axis_fields)

ALPHAS = [0.05, 0.10, 0.20]
CITYSCAPES_MODELS = [
    "segformer_b2_cityscapes", "segformer_b5_cityscapes",
    "mask2former_swinb_cityscapes", "segformer_b2_cityscapes_mcdrop8",
]


def _rho_note(df: pd.DataFrame) -> str:
    """Capture level of a single-rho frame, as a caption clause.

    Averaging the two capture levels can hide a violation at one behind a
    compliant value at the other, so these tables report one level and say
    which.
    """
    vals = sorted(set(np.round(df["rho"].dropna().unique(), 3)))
    if len(vals) == 1:
        return f" Capture level $\\rho={vals[0]}$."
    return ""


def table_validity_cityscapes(df: pd.DataFrame) -> str:
    """T1: in-distribution validity/efficiency per model and alpha."""
    sub = df[(df.block == "e1") & (df.method == "region_crc")]
    cells = cell_means(sub, ["model", "alpha"])
    rows = []
    for model in CITYSCAPES_MODELS:
        parts = [MODEL_LABELS[model]]
        for alpha in ALPHAS:
            row = cells[(cells.model == model) & (cells.alpha == alpha)]
            if row.empty:
                parts += ["--", "--"]
            else:
                parts += [fmt(row.region_fnr.iloc[0]), fmt(row.marked_area.iloc[0])]
        rows.append(" & ".join(parts) + r" \\")
    header = ("Model & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{$\alpha={a:g}$}}" for a in ALPHAS))
    subheader = " & " + " & ".join(["FNR & Area"] * len(ALPHAS)) + r" \\"
    body = subheader + "\n" + r"\midrule" + "\n" + "\n".join(rows)
    return latex_table(
        body,
        "In-distribution region-level risk control on Cityscapes.",
        "tab:validity_cityscapes", "l" + "cc" * len(ALPHAS), header)


def table_baselines(df: pd.DataFrame) -> str:
    """T2: method comparison at alpha=0.2 (region FNR / area), Cityscapes."""
    sub = df[(df.block == "e1") & (df.alpha == 0.20)]
    cells = cell_means(sub, ["model", "method", "rho"])
    # x16 is dilation CRC: the argmax mask dilated by a calibrated radius
    # (stage 30 tables, run on "<model>__dilation"). It is the fourth panel's
    # requested geometric baseline and sits under each model's block.
    extra = df[df.block.isin(["x4", "x5", "x16"]) & (df.alpha == 0.20)]
    extra_cells = cell_means(extra, ["model", "method", "rho"]) if len(extra) else None
    methods = ["argmax", "heuristic", "pixel_crc", "region_crc"]
    labels = {"argmax": "Argmax", "heuristic": "Uncorrected",
              "pixel_crc": "Pixel CRC", "region_crc": "Region CRC (ours)"}
    # Baseline rows sourced from the x4/x5 runs: LAC-style global threshold
    # (pixel-coverage target) and the temperature-scaled posterior variants.
    extra_rows = [
        ("lac_global", "", "LAC (marginal)"),
        ("lac_classcond", "", "LAC (class-cond.)"),
        ("heuristic", "_tempscaled", "Uncorrected (temp.)"),
        ("region_crc", "_tempscaled", "Region CRC (temp.)"),
        ("region_crc", "__dilation", "Dilation CRC"),
    ]

    def fetch(frame, model, method, rho):
        if frame is None:
            return None
        row = frame[(frame.model == model) & (frame.method == method)
                    & (frame.rho == rho)]
        return None if row.empty else row

    rows = []
    emitted: set[str] = set()
    for model in CITYSCAPES_MODELS:
        for i, method in enumerate(methods):
            parts = [MODEL_LABELS[model] if i == 0 else "", labels[method]]
            for rho in (0.1, 0.5):
                row = fetch(cells, model, method, rho)
                if row is None:
                    parts += ["--", "--"]
                else:
                    parts += [fmt(row.region_fnr.iloc[0]), fmt(row.marked_area.iloc[0])]
            rows.append(" & ".join(parts) + r" \\")
        for method, suffix, label in extra_rows:
            found = {rho: fetch(extra_cells, model + suffix, method, rho)
                     for rho in (0.1, 0.5)}
            if all(v is None for v in found.values()):
                continue
            emitted.add(method)
            parts = ["", label]
            for rho in (0.1, 0.5):
                row = found[rho]
                if row is None:
                    parts += ["--", "--"]
                else:
                    parts += [fmt(row.region_fnr.iloc[0]), fmt(row.marked_area.iloc[0])]
            rows.append(" & ".join(parts) + r" \\")
        rows.append(r"\addlinespace")
    header = (r"Model & Method & \multicolumn{2}{c}{$\rho=0.1$} & "
              r"\multicolumn{2}{c}{$\rho=0.5$}")
    subheader = " & & " + " & ".join(["FNR & Area"] * 2) + r" \\"
    body = subheader + "\n" + r"\midrule" + "\n" + "\n".join(rows[:-1])
    # The caption describes the rows that were actually written. The
    # class-conditional LAC arm is produced only by a stage-5 run made with
    return latex_table(
        body,
        "Method comparison at $\\alpha=0.2$ on Cityscapes (in-distribution), "
        "at both capture levels.",
        "tab:baselines", "llcccc", header, size="scriptsize", colsep="2.5pt")


def table_breakdown(df: pd.DataFrame) -> str:
    """T3: shift breakdown on ACDC: source-calibrated FNR vs alpha."""
    sub = df[(df.block == "e2") & (df.method == "region_crc")]
    cells = cell_means(sub, ["model", "condition", "alpha"])
    conds = ["fog", "night", "rain", "snow"]
    rows = []
    for model in CITYSCAPES_MODELS:
        parts = [MODEL_LABELS[model]]
        for alpha in ALPHAS:
            for cond in conds:
                row = cells[(cells.model == model) & (cells.condition == cond)
                            & (cells.alpha == alpha)]
                val = row.region_fnr.iloc[0] if not row.empty else np.nan
                area = row.marked_area.iloc[0] if not row.empty else np.nan
                cell = fmt(val)
                if not np.isnan(val) and val > alpha:
                    cell = rf"\textbf{{{cell}}}"
                # A compliant cell bought by marking the whole image is not a
                # success, and at alpha=0.05 most of this table is that case.
                if not np.isnan(area) and area > 0.99:
                    cell += "$^{\\ast}$"
                parts.append(cell)
        rows.append(" & ".join(parts) + r" \\")
    header = "Model & " + " & ".join(
        rf"\multicolumn{{4}}{{c}}{{$\alpha={a:g}$}}" for a in ALPHAS)
    subheader = " & " + " & ".join(
        " & ".join(c[:1].upper() + c[1:] for c in conds) for _ in ALPHAS) + r" \\"
    body = subheader + "\n" + r"\midrule" + "\n" + "\n".join(rows)
    return latex_table(
        body,
        "Region FNR on ACDC conditions under source (Cityscapes) "
        "calibration.",
        "tab:breakdown", "l" + "cccc" * len(ALPHAS), header, star=True)


def table_tier_a(df: pd.DataFrame) -> str:
    """T4: tier-A recovery and applicability vs n_t (Cityscapes->ACDC)."""
    sub = df[(df.block == "e3") & (df.method == "region_crc")]
    # All four Cityscapes models. The third panel (2026-09-02, M4) had
    # restricted the table to the SegFormer variants because Mask2Former's
    # nine cells were the degenerate lambda_max solution on the uniform
    # threshold grid; on the log-tail grid (2026-09-08) the same checkpoint
    # calibrates to interior thresholds in every cell, so the restriction has
    # no basis and the table matches the validity count of the text, which was
    # always over all four models.
    feas = sub[sub.feasible.astype(bool)]
    # The infeasibility column is class-balanced. Averaging FNR and area over
    # raw feasible rows instead would weight whichever class still has feasible
    # draws, and at the tight levels that is a single class. Both columns are
    # therefore averaged per class first, then over classes.
    percls = cell_means(feas, ["n_target", "class_name", "alpha"])
    counts = feas.groupby(["n_target", "alpha"]).size()
    infeas = infeasibility(sub, ["n_target", "class_name", "alpha"])
    rows = []
    for nt in (25, 50, 100):
        parts = [str(nt)]
        for alpha in ALPHAS:
            row = percls[(percls.n_target == nt) & np.isclose(percls.alpha, alpha)]
            n_draws = int(counts.get((nt, alpha), 0))
            if row.empty:
                parts += ["--", "--"]
            else:
                parts += [fmt(row.region_fnr.mean()), fmt(row.marked_area.mean())]
            inf = infeas[(infeas.n_target == nt) & (infeas.alpha == alpha)]
            rate = inf.infeasible_rate.mean()
            # A rate printed as 1.00 while an FNR is printed beside it would
            # imply that FNR came from no draws at all; bound it instead.
            cell = r"$>$0.99" if 0.995 <= rate < 1.0 else fmt(rate, 2)
            parts.append(rf"{cell} ({n_draws})")
        rows.append(" & ".join(parts) + r" \\")
    header = "$n_t$ & " + " & ".join(
        rf"\multicolumn{{3}}{{c}}{{$\alpha={a:g}$}}" for a in ALPHAS)
    # "Inf." reads as infinity or inference before it reads as infeasible.
    subheader = " & " + " & ".join(["FNR & Area & Infeas.\\ ($n$)"] * len(ALPHAS)) + r" \\"
    body = subheader + "\n" + r"\midrule" + "\n" + "\n".join(rows)
    return latex_table(
        body,
        "Tier A on ACDC: exact recalibration from $n_t$ labeled target "
        "images.",
        # The feasible-draw counts widen the table past one IEEE column.
        "tab:tier_a", "l" + "ccc" * len(ALPHAS), header, star=True)



def table_marida_tier_a(df: pd.DataFrame) -> str:
    """T12: tier A on the held-out MARIDA spring (fifth referee panel).

    Rows are the labeled-target budget crossed with the draw scheme: patches
    drawn uniformly from the spring test group, and whole acquisition scenes
    assigned to the calibration side so that no test patch shares a scene
    with a calibration patch. Columns as in Table VI; the single critical
    class makes the class balancing a no-op. Marked area is also given in
    hectares of a 6.55 km^2 patch, the unit a photointerpreter budgets in.
    """
    sub = df[(df.block == "h4") & (df.method == "region_crc")].copy()
    if sub.empty:
        return latex_table(r"\multicolumn{10}{c}{h4 runs absent} \\",
                           "Tier A on held-out MARIDA spring.", "tab:marida_tier_a",
                           "l" + "ccc" * len(ALPHAS), "", star=True)
    sub["scene_disjoint"] = sub.experiment.str.contains("scenedisjoint")
    feas = sub[sub.feasible.astype(bool)]
    percls = cell_means(feas, ["n_target", "scene_disjoint", "class_name", "alpha"])
    counts = feas.groupby(["n_target", "scene_disjoint", "alpha"]).size()
    infeas = infeasibility(sub, ["n_target", "scene_disjoint", "class_name", "alpha"])
    rows = []
    for nt in (25, 50):
        for sd, label in ((False, "patches"), (True, "scene-disjoint")):
            parts = [f"{nt}, {label}"]
            for alpha in ALPHAS:
                row = percls[(percls.n_target == nt) & (percls.scene_disjoint == sd)
                             & np.isclose(percls.alpha, alpha)]
                n_draws = int(counts.get((nt, sd, alpha), 0))
                if row.empty or n_draws == 0:
                    parts += ["--", "--"]
                else:
                    area = float(row.marked_area.mean())
                    parts += [fmt(float(row.region_fnr.mean())),
                              f"{fmt(area)} ({655.36 * area:.0f}\,ha)"]
                inf = infeas[(infeas.n_target == nt) & (infeas.scene_disjoint == sd)
                             & (infeas.alpha == alpha)]
                rate = float(inf.infeasible_rate.mean())
                cell = r"$>$0.99" if 0.995 <= rate < 1.0 else ("1.00" if rate >= 1.0 else fmt(rate, 2))
                parts.append(rf"{cell} ({n_draws})")
            rows.append(" & ".join(parts) + r" \\")
    header = "$n_t$, draw & " + " & ".join(
        rf"\multicolumn{{3}}{{c}}{{$\alpha={a:g}$}}" for a in ALPHAS)
    subheader = " & " + " & ".join(["FNR & Area & Infeas.\\ ($n$)"] * len(ALPHAS)) + r" \\"
    body = subheader + "\n" + r"\midrule" + "\n" + "\n".join(rows)
    return latex_table(
        body,
        "Tier A on held-out MARIDA spring: recalibration from $n_t$ labeled "
        "spring patches.",
        "tab:marida_tier_a", "l" + "ccc" * len(ALPHAS), header, star=True)


def table_loveda(df: pd.DataFrame) -> str:
    """T6: LoveDA validity and directional breakdown."""
    l1 = cell_means(df[(df.block == "l1") & (df.method == "region_crc")],
                    ["model", "alpha"])
    l2 = cell_means(df[(df.block == "l2") & (df.method == "region_crc")],
                    ["condition", "alpha"])
    rows = []
    for model, label in (("segformer_b2_loveda_urban", "Urban (in-domain)"),
                         ("segformer_b2_loveda_rural", "Rural (in-domain)")):
        parts = [label]
        for alpha in ALPHAS:
            row = l1[(l1.model == model) & (l1.alpha == alpha)]
            parts += [fmt(row.region_fnr.iloc[0]), fmt(row.marked_area.iloc[0])]
        rows.append(" & ".join(parts) + r" \\")
    rows.append(r"\addlinespace")
    for cond, label in (("urban2rural", r"Urban$\to$Rural"),
                        ("rural2urban", r"Rural$\to$Urban")):
        parts = [label]
        for alpha in ALPHAS:
            row = l2[(l2.condition == cond) & (l2.alpha == alpha)]
            val = row.region_fnr.iloc[0]
            cell = fmt(val)
            if val > alpha:
                cell = rf"\textbf{{{cell}}}"
            parts += [cell, fmt(row.marked_area.iloc[0])]
        rows.append(" & ".join(parts) + r" \\")
    header = "Setting & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{$\alpha={a:g}$}}" for a in ALPHAS)
    subheader = " & " + " & ".join(["FNR & Area"] * len(ALPHAS)) + r" \\"
    body = subheader + "\n" + r"\midrule" + "\n" + "\n".join(rows)
    return latex_table(
        body,
        "LoveDA: in-domain validity and cross-domain breakdown under source "
        "calibration.",
        "tab:loveda", "l" + "cc" * len(ALPHAS), header)


def table_marida(df: pd.DataFrame) -> str:
    """T7: MARIDA official split and the held-out tile and season axes.

    Each held-out axis has its own model, fitted without any patch of that
    region or season and without the calibration group, so a row reads as the
    behaviour of a detector meeting that axis for the first time. Only the
    official axis carries both a single model and a deep ensemble.

    A cell is set in bold when the whole scene-level bootstrap interval lies
    above the level. Marking cells by the point estimate alone would promote
    differences these partitions cannot resolve: a held-out tile can contain
    as few as three acquisition scenes.
    """
    def _cells(sub, label):
        parts = [label]
        for alpha in ALPHAS:
            row = sub[sub.alpha == alpha]
            if row.empty:
                parts.append("--")
                continue
            val = float(row.region_fnr.iloc[0])
            lo = float(row.fnr_boot_lo.iloc[0]) if "fnr_boot_lo" in row else float("nan")
            cell = fmt(val)
            if lo == lo and lo > alpha:
                cell = rf"\textbf{{{cell}}}"
            parts.append(cell)
        return " & ".join(parts) + r" \\"

    def _means(sub, keys):
        agg = cell_means(sub, keys)
        boot = sub.groupby(keys, dropna=False)[["fnr_boot_lo", "fnr_boot_hi"]].mean()
        return agg.merge(boot.reset_index(), on=keys, how="left")

    rows = []
    h1 = _means(df[(df.block == "h1") & (df.method == "region_crc")],
                ["model", "alpha"])
    for model in ("marida_unet_official_holdout", "marida_unet_official_holdout_ens5"):
        sub = h1[h1.model == model]
        if sub.empty:
            continue
        suffix = "ensemble-5" if model.endswith("_ens5") else "single"
        rows.append(_cells(sub, f"Official split, U-Net {suffix}"))

    rows.append(r"\addlinespace")
    h2 = _means(df[(df.block == "h2") & (df.method == "region_crc")],
                ["region", "alpha"])
    for region in sorted(h2.region.dropna().unique()):
        rows.append(_cells(h2[h2.region == region], f"Held-out tile {region}"))

    rows.append(r"\addlinespace")
    h3 = _means(df[(df.block == "h3") & (df.method == "region_crc")],
                ["season", "alpha"])
    for season in ("winter", "spring", "summer", "autumn"):
        sub = h3[h3.season == season]
        if sub.empty:
            continue
        rows.append(_cells(sub, f"Held-out season ({season})"))

    header = "Axis & " + " & ".join(rf"$\alpha={a:g}$" for a in ALPHAS)
    body = "\n".join(rows)
    return latex_table(
        body,
        "MARIDA marine-debris region FNR under region-level CRC.",
        "tab:marida", "lccc", header)


def table_ablations(df: pd.DataFrame) -> str:
    """T8: shared-threshold and size-weighted-loss ablations (Cityscapes)."""
    x1 = df[(df.block == "x1") & df.experiment.str.contains("segformer|mask2former")]
    per = cell_means(x1[(x1.method == "region_crc")], ["model", "alpha"])
    shared = cell_means(
        x1[(x1.method == "shared_crc") & (x1.class_name != "max_over_classes")],
        ["model", "alpha"])
    shared_max = cell_means(
        x1[(x1.method == "shared_crc") & (x1.class_name == "max_over_classes")],
        ["model", "alpha"])
    rows = []
    for model in CITYSCAPES_MODELS:
        parts = [MODEL_LABELS[model]]
        for alpha in (0.10, 0.20):
            p = per[(per.model == model) & (per.alpha == alpha)]
            s = shared[(shared.model == model) & (shared.alpha == alpha)]
            m = shared_max[(shared_max.model == model) & (shared_max.alpha == alpha)]
            parts += [fmt(p.marked_area.iloc[0], 2), fmt(s.marked_area.iloc[0], 2),
                      fmt(m.region_fnr.iloc[0])]
        rows.append(" & ".join(parts) + r" \\")
    header = ("Model & " + " & ".join(
        rf"\multicolumn{{3}}{{c}}{{$\alpha={a:g}$}}" for a in (0.1, 0.2)))
    subheader = (" & " + " & ".join(
        [r"Area\textsubscript{per} & Area\textsubscript{sh} & FNR\textsubscript{max}"] * 2)
        + r" \\")
    body = subheader + "\n" + r"\midrule" + "\n" + "\n".join(rows)
    return latex_table(
        body,
        "Shared-threshold ablation on Cityscapes.",
        "tab:ablations", "l" + "ccc" * 2, header, size="scriptsize", colsep="2.5pt")


CRITICAL_CLASSES = ["person", "rider", "bicycle"]
# The full model labels overflow an IEEE column at twelve rows by eight
# columns; the caption names the table they abbreviate.
SHORT_MODEL_LABELS = {
    "segformer_b2_cityscapes": "B2",
    "segformer_b5_cityscapes": "B5",
    "mask2former_swinb_cityscapes": "M2F",
    "segformer_b2_cityscapes_mcdrop8": "B2-MC",
}


def table_perclass(df: pd.DataFrame) -> str:
    """T2: the in-distribution matrix resolved per class, at both capture levels.

    The other Cityscapes tables average the three critical classes, which is a
    weaker check than the guarantee: a class mean below the level does not
    imply every class is. This table is the 72-cell claim itself, and marking
    the cells whose selected threshold covers essentially the whole image also
    makes the efficiency caveat checkable.
    """
    sub = df[(df.block == "e1") & (df.method == "region_crc")]
    cells = cell_means(sub, ["model", "class_name", "alpha", "rho"])
    rows, n_cells, n_over, n_full = [], 0, 0, 0
    for model in CITYSCAPES_MODELS:
        for k, cname in enumerate(CRITICAL_CLASSES):
            label = SHORT_MODEL_LABELS[model] if k == 0 else ""
            parts = [label, cname]
            for rho in (0.5, 0.1):
                for alpha in ALPHAS:
                    row = cells[(cells.model == model)
                                & (cells.class_name == cname)
                                & np.isclose(cells.alpha, alpha)
                                & np.isclose(cells.rho, rho)]
                    if row.empty:
                        parts.append("--")
                        continue
                    val = float(row.region_fnr.iloc[0])
                    area = float(row.marked_area.iloc[0])
                    n_cells += 1
                    n_over += val > alpha
                    n_full += area > 0.99
                    parts.append(fmt(val) + ("$^{\\ast}$" if area > 0.99 else ""))
            rows.append(" & ".join(parts) + r" \\")
        if model != CITYSCAPES_MODELS[-1]:
            rows.append(r"\addlinespace")
    header = ("Model & Class & "
              + " & ".join(rf"\multicolumn{{3}}{{c}}{{$\rho={r:g}$}}"
                           for r in (0.5, 0.1)))
    subheader = (" & & " + " & ".join(rf"$\alpha={a:g}$" for a in ALPHAS) * 1
                 + " & " + " & ".join(rf"$\alpha={a:g}$" for a in ALPHAS)
                 + r" \\")
    body = subheader + "\n" + r"\midrule" + "\n" + "\n".join(rows)
    caption = "In-distribution region FNR on Cityscapes, resolved per class."
    if n_over:
        caption += f" WARNING: {n_over} cells exceed their level."
    return latex_table(body, caption, "tab:perclass",
                       "ll" + "ccc" * 2, header, size="scriptsize", colsep="2pt")


def table_loveda_perclass(df: pd.DataFrame) -> str:
    """T7: the LoveDA in-domain cells resolved per class, at both capture levels.

    The class average of Table~\ref{tab:loveda} hides two facts a reader needs:
    which class carries the exceedances, and that water at the tight levels is
    met only by marking the whole image.
    """
    sub = df[(df.block == "l1") & (df.method == "region_crc")]
    cells = cell_means(sub, ["model", "class_name", "alpha", "rho"])
    classes = sorted(cells.class_name.unique())
    rows, n_cells, n_over, n_full = [], 0, 0, 0
    models = [("segformer_b2_loveda_urban", "Urban"),
              ("segformer_b2_loveda_rural", "Rural")]
    for model, label in models:
        for k, cname in enumerate(classes):
            parts = [label if k == 0 else "", cname]
            for rho in (0.5, 0.1):
                for alpha in ALPHAS:
                    row = cells[(cells.model == model)
                                & (cells.class_name == cname)
                                & np.isclose(cells.alpha, alpha)
                                & np.isclose(cells.rho, rho)]
                    if row.empty:
                        parts.append("--")
                        continue
                    val = float(row.region_fnr.iloc[0])
                    area = float(row.marked_area.iloc[0])
                    n_cells += 1
                    n_over += val > alpha
                    n_full += area > 0.99
                    cell = fmt(val)
                    if val > alpha:
                        cell = rf"\textbf{{{cell}}}"
                    parts.append(cell + ("$^{\\ast}$" if area > 0.99 else ""))
            rows.append(" & ".join(parts) + r" \\")
        if model != models[-1][0]:
            rows.append(r"\addlinespace")
    header = ("Domain & Class & "
              + " & ".join(rf"\multicolumn{{3}}{{c}}{{$\rho={r:g}$}}"
                           for r in (0.5, 0.1)))
    subheader = (" & & " + " & ".join(rf"$\alpha={a:g}$" for a in ALPHAS)
                 + " & " + " & ".join(rf"$\alpha={a:g}$" for a in ALPHAS)
                 + r" \\")
    body = subheader + "\n" + r"\midrule" + "\n" + "\n".join(rows)
    caption = "LoveDA in-domain region FNR resolved per class."
    return latex_table(body, caption, "tab:loveda_perclass",
                       "ll" + "ccc" * 2, header, size="scriptsize", colsep="2pt")


def table_perclass_area(df: pd.DataFrame) -> str:
    """T8: marked area for the per-class cells of Table~\ref{tab:perclass}.

    The efficiency spread across classes is the operator's real cost and
    appears otherwise only in prose.
    """
    segformer = [m for m in CITYSCAPES_MODELS if m.startswith("segformer")]
    sub = df[(df.block == "e1") & (df.method == "region_crc")]
    cells = cell_means(sub, ["model", "class_name", "alpha", "rho"])
    rows, lo, hi = [], np.inf, -np.inf
    for model in CITYSCAPES_MODELS:
        for k, cname in enumerate(CRITICAL_CLASSES):
            label = SHORT_MODEL_LABELS[model] if k == 0 else ""
            parts = [label, cname]
            for rho in (0.5, 0.1):
                for alpha in ALPHAS:
                    row = cells[(cells.model == model)
                                & (cells.class_name == cname)
                                & np.isclose(cells.alpha, alpha)
                                & np.isclose(cells.rho, rho)]
                    if row.empty:
                        parts.append("--")
                        continue
                    area = float(row.marked_area.iloc[0])
                    if (np.isclose(alpha, 0.20) and np.isclose(rho, 0.5)
                            and model in segformer):
                        lo, hi = min(lo, area), max(hi, area)
                    parts.append(fmt(area))
            rows.append(" & ".join(parts) + r" \\")
        if model != CITYSCAPES_MODELS[-1]:
            rows.append(r"\addlinespace")
    header = ("Model & Class & "
              + " & ".join(rf"\multicolumn{{3}}{{c}}{{$\rho={r:g}$}}"
                           for r in (0.5, 0.1)))
    subheader = (" & & " + " & ".join(rf"$\alpha={a:g}$" for a in ALPHAS)
                 + " & " + " & ".join(rf"$\alpha={a:g}$" for a in ALPHAS)
                 + r" \\")
    body = subheader + "\n" + r"\midrule" + "\n" + "\n".join(rows)
    caption = "Mean marked fraction of the image, resolved per class."
    return latex_table(body, caption, "tab:perclass_area",
                       "ll" + "ccc" * 2, header, size="scriptsize", colsep="2pt")


TABLES = {
    "validity_cityscapes": table_validity_cityscapes,
    "perclass": table_perclass,
    "baselines": table_baselines,
    "breakdown": table_breakdown,
    "tier_a": table_tier_a,
    "loveda": table_loveda,
    "loveda_perclass": table_loveda_perclass,
    "perclass_area": table_perclass_area,
    "marida": table_marida,
    "marida_tier_a": table_marida_tier_a,
    "ablations": table_ablations,
}


# table_baselines reports both capture levels side by side and filters rho
# itself. The other builders take a single capture level, because averaging
# the two can hide a violation at one behind a compliant value at the other.
BOTH_RHO_TABLES = {"baselines", "perclass", "loveda_perclass",
                   "perclass_area"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", type=float, default=0.5,
                        help="capture level for the single-rho tables; the "
                             "main article reports 0.5 and the released "
                             "artifacts carry 0.1 under a -rho0 suffix")
    parser.add_argument("--suffix", default="",
                        help="appended to every output filename, so a second "
                             "capture level can be generated without "
                             "overwriting the main tables; a value starting "
                             "with a dash must be passed as --suffix=-rho0")
    args = parser.parse_args()

    df = parse_axis_fields(load_experiments())
    # Canonical duplicate of the b2 in-distribution run: keep the matrix copy.
    df = df[df.experiment != "e1_cs_indist_b2"]
    single = df[np.isclose(df.rho, args.rho)]
    if single.empty:
        print(f"ERROR: no rows at rho={args.rho}")
        return 1
    out_dir = results_dir("tables")
    written = []
    for name, builder in TABLES.items():
        tex = builder(df if name in BOTH_RHO_TABLES else single)
        path = out_dir / f"{name}{args.suffix}.tex"
        path.write_text(tex)
        written.append(path.name)
        print(f"wrote {path}")
    manifest = {
        "tables": written,
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
