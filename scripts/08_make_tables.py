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
        "In-distribution region-level risk control on Cityscapes. "
        "Mean region FNR and marked-area fraction over 100 calibration draws. "
        "Each cell averages the three critical classes; the guarantee is per "
        "class, and Section~\\ref{sec:exp_indist} reports the per-class cells." + _rho_note(df),
        "tab:validity_cityscapes", "l" + "cc" * len(ALPHAS), header)


def table_baselines(df: pd.DataFrame) -> str:
    """T2: method comparison at alpha=0.2 (region FNR / area), Cityscapes."""
    sub = df[(df.block == "e1") & (df.alpha == 0.20)]
    cells = cell_means(sub, ["model", "method", "rho"])
    extra = df[df.block.isin(["x4", "x5"]) & (df.alpha == 0.20)]
    extra_cells = cell_means(extra, ["model", "method", "rho"]) if len(extra) else None
    methods = ["argmax", "heuristic", "pixel_crc", "region_crc"]
    labels = {"argmax": "Argmax", "heuristic": "Uncorrected",
              "pixel_crc": "Pixel CRC", "region_crc": "Region CRC (ours)"}
    # Baseline rows sourced from the x4/x5 runs: LAC-style global threshold
    # (pixel-coverage target) and the temperature-scaled posterior variants.
    extra_rows = [
        ("lac_global", "", "LAC (global)"),
        ("heuristic", "_tempscaled", "Uncorrected (temp.)"),
        ("region_crc", "_tempscaled", "Region CRC (temp.)"),
    ]

    def fetch(frame, model, method, rho):
        if frame is None:
            return None
        row = frame[(frame.model == model) & (frame.method == method)
                    & (frame.rho == rho)]
        return None if row.empty else row

    rows = []
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
    return latex_table(
        body,
        "Method comparison at $\\alpha=0.2$ on Cityscapes (in-distribution), "
        "at both capture levels. LAC denotes a single global threshold tuned "
        "to an all-class pixel-coverage target, the marginal variant of the "
        "least-ambiguous set-valued classifier; the temperature-scaled rows "
        "use one scalar fitted by negative log-likelihood on the seed-0 "
        "calibration list and reused across the draws; refitting it on each "
        "draw's own list moves every tempered cell by at most $0.0006$ "
        "(Section~\\ref{sec:exp_indist}). FNR is the "
        "mean region-miss loss over test images containing the class, "
        "averaged over the three critical classes; Area is "
        "the mean marked fraction over all test images.",
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
                cell = fmt(val)
                if not np.isnan(val) and val > alpha:
                    cell = rf"\textbf{{{cell}}}"
                parts.append(cell)
        rows.append(" & ".join(parts) + r" \\")
    header = "Model & " + " & ".join(
        rf"\multicolumn{{4}}{{c}}{{$\alpha={a:g}$}}" for a in ALPHAS)
    subheader = " & " + " & ".join(
        " & ".join(c[:1].upper() + c[1:] for c in conds) for _ in ALPHAS) + r" \\"
    body = subheader + "\n" + r"\midrule" + "\n" + "\n".join(rows)
    return latex_table(
        body,
        "Region FNR on ACDC conditions under source (Cityscapes) calibration. "
        "Bold entries violate the nominal level: the in-distribution "
        "guarantee does not survive the shift. "
        "Each cell averages the three critical classes; the guarantee is per "
        "class, and Section~\\ref{sec:exp_indist} reports the per-class cells." + _rho_note(df),
        "tab:breakdown", "l" + "cccc" * len(ALPHAS), header, star=True)


def table_tier_a(df: pd.DataFrame) -> str:
    """T4: tier-A recovery and applicability vs n_t (Cityscapes->ACDC)."""
    sub = df[(df.block == "e3") & (df.method == "region_crc")]
    feas = sub[sub.feasible.astype(bool)]
    cells = cell_means(feas, ["n_target", "alpha"])
    infeas = infeasibility(sub, ["n_target", "class_name", "alpha"])
    rows = []
    for nt in (25, 50, 100):
        parts = [str(nt)]
        for alpha in ALPHAS:
            row = cells[(cells.n_target == nt) & (cells.alpha == alpha)]
            if row.empty:
                parts += ["--", "--"]
            else:
                parts += [fmt(row.region_fnr.iloc[0]), fmt(row.marked_area.iloc[0])]
            inf = infeas[(infeas.n_target == nt) & (infeas.alpha == alpha)]
            rate = inf.infeasible_rate.mean()
            # A rate printed as 1.00 while an FNR is printed beside it would
            # imply that FNR came from no draws at all; bound it instead.
            parts.append(r"$>$0.99" if 0.995 <= rate < 1.0 else fmt(rate, 2))
        rows.append(" & ".join(parts) + r" \\")
    header = "$n_t$ & " + " & ".join(
        rf"\multicolumn{{3}}{{c}}{{$\alpha={a:g}$}}" for a in ALPHAS)
    subheader = " & " + " & ".join(["FNR & Area & Inf."] * len(ALPHAS)) + r" \\"
    body = subheader + "\n" + r"\midrule" + "\n" + "\n".join(rows)
    return latex_table(
        body,
        "Tier A on ACDC: exact recalibration from $n_t$ labeled target "
        "images. FNR and marked area are averaged over feasible draws only; "
        "Inf.\\ is the fraction of draws where $\\alpha$ is unattainable "
        "because too few of the $n_t$ images contain the class. Where Inf.\\ "
        "approaches one, the paired FNR and area rest on very few draws and "
        "are indicative only." + _rho_note(df),
        "tab:tier_a", "l" + "ccc" * len(ALPHAS), header)


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
        "calibration. Bold entries violate the nominal level; the shift is "
        "strongly directional. Each cell averages the two critical classes; "
        "Section~\\ref{sec:exp_loveda} resolves the in-domain cells per "
        "class." + _rho_note(df),
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
        "MARIDA marine-debris region FNR under region-level CRC. One model per "
        "held-out axis, fitted without that tile or season and without the "
        "calibration group; the official axis carries a single model and a "
        "five-member ensemble. The tile axis is a within-site hold-out "
        "(Section~\\ref{sec:setup}). Partitions are fixed, so cells are single "
        "measurements. Bold marks the cells whose whole 95\\% scene-level "
        "bootstrap interval lies above the level; intervals are in "
        "Table~\\ref{tab:uncertainty_marida}." + _rho_note(df),
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
        "Shared-threshold ablation on Cityscapes: mean marked area under "
        "independent per-class thresholds (Area\\textsubscript{per}) versus a "
        "single shared threshold (Area\\textsubscript{sh}), and the "
        "max-over-classes FNR the shared threshold controls. Coupling to the "
        "hardest class inflates the marked area by up to $3.0\\times$."
        + _rho_note(df),
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
    caption = (
        f"In-distribution region FNR on Cityscapes, resolved per class over "
        f"the {n_cells} cells behind the class averages of "
        f"Table~\\ref{{tab:validity_cityscapes}}. No cell exceeds its level. "
        f"$\\ast$ marks the {n_full} cells where the level is met only by "
        f"marking more than $99\\%$ of the image, so the bound holds but the "
        f"mask is uninformative. Model labels abbreviate those of "
        f"Table~\\ref{{tab:validity_cityscapes}}; means over 100 calibration "
        f"draws.")
    if n_over:
        caption += f" WARNING: {n_over} cells exceed their level."
    return latex_table(body, caption, "tab:perclass",
                       "ll" + "ccc" * 2, header, size="scriptsize", colsep="2pt")


TABLES = {
    "validity_cityscapes": table_validity_cityscapes,
    "perclass": table_perclass,
    "baselines": table_baselines,
    "breakdown": table_breakdown,
    "tier_a": table_tier_a,
    "loveda": table_loveda,
    "marida": table_marida,
    "ablations": table_ablations,
}


# table_baselines reports both capture levels side by side and filters rho
# itself. The other builders take a single capture level, because averaging
# the two can hide a violation at one behind a compliant value at the other.
BOTH_RHO_TABLES = {"baselines", "perclass"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", type=float, default=0.5,
                        help="capture level for the single-rho tables; the "
                             "main article reports 0.5 and the released "
                             "artifacts carry 0.1 under a -rho01 suffix")
    parser.add_argument("--suffix", default="",
                        help="appended to every output filename, so a second "
                             "capture level can be generated without "
                             "overwriting the main tables")
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
