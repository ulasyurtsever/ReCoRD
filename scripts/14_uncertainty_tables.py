#!/usr/bin/env python
"""Stage 14: uncertainty columns for the validity tables.

Every reported cell is an estimate printed to three decimals, and a cell is
marked as a violation when it exceeds the nominal level.  With 100 seeded
draws, or with 223 ground-truth components in the whole MARIDA test split, some
of those exceedances are inside the noise.  This script
attaches an interval to each cell so the bolding can be justified rather than
asserted.

Two regimes, two intervals:

seeded runs (Cityscapes/ACDC, LoveDA)
    the draws are the replicates, so the standard error of the mean over seeds
    is the right summary; a cell is called a violation only when the level lies
    below the lower end of the mean +/- 1.96 SE interval.

fixed partitions (MARIDA)
    there is a single partition and no seed variance, so the interval comes
    from a bootstrap whose resampling unit is the acquisition scene.  A tile
    held out of training can rest on as few as three scenes, and the patches
    of one scene share illumination, sea state and annotator, so neither a
    binomial on the miss count nor a patch bootstrap would carry the right
    precision.

Output: results/tables/uncertainty_<block>.tex plus a printed summary of which
bolded cells survive their interval.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from record.paths import results_dir
from record.reporting import (MODEL_LABELS, fmt, latex_table, load_experiments,
                              parse_axis_fields)

Z = 1.959963984540054  # two-sided 95%


def seeded_cells(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean, SE over seeds, and a violation verdict per cell."""
    sub = frame[frame["feasible"].astype(bool) | (frame["method"] == "argmax")]
    grouped = sub.groupby(by, dropna=False)["region_fnr"]
    out = grouped.agg(mean="mean", sd="std", n="size").reset_index()
    out["se"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    out["lo"] = out["mean"] - Z * out["se"]
    out["hi"] = out["mean"] + Z * out["se"]
    # A violation only counts when the whole interval sits above the level.
    out["violates"] = out["lo"] > out["alpha"]
    out["borderline"] = (out["mean"] > out["alpha"]) & ~out["violates"]
    return out


def fixed_cells(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Scene-level bootstrap intervals for single-partition (MARIDA) cells.

    The reported risk is a mean over test images of the within-image miss
    fraction, but the images of one acquisition scene are not independent
    draws, so the scene is the resampling unit. A binomial interval on a miss
    count would also be a different estimand, since image component counts are
    unequal and the image mean times the component total is not a miss count.
    The interval is read from the ``fnr_boot_lo`` / ``fnr_boot_hi`` columns
    that ``05_run_experiments.py`` writes when it is run with
    ``--bootstrap-ci`` and ``--bootstrap-clusters``. The component tally is
    read from ``n_missed_components``, which counts components whose captured
    fraction falls below rho at the selected threshold; it is a different
    estimand from the image-averaged rate and is not derived from it.
    """
    rows = []
    for key, sub in frame.groupby(by, dropna=False):
        mean = float(sub["region_fnr"].mean())
        n_comp = sub["n_test_components"].dropna()
        n = int(n_comp.iloc[0]) if len(n_comp) else 0
        missed = sub["n_missed_components"].dropna()
        if not len(missed):
            raise SystemExit(
                "MARIDA CSVs carry no n_missed_components column; re-run the "
                "MARIDA blocks with scripts/run_marida_holdout.sh so the "
                "component tally is measured rather than inferred from the "
                "image-averaged rate")
        k = int(missed.iloc[0])
        if "fnr_boot_lo" in sub.columns and sub["fnr_boot_lo"].notna().any():
            lo = float(sub["fnr_boot_lo"].mean())
            hi = float(sub["fnr_boot_hi"].mean())
        else:
            raise SystemExit(
                "MARIDA CSVs carry no bootstrap columns; re-run the MARIDA "
                "blocks with scripts/run_marida_holdout.sh, which passes "
                "--bootstrap-ci and --bootstrap-clusters")
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(mean=mean, n_components=n, k_missed=k, lo=lo, hi=hi)
        rec["violates"] = lo > rec["alpha"]
        rec["borderline"] = (mean > rec["alpha"]) and not rec["violates"]
        rows.append(rec)
    return pd.DataFrame(rows)


def emit(cells: pd.DataFrame, keycols: list[str], caption: str, label: str,
         path, show_components: bool = False) -> None:
    """Write one uncertainty table.

    Bold marks a cell whose whole interval sits above the level; underline
    marks a cell above the level by less than its own interval.
    """
    cols = keycols + ["$\\alpha$", "FNR", "95\\% CI"]
    if show_components:
        cols += ["\\#\\,cmp.", "miss", "pooled"]
    header = " & ".join(cols)

    body_rows = []
    for _, r in cells.iterrows():
        cell = fmt(r["mean"], 3)
        if r["violates"]:
            cell = f"\\textbf{{{cell}}}"
        elif r["borderline"]:
            cell = f"\\underline{{{cell}}}"
        vals = [str(r[c]).replace("_", r"\_") for c in keycols]
        vals += [fmt(r["alpha"], 2), cell,
                 f"[{fmt(r['lo'], 3)}, {fmt(r['hi'], 3)}]"]
        if show_components:
            pooled = (r["k_missed"] / r["n_components"]
                      if r["n_components"] else float("nan"))
            vals += [str(int(r["n_components"])), str(int(r["k_missed"])),
                     fmt(pooled, 3)]
        body_rows.append(" & ".join(vals) + r" \\")

    colspec = "l" * len(keycols) + "c" * (len(cols) - len(keycols))
    # The MARIDA table carries three extra count columns beside long axis
    # labels, so it needs tighter padding to stay inside the IEEE column.
    colsep = "2pt" if show_components else "3pt"
    path.write_text(latex_table("\n".join(body_rows), caption, label,
                                colspec, header, colsep=colsep))
    print(f"wrote {path}  ({len(body_rows)} cells)")


def marida_axis_label(experiment: str) -> str:
    """Human-readable row label, matching the rows of the MARIDA table."""
    if experiment.startswith("h1_"):
        kind = "ensemble-5" if experiment.endswith("_ens5") else "single"
        return f"Official split, {kind}"
    parts = experiment.split("__")
    if experiment.startswith("h2_"):
        return f"Held-out tile {parts[1]}"
    if experiment.startswith("h3_"):
        return f"Held-out season ({parts[1]})"
    return experiment.replace("_", r"\_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", type=float, default=0.5)
    args = parser.parse_args()

    tab_dir = results_dir("tables")

    # --- seeded: in-distribution Cityscapes -------------------------------
    ind = parse_axis_fields(load_experiments("e1_indist*"))
    ind = ind[np.isclose(ind["rho"], args.rho) & (ind["method"] == "region_crc")]
    cells = seeded_cells(ind, ["model", "class_name", "alpha"])
    cells["model"] = cells["model"].map(lambda m: MODEL_LABELS.get(m, m))
    emit(cells, ["model", "class_name"],
         f"In-distribution region FNR at $\\rho={args.rho}$ with standard "
         "errors over 100 seeded calibration draws. Bold marks cells whose "
         "whole 95\\% interval lies above the nominal level; underline marks "
         "cells above the level by less than the interval, which the seed "
         "variance cannot separate from noise.",
         "tab:uncertainty_indist", tab_dir / "uncertainty_indist.tex")

    # --- seeded: ACDC breakdown -------------------------------------------
    brk = parse_axis_fields(load_experiments("e2_break*"))
    brk = brk[np.isclose(brk["rho"], args.rho) & (brk["method"] == "region_crc")]
    cells = seeded_cells(brk, ["model", "condition", "alpha"])
    cells["model"] = cells["model"].map(lambda m: MODEL_LABELS.get(m, m))
    emit(cells, ["model", "condition"],
         f"Region FNR on ACDC at $\\rho={args.rho}$ under source calibration, "
         "with standard errors over seeded draws. Bold and underline as in "
         "the previous table.",
         "tab:uncertainty_breakdown", tab_dir / "uncertainty_breakdown.tex")

    # --- fixed partitions: MARIDA -----------------------------------------
    try:
        mar = parse_axis_fields(load_experiments("h[123]_*"))
    except FileNotFoundError:
        print("no MARIDA experiments found; skipping the fixed-partition table")
        return 0
    mar = mar[np.isclose(mar["rho"], args.rho) & (mar["method"] == "region_crc")]
    missing = [c for c in ("n_test_components", "n_missed_components")
               if c not in mar.columns]
    if missing:
        print(f"WARNING: {', '.join(missing)} missing from the MARIDA CSVs. "
              "Re-run scripts/run_marida_holdout.sh so the component columns "
              "are present; the table is skipped.")
        return 0
    cells = fixed_cells(mar, ["experiment", "alpha"])
    cells["Axis"] = cells["experiment"].map(marida_axis_label)
    # Row order follows Table~\ref{tab:marida} so the two can be read
    # cell by cell: official split, then tiles, then seasons in calendar order.
    season_rank = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}

    def _rank(exp: str) -> tuple:
        if exp.startswith("h1_"):
            return (0, 1 if exp.endswith("_ens5") else 0, "")
        if exp.startswith("h2_"):
            return (1, 0, exp.split("__")[1])
        return (2, season_rank.get(exp.split("__")[1], 9), "")

    cells = (cells.assign(_r=cells["experiment"].map(_rank))
                  .sort_values(["_r", "alpha"], kind="stable")
                  .drop(columns="_r").reset_index(drop=True))
    emit(cells, ["Axis"],
         f"MARIDA region FNR at $\\rho={args.rho}$ with 95\\% scene-level "
         "bootstrap intervals. These partitions are fixed, so the uncertainty "
         "is not seed variance. The resampling unit is the acquisition scene: "
         "the patches of one scene share illumination, sea state and "
         "annotator, and a held-out tile can hold as few as three scenes, so "
         "resampling patches would report a precision the design does not "
         "carry; intervals are 95\\% percentile intervals over 4000 cluster "
         "resamples. Bold marks the cells whose whole interval lies above the "
         "level. \\#\\,comp.\\ is the number of ground-truth components in the "
         "test group, missed is how many of them fall below the capture level "
         "at the selected threshold, and pooled is their ratio. The FNR column "
         "is the controlled quantity and weights images equally; the pooled "
         "column weights components equally and is not the quantity the "
         "guarantee bounds. The two differ whenever images carry unequal "
         "component counts.",
         "tab:uncertainty_marida", tab_dir / "uncertainty_marida.tex",
         show_components=True)  # narrow columns: the row labels are long

    n_bold = int(cells["violates"].sum())
    n_border = int(cells["borderline"].sum())
    print(f"MARIDA: {n_bold} cells violate with the whole interval above the "
          f"level; {n_border} exceed the level but not beyond the interval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
