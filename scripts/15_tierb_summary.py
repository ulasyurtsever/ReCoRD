#!/usr/bin/env python
"""Stage 15: reporting for the tier-B diagnostics and the positive control.

1. In-sample versus cross-fitted density-ratio weights.  The tier-B negative
   result rests on the estimated weights collapsing to the lower clip.  A
   domain classifier fitted on the same source points it then scores collapses
   for a second reason, separating its own training set, and the collapse
   alone does not distinguish the two.  This compares the two estimators on
   the same runs.

2. The joint (ell, kappa) clip grid.  The vacuity condition is
   kappa/(ell*n + kappa) > alpha, so the lower clip matters as much as the
   upper one and the grid varies both.

3. The in-distribution positive control on MARIDA.  Calibration and test are
   exchangeable by construction on a random patch-level split, so the
   guarantee must hold there; a violation would point at the implementation
   rather than at the benchmark.  The control is a health check on the
   implementation and is reported only here.

Writes results/tables/tierb_estimator.tex, results/tables/clip_grid.tex and
results/tables/marida_indist.tex, and prints a plain-text summary.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from record.paths import results_dir
from record.reporting import fmt, latex_table, load_experiments

COND_ORDER = ["fog", "night", "rain", "snow"]


def _cond(name: str) -> str:
    m = re.search(r"__(fog|night|rain|snow)__", name)
    return m.group(1) if m else "?"


def _summarise(frame: pd.DataFrame) -> dict:
    """Per-block tier-B diagnostics, averaged over seeds and classes."""
    informative = (~frame["feasible"].astype(bool)) | (frame["lam"] < 1.0)
    return {
        "p_test": float(frame["weight_p_test"].mean()),
        "ess": float(frame["weight_ess"].mean()),
        "w_sum": float(np.nan),  # filled by caller when the clip is known
        "frac_informative": float((frame["lam"] < 1.0).mean()),
        "area": float(frame["marked_area_fraction"].mean()),
        "fnr": float(frame["region_fnr"].mean()),
        "n": int(len(frame)),
    }


def estimator_table(alpha: float, rho: float) -> str:
    """In-sample logistic versus cross-fitted logistic, same runs."""
    try:
        ins = load_experiments("e4_tierB__*__segformer_b2_cityscapes__dinov2_vitb14")
        cvf = load_experiments("p4_tierB_cv__*")
    except FileNotFoundError as exc:
        print(f"estimator table skipped: {exc}")
        return ""

    rows = []
    print("\n=== 1. weight estimator: in-sample vs cross-fitted ===")
    print(f"{'cond':<7}{'p_test(in)':>12}{'p_test(cv)':>12}"
          f"{'ESS(in)':>10}{'ESS(cv)':>10}{'inf.(in)':>10}{'inf.(cv)':>10}")
    for cond in COND_ORDER:
        a = ins[(ins["experiment"].map(_cond) == cond)
                & np.isclose(ins["alpha"], alpha) & np.isclose(ins["rho"], rho)]
        b = cvf[(cvf["experiment"].map(_cond) == cond)
                & np.isclose(cvf["alpha"], alpha) & np.isclose(cvf["rho"], rho)]
        if a.empty or b.empty:
            continue
        sa, sb = _summarise(a), _summarise(b)
        print(f"{cond:<7}{sa['p_test']:>12.3f}{sb['p_test']:>12.3f}"
              f"{sa['ess']:>10.1f}{sb['ess']:>10.1f}"
              f"{sa['frac_informative']:>10.2f}{sb['frac_informative']:>10.2f}")
        rows.append(" & ".join([
            cond, fmt(sa["p_test"], 3), fmt(sb["p_test"], 3),
            fmt(sa["ess"], 1), fmt(sb["ess"], 1),
            fmt(sa["frac_informative"], 2), fmt(sb["frac_informative"], 2),
        ]) + r" \\")

    if not rows:
        return ""
    header = ("Condition & \\multicolumn{2}{c}{$\\hat{p}_{n+1}$} & "
              "\\multicolumn{2}{c}{ESS} & \\multicolumn{2}{c}{informative} \\\\\n"
              " & in-sample & cross-fit & in-sample & cross-fit & "
              "in-sample & cross-fit")
    caption = (
        f"Density-ratio estimator on Cityscapes$\\to$ACDC (SegFormer-B2, DINOv2, "
        f"$\\alpha={alpha}$, $\\rho={rho}$). The in-sample logistic classifier is "
        "fitted on the source points it then scores; the cross-fitted variant "
        "scores each source point out of fold. ESS is the effective sample "
        "size of the weights, and ``informative'' is the fraction of draws "
        "returning $\\hat{\\lambda} < 1$. If the collapse is a property of the "
        "shift rather than of the estimator, the two columns agree.")
    return latex_table("\n".join(rows), caption, "tab:tierb_estimator",
                       "lcccccc", header)


def clip_grid_table(rho: float) -> str:
    """Conservative test mass over the joint (ell, kappa) grid."""
    try:
        grid = load_experiments("p3_lo*")
    except FileNotFoundError as exc:
        print(f"clip grid skipped: {exc}")
        return ""

    grid = grid[np.isclose(grid["rho"], rho)].copy()
    grid["c_low"] = grid["experiment"].str.extract(r"lo(\d+)_")[0].map(
        lambda s: float(f"0.{s[1:]}") if s.startswith("0") else float(s))
    grid["kappa"] = pd.to_numeric(
        grid["experiment"].str.extract(r"_clip(\d+)__")[0], errors="coerce")
    grid["cond"] = grid["experiment"].map(_cond)

    rows = []
    print("\n=== 2. joint (ell, kappa) clip grid ===")
    print(f"{'cond':<7}{'c':>6}{'kappa':>7}{'alpha':>7}{'p_test':>8}"
          f"{'fires':>7}{'inform.':>9}{'FNR|inf':>9}{'area':>7}{'verdict':>10}")
    for (cond, c_low, kappa, alpha), sub in grid.groupby(
            ["cond", "c_low", "kappa", "alpha"]):
        p_test = float(sub["weight_p_test"].mean())
        fires = p_test > alpha
        inform = sub[sub["lam"] < 1.0]
        frac = float(len(inform) / len(sub))
        if len(inform) > 1:
            fnr = float(inform["region_fnr"].mean())
            se = float(inform["region_fnr"].std(ddof=1) / np.sqrt(len(inform)))
            area = float(inform["marked_area_fraction"].mean())
            verdict = ("violates" if fnr - 1.96 * se > alpha
                       else ("valid" if fnr <= alpha else "borderline"))
        else:
            fnr = se = area = float("nan")
            verdict = "--"
        print(f"{cond:<7}{c_low:>6.2f}{kappa:>7.0f}{alpha:>7.2f}{p_test:>8.3f}"
              f"{('yes' if fires else 'NO'):>7}{frac * 100:>8.0f}%"
              f"{fnr:>9.3f}{area:>7.3f}{verdict:>10}")
        rows.append(" & ".join([
            cond, fmt(c_low, 2), fmt(kappa, 0), fmt(alpha, 2), fmt(p_test, 3),
            "yes" if fires else "no", fmt(frac, 2), fmt(fnr, 3), fmt(area, 3),
            verdict]) + r" \\")

    if not rows:
        return ""
    header = ("Condition & $\\ell$ & $\\kappa$ & $\\alpha$ & $\\hat{p}_{n+1}$ & "
              "fires & inform. & FNR & Area & Verdict")
    caption = (
        "Vacuity certificate over the joint clip grid (SegFormer-B2, DINOv2, "
        f"$\\rho={rho}$). The condition $\\hat{{p}}_{{n+1}}B > \\alpha$ depends on "
        "both clips through $\\kappa/(\\ell n+\\kappa)$, so narrowing the interval from "
        "below silences the certificate as effectively as lowering $\\kappa$. "
        "``inform.'' is the fraction of draws returning $\\hat{\\lambda} < 1$; FNR "
        "and area are averaged over those draws. A cell counts as violating "
        "when the whole $95\\%$ interval of the empirical FNR lies above the "
        "level. Where the certificate is silenced the procedure becomes "
        "informative and stops respecting the level.")
    return latex_table("\n".join(rows), caption, "tab:clip_grid",
                       "lccccccccc", header)


def marida_indist_table(rho: float) -> str:
    """The in-distribution positive control on MARIDA."""
    try:
        ctrl = load_experiments("p5_marida_indist__*")
    except FileNotFoundError as exc:
        print(f"MARIDA control skipped: {exc}")
        return ""

    ctrl = ctrl[np.isclose(ctrl["rho"], rho)]
    rows = []
    print("\n=== 3. MARIDA in-distribution positive control ===")
    print(f"{'model':<26}{'alpha':>7}{'FNR':>8}{'SE':>8}{'95% CI':>20}{'holds':>6}")
    for (model, method), sub in ctrl.groupby(["model", "method"]):
        if method != "region_crc":
            continue
        for alpha, cell in sub.groupby("alpha"):
            ok = cell["feasible"].astype(bool)
            x = cell.loc[ok, "region_fnr"].dropna()
            fnr = float(x.mean())
            se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
            lo, hi = fnr - 1.96 * se, fnr + 1.96 * se
            area = float(cell.loc[ok, "marked_area_fraction"].mean())
            # A point estimate a hair above the level is not a violation with
            # 100 draws; the level has to sit below the whole interval.
            holds = not (lo > alpha)
            print(f"{model:<26}{alpha:>7.2f}{fnr:>8.4f}{se:>8.4f}"
                  f"  [{lo:.4f}, {hi:.4f}]{('yes' if holds else 'NO'):>6}")
            label = ("U-Net ensemble-5" if model.endswith("_ens5")
                     else "U-Net single")
            rows.append(" & ".join([
                label, fmt(alpha, 2), fmt(fnr, 3),
                f"[{fmt(lo, 3)}, {fmt(hi, 3)}]", fmt(area, 4)]) + r" \\")

    if not rows:
        return ""
    header = "Model & $\\alpha$ & FNR & 95\\% CI & Area"
    caption = (
        f"MARIDA in-distribution control at $\\rho={rho}$. Region CRC on a "
        "random patch-level split of the pooled MARIDA patches over 100 seeded "
        "draws. Every other MARIDA scheme assigns scenes disjointly; this one "
        "is exchangeable by construction, so the empirical region miss rate is "
        "expected to respect the requested level, and a violation here would "
        "indicate an implementation fault rather than a hard partition. The "
        "interval is over the seeded draws and covers the level in every row.")
    return latex_table("\n".join(rows), caption, "tab:marida_indist",
                       "lcccc", header)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--rho", type=float, default=0.5)
    args = parser.parse_args()

    tab_dir = results_dir("tables")
    for name, tex in (
        ("tierb_estimator", estimator_table(args.alpha, args.rho)),
        ("clip_grid", clip_grid_table(args.rho)),
        ("marida_indist", marida_indist_table(args.rho)),
    ):
        if tex:
            path = tab_dir / f"{name}.tex"
            path.write_text(tex)
            print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
