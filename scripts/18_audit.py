#!/usr/bin/env python
"""Regression suite over the released result files.

Recomputes every published quantity from ``results/experiments/*.csv`` on a
code path independent of the table and figure builders, and compares it with
the value the pipeline produced when those files were generated. A failure
means the pipeline has stopped reproducing a published value, which is the
failure mode a spot check misses.

Three verdicts per check:

    [OK]   id  quantity ............. recomputed value
    [FAIL] id  quantity ............. expected vs recomputed
    [NOTE] id  a condition that needs a human decision

Qualitative properties ("holds in every cell", "nonincreasing at every
budget", "no cell exceeds the level") are checked as strictly as the numeric
ones. Identifiers are stable across runs so a failure can be traced to a
single quantity.

Usage:
    python scripts/18_audit.py            # all sections
    python scripts/18_audit.py --only M   # one section
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import traceback

import numpy as np
import pandas as pd

from record.grid import FP_SUBGRID_INDICES, LAMBDA_GRID, fp_subgrid_slot
from record.paths import results_dir, splits_dir

LAM_MAX = 1000


def paths_root():
    return results_dir('tables').parent.parent

TOL = 5e-4          # a printed three-decimal value must agree to half a unit
# How far the realized test risk of a conformal threshold may sit above its
# nominal level before it stops being sampling noise. Used only where the
# claim is that a method lands ON its target, not that it never exceeds it.
PIXEL_TARGET_SLACK = 0.02
# How close the class-conditional LAC row has to sit to the pixel-CRC row
# before the two count as the same operating point. Wide enough for the
# threshold grid to land a step apart, far narrower than the gap to the
# marginal LAC row, which is above 0.7.
LAC_PIXEL_GAP = 0.02
N_FAIL = 0
N_OK = 0
N_NOTE = 0
SECTION = ""


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------
def load(pattern: str) -> pd.DataFrame:
    exp = results_dir("experiments")
    paths = sorted(glob.glob(str(exp / f"{pattern}.csv")))
    paths = [p for p in paths if not p.endswith("_triage.csv")]
    if not paths:
        raise FileNotFoundError(pattern)
    frames = []
    for p in paths:
        f = pd.read_csv(p)
        f["experiment"] = os.path.basename(p)[:-4]
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out["alpha"] = out["alpha"].astype(float)
    out["rho"] = out["rho"].astype(float)
    return out


def load_triage(pattern: str = "*_triage") -> pd.DataFrame:
    exp = results_dir("experiments")
    frames = []
    for p in sorted(glob.glob(str(exp / f"{pattern}.csv"))):
        f = pd.read_csv(p)
        f["experiment"] = os.path.basename(p)[: -len("_triage.csv")]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def sel(df, **kw):
    """Filter with float tolerance on alpha/rho."""
    out = df
    for k, v in kw.items():
        if v is None:
            continue
        if k in ("alpha", "rho"):
            out = out[np.isclose(out[k].astype(float), v)]
        elif isinstance(v, (list, tuple, set)):
            out = out[out[k].isin(list(v))]
        else:
            out = out[out[k] == v]
    return out


def feas(df):
    """Reporting convention: feasible draws only, argmax exempt."""
    if "feasible" not in df.columns:
        return df
    return df[df["feasible"].astype(bool) | (df["method"] == "argmax")]


def mean(df, col="region_fnr", feasible_only=True):
    d = feas(df) if feasible_only else df
    if d.empty:
        return float("nan")
    return float(d[col].mean())


def head(title):
    global SECTION
    SECTION = title
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


def ok(cid, text, detail=""):
    global N_OK
    N_OK += 1
    print(f"[OK]   {cid:>4}  {text}" + (f"  |  {detail}" if detail else ""))


def fail(cid, text, detail=""):
    global N_FAIL
    N_FAIL += 1
    print(f"[FAIL] {cid:>4}  {text}" + (f"  |  {detail}" if detail else ""))


def note(cid, text, detail=""):
    global N_NOTE
    N_NOTE += 1
    print(f"[NOTE] {cid:>4}  {text}" + (f"  |  {detail}" if detail else ""))


def near(cid, text, expected, measured, tol=TOL):
    if measured is None or (isinstance(measured, float) and np.isnan(measured)):
        fail(cid, text, f"expected {expected}, measured NaN")
    elif abs(measured - expected) <= tol:
        ok(cid, text, f"{measured:.4f}")
    else:
        fail(cid, text, f"expected {expected}, measured {measured:.4f}")


def rng_claim(cid, text, lo, hi, values, tol=TOL):
    """Check that the measured min and max match the reported range."""
    v = np.asarray([x for x in values if not np.isnan(x)], float)
    if v.size == 0:
        fail(cid, text, "no values")
        return
    mn, mx = v.min(), v.max()
    if abs(mn - lo) <= tol and abs(mx - hi) <= tol:
        ok(cid, text, f"[{mn:.4f}, {mx:.4f}]")
    else:
        fail(cid, text, f"paper [{lo}, {hi}], measured [{mn:.4f}, {mx:.4f}]")


def truth(cid, text, condition, detail=""):
    (ok if condition else fail)(cid, text, detail)


CS_MODELS = ["segformer_b2_cityscapes", "segformer_b5_cityscapes",
             "mask2former_swinb_cityscapes", "segformer_b2_cityscapes_mcdrop8"]
SEGF = ["segformer_b2_cityscapes", "segformer_b5_cityscapes",
        "segformer_b2_cityscapes_mcdrop8"]
ALPHAS = [0.05, 0.10, 0.20]
CONDS = ["fog", "night", "rain", "snow"]


def cellmean(df, keys, col="region_fnr", feasible_only=True):
    d = feas(df) if feasible_only else df
    return d.groupby(keys, dropna=False)[col].mean()


# --------------------------------------------------------------------------
# D/E. setup + in-distribution validity
# --------------------------------------------------------------------------
def section_indist():
    head("D/E  Setup and in-distribution validity")
    e1 = load("e1_indist__*")

    truth(43, "alpha grid is {0.05,0.1,0.2}",
          sorted(np.unique(np.round(e1.alpha, 3))) == [0.05, 0.1, 0.2],
          str(sorted(np.unique(np.round(e1.alpha, 3)))))
    truth(27, "rho grid is {0.1,0.5}",
          sorted(np.unique(np.round(e1.rho, 3))) == [0.1, 0.5],
          str(sorted(np.unique(np.round(e1.rho, 3)))))
    truth(44, "100 seeds per in-distribution experiment",
          e1.groupby("experiment").seed.nunique().eq(100).all(),
          str(e1.groupby("experiment").seed.nunique().to_dict()))
    truth(36, "critical classes are person/rider/bicycle",
          sorted(e1.class_name.unique()) == ["bicycle", "person", "rider"],
          str(sorted(e1.class_name.unique())))
    note(29, "n (calibration images containing the class), Cityscapes",
         f"{int(e1.n_cal_images.min())}-{int(e1.n_cal_images.max())} "
         "(26-440 across all three benchmarks)")
    truth(30, "lambda grid has 1001 points",
          int(e1.lam_index.max()) == 1000, f"max lam_index={int(e1.lam_index.max())}")

    # --- 1 / 51 / 245: 72 cells, no violation ---
    r = sel(e1, method="region_crc")
    cells = cellmean(r, ["model", "class_name", "alpha", "rho"])
    truth(51, "in-distribution matrix has exactly 72 cells",
          len(cells) == 72, f"{len(cells)} cells")
    # Abstract and Section IV-A: validity is met by marking almost everything
    # in part of the matrix, so the count of near-saturated cells is reported.
    areas = cellmean(r, ["model", "class_name", "alpha", "rho"],
                     col="marked_area_fraction")
    n_sat = int((areas.values > 0.99).sum())
    truth(53, "no in-distribution cell marks more than 99% of the image (log-tail grid)",
          n_sat == 0, f"{n_sat} cells")
    truth(53, "the largest per-class marked area is at most 29% (Mask2Former, person, a=0.05)",
          0.28 <= float(areas.values.max()) <= 0.29,
          f"max {float(areas.values.max()):.4f} at {areas.idxmax()}")
    lev = cells.index.get_level_values("alpha").values
    viol = cells[cells.values > lev + 1e-12]
    truth(1, "no in-distribution cell exceeds its level",
          len(viol) == 0,
          "worst margin %.4f" % float((cells.values - lev).max()) if len(viol) == 0
          else f"{len(viol)} violations: {viol.to_dict()}")

    # --- 52: upper end of the 95% seed interval below the level ---
    g = feas(r).groupby(["model", "class_name", "alpha", "rho"])["region_fnr"]
    m, s, n = g.mean(), g.std(ddof=1), g.size()
    hi = m + 1.959963984540054 * s / np.sqrt(n.clip(lower=1))
    lev = hi.index.get_level_values("alpha").values
    over = hi[hi.values > lev + 1e-12]
    lo_ = m - 1.959963984540054 * s / np.sqrt(n.clip(lower=1))
    exc = over.values - over.index.get_level_values("alpha").values
    truth(52, "41 of the 72 cells have the upper end of the 95% interval above "
              "the level, by less than 0.004",
          len(over) == 41 and float(exc.max()) < 0.004,
          f"{len(over)} cells, max excess {float(exc.max()):.4f}"
          if len(over) else "none")
    lev2 = lo_.index.get_level_values("alpha").values
    truth("52b", "no cell has a 95% LOWER end above the level (violation test)",
          bool((lo_.values <= lev2 + 1e-12).all()),
          "max lower end - alpha = %.4f" % float((lo_.values - lev2).max()))

    # --- 54 / 2 / 231: marked area 2.5-9% at alpha=0.2 ---
    a = [mean(sel(r, model=mo, alpha=0.20, rho=0.5), "marked_area_fraction")
         for mo in SEGF]
    rng_claim(54, "SegFormer marked area 1.4-2.4% at alpha=0.2", 0.0139, 0.0240, a)

    # --- 55: alpha=0.05 marks most of the image ---
    a05 = {mo: mean(sel(r, model=mo, alpha=0.05, rho=0.5), "marked_area_fraction")
           for mo in CS_MODELS}
    truth(55, "alpha=0.05 costs 3.9-4.9% of the image on the SegFormer variants",
          0.039 - 5e-4 <= min(a05[m] for m in SEGF) and max(a05[m] for m in SEGF) <= 0.049 + 5e-4,
          str({k: round(v, 3) for k, v in a05.items()}))
    near(55, "alpha=0.05 costs 27% on Mask2Former", 0.272,
         a05["mask2former_swinb_cityscapes"])

    # --- 57: MC-dropout 36% vs B2 63% at alpha=0.1 ---
    near(57, "MC-dropout area at alpha=0.1 is 0.035",
         0.035, mean(sel(r, model="segformer_b2_cityscapes_mcdrop8", alpha=0.10,
                         rho=0.5), "marked_area_fraction"))
    near(57, "B2 area at alpha=0.1 is 0.035 (the same as MC-dropout)",
         0.035, mean(sel(r, model="segformer_b2_cityscapes", alpha=0.10,
                         rho=0.5), "marked_area_fraction"))

    # --- 59: Mask2Former degenerate at rho=0.5 ---
    m2f = sel(r, model="mask2former_swinb_cityscapes", rho=0.5)
    truth(59, "Mask2Former never returns lambda_max in distribution (log-tail grid)",
          bool((m2f.lam_index < LAM_MAX).all()),
          f"{int((m2f.lam_index >= LAM_MAX).sum())} of {len(m2f)} draws at lambda_max")
    m2f_cut = {al: float(1.0 - sel(m2f, alpha=al).lam.mean()) for al in ALPHAS}
    truth(59, "Mask2Former mean cutoff 1-lambda is 3e-6 / 1.4e-5 / 1.9e-4 at the three levels",
          abs(m2f_cut[0.05] - 3.0e-6) < 0.5e-6 and abs(m2f_cut[0.10] - 1.4e-5) < 0.1e-5
          and abs(m2f_cut[0.20] - 1.9e-4) < 0.1e-4,
          str({k: f"{v:.2e}" for k, v in m2f_cut.items()}))
    seg_cut = {al: [float(1.0 - sel(r, model=mo, alpha=al, rho=0.5).lam.mean()) for mo in SEGF]
               for al in ALPHAS}
    truth(59, "SegFormer mean cutoffs span 1.8-6e-4, 1-2e-3 and 6-13e-3",
          1.7e-4 <= min(seg_cut[0.05]) and max(seg_cut[0.05]) <= 6e-4
          and 1e-3 <= min(seg_cut[0.10]) and max(seg_cut[0.10]) <= 2.1e-3
          and 6e-3 <= min(seg_cut[0.20]) and max(seg_cut[0.20]) <= 13.1e-3,
          str({k: [f"{x:.2e}" for x in v] for k, v in seg_cut.items()}))
    m2f_ratio = [mean(sel(m2f, alpha=al), "marked_area_fraction")
                 / mean(sel(r, model=mo, alpha=al, rho=0.5), "marked_area_fraction")
                 for al in ALPHAS for mo in SEGF]
    rng_claim(59, "Mask2Former costs two to seven times the SegFormer area", 2.0, 7.0,
              m2f_ratio, 0.3)

    # --- 61-64: in-distribution Cityscapes cells ---
    tab2 = {
        "segformer_b2_cityscapes": [(0.046, 0.049), (0.096, 0.035), (0.195, 0.023)],
        "segformer_b5_cityscapes": [(0.045, 0.039), (0.095, 0.022), (0.194, 0.014)],
        "mask2former_swinb_cityscapes": [(0.044, 0.272), (0.095, 0.134), (0.193, 0.053)],
        "segformer_b2_cityscapes_mcdrop8": [(0.046, 0.049), (0.095, 0.035), (0.194, 0.024)],
    }
    for cid, (mo, rows) in zip([61, 62, 63, 64], tab2.items()):
        for al, (f_, a_) in zip(ALPHAS, rows):
            d = sel(r, model=mo, alpha=al, rho=0.5)
            near(cid, f"in-dist {mo} a={al} FNR", f_, mean(d, "region_fnr"))
            near(cid, f"in-dist {mo} a={al} area", a_,
                 mean(d, "marked_area_fraction"))


# --------------------------------------------------------------------------
# F. baselines
# --------------------------------------------------------------------------
def section_baselines():
    head("F  Method baselines and the area-matched comparison")
    e1 = load("e1_indist__*")
    lac = load("x4_lac__*")
    tmp = load("x5_temp__*")

    # 65 / 11: argmax 38-68% in distribution
    vals = [mean(sel(e1, method="argmax", model=mo, alpha=0.20, rho=rh),
                 "region_fnr")
            for mo in CS_MODELS for rh in (0.1, 0.5)]
    rng_claim(65, "argmax misses 38-68% of regions in distribution",
              0.376, 0.684, vals)

    # 87: caption "argmax misses the majority of regions"
    below = {f"{mo}|rho={rh}": round(v, 3)
             for (mo, rh), v in zip([(mo, rh) for mo in CS_MODELS for rh in (0.1, 0.5)],
                                    vals) if v <= 0.5}
    truth(87, "caption: argmax misses 38-68% of regions",
          abs(min(vals) - 0.376) < 5e-4 and abs(max(vals) - 0.684) < 5e-4,
          f"[{min(vals):.4f}, {max(vals):.4f}]")

    # 67 / 17 / 73: pixel CRC worst region FNR 0.257
    pix = {mo: mean(sel(e1, method="pixel_crc", model=mo, alpha=0.20, rho=0.5),
                    "region_fnr") for mo in CS_MODELS}
    near(67, "pixel CRC worst region FNR at alpha=0.2, rho=0.5 is 0.260",
         0.260, max(pix.values()))
    truth(4, "pixel CRC violates the region target in distribution on all four models",
          min(pix.values()) > 0.20, str({k: round(v, 3) for k, v in pix.items()}))
    rng_claim(4, "pixel CRC region FNR at rho=0.5 spans 0.241-0.260", 0.2408, 0.2601,
              list(pix.values()))

    # 68: paired (FNR, area) on B5
    b5 = sel(e1, model="segformer_b5_cityscapes", alpha=0.20, rho=0.5)
    near(68, "B5 pixel CRC area 1.1%", 0.011,
         mean(sel(b5, method="pixel_crc"), "marked_area_fraction"))
    near(68, "B5 region CRC 0.194 at 1.4% area", 0.194,
         mean(sel(b5, method="region_crc"), "region_fnr"))
    near(68, "B5 region CRC area 1.4%", 0.014,
         mean(sel(b5, method="region_crc"), "marked_area_fraction"))

    # 76: the false-positive component count is WITHDRAWN from the article.
    # The stored column is read at the nearest point of a 25-step subgrid, and
    # every selected threshold sits in the top 20 grid points, so most draws are
    # read at lambda_max, where the whole image is one component and the count
    # degenerates to the class-absence indicator. Guard against the column
    # silently returning to the text before the subgrid is fixed.
    # The slot must never sit above the calibrated threshold. Computed with the
    # shipped helper rather than a local copy of the subgrid, so that a change
    # to the grid cannot leave this check certifying the old behaviour.
    for _mo in ("segformer_b2_cityscapes", "segformer_b5_cityscapes",
                "segformer_b2_cityscapes_mcdrop8"):
        _d = sel(e1, method="region_crc", model=_mo, alpha=0.20, rho=0.5)
        _up = float(np.mean([FP_SUBGRID_INDICES[fp_subgrid_slot(int(_x))] > _x
                             for _x in _d.lam_index.values]))
        truth(76, f"{_mo}: no FP slot rounds above its calibrated threshold",
              _up == 0.0, f"snap-up fraction {_up:.3f}")
    a_b2 = mean(sel(e1, method="region_crc", model="segformer_b2_cityscapes",
                    alpha=0.20, rho=0.5), "marked_area_fraction")
    a_b5 = mean(sel(e1, method="region_crc", model="segformer_b5_cityscapes",
                    alpha=0.20, rho=0.5), "marked_area_fraction")
    truth(77, "B5 marks a smaller area than B2 (1.4% against 2.3%)",
          a_b5 < a_b2 and abs(a_b5 - 0.014) < 5e-4 and abs(a_b2 - 0.023) < 5e-4,
          f"B5 area={a_b5:.3f} vs B2 area={a_b2:.3f} at alpha=0.2, rho=0.5")

    # 79: the new per-class LoveDA and breakdown conventions
    lov = sel(load("l1_*"), method="region_crc")
    lcells = lov[lov.feasible.astype(bool)].groupby(
        ["model", "class_name", "alpha", "rho"]).agg(
        fnr=("region_fnr", "mean"), area=("marked_area_fraction", "mean"))
    truth(79, "LoveDA per-class table covers 24 in-domain cells",
          len(lcells) == 24, f"{len(lcells)} cells")
    vac = lcells[lcells.area > 0.99]
    truth(79, "no LoveDA in-domain cell marks >99% of the image",
          len(vac) == 0, str(sorted(vac.index.tolist())))
    l05 = lcells[np.isclose(lcells.index.get_level_values("rho"), 0.5)]
    a20 = l05[np.isclose(l05.index.get_level_values("alpha"), 0.2)].area
    a05 = l05[np.isclose(l05.index.get_level_values("alpha"), 0.05)].area
    truth(79, "LoveDA in-domain masks cover 13-23% at a=0.2 and 26-46% at a=0.05 (class-averaged, rho=0.5)",
          0.125 <= float(a20.groupby(level=["model", "rho"]).mean().min()) and
          float(a20.groupby(level=["model", "rho"]).mean().max()) <= 0.235 and
          0.255 <= float(a05.groupby(level=["model", "rho"]).mean().min()) and
          float(a05.groupby(level=["model", "rho"]).mean().max()) <= 0.465,
          f"a=0.2 {a20.groupby(level=['model', 'rho']).mean().round(3).to_dict()}; "
          f"a=0.05 {a05.groupby(level=['model', 'rho']).mean().round(3).to_dict()}")
    over = lcells[lcells.fnr > lcells.index.get_level_values("alpha")]
    truth(79, "exactly seven LoveDA in-domain cells exceed their level",
          len(over) == 7, str(sorted(over.index.tolist())))

    brk = sel(load("e2_break__*"), method="region_crc", rho=0.5)
    bcells = brk[brk.feasible.astype(bool)].groupby(
        ["model", "experiment", "alpha"]).agg(
        area=("marked_area_fraction", "mean"))
    star05 = bcells[(bcells.index.get_level_values("alpha") == 0.05)
                    & (bcells.area > 0.99)]
    models05 = sorted({m for m, _, _ in star05.index})
    truth(79, "no breakdown cell marks more than 99% of the image at any level",
          len(star05) == 0 and int((bcells.area > 0.99).sum()) == 0,
          f"{int((bcells.area > 0.99).sum())} starred cells over models {models05}")

    # 78: size strata
    for cid, mo in [(78, "segformer_b2_cityscapes"), (78, "segformer_b5_cityscapes")]:
        d = sel(e1, method="region_crc", model=mo, alpha=0.20, rho=0.5)
        s0, s1, s2 = (mean(d, f"fnr_stratum{i}") for i in range(3))
        note(cid, f"size strata {mo}", f"small={s0:.3f} mid={s1:.3f} large={s2:.3f}")

    # 80: LAC
    note(80, "LAC methods present", str(sorted(lac.method.unique())))
    lacm = [m for m in lac.method.unique() if "lac" in m]
    if lacm:
        v02 = list(cellmean(sel(lac, method=lacm[0], alpha=0.20, rho=0.5),
                            ["model"]).values)
        v05 = list(cellmean(sel(lac, method=lacm[0], alpha=0.05, rho=0.5),
                            ["model"]).values)
        rng_claim(80, "LAC misses 88-95% at alpha=0.2", 0.8815, 0.9547, v02, 0.002)
        rng_claim(80, "LAC misses 51-61% at alpha=0.05", 0.510, 0.612, v05, 0.002)

    # 83: tempered uncorrected mean 0.197
    th = sel(tmp, method="heuristic", alpha=0.20)
    near(83, "tempered uncorrected mean FNR at alpha=0.2 is 0.197",
         0.197, mean(th, "region_fnr"), 0.001)
    per = cellmean(th, ["model", "class_name", "rho"])
    truth(83, "tempered uncorrected still crosses the level in individual cells",
          (per > 0.20).any(), f"{int((per > 0.20).sum())} of {len(per)} cells above 0.2")

    # 84: tempered region CRC valid everywhere
    tr = sel(tmp, method="region_crc")
    c = cellmean(tr, ["model", "class_name", "alpha", "rho"])
    lev = c.index.get_level_values("alpha").values
    truth(84, "tempered region CRC valid in every configuration",
          bool((c.values <= lev + 1e-12).all()),
          f"{int((c.values > lev).sum())} violations")

    # 85: tempering leaves the area where it was (B5 1.4% -> 1.5%, B2 2.3% -> 2.4%)
    b5t = [mo for mo in tr.model.unique() if "b5" in mo]
    b2t = [mo for mo in tr.model.unique() if "b2" in mo]
    near(85, "tempered B5 area at alpha=0.2, rho=0.5 is 0.015", 0.015,
         mean(sel(tr, model=b5t, alpha=0.20, rho=0.5), "marked_area_fraction"))
    near(85, "tempered B2 area at alpha=0.2, rho=0.5 is 0.024", 0.024,
         mean(sel(tr, model=b2t, alpha=0.20, rho=0.5), "marked_area_fraction"))
    truth(85, "tempering moves the region-CRC area by at most a tenth of a point",
          abs(mean(sel(tr, model=b5t, alpha=0.20, rho=0.5), "marked_area_fraction")
              - mean(sel(e1, method="region_crc", model="segformer_b5_cityscapes",
                         alpha=0.20, rho=0.5), "marked_area_fraction")) <= 0.001
          and abs(mean(sel(tr, model=b2t, alpha=0.20, rho=0.5), "marked_area_fraction")
                  - mean(sel(e1, method="region_crc", model="segformer_b2_cityscapes",
                             alpha=0.20, rho=0.5), "marked_area_fraction")) <= 0.001,
          "")

    # 90-111: every cell of the method-baseline comparison
    table3 = {
        ("segformer_b2_cityscapes", "argmax"): [(0.586, 0.007), (0.676, 0.007)],
        ("segformer_b2_cityscapes", "heuristic"): [(0.199, 0.019), (0.199, 0.023)],
        ("segformer_b2_cityscapes", "pixel_crc"): [(0.174, 0.021), (0.241, 0.021)],
        ("segformer_b2_cityscapes", "region_crc"): [(0.194, 0.020), (0.195, 0.023)],
        ("segformer_b5_cityscapes", "argmax"): [(0.376, 0.007), (0.470, 0.007)],
        ("segformer_b5_cityscapes", "heuristic"): [(0.199, 0.011), (0.199, 0.014)],
        ("segformer_b5_cityscapes", "pixel_crc"): [(0.202, 0.011), (0.260, 0.011)],
        ("segformer_b5_cityscapes", "region_crc"): [(0.194, 0.012), (0.194, 0.014)],
        ("mask2former_swinb_cityscapes", "argmax"): [(0.548, 0.007), (0.633, 0.007)],
        ("mask2former_swinb_cityscapes", "heuristic"): [(0.197, 0.032), (0.198, 0.051)],
        ("mask2former_swinb_cityscapes", "pixel_crc"): [(0.195, 0.033), (0.260, 0.033)],
        ("mask2former_swinb_cityscapes", "region_crc"): [(0.193, 0.033), (0.193, 0.053)],
        ("segformer_b2_cityscapes_mcdrop8", "argmax"): [(0.592, 0.007), (0.684, 0.007)],
        ("segformer_b2_cityscapes_mcdrop8", "heuristic"): [(0.199, 0.019), (0.199, 0.024)],
        ("segformer_b2_cityscapes_mcdrop8", "pixel_crc"): [(0.172, 0.021), (0.244, 0.021)],
        ("segformer_b2_cityscapes_mcdrop8", "region_crc"): [(0.194, 0.020), (0.194, 0.024)],
    }
    for (mo, me), rows in table3.items():
        for rh, (f_, a_) in zip((0.1, 0.5), rows):
            d = sel(e1, model=mo, method=me, alpha=0.20, rho=rh)
            near(90, f"baseline {mo}/{me} rho={rh} FNR", f_, mean(d, "region_fnr"))
            near(90, f"baseline {mo}/{me} rho={rh} area", a_,
                 mean(d, "marked_area_fraction"))
    # 957: the abstract's scope for the pixel-CRC sentence (third panel, M3).
    cs_models = sorted({mo for mo, _ in table3})
    pix = {(mo, rh): mean(sel(e1, model=mo, method="pixel_crc", alpha=0.20, rho=rh))
           for mo in cs_models for rh in (0.1, 0.5)}
    segf = [m for m in cs_models if m.startswith("segformer")]
    truth(957, "abstract: at rho=0.5 pixel CRC misses the region target on all "
               "four Cityscapes models",
          all(pix[(m, 0.5)] > 0.20 for m in cs_models),
          str({m.split("_")[1]: round(pix[(m, 0.5)], 3) for m in cs_models}))
    note(957, "at rho=0.1 pixel CRC exceeds the level only on B5, by 0.002; the "
              "abstract's sentence is scoped to rho=0.5",
         str({m.split("_")[1]: round(pix[(m, 0.1)], 3) for m in cs_models}))

    # 94/101: LAC rows; 95/96/102/103: tempered-posterior rows
    lacm = [m for m in lac.method.unique() if "lac" in m]
    lac_rows = {"segformer_b2": [(0.865, 0.003), (0.955, 0.003)],
                "segformer_b5": [(0.694, 0.004), (0.882, 0.004)]}
    for key, rows in lac_rows.items():
        mos = [m for m in lac.model.unique() if m.startswith(key)]
        for rh, (f_, a_) in zip((0.1, 0.5), rows):
            d = sel(lac, method=lacm[0], model=mos, alpha=0.20, rho=rh)
            near(94, f"baseline {key} LAC rho={rh} FNR", f_, mean(d))
            near(94, f"baseline {key} LAC rho={rh} area", a_,
                 mean(d, "marked_area_fraction"))
    tmp_rows = {("segformer_b2", "heuristic"): [(0.199, 0.020), (0.199, 0.024)],
                ("segformer_b2", "region_crc"): [(0.193, 0.020), (0.193, 0.024)],
                ("segformer_b5", "heuristic"): [(0.197, 0.012), (0.197, 0.014)],
                ("segformer_b5", "region_crc"): [(0.192, 0.012), (0.192, 0.015)]}
    for (key, me), rows in tmp_rows.items():
        mos = [m for m in tmp.model.unique() if m.startswith(key)]
        for rh, (f_, a_) in zip((0.1, 0.5), rows):
            d = sel(tmp, method=me, model=mos, alpha=0.20, rho=rh)
            near(95, f"baseline {key}/{me} (tempered) rho={rh} FNR", f_, mean(d))
            near(95, f"baseline {key}/{me} (tempered) rho={rh} area", a_,
                 mean(d, "marked_area_fraction"))

    # 70-72: area-matched Pareto
    try:
        par = load("p1_pareto__*")
    except FileNotFoundError:
        note(70, "pareto sweep CSVs missing")
        return
    for mo, area_q, exp_r, exp_p in [("segformer_b2_cityscapes", 0.0188, 0.282, 0.287),
                                     ("segformer_b5_cityscapes", 0.0685, 0.041, 0.047)]:
        d = sel(par, model=mo, rho=0.5)
        curves = {}
        for me in ("region_crc", "pixel_crc"):
            g = cellmean(sel(d, method=me), ["alpha"], "marked_area_fraction")
            f = cellmean(sel(d, method=me), ["alpha"], "region_fnr")
            o = np.argsort(g.values)
            curves[me] = (g.values[o], f.values[o])
        lo = max(curves["region_crc"][0].min(), curves["pixel_crc"][0].min())
        hi = min(curves["region_crc"][0].max(), curves["pixel_crc"][0].max())
        if not (lo <= area_q <= hi):
            fail(71, f"{mo}: quoted matched area {area_q} outside the measured band",
                 f"band [{lo:.3f}, {hi:.3f}]")
            continue
        rr = float(np.interp(area_q, *curves["region_crc"]))
        pp = float(np.interp(area_q, *curves["pixel_crc"]))
        near(71, f"{mo} region CRC at {area_q:.1%} area", exp_r, rr, 0.003)
        near(71, f"{mo} pixel CRC at {area_q:.1%} area", exp_p, pp, 0.003)
        grid = np.linspace(lo, hi, 200)
        diff = np.interp(grid, *curves["pixel_crc"]) - np.interp(grid, *curves["region_crc"])
        note(70, f"{mo} area-matched advantage of region CRC over the band",
             f"min={diff.min():+.3f} max={diff.max():+.3f} (band {lo:.3f}-{hi:.3f})")


# --------------------------------------------------------------------------
# G. ablations
# --------------------------------------------------------------------------
def section_lac_detail():
    """Section IV-B quotes LAC at both capture levels and its marked area."""
    lac = sel(load("x4_lac__*"), method="lac_global")
    per = lac.groupby(["model", "alpha", "rho"])[
        ["region_fnr", "marked_area_fraction"]].mean()
    for rho, (lo20, hi20), (lo05, hi05) in [
            (0.5, (0.8815, 0.9547), (0.5094, 0.6117)),
            (0.1, (0.6937, 0.8646), (0.4118, 0.5314))]:
        v20 = [v for (m, a, rh), v in per["region_fnr"].items()
               if abs(a - 0.2) < 1e-9 and abs(rh - rho) < 1e-9]
        v05 = [v for (m, a, rh), v in per["region_fnr"].items()
               if abs(a - 0.05) < 1e-9 and abs(rh - rho) < 1e-9]
        rng_claim(96, f"LAC region FNR at alpha=0.2, rho={rho}", lo20, hi20, v20)
        rng_claim(96, f"LAC region FNR at alpha=0.05, rho={rho}", lo05, hi05, v05)
    a20 = [v for (m, a, rh), v in per["marked_area_fraction"].items()
           if abs(a - 0.2) < 1e-9 and abs(rh - 0.5) < 1e-9]
    a05 = [v for (m, a, rh), v in per["marked_area_fraction"].items()
           if abs(a - 0.05) < 1e-9 and abs(rh - 0.5) < 1e-9]
    rng_claim(97, "LAC marked area at alpha=0.2 is 0.3-0.4% of the image",
              0.0028, 0.0038, a20)
    rng_claim(97, "LAC marked area at alpha=0.05 is 0.7-0.9%", 0.0070, 0.0090, a05)
    _lac_classcond_and_pixel_fnr()


def _x4_argv() -> list[dict]:
    """The recorded argv of every x4_lac run, from the stage-5 meta sidecars."""
    out = []
    for path in sorted(glob.glob(str(results_dir("experiments") / "x4_lac__*.meta.json"))):
        with open(path) as f:
            out.append(json.load(f).get("argv", {}))
    return out


def _lac_classcond_and_pixel_fnr():
    """The class-conditional LAC arm and the realized pixel FNR must be read.

    Both are optional flags of stage 5, and an earlier version of this check
    decided whether to demand them by reading the run's own meta sidecar. That
    made the guard depend on the artefact under test: rerunning the block with
    the narrower flag set overwrites the CSV *and* the sidecar, so the check
    stopped asking for exactly the rows the rerun had just dropped, and the
    audit passed while Table II lost two rows. The demand is therefore
    unconditional. Table II prints the class-conditional row and Section V
    quotes both the region-FNR and the marked-area gap, so once the x4 block
    exists at all, these rows are published content and their absence is a
    failure, whatever the run asked for. The recorded argv is still read, but
    only to say which invocation produced a shortfall.
    """
    raw = load("x4_lac__*")
    argv = _x4_argv()
    asked_cc = any("class_conditional" in (a.get("lac_variants") or []) for a in argv)
    asked_px = any(a.get("measure_pixel_fnr") for a in argv)

    cc = raw[raw["method"] == "lac_classcond"]
    truth(98, "class-conditional LAC rows present in x4_lac__*",
          not cc.empty,
          f"{len(cc)} rows" if not cc.empty else
          "x4_lac ran without --lac-variants marginal class_conditional "
          f"(sidecars recorded the request: {asked_cc}); Table II loses its "
          "class-conditional row and Section V its two gaps")

    if not cc.empty:
        # What the arm actually shows, measured rather than hoped for: given
        # its own class's pixel-coverage target, class-conditional LAC lands
        # on the pixel-CRC row. That is the honest answer to the referee --
        # the fair version of LAC is the pixel baseline the article already
        # reports -- and it is checked over every cell of the matrix, not one
        # convenient corner, because the coincidence is what is being claimed.
        cell = feas(raw).groupby(["method", "model", "alpha", "rho"]).agg(
            reg=("region_fnr", "mean"),
            area=("marked_area_fraction", "mean")).reset_index()
        key = ["model", "alpha", "rho"]
        ccc = cell[cell["method"] == "lac_classcond"].set_index(key)
        pxc = cell[cell["method"] == "pixel_crc"].set_index(key)
        shared = ccc.index.intersection(pxc.index)
        if not len(shared):
            fail(98, "no cell has both a class-conditional LAC and a pixel-CRC row")
        else:
            dreg = (ccc.loc[shared, "reg"] - pxc.loc[shared, "reg"]).abs()
            darea = (ccc.loc[shared, "area"] - pxc.loc[shared, "area"]).abs()
            truth(98, "class-conditional LAC coincides with pixel CRC in every cell",
                  float(dreg.max()) <= LAC_PIXEL_GAP,
                  f"{len(shared)} cells; largest region-FNR gap {float(dreg.max()):.4f}, "
                  f"largest area gap {float(darea.max()):.4f}, against a "
                  f"{LAC_PIXEL_GAP:g} margin")
            # The two gaps the section prints, to the precision it prints them.
            near(98, "largest class-conditional / pixel-CRC region-FNR gap",
                 0.001, float(dreg.max()), tol=5e-4)
            near(98, "largest class-conditional / pixel-CRC marked-area gap",
                 0.0004, float(darea.max()), tol=5e-4)

    has_px = "realized_pixel_fnr" in raw.columns
    truth(99, "realized_pixel_fnr column present in x4_lac__*", has_px,
          "x4_lac ran without --measure-pixel-fnr (sidecars recorded the "
          f"request: {asked_px}); the pixel-level target Section V reports "
          "is then asserted rather than measured" if not has_px else "")
    if has_px:
        px = raw.dropna(subset=["realized_pixel_fnr"])
        if px.empty or px[px["method"].isin(LAC_METHODS)].empty:
            fail(99, "realized_pixel_fnr is all-NaN on the LAC rows")
        else:
            _pixel_target_checks(px)


# Which pixel quantity each method is entitled to be judged against. The
# column measures the realized pixel FNR OF THE CRITICAL CLASS the row is
# scored on. For lac_classcond and pixel_crc that is the quantity the
# threshold was calibrated to, so the level is a promise. For lac_global it
# is not: that threshold is set by an all-class pixel-coverage target, in
# which the three critical classes are a rounding error, so its per-class
# pixel FNR is free to be anything -- and how far above the level it lands
# is the finding, not a violation.
LAC_METHODS = ("lac_global", "lac_classcond")
OWN_TARGET_METHODS = ("lac_classcond", "pixel_crc")
# How far above its own level a method that was calibrated to this exact
# quantity may land before it stops being sampling noise.
# The marginal threshold has to miss the critical classes' pixels by a wide
# margin, or the article's whole reading of that row is wrong.
MARGINAL_PIXEL_EXCESS = 0.20
# ... and its region loss has to stand clear of the pixel-CRC row.
MARGINAL_REGION_SEPARATION = 0.35


def _pixel_target_checks(px):
    """Judge each method against the pixel quantity it actually targets."""
    cell = px.groupby(["method", "model", "alpha", "rho"]).agg(
        pix=("realized_pixel_fnr", "mean"), reg=("region_fnr", "mean")).reset_index()

    own = cell[cell["method"].isin(OWN_TARGET_METHODS)]
    over = own[own["pix"] > own["alpha"] + PIXEL_TARGET_SLACK]
    worst = float((own["pix"] - own["alpha"]).max()) if len(own) else float("nan")
    truth(99, "a threshold calibrated to a class's own pixel risk lands on that "
              "level (class-conditional LAC, pixel CRC)",
          bool(len(own)) and over.empty,
          f"{len(own)} cells, worst overshoot {worst:+.4f} against a "
          f"{PIXEL_TARGET_SLACK:g} margin" if len(own) and over.empty
          else f"{len(over)} of {len(own)} cells exceed their level by more "
               f"than {PIXEL_TARGET_SLACK:g} (worst {worst:+.4f})")

    # The marginal variant, charged with the same quantity, is nowhere near
    # it: the all-class threshold under-covers the critical classes at the
    # PIXEL level too, so the region-level failure reported in the article is
    # not an artifact of having moved to regions.
    mg = cell[cell["method"] == "lac_global"]
    short = float((mg["pix"] - mg["alpha"]).min()) if len(mg) else float("nan")
    truth(99, "the marginal LAC threshold misses the critical classes' pixels "
              "as well, not only their regions",
          bool(len(mg)) and short > MARGINAL_PIXEL_EXCESS,
          f"{len(mg)} cells, smallest excess over the level {short:+.4f}; "
          f"pixel FNR spans {mg['pix'].min():.3f}-{mg['pix'].max():.3f}"
          if len(mg) else "no lac_global cell")

    # And it stays far from the pixel-CRC row in region loss, which is what
    # makes the two LAC variants worth reporting separately.
    key = ["model", "alpha", "rho"]
    gap = (mg.set_index(key)["reg"]
           - cell[cell["method"] == "pixel_crc"].set_index(key)["reg"]).dropna()
    truth(99, "marginal LAC stays far above the pixel-CRC row in region loss",
          bool(len(gap)) and float(gap.min()) > MARGINAL_REGION_SEPARATION,
          f"{len(gap)} cells, smallest separation {float(gap.min()):+.4f}"
          if len(gap) else "no comparable cell")

    # The per-level values Section IV-B prints. The realized pixel FNR does
    # not depend on rho, so one number per (method, level) is the whole claim.
    by_level = cell.groupby(["method", "alpha"])["pix"].mean()
    for alpha, want in ((0.05, 0.46), (0.10, 0.67), (0.20, 0.82)):
        near(100, f"marginal LAC pixel FNR on the critical class at alpha={alpha:g}",
             want, float(by_level[("lac_global", alpha)]), tol=5e-3)
    excess = [float(by_level[("lac_global", a)] - a) for a in (0.05, 0.10, 0.20)]
    rng_claim(100, "the marginal excess over the level spans 0.41-0.62",
              0.41, 0.62, excess, tol=5e-3)
    for alpha, want in ((0.05, 0.046), (0.10, 0.095), (0.20, 0.194)):
        near(100, f"class-conditional LAC pixel FNR at alpha={alpha:g}",
             want, float(by_level[("lac_classcond", alpha)]), tol=5e-4)

    # What Table II prints: the alpha=0.2 rows, rounded as the table rounds.
    tab = cell[np.isclose(cell["alpha"], 0.20)]
    key = ["model", "rho"]
    cc = tab[tab["method"] == "lac_classcond"].set_index(key)
    px = tab[tab["method"] == "pixel_crc"].set_index(key)
    shared = cc.index.intersection(px.index)
    dprint = (cc.loc[shared, "reg"].round(3) - px.loc[shared, "reg"].round(3)).abs()
    truth(100, "as printed in Table II the two rows agree to 0.002 in FNR",
          bool(len(shared)) and float(dprint.max()) <= 0.002 + 1e-12,
          f"{len(shared)} printed cells, largest printed gap {float(dprint.max()):.3f}")


def section_ablations():
    head("G  Shared-threshold and size-weighted ablations")
    e1 = load("e1_indist__*")
    sh = load("x1_shared__*")
    sw = load("x2_sizew__*")

    cs = sel(sh, model=CS_MODELS, method="shared_crc")
    c = cellmean(cs, ["model", "alpha", "rho"], "controlled_risk")
    lev = c.index.get_level_values("alpha").values
    truth(112, "shared threshold preserves validity on the worst class in every cell",
          bool((c.values <= lev + 1e-12).all()),
          f"{int((c.values > lev).sum())} violations")

    per = [mean(sel(e1, method="region_crc", model=mo, alpha=0.10, rho=0.5),
                "marked_area_fraction") for mo in SEGF]
    shd = [mean(sel(cs, model=mo, alpha=0.10, rho=0.5), "marked_area_fraction")
           for mo in SEGF]
    rng_claim(113, "per-class area 2.2-3.5% at alpha=0.1", 0.0221, 0.0352, per)
    rng_claim(113, "shared area 3.4-4.8% at alpha=0.1", 0.0339, 0.0481, shd)
    m2f_per = mean(sel(e1, method="region_crc", model="mask2former_swinb_cityscapes",
                       alpha=0.10, rho=0.5), "marked_area_fraction")
    m2f_shd = mean(sel(cs, model="mask2former_swinb_cityscapes", alpha=0.10, rho=0.5),
                   "marked_area_fraction")
    truth(113, "Mask2Former shared threshold at alpha=0.1: 13% -> 24%",
          abs(m2f_per - 0.134) < 5e-4 and abs(m2f_shd - 0.237) < 5e-4,
          f"{m2f_per:.4f} -> {m2f_shd:.4f}")

    # 116: ablation caption "up to an order of magnitude"
    ratios = {}
    for mo in CS_MODELS:
        for al in (0.10, 0.20):
            p = mean(sel(e1, method="region_crc", model=mo, alpha=al, rho=0.5),
                     "marked_area_fraction")
            s = mean(sel(cs, model=mo, alpha=al, rho=0.5), "marked_area_fraction")
            if p > 0:
                ratios[f"{mo}|a={al}"] = s / p
    worst = max(ratios.values())
    truth(116, "the shared threshold inflates the marked area by up to 2.2x (Mask2Former, a=0.2)",
          2.15 <= worst <= 2.25 and max(ratios, key=ratios.get).startswith("mask2former"),
          f"largest measured ratio = {worst:.2f}x "
                               f"({max(ratios, key=ratios.get)})")

    # 114: size-weighted loss
    d = sel(sw, alpha=0.20, rho=0.5)
    cr = mean(d, "controlled_risk")
    fnrs = [mean(sel(d, model=mo), "region_fnr") for mo in
            ["segformer_b2_cityscapes", "segformer_b5_cityscapes"]]
    near(114, "size-weighted controlled risk at alpha=0.2", 0.19, cr, 0.006)
    rng_claim(114, "size-weighted unweighted region FNR 0.26-0.30",
              0.264, 0.299, fnrs, 0.006)


# --------------------------------------------------------------------------
# H. breakdown under shift
# --------------------------------------------------------------------------
def section_breakdown():
    head("H  Breakdown under weather shift")
    e2 = load("e2_break__*")
    e1 = load("e1_indist__*")
    r = sel(e2, method="region_crc")

    c = cellmean(sel(r, model=SEGF, alpha=0.20, rho=0.5), ["model", "experiment"])
    n_viol = int((c.values > 0.20 + 1e-12).sum())
    truth(124, "SegFormer variants violate on all twelve model-condition cells",
          n_viol == len(c) == 12, f"{n_viol} of {len(c)} cells violate")
    worst = c.max() / 0.20
    near(124, "worst violation ratio at alpha=0.2 is 2.23x", 2.23, float(worst), 0.01)
    note(124, "worst cell", f"{c.idxmax()} = {c.max():.4f}")

    g125 = feas(sel(r, model=SEGF, alpha=0.20, rho=0.5)).groupby(["model", "experiment"])["region_fnr"]
    lo125 = g125.mean() - 1.959963984540054 * g125.std(ddof=1) / np.sqrt(g125.size())
    truth(125, "every one of the twelve SegFormer cells at alpha=0.2 exceeds the level "
               "beyond the Monte Carlo error of the draws",
          len(lo125) == 12 and bool((lo125.values > 0.20).all()),
          f"{int((lo125.values > 0.20).sum())} of {len(lo125)} lower ends above 0.2")

    c1 = cellmean(sel(r, model=SEGF, alpha=0.10, rho=0.5), ["experiment"])
    near(126, "worst violation ratio at alpha=0.1 is 2.74x (B5 at night)", 2.74,
         float(c1.max() / 0.10), 0.02)
    c1m = cellmean(sel(r, model=SEGF, alpha=0.10, rho=0.5), ["model", "experiment"])
    truth(126, "the SegFormer variants violate on all twelve cells at alpha=0.1 too",
          int((c1m.values > 0.10).sum()) == 12, f"{int((c1m.values > 0.10).sum())} of {len(c1m)}")

    mc = cellmean(sel(r, model="segformer_b2_cityscapes_mcdrop8", alpha=0.10,
                      rho=0.5), ["experiment"])
    mc = {k.split("__")[1]: v for k, v in mc.items()}
    truth(127, "MC-dropout breaks under every condition at alpha=0.1",
          all(mc[k] > 0.10 for k in CONDS),
          str({k: round(v, 3) for k, v in mc.items()}))

    c05 = cellmean(sel(r, alpha=0.05, rho=0.5), ["model", "experiment"])
    c05s = cellmean(sel(r, model=SEGF, alpha=0.05, rho=0.5), ["model", "experiment"])
    truth(128, "at alpha=0.05 the SegFormer variants violate on all twelve cells, "
               "by up to 2.71x (MC-dropout under fog)",
          int((c05s.values > 0.05).sum()) == 12 and abs(float(c05s.max()) / 0.05 - 2.71) < 0.02
          and c05s.idxmax() == ("segformer_b2_cityscapes_mcdrop8",
                                "e2_break__fog__segformer_b2_cityscapes_mcdrop8"),
          f"{int((c05s.values > 0.05).sum())} of 12; max={c05s.max():.4f} at {c05s.idxmax()}")
    truth(128, "13 of the 16 model-condition cells violate at alpha=0.05",
          int((c05.values > 0.05).sum()) == 13, f"{int((c05.values > 0.05).sum())} of {len(c05)}")

    cls = cellmean(sel(r, alpha=0.20, rho=0.5), ["model", "experiment", "class_name"])
    near(129, "worst class-level cell is 0.522", 0.522, float(cls.max()))
    note(129, "worst class-level cell identity", str(cls.idxmax()))

    an = [mean(sel(e2, method="argmax", model=mo, alpha=0.20, rho=rh,
                   experiment=f"e2_break__night__{mo}"))
          for mo in CS_MODELS for rh in (0.1, 0.5)]
    rng_claim(130, "argmax misses 68-83% of regions at night", 0.679, 0.835, an)

    m2f = sel(r, model="mask2former_swinb_cityscapes", rho=0.5)
    m2c = cellmean(m2f, ["experiment", "alpha"])
    m2v = {(k[0].split("__")[1], k[1]): v > k[1] for k, v in m2c.items()}
    truth(131, "Mask2Former under shift violates in fog at every level and in snow at "
               "alpha>=0.1, and holds under night and rain",
          all(m2v[("fog", a)] for a in ALPHAS) and m2v[("snow", 0.1)] and m2v[("snow", 0.2)]
          and not m2v[("snow", 0.05)] and not any(m2v[(c_, a)] for c_ in ("night", "rain") for a in ALPHAS),
          str({k: round(v, 3) for k, v in m2c.items()}))
    near(131, "Mask2Former under fog at alpha=0.05 exceeds the level 2.05x", 2.05,
         float(m2c[("e2_break__fog__mask2former_swinb_cityscapes", 0.05)]) / 0.05, 0.01)

    # The KS statistic is identically zero when lambda-hat = lambda_max, since
    # both sides then mark every pixel. Those draws cannot fire and are excluded
    # from the reported rates, as the text now states.
    mon = sel(r, rho=0.5)
    mon = mon[mon.model.isin(SEGF)]
    degen = mon[mon.lam_index == LAM_MAX]
    truth(132, "the monitor is identically silent when lambda-hat = lambda_max",
          bool(np.allclose(degen.monitor_ks, 0.0))
          and bool((~degen.monitor_flag.astype(bool)).all()),
          f"{len(degen)} degenerate draws, max KS {float(degen.monitor_ks.max()):.2e}")
    truth(132, "four of the 10 800 SegFormer region-CRC draws at rho=0.5 are degenerate",
          len(mon) == 10800 and int((mon.lam_index == LAM_MAX).sum()) == 4,
          f"{int((mon.lam_index == LAM_MAX).sum())} of {len(mon)}")
    usable = mon[mon.lam_index < LAM_MAX]
    fl = usable.groupby("experiment")["monitor_flag"].mean()
    rng_claim(132, "monitor fires in 54-100% of the non-degenerate region-CRC "
                   "(draw, class, level) configurations at rho=0.5",
              0.536, 1.00, fl.values, 0.006)
    ind = sel(e1, method="region_crc", rho=0.5)
    ind = ind[ind.model.isin(SEGF)]
    ind = ind[ind.lam_index < LAM_MAX]
    fa = ind.groupby("model")["monitor_flag"].mean()
    rng_claim(132, "in-distribution false-alarm rate 0.6-1.2% on the same subset",
              0.0056, 0.0122, fa.values, 0.0006)
    note(132, "nominal KS level is 1%; the realized in-distribution rate is",
         f"{float(ind.monitor_flag.mean()):.4f} pooled over the three variants")

    by_cond = {}
    for c_ in CONDS:
        sub = sel(r, rho=0.5)
        sub = sub[sub.model.isin(SEGF) & sub.experiment.str.contains(f"__{c_}__")]
        cellsc = cellmean(sel(sub, alpha=0.20), ["model"])
        by_cond[c_] = (float(cellsc.mean()), float(cellsc.max()),
                       float(sub[sub.lam_index < LAM_MAX].monitor_flag.mean()))
    for c_, want in (("night", 0.637), ("rain", 0.798),
                     ("snow", 0.948), ("fog", 1.000)):
        near(133, f"{c_} fires in {want:.0%} of non-degenerate draws",
             want, by_cond[c_][2], 0.002)
    worst_mean = max(by_cond, key=lambda k: by_cond[k][0])
    worst_cell = max(by_cond, key=lambda k: by_cond[k][1])
    least_flag = min(by_cond, key=lambda k: by_cond[k][2])
    truth(133, "night is the LEAST FLAGGED condition (non-degenerate draws)",
          least_flag == "night",
          str({k: round(v[2], 3) for k, v in by_cond.items()}))
    truth(133, "night produces the single worst cell",
          worst_cell == "night",
          f"worst by single cell = {worst_cell}, by mean = {worst_mean}; " +
          str({k: (round(v[0], 3), round(v[1], 3)) for k, v in by_cond.items()}))

    # 135-138: every cell of the ACDC breakdown
    table5 = {
        "segformer_b2_cityscapes": [0.122, 0.080, 0.125, 0.121, 0.195, 0.174,
                                    0.213, 0.226, 0.348, 0.320, 0.405, 0.361],
        "segformer_b5_cityscapes": [0.071, 0.120, 0.098, 0.097, 0.106, 0.274,
                                    0.166, 0.157, 0.219, 0.445, 0.351, 0.239],
        "mask2former_swinb_cityscapes": [0.102, 0.011, 0.019, 0.048, 0.176, 0.042,
                                         0.069, 0.129, 0.272, 0.165, 0.181, 0.233],
        "segformer_b2_cityscapes_mcdrop8": [0.135, 0.077, 0.126, 0.128, 0.212,
                                            0.150, 0.215, 0.240, 0.328, 0.305,
                                            0.404, 0.340],
    }
    for mo, vals in table5.items():
        k = 0
        for al in ALPHAS:
            for c_ in CONDS:
                d = sel(r, model=mo, alpha=al, rho=0.5,
                        experiment=f"e2_break__{c_}__{mo}")
                near(135, f"breakdown {mo} a={al} {c_}", vals[k], mean(d))
                k += 1


# --------------------------------------------------------------------------
# I. tier A
# --------------------------------------------------------------------------
# Experiment-name fragments that must never appear inside the published
# tier-A selection. Each one names a separate arm that answers a different
# question; averaged into the published cells it moves them silently.
TIER_A_FOREIGN = ("seqdisjoint",)


def section_tier_a():
    head("I  Tier A recalibration")
    a = load("e3_tierA*")
    # A new arm named into the e3 block would be swept in here and into the
    # tables, which filter on block == "e3". That happened once, with
    # e3_tierA25_seqdisjoint__*: 648 published cells became 694 and the
    # class-balanced means moved. Catch it by name rather than by noticing
    # the numbers drifted.
    foreign = sorted({e for e in a.experiment.unique()
                      if any(tok in e for tok in TIER_A_FOREIGN)})
    truth(139, "the published tier-A selection holds no foreign arm",
          not foreign,
          f"{len(foreign)} foreign experiment(s) inside 'e3_tierA*': "
          f"{', '.join(foreign)}; rename them out of the e3 block"
          if foreign else f"{a.experiment.nunique()} experiments, all published tier A")
    a["n_target"] = pd.to_numeric(a.experiment.str.extract(r"tierA(\d+)")[0])
    r = sel(a, method="region_crc")

    g140 = feas(r).groupby(["n_target", "experiment", "class_name", "alpha", "rho"])
    res = g140["region_fnr"].agg(["mean", "size"])
    lev = res.index.get_level_values("alpha").values
    bad = res[res["mean"].values > lev + 1e-12]
    sd140 = g140["region_fnr"].std(ddof=1)
    lo140 = res["mean"] - 1.959963984540054 * sd140 / np.sqrt(res["size"])
    # a single-draw cell has no interval; it cannot be separable
    bad_lo = lo140.loc[bad.index].fillna(-np.inf)
    n_sep = int((bad_lo.values > bad.index.get_level_values("alpha").values).sum())
    truth(140, "tier A holds in 610 of the 648 feasible cells; none of the 38 "
               "exceptions is separable from the Monte Carlo error of the draws",
          len(res) == 648 and len(bad) == 38 and n_sep == 0,
          f"{len(bad)} of {len(res)} cells above the level; {n_sep} separable")
    small = bad[bad["size"] <= 10]
    truth(140, "26 of the exceptions rest on ten or fewer feasible draws; the largest, "
               "0.265 against alpha=0.1 (B5, bicycle, fog, n_t=50), is a single draw",
          len(small) == 26 and abs(float(bad["mean"].max()) - 0.265) < 5e-4
          and int(bad.loc[bad["mean"].idxmax(), "size"]) == 1
          and bad["mean"].idxmax()[:5] == (50, "e3_tierA50__fog__segformer_b5_cityscapes",
                                           "bicycle", 0.1, 0.5),
          f"{len(small)} small cells; worst {bad['mean'].max():.3f} over "
          f"{int(bad.loc[bad['mean'].idxmax(), 'size'])} draws at {bad['mean'].idxmax()}")
    n_lmax = int((feas(r).lam_index >= LAM_MAX).sum())
    note(140, "tier-A draws that return lambda_max (valid, uninformative), all models",
         f"{n_lmax} of {len(feas(r))} feasible draws; by model "
         + str(feas(r)[feas(r).lam_index >= LAM_MAX].groupby("model").size().to_dict()))
    big = bad[bad["size"] > 10]
    excess = big["mean"] - big.index.get_level_values("alpha")
    truth(140, "the other twelve rest on 27-41 draws, are all rider cells, eleven at "
               "alpha=0.2, and exceed by at most 0.06 (B5, rider, rain)",
          len(big) == 12 and int(big["size"].min()) == 27 and int(big["size"].max()) == 41
          and all(k[2] == "rider" for k in big.index)
          and sum(np.isclose(k[3], 0.2) for k in big.index) == 11
          and 0.055 <= float(excess.max()) <= 0.06
          and excess.idxmax()[1] == "e3_tierA50__rain__segformer_b5_cityscapes",
          f"{len(big)} cells, draws {sorted(big['size'].unique().tolist())}, "
          f"classes {sorted(set(k[2] for k in big.index))}, max excess {float(excess.max()):.4f} "
          f"at {excess.idxmax()}")

    sd5 = feas(sel(r, rho=0.5)).groupby(
        ["n_target", "experiment", "class_name", "alpha"])["lam"].std(ddof=1)
    sdall = feas(r).groupby(["n_target", "experiment", "class_name", "alpha",
                             "rho"])["lam"].std(ddof=1)
    truth(141, "sd of lambda-hat below 0.09 in every tier-A cell (rho=0.5)",
          float(np.nanmax(sd5.values)) < 0.09,
          f"max sd = {np.nanmax(sd5.values):.4f} at {sd5.idxmax()}")
    note(141, "same over both capture levels",
         f"max sd = {np.nanmax(sdall.values):.4f} at {sdall.idxmax()}")

    # Table VI is class-balanced: each column is averaged per class and then
    # over classes, so no cell is dominated by whichever class still has
    # feasible draws at a tight level.
    # Table VI is over all four models again (2026-09-09): on the log-tail grid
    # Mask2Former calibrates to interior thresholds in every tier-A cell, so
    # the SegFormer-only restriction of the third panel (M4) has no basis.
    def _balanced(n, alpha, col="region_fnr", models=None):
        d = sel(r, n_target=n, alpha=alpha, rho=0.5)
        d = d[d.feasible.astype(bool)]
        d = d[d.model.isin(CS_MODELS if models is None else models)]
        return float(d.groupby("class_name")[col].mean().mean())

    ar = [_balanced(n, 0.20, "marked_area_fraction") for n in (25, 50, 100)]
    rng_claim(142, "tier-A marked area 11-18% at alpha=0.2 (class-balanced, four models)",
              0.1069, 0.1755, ar, 0.006)
    truth(142, "area falls by seven points of the image between n_t=25 and 100",
          0.065 <= ar[0] - ar[2] <= 0.075 and ar[0] > ar[1] > ar[2],
          str([round(x, 3) for x in ar]))
    ars = [_balanced(n, 0.20, "marked_area_fraction", SEGF) for n in (25, 50, 100)]
    rng_claim(142, "SegFormer-only tier-A area 8-14% at alpha=0.2", 0.0773, 0.1436, ars, 0.006)
    arm = [_balanced(n, 0.20, "marked_area_fraction", ["mask2former_swinb_cityscapes"])
           for n in (25, 50, 100)]
    rng_claim(142, "Mask2Former tier-A area 20-27% at alpha=0.2", 0.1955, 0.2712, arm, 0.006)

    inf25 = 1 - sel(r, n_target=25, alpha=0.05).feasible.astype(bool).mean()
    inf100 = 1 - sel(r, n_target=100, alpha=0.05).feasible.astype(bool).mean()
    truth(145, "alpha=0.05 unattainable in >99% of n_t=25 draws",
          inf25 > 0.99, f"{inf25:.4f}")
    near(145, "infeasible fraction at n_t=100, alpha=0.05 is 0.47", 0.47, inf100, 0.006)

    table6 = {25: [(0.025, 0.259, 0.99), (0.066, 0.226, 0.74), (0.140, 0.176, 0.38)],
              50: [(0.026, 0.358, 0.80), (0.068, 0.272, 0.45), (0.169, 0.125, 0.19)],
              100: [(0.030, 0.388, 0.47), (0.077, 0.241, 0.27), (0.167, 0.107, 0.03)]}
    for n, rows in table6.items():
        for al, (f_, ar_, inf_) in zip(ALPHAS, rows):
            d = sel(r, n_target=n, alpha=al, rho=0.5)
            near(149, f"tier A n_t={n} a={al} FNR (class-balanced)",
                 f_, _balanced(n, al))
            near(149, f"tier A n_t={n} a={al} area (class-balanced)",
                 ar_, _balanced(n, al, "marked_area_fraction"))
            got = 1 - d.feasible.astype(bool).mean()
            if n == 25 and al == 0.05:
                truth(149, f"tier A n_t=25 a=0.05 infeasible >0.99", got > 0.99, f"{got:.4f}")
            else:
                near(149, f"tier A n_t={n} a={al} infeasible", inf_, got, 0.006)
    # 958: the reason Mask2Former is kept out of Table VI (third panel, M4).
    m2f = sel(r, model="mask2former_swinb_cityscapes", rho=0.5)
    m2f = m2f[m2f.feasible.astype(bool)]
    m2f_area = (m2f.groupby(["n_target", "alpha", "class_name"])["marked_area_fraction"]
                .mean().groupby(level=[0, 1]).mean())
    truth(958, "Mask2Former marks 20-77% of the image over its nine tier-A cells "
               "(class-balanced), so it belongs in Table VI again",
          len(m2f_area) == 9 and abs(float(m2f_area.min()) - 0.196) < 5e-4
          and abs(float(m2f_area.max()) - 0.765) < 5e-4,
          f"min {float(m2f_area.min()):.3f}, max {float(m2f_area.max()):.3f}, cells {len(m2f_area)}, "
          f"{int((m2f.lam_index >= LAM_MAX).sum())} of {len(m2f)} draws at lambda_max")
    # 959: what the n_t=25, alpha=0.05 cell is made of (third panel, M5).
    cell = sel(r, n_target=25, alpha=0.05, rho=0.5)
    cell = cell[cell.feasible.astype(bool)]
    truth(959, "n_t=25, alpha=0.05: sixteen feasible draws, all person "
               "under snow, four seeds per model",
          len(cell) == 16 and set(cell.class_name) == {"person"}
          and bool(cell.experiment.str.contains("snow").all())
          and bool(cell.groupby("model").size().eq(4).all()),
          f"{len(cell)} draws, classes {sorted(set(cell.class_name))}, "
          f"conditions {sorted(set(cell.experiment.str.extract(r'__(\w+?)__')[0]))}")


# --------------------------------------------------------------------------
# J. tier B
# --------------------------------------------------------------------------

    _seqdisjoint_tier_a_claims()

def section_tier_b():
    head("J  Tier B and the clip window")
    tb = load("e4_tierB*")
    try:
        l4 = load("l4_tierB__*")
        tb_all = pd.concat([tb, l4], ignore_index=True)
    except FileNotFoundError:
        tb_all = tb
        note(153, "LoveDA tier-B CSVs missing from this run")

    w = sel(tb_all, method="weighted_crc")
    # Against the grid's last index, not against this column's own maximum.
    # The earlier form compared w.lam with w.lam.max(), which is true of any
    # constant column: replacing every threshold with an informative value --
    # the exact negation of this claim -- left the assertion passing.
    truth(153, "tier B returns lambda_max in EVERY configuration of the main matrix",
          bool((w.lam_index >= LAM_MAX).all()),
          f"{int((w.lam_index < LAM_MAX).sum())} of {len(w)} rows informative; "
          f"lam range [{w.lam.min():.4f}, {w.lam.max():.4f}] against "
          f"lambda_max = {LAMBDA_GRID[LAM_MAX]:g}")
    truth(153, "tier B FNR is 0 and area is 1 in the main matrix",
          bool(np.allclose(w.region_fnr, 0.0)) and
          bool(np.allclose(w.marked_area_fraction, 1.0)),
          f"max FNR={w.region_fnr.max():.4f} min area={w.marked_area_fraction.min():.4f}")

    log_ = w[~w.experiment.str.contains("knn")]
    e_ = log_.weight_ess.dropna()
    truth(156, "ESS equals n to three figures (logistic matrix)",
          bool(np.allclose(e_.values, log_.loc[e_.index, "n_cal_images"].values,
                           rtol=5e-3)),
          f"ess/n ratio range [{(e_ / log_.loc[e_.index, 'n_cal_images']).min():.4f}, "
          f"{(e_ / log_.loc[e_.index, 'n_cal_images']).max():.4f}]")
    knn = w[w.experiment.str.contains("knn")]
    if not knn.empty:
        note(156, "the kNN estimator in the same matrix does NOT give ESS = n",
             f"ess range [{knn.weight_ess.min():.1f}, {knn.weight_ess.max():.1f}] "
             f"against n = {int(knn.n_cal_images.mean())}")

    # 158: cross-fitting
    try:
        cv = sel(load("p4_tierB_cv__*"), method="weighted_crc")
        p_in = mean(sel(w, embedding=None), "weight_p_test", feasible_only=False)
        rng_claim(158, "cross-fitted p_test 0.706-0.708",
                  0.706, 0.708,
                  cv.groupby("experiment")["weight_p_test"].mean().values, 0.0006)
        rng_claim(158, "cross-fitted ESS 161-167", 161, 167,
                  cv.groupby("experiment")["weight_ess"].mean().values, 0.6)
        truth(158, "cross-fitting leaves zero informative draws",
              bool((cv.lam_index >= LAM_MAX).all()),
              f"{int((cv.lam_index < LAM_MAX).sum())} informative; "
              f"lam range [{cv.lam.min():.4f}, {cv.lam.max():.4f}] against "
              f"lambda_max = {LAMBDA_GRID[LAM_MAX]:g}")
    except FileNotFoundError:
        note(158, "cross-fitted CSVs missing")

    # 160/161/162-168: the (c, kappa) ratio sweep
    try:
        p3 = load("p3_lo*")
    except FileNotFoundError:
        note(160, "ratio-sweep CSVs missing")
        return
    lo = p3.experiment.str.extract(r"lo(\d+)_")[0].astype(float) / 100.0
    hi = p3.experiment.str.extract(r"clip(\d+)__")[0].astype(float)
    p3 = p3.assign(c_lo=lo.values, kappa=hi.values, ratio=(hi / lo).values)
    p3 = p3.assign(cond=p3.experiment.str.extract(r"clip\d+__(\w+?)__")[0].values)
    p3 = sel(p3, method="weighted_crc")
    p3f = p3[p3.cond == "fog"]        # the text and Fig. 8 describe the fog axis

    pair = {}
    for (c_, k_) in [(0.05, 2.0), (0.50, 20.0)]:
        d = p3[np.isclose(p3.c_lo, c_) & np.isclose(p3.kappa, k_)]
        if not d.empty:
            pair[(c_, k_)] = (float(d.weight_p_test.mean()),
                              float(d.lam.mean()),
                              float(d.region_fnr.mean()),
                              float(d.marked_area_fraction.mean()))
    if len(pair) == 2:
        a, b = list(pair.values())
        truth(160, "(0.05,2) and (0.50,20) give identical behaviour",
              all(abs(x - y) < 1e-6 for x, y in zip(a, b)),
              f"{tuple(round(x,4) for x in a)} vs {tuple(round(x,4) for x in b)}")
        near(160, "both give p_test = 0.198", 0.198, a[0], 0.0006)

    for al, targets in [(0.20, {50: (0.187, 0.007, 0.829), 40: (0.667, 0.051, 0.352),
                                20: (0.997, 0.207, 0.0085), 10: (0.987, 0.281, 0.0054),
                                8: (0.997, 0.294, None), 4: (0.993, 0.325, None)})]:
        for ratio, (inf_frac, risk, area) in targets.items():
            d = p3f[np.isclose(p3f.ratio, ratio) & np.isclose(p3f.alpha, al)
                    & np.isclose(p3f.rho, 0.5)]
            if d.empty:
                note(163, f"ratio {ratio} not measured at alpha={al}")
                continue
            got_inf = float((d.lam < d.lam.max() - 1e-12).mean())
            got_risk = float(d.region_fnr.mean())
            got_area = float(d.marked_area_fraction.mean())
            near(163, f"ratio {ratio}: informative fraction", inf_frac, got_inf, 0.02)
            near(163, f"ratio {ratio}: risk over ALL draws", risk, got_risk, 0.002)
            if area is not None:
                near(163, f"ratio {ratio}: marked area", area, got_area, 0.006)

    big = p3[(p3.ratio >= 80) & np.isclose(p3.alpha, 0.20)]
    truth(162, "at alpha=0.2 ratios of 80 and above leave nothing informative",
          bool((big.lam_index >= LAM_MAX).all()) if len(big) else False,
          f"{int((big.lam_index < LAM_MAX).sum())} informative of {len(big)}")
    # the two valid cells, conditioned on the informative draws
    for ratio, want in ((40, 0.076), (50, 0.037)):
        d_ = p3f[np.isclose(p3f.ratio, ratio) & np.isclose(p3f.alpha, 0.20) & np.isclose(p3f.rho, 0.5)]
        d_ = d_[d_.lam_index < LAM_MAX]
        near(163, f"ratio {ratio}: risk conditioned on the informative draws", want,
             float(d_.region_fnr.mean()), 0.002)
    d20 = p3f[np.isclose(p3f.ratio, 20) & np.isclose(p3f.alpha, 0.20) & np.isclose(p3f.rho, 0.5)]
    lo20 = float(d20.region_fnr.mean() - 1.959963984540054 * d20.region_fnr.std(ddof=1) / np.sqrt(len(d20)))
    truth(163, "ratio 20 at alpha=0.2 violates with the level outside its interval",
          lo20 > 0.20, f"lower end {lo20:.4f}")

    a10 = p3f[np.isclose(p3f.alpha, 0.10) & np.isclose(p3f.rho, 0.5)]
    inf_by_ratio = a10.assign(inf=(a10.lam_index < LAM_MAX)).groupby("ratio")["inf"].mean()
    risk_by_ratio = a10.groupby("ratio")["region_fnr"].mean()
    area_by_ratio = a10.groupby("ratio")["marked_area_fraction"].mean()
    truth(167, "at alpha=0.1 (fog, rho=0.5) ratios 4-10 are informative in every draw, "
               "ratio 20 in 37%, and nothing above",
          bool((inf_by_ratio[[4.0, 8.0, 10.0]] >= 0.99).all()) and abs(inf_by_ratio[20.0] - 0.37) < 0.01
          and bool((inf_by_ratio[inf_by_ratio.index > 20] == 0).all()),
          str({round(k, 1): round(v, 3) for k, v in inf_by_ratio.items()}))
    truth(167, "ratio 20 is the valid one at alpha=0.1: 0.017 at 65% area",
          abs(risk_by_ratio[20.0] - 0.017) < 0.002 and abs(area_by_ratio[20.0] - 0.649) < 0.01,
          f"risk {risk_by_ratio[20.0]:.4f} area {area_by_ratio[20.0]:.3f}")
    rng_claim(167, "ratios 10 and below violate at alpha=0.1 on fog: 0.106-0.170",
              0.106, 0.170, [risk_by_ratio[k] for k in (4.0, 8.0, 10.0)], 0.001)
    n10 = p3[(p3.cond == "night") & np.isclose(p3.alpha, 0.10) & np.isclose(p3.rho, 0.5)]
    nr = n10.groupby("ratio")["region_fnr"].mean()
    truth(167, "on night at alpha=0.1 ratio 4 violates (0.146) and ratios 8 and 10 hold (0.092, 0.069)",
          nr[4.0] > 0.10 and abs(nr[4.0] - 0.146) < 0.001 and abs(nr[8.0] - 0.092) < 0.001
          and abs(nr[10.0] - 0.069) < 0.001,
          str({round(k, 1): round(v, 3) for k, v in nr.items()}))
    n20 = p3[(p3.cond == "night") & np.isclose(p3.alpha, 0.20) & np.isclose(p3.rho, 0.5)]
    near(167, "night, ratio 20 at alpha=0.2: 0.178", 0.178,
         float(n20[np.isclose(n20.ratio, 20)].region_fnr.mean()), 0.001)

    a05 = p3f[np.isclose(p3f.alpha, 0.05) & np.isclose(p3f.rho, 0.5)]
    inf05 = a05.assign(inf=(a05.lam_index < LAM_MAX)).groupby("ratio")["inf"].mean()
    risk05 = a05.groupby("ratio")["region_fnr"].mean()
    area05 = a05.groupby("ratio")["marked_area_fraction"].mean()
    truth(168, "at alpha=0.05 (fog, rho=0.5) ratio 4 is informative in 99% of draws and "
               "violates (0.070); ratios 8 and 10 are valid with 67% and 33% informative "
               "at 36% and 70% area; nothing at 20 and above",
          inf05[4.0] >= 0.99 and risk05[4.0] > 0.05 and abs(risk05[4.0] - 0.070) < 0.001
          and abs(inf05[8.0] - 0.667) < 0.01 and abs(inf05[10.0] - 0.327) < 0.01
          and risk05[8.0] <= 0.05 and risk05[10.0] <= 0.05
          and abs(area05[8.0] - 0.364) < 0.01 and abs(area05[10.0] - 0.704) < 0.01
          and bool((inf05[inf05.index >= 20] == 0).all()),
          "inf " + str({round(k, 1): round(v, 3) for k, v in inf05.items()}) + " risk "
          + str({round(k, 1): round(v, 3) for k, v in risk05.items()}))

    # 170: reduced level
    d20 = p3[np.isclose(p3.ratio, 20) & np.isclose(p3.alpha, 0.20)]
    if not d20.empty:
        p = float(d20.weight_p_test.mean())
        near(170, "reduced level (alpha - p)/(1-p) at ratio 20 is 0.101",
             0.101, (0.20 - p) / (1 - p), 0.002)


# --------------------------------------------------------------------------
# K/L. LoveDA and MARIDA
# --------------------------------------------------------------------------

    _tierb_test_charge_claims()

def _seqdisjoint_tier_a_claims():
    """Section IV-E: tier A with whole driving sequences held out.

    The paragraph claims the recovery is not proximity within a drive. The
    checks are the cells that carry a feasible draw under both schemes, and
    the three quantities the text prints from them.
    """
    paths = sorted(glob.glob(str(results_dir("experiments")
                                 / "x13_tierA25_seqdisjoint__*.csv")))
    if not paths:
        note(245, "sequence-disjoint tier A not run",
             "run scripts/28_acdc_sequence_schemes.py and the stage-5 arm")
        return

    def cells(pattern):
        frames = []
        for path in sorted(glob.glob(str(results_dir("experiments") / pattern))):
            f = pd.read_csv(path)
            f["experiment"] = os.path.basename(path)[:-4]
            frames.append(f)
        d = pd.concat(frames, ignore_index=True)
        d = d[d["method"] == "region_crc"].copy()
        d["cond"] = d["experiment"].str.extract(r"__(fog|night|rain|snow)__")[0]
        f = feas(d)
        g = (f.groupby(["cond", "alpha", "rho", "class_name"])
              [["region_fnr", "marked_area_fraction"]].mean()
              .groupby(["cond", "alpha", "rho"]).mean())
        return g

    sq = cells("x13_tierA25_seqdisjoint__*.csv")
    pub = cells("e3_tierA25__*segformer_b2_cityscapes.csv")
    j = pub.join(sq, lsuffix="_pub", rsuffix="_sq", how="inner")
    truth(245, "eighteen tier-A cells carry a feasible draw under both schemes",
          len(j) == 18, f"{len(j)} cells")

    lev = j.index.get_level_values("alpha").values
    over_idx = j.index[j["region_fnr_sq"].values > lev]
    truth(245, "two sequence-disjoint cells exceed their level, the fog cells at "
               "alpha=0.1 (0.102 and 0.104 against 0.070 and 0.076), and no published cell does",
          len(over_idx) == 2 and all(k[0] == "fog" and np.isclose(k[1], 0.1) for k in over_idx)
          and int((j["region_fnr_pub"].values > lev).sum()) == 0
          and abs(float(j.loc[("fog", 0.1, 0.1), "region_fnr_sq"]) - 0.102) < 5e-4
          and abs(float(j.loc[("fog", 0.1, 0.5), "region_fnr_sq"]) - 0.104) < 5e-4
          and abs(float(j.loc[("fog", 0.1, 0.1), "region_fnr_pub"]) - 0.070) < 5e-4
          and abs(float(j.loc[("fog", 0.1, 0.5), "region_fnr_pub"]) - 0.076) < 5e-4,
          f"{len(over_idx)} of {len(j)} cells above the level: {list(over_idx)}; worst "
          f"{float(j['region_fnr_sq'].max()):.4f}")
    # the level lies inside the draw interval of both, and both rest on person alone
    sq_paths = sorted(glob.glob(str(results_dir("experiments") / "x13_tierA25_seqdisjoint__fog__*.csv")))
    sqf = pd.concat([pd.read_csv(p_) for p_ in sq_paths], ignore_index=True)
    sqf = feas(sqf[(sqf["method"] == "region_crc") & np.isclose(sqf["alpha"], 0.1)])
    dl = sqf.groupby(["rho", "seed"])["region_fnr"].mean().groupby("rho")
    lo_fog = dl.mean() - 1.959963984540054 * dl.std(ddof=1) / np.sqrt(dl.size())
    truth(245, "the level lies inside the draw interval of both fog cells, which rest on "
               "23 person draws",
          bool((lo_fog.values <= 0.1).all()) and set(sqf.class_name) == {"person"}
          and bool((dl.size() == 23).all()),
          f"lower ends {lo_fog.round(4).to_dict()}, classes {sorted(set(sqf.class_name))}, "
          f"draws {dl.size().to_dict()}")
    near(246, "largest published tier-A cell at n_t=25", 0.159,
         float(j["region_fnr_pub"].max()), tol=5e-4)
    near(246, "largest sequence-disjoint cell at n_t=25", 0.181,
         float(j["region_fnr_sq"].max()), tol=5e-4)
    truth(246, "the published maximum is night (a=0.2, rho=0.1) and the disjoint maximum night (a=0.2, rho=0.5), both below 0.2",
          j["region_fnr_pub"].idxmax() == ("night", 0.2, 0.1) and j["region_fnr_sq"].idxmax() == ("night", 0.2, 0.5),
          f"{j['region_fnr_pub'].idxmax()} -> {j['region_fnr_sq'].idxmax()}")
    near(246, "the night cell moves from 0.159 to 0.178 under the disjoint scheme", 0.178,
         float(j.loc[j["region_fnr_pub"].idxmax(), "region_fnr_sq"]), tol=5e-4)
    move = j["region_fnr_sq"] - j["region_fnr_pub"]
    near(246, "largest single movement in region FNR", 0.036,
         float(move.max()), tol=5e-4)
    near(246, "largest negative movement in region FNR", -0.038,
         float(move.min()), tol=5e-4)
    truth(246, "the largest movements are both on snow (a=0.2 rho=0.5 up, a=0.1 rho=0.1 down)",
          move.idxmax() == ("snow", 0.20, 0.5) and move.idxmin() == ("snow", 0.10, 0.1),
          f"{move.idxmax()} / {move.idxmin()}")
    area = j["marked_area_fraction_sq"] - j["marked_area_fraction_pub"]
    truth(247, "marked area is smaller in fifteen of the eighteen cells",
          int((area < 0).sum()) == 15, f"{int((area < 0).sum())} of {len(area)}")
    truth(247, "the largest area reduction is 19 points (fog, a=0.1, rho=0.1)",
          0.185 <= float(-area.min()) <= 0.195 and area.idxmin() == ("fog", 0.10, 0.1),
          f"largest reduction {float(-area.min()):.4f} at {area.idxmin()}")

    _seqdisjoint_frame_cost()
    _sequence_overlap_claims()
    _panel4_arms()


def _marida_confidence_size_strata():
    """Section IV-H: the confidence gap is confounded by component size.

    The paragraph claims that stratifying by size removes most of the apparent
    confidence effect. That is a negative claim about the confidence layer, so
    the stratum rates it prints are checked directly against the component
    table rather than inferred from the pooled numbers.
    """
    path = results_dir("experiments") / "x11_marida_confidence__components.csv"
    if not path.exists():
        note(253, "MARIDA component table absent", str(path))
        return
    d = pd.read_csv(path)
    t = d[d["official_group"] == "test"]
    col = "missed__alpha0.2__rho0.5"
    truth(253, "the official test group holds 223 components",
          len(t) == 223, f"{len(t)} components")

    hi, lo = t[t["any_high"]], t[~t["any_high"]]
    truth(253, "91 of the 158 no-High components are a single pixel, "
               "against 22 of the 65 any-High",
          (int((lo["size_px"] == 1).sum()), len(lo),
           int((hi["size_px"] == 1).sum()), len(hi)) == (91, 158, 22, 65),
          f"{int((lo['size_px'] == 1).sum())} of {len(lo)} and "
          f"{int((hi['size_px'] == 1).sum())} of {len(hi)}")

    def rate(frame, lo_px, hi_px):
        g = frame[(frame["size_px"] >= lo_px) & (frame["size_px"] <= hi_px)]
        return (float(g[col].mean()) if len(g) else float("nan")), len(g)

    for px, want_hi, want_lo in ((1, 0.227, 0.220), (2, 0.100, 0.143)):
        rh, nh = rate(hi, px, px)
        rl, nl = rate(lo, px, px)
        near(253, f"any-High miss rate at {px} px", want_hi, rh, tol=5e-4)
        near(253, f"no-High miss rate at {px} px", want_lo, rl, tol=5e-4)

    big_hi, n_big_hi = rate(hi, 3, 10 ** 9)
    big_lo, n_big_lo = rate(lo, 3, 10 ** 9)
    truth(254, "at three pixels or more the any-High group misses none of its "
               "13 components against one of the 11 others",
          (n_big_hi, n_big_lo) == (13, 11)
          and big_hi == 0.0 and abs(big_lo * n_big_lo - 1.0) < 1e-9,
          f"any-High {big_hi:.3f} of {n_big_hi}, no-High {big_lo:.3f} of {n_big_lo}")

    # The point of the paragraph: the pooled gap is much larger than any
    # within-stratum gap, which is what "confounded by size" means.
    pooled = float(lo[col].mean()) - float(hi[col].mean())
    within = max(abs(rate(lo, px, px)[0] - rate(hi, px, px)[0]) for px in (1, 2))
    truth(254, "the pooled confidence gap exceeds every within-stratum gap",
          pooled > within,
          f"pooled {pooled:+.3f} against a largest within-stratum gap of {within:.3f}")


def _sequence_overlap_claims():
    """Section IV-A: at n_t=25, the share of test frames whose driving sequence
    also supplied a calibration frame. The text once said 91-96%; measured from
    the committed schemes it is a median of 92-100% by condition over a per-draw
    range that reaches down to 70% (third panel, M2)."""
    stats = {}
    for cond in ("fog", "night", "rain", "snow"):
        path = splits_dir() / f"acdc_{cond}_targetcal25.json"
        if not path.exists():
            note(956, f"{cond} target-calibration scheme absent", str(path))
            return
        with open(path) as f:
            seeds = json.load(f)["seeds"]
        fr = []
        for v in seeds.values():
            cal = {i.split("/")[2] for i in v["target_calibration"]}
            fr.append(float(np.mean([i.split("/")[2] in cal for i in v["test"]])))
        stats[cond] = (float(np.median(fr)), float(min(fr)), float(max(fr)))
    meds = [v[0] for v in stats.values()]
    mins = [v[1] for v in stats.values()]
    rng_claim(956, "median sequence overlap at n_t=25 is 92-100% by condition",
              0.922, 1.000, meds, 0.005)
    truth(956, "no draw falls below 70% overlap", min(mins) >= 0.70,
          "per-condition minima " + str({k: round(v[1], 3) for k, v in stats.items()}))
    truth(956, "the retired 91-96% range was wrong at both ends",
          max(meds) > 0.96 and min(mins) < 0.91,
          f"max median {max(meds):.3f}, min draw {min(mins):.3f}")


def _panel4_arms():
    """Fourth panel (2026-09-07): the arms the article now quotes.

    x13 at n_t = 50/100, tier B without shift (x14), source CRC at the
    reduced level (x15) and dilation CRC (x16). Each check pins a sentence of
    Section IV; x17 (Mask2Former on the log-tail grid) is reported separately
    once its role in the tables is decided.
    """
    import glob as _glob
    exp = results_dir("experiments")

    def frame(pattern):
        paths = sorted(_glob.glob(str(exp / pattern)))
        if not paths:
            return None
        fs = []
        for path in paths:
            f = pd.read_csv(path)
            f["experiment"] = os.path.basename(path)[:-4]
            fs.append(f)
        return pd.concat(fs, ignore_index=True)

    # 961: sequence-disjoint tier A at the two larger budgets.
    for n_t, want_sq, want_pub, n_over_sq, n_over_pub in ((50, 0.184, 0.191, 2, 2), (100, 0.187, 0.183, 3, 0)):
        sq = frame(f"x13_tierA{n_t}_seqdisjoint__*.csv")
        pub = frame(f"e3_tierA{n_t}__*segformer_b2_cityscapes.csv")
        if sq is None or pub is None:
            note(961, f"x13 at n_t={n_t} absent")
            continue
        def cells(d):
            d = d[d["method"] == "region_crc"].copy()
            d["cond"] = d["experiment"].str.extract(r"__(fog|night|rain|snow)__")[0]
            f = feas(d)
            return (f.groupby(["cond", "alpha", "rho", "class_name"])["region_fnr"].mean()
                     .groupby(["cond", "alpha", "rho"]).mean())
        j = pd.concat([cells(pub).rename("pub"), cells(sq).rename("sq")], axis=1, join="inner")
        lev = j.index.get_level_values("alpha").values
        o_sq = j.index[j["sq"].values > lev + 1e-12]
        o_pub = j.index[j["pub"].values > lev + 1e-12]
        truth(961, f"n_t={n_t}: {n_over_sq} disjoint and {n_over_pub} published cells exceed "
                   "their level, all at alpha=0.05",
              len(o_sq) == n_over_sq and len(o_pub) == n_over_pub
              and all(np.isclose(k[1], 0.05) for k in list(o_sq) + list(o_pub)),
              f"{len(j)} cells; disjoint over {list(o_sq)}; published over {list(o_pub)}")
        truth(961, f"n_t={n_t}: the largest cells are the night cells at alpha=0.2, rho=0.5",
              j["sq"].idxmax() == ("night", 0.2, 0.5) and j["pub"].idxmax() in
              (("night", 0.2, 0.5), ("fog", 0.2, 0.5)),
              f"{j['sq'].idxmax()} / {j['pub'].idxmax()}")
        near(961, f"n_t={n_t}: largest sequence-disjoint cell", want_sq, float(j["sq"].max()), 5e-4)
        near(961, f"n_t={n_t}: largest published cell", want_pub, float(j["pub"].max()), 5e-4)

    # 962: tier B without shift.
    x14 = frame("x14_tierB_noshift__*.csv")
    if x14 is None:
        note(962, "x14 absent")
    else:
        x14["arm"] = x14["experiment"].str.extract(r"__(dinov2_vitb14|clip_vitb16)__clip(\d+)").apply(
            lambda r: f"{r[0]}:{r[1]}", axis=1)
        w = x14[x14["method"] == "weighted_crc"]
        for arm, want_p in (("dinov2_vitb14:20", 0.684), ("dinov2_vitb14:5", 0.356),
                            ("dinov2_vitb14:2", 0.183), ("clip_vitb16:20", 0.574)):
            d = w[w["arm"] == arm]
            near(962, f"no-shift conservative test mass, {arm}", want_p,
                 float(d["weight_p_test"].mean()), 0.005)
        d20 = w[w["arm"] == "dinov2_vitb14:20"]
        k = 20.0
        near(962, "no-shift sum of source weights at kappa=20 is about 9", 9.2,
             float((k / d20["weight_p_test"] - k).mean()), 0.5)
        for arm in ("dinov2_vitb14:20", "dinov2_vitb14:5", "clip_vitb16:20"):
            d = w[w["arm"] == arm]
            truth(962, f"no informative draw at any level, {arm}",
                  bool((d["lam"] >= 1.0 - 1e-12).all()),
                  f"{int((d['lam'] < 1.0 - 1e-12).sum())} informative of {len(d)}")
        d2 = w[(w["arm"] == "dinov2_vitb14:2") & np.isclose(w["alpha"], 0.2)]
        inf2 = float((d2["lam"] < 1.0 - 1e-12).mean())
        near(962, "kappa=2, alpha=0.2: two thirds of the draws informative", 0.667, inf2, 0.01)
        rng_claim(962, "kappa=2, alpha=0.2: marked area 36-37%", 0.363, 0.369,
                  list(d2.groupby("rho")["marked_area_fraction"].mean().values), 0.005)
        r = x14[(x14["method"] == "region_crc") & np.isclose(x14["alpha"], 0.2)]
        rng_claim(962, "unweighted region CRC on the same splits marks 2.0-2.3%", 0.0196, 0.0234,
                  list(r.groupby("rho")["marked_area_fraction"].mean().values), 0.0005)
        dc = w[w["arm"] == "clip_vitb16:20"]
        near(962, "no-shift sum of source weights at kappa=20 with CLIP is about 16", 15.9,
             float((k / dc["weight_p_test"] - k).mean()), 0.5)

    # 963: source CRC at the reduced level reproduces tier B.
    x15 = frame("x15_source_reduced__*.csv")
    p3 = frame("p3_lo*.csv")
    if x15 is None or p3 is None:
        note(963, "x15 or p3 absent")
    else:
        p3 = p3[p3["method"] == "weighted_crc"].copy()
        lo = p3["experiment"].str.extract(r"lo(\d{3})_")[0].astype(int) / 100
        hi = p3["experiment"].str.extract(r"clip(\d+)__")[0].astype(int)
        p3["ratio"] = (hi / lo).round(0)
        p3["cond"] = p3["experiment"].str.extract(r"__(fog|night)__")[0]
        x15 = x15[x15["method"] == "region_crc"].copy()
        x15["cond"] = x15["experiment"].str.extract(r"__(fog|night)__")[0]
        gaps = {}
        for cond in ("fog", "night"):
            for ratio, a_red in ((4, 0.1801), (8, 0.1602), (10, 0.1503), (20, 0.1009)):
                s_ = x15[(x15["cond"] == cond) & np.isclose(x15["alpha"], a_red)
                         & np.isclose(x15["rho"], 0.5)]["region_fnr"].mean()
                t_ = p3[(p3["cond"] == cond) & np.isclose(p3["ratio"], ratio)
                        & np.isclose(p3["alpha"], 0.2) & np.isclose(p3["rho"], 0.5)]["region_fnr"].mean()
                gaps[(cond, ratio)] = float(s_ - t_)
        close = [abs(v) for (c, r), v in gaps.items()]
        truth(963, "source CRC at the reduced level is within 0.011 of tier B at ratios 4, 8, 10 and 20",
              bool(close) and max(close) <= 0.011, str({k: round(v, 3) for k, v in gaps.items()}))
        near(963, "fog, ratio 8: 0.287 against 0.294", 0.287,
             float(x15[(x15["cond"] == "fog") & np.isclose(x15["alpha"], 0.1602)
                       & np.isclose(x15["rho"], 0.5)]["region_fnr"].mean()), 0.001)
        near(963, "fog, ratio 10: 0.274 against 0.281", 0.274,
             float(x15[(x15["cond"] == "fog") & np.isclose(x15["alpha"], 0.1503)
                       & np.isclose(x15["rho"], 0.5)]["region_fnr"].mean()), 0.001)

    # 964: dilation CRC.
    x16 = frame("x16_dilation__*.csv")
    e1 = frame("e1_indist__*.csv")
    if x16 is None:
        note(964, "x16 absent")
    else:
        d = x16[x16["method"] == "region_crc"]
        tight = d[d["alpha"] < 0.15]
        truth(964, "dilation CRC needs the whole image at alpha <= 0.1 for every model",
              bool((tight["lam"] >= 1.0 - 1e-12).all()),
              f"{int((tight['lam'] < 1.0 - 1e-12).sum())} interior radii of {len(tight)}")
        b5 = d[(d["model"] == "segformer_b5_cityscapes__dilation") & np.isclose(d["alpha"], 0.2)]
        # The radius is an index quantity (0.1 px per grid step on every grid),
        # never lambda itself: on the log-tail grid lambda*100 is meaningless.
        rng_claim(964, "SegFormer-B5 at alpha=0.2: mean radius 71-75 px counting the "
                       "saturated draws at 100 px", 70.8, 74.8,
                  list((b5.groupby("rho")["lam_index"].mean() / 10.0).values), 0.5)
        interior = b5[b5["lam_index"] < LAM_MAX]
        rng_claim(964, "an interior radius in 56-58% of the draws", 0.563, 0.58,
                  list(b5.groupby("rho")["lam_index"].apply(lambda s: float((s < LAM_MAX).mean())).values), 0.005)
        rng_claim(964, "of about 50-55 px", 49.6, 55.3,
                  list((interior.groupby("rho")["lam_index"].mean() / 10.0).values), 0.5)
        rng_claim(964, "and 44-46% marked area", 0.4434, 0.4621,
                  list(b5.groupby("rho")["marked_area_fraction"].mean().values), 0.005)
        others = d[(d["model"] != "segformer_b5_cityscapes__dilation") & np.isclose(d["alpha"], 0.2)]
        truth(964, "the other three models need the whole image at alpha=0.2 as well",
              bool((others["lam"] >= 1.0 - 1e-12).all()),
              f"{int((others['lam'] < 1.0 - 1e-12).sum())} interior radii of {len(others)}")
        if e1 is not None:
            key = ["class_name", "alpha", "rho", "seed"]
            a_ = x16[x16["method"] == "argmax"].assign(model=lambda f: f["model"].str.replace("__dilation", "", regex=False))
            b_ = e1[e1["method"] == "argmax"]
            m = a_.merge(b_, on=["model"] + key, suffixes=("_dil", "_e1"))
            truth(964, "the dilation family's argmax rows coincide with the published argmax rows",
                  len(m) > 0 and float((m["region_fnr_dil"] - m["region_fnr_e1"]).abs().max()) < 1e-9,
                  f"{len(m)} rows matched")


def _seqdisjoint_frame_cost():
    """Section IV-E: what holding out whole drives costs in test frames."""
    paths = sorted(glob.glob(str(splits_dir()
                                 / "acdc_*_targetcal25_seqdisjoint.meta.json")))
    if not paths:
        note(248, "sequence-disjoint scheme metadata absent", str(splits_dir()))
        return
    lost = {}
    for path in paths:
        with open(path) as f:
            m = json.load(f)
        lost[m["condition"]] = (m["test_frames_in_published_scheme"]
                                - m["test_frames"]["median"])
    truth(248, "four conditions carry a sequence-disjoint scheme at n_t=25",
          len(lost) == 4, ", ".join(f"{k} {v:.0f}" for k, v in sorted(lost.items())))
    # A containment claim, not an endpoint claim: the text says the cost lies
    # between 25 and 59 frames, and the largest is rain's 58.5.
    v = list(lost.values())
    truth(248, "holding out whole drives costs between 25 and 59 test frames "
               "at the median",
          bool(v) and min(v) >= 25 and max(v) <= 59,
          "per condition: " + ", ".join(f"{k} {lost[k]:g}" for k in sorted(lost)))


def _marida_confidence_claims():
    """Section IV-H: MARIDA annotation confidence.

    The pixel split is checked against MARIDA's own published total of 3399
    annotated debris pixels, so a change in how the rasters are read shows up
    as a disagreement with the source rather than as a new number.
    """
    path = results_dir("experiments") / "x11_marida_confidence.json"
    if not path.exists():
        note(250, "MARIDA confidence layer not read",
             "run scripts/26_marida_confidence.py")
        return
    with open(path) as f:
        d = json.load(f)
    px = d["pixels_by_confidence"]
    truth(250, "3399 annotated debris pixels, 1625 High / 1235 Moderate / 539 Low",
          (px["high"], px["moderate"], px["low"]) == (1625, 1235, 539)
          and sum(px.values()) == 3399,
          f"{px}, total {sum(px.values())}")
    truth(250, "1330 debris components over 373 patches",
          (d["n_components"], d["n_debris_patches"]) == (1330, 373),
          f"{d['n_components']} components, {d['n_debris_patches']} patches")
    near(250, "44.4% of components carry at least one High pixel",
         0.444, float(d["frac_any_high"]), tol=5e-4)

    rows = {(round(r["alpha"], 2), round(r["rho"], 1)): r
            for r in d["official_split_operating_points"]}
    r = rows[(0.20, 0.5)]
    truth(251, "158 of the 223 official test components carry no High pixel",
          (r["n_test_components_no_high"], r["n_test_components"]) == (158, 223),
          f"{r['n_test_components_no_high']} of {r['n_test_components']}")
    near(251, "single model at alpha=0.2: miss rate on any-High components",
         0.123, float(r["miss_rate_component_avg_any_high"]), tol=5e-4)
    near(251, "single model at alpha=0.2: miss rate on no-High components",
         0.184, float(r["miss_rate_component_avg_no_high"]), tol=5e-4)
    r5 = rows[(0.05, 0.5)]
    near(251, "single model at alpha=0.05: any-High", 0.046,
         float(r5["miss_rate_component_avg_any_high"]), tol=5e-4)
    near(251, "single model at alpha=0.05: no-High", 0.076,
         float(r5["miss_rate_component_avg_no_high"]), tol=5e-4)
    truth(251, "the no-High group is missed more often at every level",
          all(rows[(a, 0.5)]["miss_rate_component_avg_no_high"]
              > rows[(a, 0.5)]["miss_rate_component_avg_any_high"]
              for a in (0.05, 0.10, 0.20)),
          "; ".join(f"a={a}: {rows[(a,0.5)]['miss_rate_component_avg_any_high']:.3f}"
                    f" vs {rows[(a,0.5)]['miss_rate_component_avg_no_high']:.3f}"
                    for a in (0.05, 0.10, 0.20)))

    ens = results_dir("experiments") / "x11_marida_confidence__ens5.json"
    if ens.exists():
        with open(ens) as f:
            e = json.load(f)
        er = {(round(r["alpha"], 2), round(r["rho"], 1)): r
              for r in e["official_split_operating_points"]}[(0.20, 0.5)]
        near(251, "ensemble at alpha=0.2: any-High", 0.169,
             float(er["miss_rate_component_avg_any_high"]), tol=5e-4)
        near(251, "ensemble at alpha=0.2: no-High", 0.234,
             float(er["miss_rate_component_avg_no_high"]), tol=5e-4)
    else:
        note(251, "the ensemble confidence run is missing",
             "rerun stage 26 with --out-stem x11_marida_confidence__ens5")

    _marida_confidence_size_strata()

    sz = d["size_by_any_high"]
    truth(252, "both confidence groups have a median size of two pixels",
          sz["any_high"]["size_px_median"] == 2.0
          and sz["no_high"]["size_px_median"] == 2.0,
          f"any-High {sz['any_high']['size_px_median']}, "
          f"no-High {sz['no_high']['size_px_median']}")
    near(252, "49.6% of no-High components are a single pixel",
         0.496, float(sz["no_high"]["frac_size_eq_1"]), tol=5e-4)
    near(252, "26.1% of any-High components are a single pixel",
         0.261, float(sz["any_high"]["frac_size_eq_1"]), tol=5e-4)
    near(252, "93.5% of no-High components are three pixels or fewer",
         0.935, float(sz["no_high"]["frac_size_le_3"]), tol=5e-4)
    near(252, "69.8% of any-High components are three pixels or fewer",
         0.698, float(sz["any_high"]["frac_size_le_3"]), tol=5e-4)


def _tierb_test_charge_claims():
    """Section IV-F: charging the test point its own estimated weight.

    The paragraph rests on two measurements -- that the four arms agree
    exactly, and that they agree because every target weight is already at
    the ceiling. Both are checked, because the second is what licenses the
    sentence saying the ceiling IS the estimate rather than a bound on it.
    """
    paths = sorted(glob.glob(str(results_dir("experiments")
                                 / "x10_tierb_test_charge__*.csv")))
    if not paths:
        note(240, "tier-B test-charge arms not computed",
             "run scripts/25_tierb_test_charge.py")
        return
    frames = []
    for path in paths:
        f = pd.read_csv(path)
        f["source"] = os.path.basename(path)[:-4]
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)

    key = ["source", "condition", "kappa", "c_low", "alpha", "rho", "seed",
           "class_name"]
    worst, compared = 0.0, 0
    for col in ("region_fnr", "marked_area_fraction", "frac_informative"):
        piv = d.pivot_table(index=key, columns="arm", values=col)
        base = piv["published/ceiling"]
        for arm in piv.columns:
            if arm == "published/ceiling":
                continue
            diff = (piv[arm] - base).abs()
            worst = max(worst, float(diff.max()))
            compared += int(diff.notna().sum())
    truth(240, "charging the test point its own weight changes nothing: the "
               "four arms agree to machine zero",
          worst == 0.0,
          f"{compared} cell comparisons over three metrics, largest "
          f"|difference| {worst:.3e}")

    cells = d.groupby(["source", "condition", "kappa"]).size()
    truth(240, "eighteen condition-clip cells, twelve on ACDC and six on LoveDA",
          len(cells) == 18
          and sum(1 for k in cells.index if "segformer_b2_cityscapes" in k[0]) == 12,
          f"{len(cells)} cells: " + ", ".join(sorted({k[0].split('__')[-1] for k in cells.index})))

    truth(241, "the median target weight equals kappa in every cell",
          bool(np.allclose(d["tgt_w_median"], d["kappa"])),
          f"largest |median - kappa| "
          f"{float((d['tgt_w_median'] - d['kappa']).abs().max()):.3e}")

    acdc = d[d["source"].str.contains("segformer_b2_cityscapes")]
    love = d[~d["source"].str.contains("segformer_b2_cityscapes")]
    near(241, "fraction of ACDC target images at the ceiling", 1.000,
         float(acdc["tgt_frac_ceiling"].min()), tol=5e-4)
    rng_claim(241, "fraction of LoveDA target images at the ceiling",
              0.991, 1.000, [float(love["tgt_frac_ceiling"].min()),
                             float(love["tgt_frac_ceiling"].max())], tol=5e-4)
    truth(241, "no target image reaches the floor",
          float(d["tgt_frac_floor"].max()) == 0.0,
          f"largest floor fraction {float(d['tgt_frac_floor'].max()):.3e}")
    truth(241, "every source weight is at the floor",
          bool(np.allclose(d["cal_w_median"], d["c_low"])),
          f"source median in [{d['cal_w_median'].min():.4f}, "
          f"{d['cal_w_median'].max():.4f}]")


def section_loveda():
    head("K  LoveDA")
    try:
        i = load("l1_indist__*")
        b = load("l2_break__*")
    except FileNotFoundError:
        note(177, "LoveDA CSVs missing")
        return
    r = sel(i, method="region_crc", rho=0.5)
    c = cellmean(r, ["experiment", "alpha"])
    lev = c.index.get_level_values("alpha").values
    truth(177, "LoveDA in-domain validity holds on both domains at every level "
               "when the two classes are averaged (per class: see check 920)",
          bool((c.values <= lev + 1e-12).all()), str({k: round(v, 3) for k, v in c.items()}))
    near(177, "tightest urban cell 0.200", 0.200,
         mean(sel(r, experiment="l1_indist__urban", alpha=0.20)))
    near(177, "tightest rural cell 0.196", 0.196,
         mean(sel(r, experiment="l1_indist__rural", alpha=0.20)))
    near(178, "LoveDA urban pixel CRC region FNR 0.220", 0.220,
         mean(sel(i, method="pixel_crc", experiment="l1_indist__urban",
                  alpha=0.20, rho=0.5)))

    u2r = cellmean(sel(b, method="region_crc", experiment="l2_break__urban2rural",
                       rho=0.5), ["alpha"])
    ratios = [u2r[a_] / a_ for a_ in ALPHAS if a_ in u2r.index]
    rng_claim(180, "urban->rural violates by 2.3-3.5x", 2.46, 3.47, ratios, 0.01)
    u2r1 = cellmean(sel(b, method="region_crc", experiment="l2_break__urban2rural",
                        rho=0.1), ["alpha"])
    rng_claim(180, "urban->rural at rho=0.1 violates by 2.3-3.2x", 2.348, 3.183,
              [u2r1[a_] / a_ for a_ in ALPHAS], 0.01)
    r2u = cellmean(sel(b, method="region_crc", experiment="l2_break__rural2urban",
                       rho=0.5), ["alpha"])
    truth(181, "rural->urban stays within the level at alpha>=0.1 when the two classes "
               "are averaged (rho=0.5) and exceeds it by 1.2x at alpha=0.05",
          r2u[0.1] <= 0.1 + 1e-12 and r2u[0.2] <= 0.2 + 1e-12
          and 1.2 <= r2u[0.05] / 0.05 <= 1.25,
          str({k: round(v, 3) for k, v in r2u.items()}))
    # Per class the same direction is not intact (third panel, M6): the text
    # now says four of twelve cells, all water, worst 1.27x at a=0.1, rho=0.1.
    pc = sel(b, method="region_crc", experiment="l2_break__rural2urban")
    if "feasible" in pc.columns:
        pc = pc[pc.feasible.astype(bool)]
    pcells = pc.groupby(["class_name", "alpha", "rho"])["region_fnr"].mean()
    pratio = pcells / pcells.index.get_level_values("alpha")
    pover = pratio[pratio > 1 + 1e-9]
    truth(181, "per class, rural->urban exceeds its level in six of the twelve "
               "cells, all of them water",
          len(pcells) == 12 and len(pover) == 6
          and all(k[0] == "water" for k in pover.index),
          f"{len(pover)} of {len(pcells)}: " + str([(k[0], k[1], k[2], round(v, 3)) for k, v in pover.items()]))
    near(181, "the worst rural->urban class cell is 1.58x", 1.583, float(pratio.max()), 0.005)
    truth(181, "and it sits at alpha=0.05, rho=0.1",
          tuple(pratio.idxmax()[1:]) == (0.05, 0.1), str(pratio.idxmax()))
    pu = sel(b, method="region_crc", experiment="l2_break__urban2rural")
    pu = pu[pu.feasible.astype(bool)] if "feasible" in pu.columns else pu
    pucells = pu.groupby(["class_name", "alpha", "rho"])["region_fnr"].mean()
    near(181, "urban->rural water reaches 0.223 against 0.05 (rho=0.5)", 0.223,
         float(pucells[("water", 0.05, 0.5)]), 0.001)

    try:
        t = load("l3_tierA*")
        t["n_target"] = pd.to_numeric(t.experiment.str.extract(r"tierA(\d+)")[0])
        tr = sel(t, method="region_crc", rho=0.5)
        ar = list(cellmean(sel(tr, alpha=0.20), ["n_target", "experiment"],
                           "marked_area_fraction").values)
        rng_claim(183, "LoveDA tier-A area 20-24% at alpha=0.2", 0.1986, 0.2352, ar, 0.006)
        cc = cellmean(tr, ["n_target", "experiment", "class_name", "alpha"])
        lv = cc.index.get_level_values("alpha").values
        truth(183, "LoveDA tier A restores the guarantee in every feasible cell",
              bool((cc.values <= lv + 1e-12).all()),
              f"{int((cc.values > lv).sum())} violations")
        inf = 1 - tr.feasible.astype(bool).mean()
        note(184, "LoveDA tier-A infeasible-draw fraction", f"{inf:.4f}")
    except FileNotFoundError:
        note(183, "LoveDA tier-A CSVs missing")


def _components(model_key="marida_unet_official_holdout"):
    """Pooled ground-truth component table, or None if it cannot be read.

    Reading parquet needs an optional engine that a bare checkout may lack.
    A missing engine is reported as a skipped check rather than an error, so
    the audit still runs where the tables cannot be opened.
    """
    import pandas as pd
    frames = []
    for d in ("marida_train", "marida_val", "marida_test"):
        p = results_dir("raw") / model_key / d / "components.parquet"
        if not p.exists():
            continue
        try:
            frames.append(pd.read_parquet(p))
        except ImportError:
            return None
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _component_summary():
    """Committed reduction of the pooled component tables (stage 24)."""
    import json
    p = results_dir("experiments") / "x9_marida_component_stats.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _n_debris_patches():
    """Number of MARIDA patches carrying at least one debris component."""
    s = _component_summary()
    if s is not None:
        return int(s["n_debris_patches"])
    comp = _components()
    return None if comp is None else int(comp["image_id"].nunique())


def _size_stats():
    """(median size in px, fraction of components at 3 px or smaller)."""
    s = _component_summary()
    if s is not None:
        return float(s["size_px_median"]), float(s["frac_size_le_3"])
    comp = _components()
    if comp is None:
        return None
    size = comp["size_px"]
    return float(size.median()), float((size <= 3).mean())


def _spring_geometry():
    """Patch, scene and tile counts of the held-out spring test group."""
    import csv
    from record.paths import splits_dir
    meta_path = splits_dir() / "marida_meta.csv"
    scheme_path = splits_dir() / "marida_holdout_season_spring.json"
    if not meta_path.exists() or not scheme_path.exists():
        return None
    import json
    ids = json.load(open(scheme_path))["seeds"]["0"]["test"]
    meta = {r["patch_id"]: r for r in csv.DictReader(open(meta_path))}
    rows = [meta[i] for i in ids if i in meta]
    if not rows:
        return None
    return {"patches": len(rows),
            "scenes": len({r["scene"] for r in rows}),
            "tiles": len({r["tile"] for r in rows})}


def _test_f1(model_key):
    """Pixel-level debris F1 recorded by scripts/10_marida_test_f1.py."""
    import json
    p = results_dir("experiments") / f"marida_test_f1__{model_key}.json"
    if not p.exists():
        return None
    return float(json.load(open(p))["f1"])


def section_marida():
    head("L  MARIDA")
    try:
        o = load("h1_official__*")
        reg = load("h2_region__*")
        sea = load("h3_season__*")
    except FileNotFoundError:
        note(192, "MARIDA held-out CSVs missing")
        return

    allm = pd.concat([o, reg, sea], ignore_index=True)
    rm = sel(allm, method="region_crc", rho=0.5)
    ro = sel(o, method="region_crc", rho=0.5)

    truth(45, "MARIDA partitions carry a single seed",
          o.seed.nunique() == 1 and reg.seed.nunique() == 1, f"{o.seed.nunique()}")
    near(199, "official test group has 223 components", 223,
         float(sel(o, alpha=0.20).n_test_components.max()), 0.5)

    # MARIDA matrix, cell by cell. One model per axis; only the official axis
    # carries both a single member and the ensemble.
    tab = {
        "h1_official__marida_unet_official_holdout": [0.066, 0.095, 0.200],
        "h1_official__marida_unet_official_holdout_ens5": [0.069, 0.098, 0.250],
        "h2_region__16PCC__marida_unet_holdout_region_16PCC": [0.000, 0.027, 0.094],
        "h2_region__16PDC__marida_unet_holdout_region_16PDC": [0.009, 0.027, 0.142],
        "h2_region__16PEC__marida_unet_holdout_region_16PEC": [0.000, 0.159, 0.333],
        "h2_region__18QYF__marida_unet_holdout_region_18QYF": [0.001, 0.096, 0.265],
        "h2_region__48PZC__marida_unet_holdout_region_48PZC": [0.000, 0.250, 0.375],
        "h3_season__winter__marida_unet_holdout_season_winter": [0.030, 0.041, 0.108],
        "h3_season__spring__marida_unet_holdout_season_spring": [0.217, 0.346, 0.430],
        "h3_season__summer__marida_unet_holdout_season_summer": [0.000, 0.005, 0.044],
        "h3_season__autumn__marida_unet_holdout_season_autumn": [0.001, 0.002, 0.052],
    }
    for exp, vals in tab.items():
        for al, v in zip(ALPHAS, vals):
            near(205, f"MARIDA {exp.split('__', 1)[1]} a={al}", v,
                 mean(sel(rm, experiment=exp, alpha=al)), 0.001)

    # 192: the official split is an in-distribution setting, so no
    # official cell may separate from its own scene-level interval.
    off_sep = [(e, a) for e in tab if e.startswith("h1_") for a in ALPHAS
               if float(sel(rm, experiment=e, alpha=a).fnr_boot_lo.mean()) > a]
    truth(192, "no official-split cell separates from its scene-level interval",
          not off_sep, str(off_sep) if off_sep else "none")

    # 194/195: every held-out tile satisfies the bound at the strictest level,
    # and no regional cell survives its own uncertainty.
    tiles = {e.split("__")[1]: [mean(sel(rm, experiment=e, alpha=a)) for a in ALPHAS]
             for e in tab if e.startswith("h2_region")}
    truth(195, "all five held-out tiles satisfy the bound at alpha=0.05",
          all(v[0] <= 0.05 + 1e-12 for v in tiles.values()),
          str({k: round(v[0], 3) for k, v in tiles.items()}))
    reg_sep = [(e, a) for e in tab if e.startswith("h2_") for a in ALPHAS
               if float(sel(rm, experiment=e, alpha=a).fnr_boot_lo.mean()) > a]
    truth(194, "no regional cell separates from its own uncertainty",
          not reg_sep, str(reg_sep) if reg_sep else "none")

    # 197: the whole matrix carries exactly three separable cells, all spring.
    sep = [(e, a) for e in tab for a in ALPHAS
           if float(sel(rm, experiment=e, alpha=a).fnr_boot_lo.mean()) > a]
    truth(197, "spring at all three levels is the only separable cell group",
          len(sep) == 3 and all("spring" in e for e, _ in sep),
          str([(e.split("__")[1], a) for e, a in sep]))
    truth(198, "winter, summer and autumn respect the level at every setting",
          all(v[i] <= ALPHAS[i] + 1e-12
              for k, v in {e.split("__")[1]: [mean(sel(rm, experiment=e, alpha=a))
                                              for a in ALPHAS]
                           for e in tab if e.startswith("h3_season")}.items()
              if k != "spring" for i in range(3)),
          "")

    spring = "h3_season__spring__marida_unet_holdout_season_spring"
    for al, ratio, lo in zip(ALPHAS, [4.35, 3.46, 2.15], [0.063, 0.148, 0.231]):
        near(196, f"spring exceeds the level by {ratio}x at a={al}", ratio,
             mean(sel(rm, experiment=spring, alpha=al)) / al, 0.02)
        near(196, f"spring interval lower end at a={al}", lo,
             float(sel(rm, experiment=spring, alpha=al).fnr_boot_lo.mean()), 0.002)
    near(196, "spring rests on 8 debris-bearing scenes", 8,
         float(sel(rm, experiment=spring, alpha=0.20).n_test_clusters.mean()), 0.5)
    near(200, "48PZC carries 13 components over 3 scenes", 13,
         float(sel(rm, experiment="h2_region__48PZC__marida_unet_holdout_region_48PZC",
                   alpha=0.20).n_test_components.mean()), 0.5)

    # 201: the threshold has no room left under the spring shift
    near(201, "spring lambda-hat at alpha=0.2", 0.741,
         mean(sel(rm, experiment=spring, alpha=0.20), "lam"), 0.002)
    sp_arg = mean(sel(allm, experiment=spring, method="argmax", rho=0.5))
    truth(201, "the calibrated mask barely improves on argmax under spring shift",
          abs(sp_arg - 0.452) < 0.005
          and mean(sel(rm, experiment=spring, alpha=0.20)) > 0.42,
          f"argmax {sp_arg:.3f} vs region CRC "
          f"{mean(sel(rm, experiment=spring, alpha=0.20)):.3f}")

    single = "h1_official__marida_unet_official_holdout"
    ens = "h1_official__marida_unet_official_holdout_ens5"

    # --- prose numbers of Section V-E that no table carries -------------
    debris = _n_debris_patches()
    if debris is None:
        fail(190, "the debris-patch count cannot be read",
             "run scripts/24_marida_component_stats.py (run_all.sh phase 7b); "
             "Section V-E quotes 373 of 1381 patches from it")
    else:
        near(190, "373 of 1381 patches contain debris", 373, float(debris), 0.5)
    near(199, "official test group spans 14 debris-bearing scenes", 14,
         float(sel(rm, experiment=single, alpha=0.20).n_test_clusters.mean()), 0.5)
    near(200, "48PZC rests on 3 scenes", 3,
         float(sel(rm, experiment="h2_region__48PZC__marida_unet_holdout_region_48PZC",
                   alpha=0.20).n_test_clusters.mean()), 0.5)
    for end, val in (("lo", 0.000), ("hi", 0.750)):
        near(200, f"48PZC interval {end} end at a=0.2", val,
             float(sel(rm, experiment="h2_region__48PZC__marida_unet_holdout_region_48PZC",
                       alpha=0.20)[f"fnr_boot_{end}"].mean()), 0.001)
    near(196, "spring exceeds the level by 1.26x at the interval lower end, a=0.05",
         1.26, float(sel(rm, experiment=spring, alpha=0.05).fnr_boot_lo.mean()) / 0.05, 0.02)
    near(201, "the spring mask marks 0.04% of the pixels at a=0.2", 0.0004,
         mean(sel(rm, experiment=spring, alpha=0.20), "marked_area_fraction"), 0.00005)
    near(203, "official single-member marked area at a=0.2 is 0.21%", 0.0021,
         mean(sel(rm, experiment=single, alpha=0.20), "marked_area_fraction"), 0.0001)
    near(203, "official ensemble marked area at a=0.2 is 0.03%", 0.0003,
         mean(sel(rm, experiment=ens, alpha=0.20), "marked_area_fraction"), 0.0001)

    # spring group geometry quoted in the prose
    geo = _spring_geometry()
    if geo:
        near(201, "held-out spring holds 134 patches", 134, float(geo["patches"]), 0.5)
        near(201, "held-out spring spans 12 scenes", 12, float(geo["scenes"]), 0.5)
        near(201, "held-out spring spans 10 MGRS tiles", 10, float(geo["tiles"]), 0.5)
    else:
        note(201, "splits/marida_meta.csv absent; spring geometry not checked")

    # component-size statistics quoted in the prose
    sizes = _size_stats()
    if sizes is not None:
        near(202, "median MARIDA component size is 2 px", 2.0, sizes[0], 0.001)
        near(202, "83% of components are 3 px or smaller", 0.83, sizes[1], 0.005)
    else:
        fail(202, "the MARIDA component-size statistics cannot be read",
             "run scripts/24_marida_component_stats.py (run_all.sh phase 7b); "
             "Section V-E and the appendix quote the median size and the "
             "3-px fraction from it")

    # pixel-level F1 of the reported detectors
    for key, want in (("marida_unet_official_holdout", 0.55),
                      ("marida_unet_official_holdout_ens5", 0.65)):
        f1 = _test_f1(key)
        if f1 is None:
            note(204, f"marida_test_f1__{key}.json absent")
        else:
            near(204, f"{key} pixel debris F1", want, f1, 0.005)

    # 203: ensemble sharper but not better
    pairs = [(mean(sel(rm, experiment=single, alpha=a), "marked_area_fraction"),
              mean(sel(rm, experiment=ens, alpha=a), "marked_area_fraction"),
              mean(sel(rm, experiment=single, alpha=a)),
              mean(sel(rm, experiment=ens, alpha=a))) for a in ALPHAS]
    truth(203, "ensemble is sharper (smaller area) but not systematically better",
          all(p[1] <= p[0] + 1e-9 for p in pairs)
          and not all(p[3] <= p[2] + 1e-9 for p in pairs),
          "areas single/ens " + str([(round(p[0], 4), round(p[1], 4)) for p in pairs])
          + " FNR " + str([(round(p[2], 3), round(p[3], 3)) for p in pairs]))


# --------------------------------------------------------------------------
# M. triage
# --------------------------------------------------------------------------

    _marida_confidence_claims()

def section_triage():
    head("M  Budgeted human review")
    tr = load_triage("x3_*_triage")
    tr = sel(tr, alpha=0.20, rho=0.5, method="region_crc").copy()
    tr["setting"] = tr.experiment.str.extract(r"x3_triage__(\w+?)__")[0]

    # 218: monotone by construction
    bad = 0
    for _, g in tr.groupby(["experiment", "seed", "class_name", "ranking"]):
        v = g.sort_values("budget")["residual_region_fnr"].values
        if np.any(np.diff(v) > 1e-12):
            bad += 1
    truth(218, "residual miss rate is nonincreasing in the budget everywhere",
          bad == 0, f"{bad} non-monotone curves")

    settings = ["official", "region_16PCC", "region_16PDC", "region_16PEC",
                "season_spring", "night_tierA50"]
    print("      the random ranking is a Monte-Carlo draw; its expectation is exact:")
    print("      E[residual(b)] = (1 - floor(b*n)/n) * residual(0);")
    print("      the figure and the caption use the closed form (1 - b),")
    print("      which differs by less than 1/n and is checked below.")
    for s in settings:
        sub = sel(tr, setting=s)
        if sub.empty:
            continue
        nseed = sub.seed.nunique()
        a = sub[sub.ranking == "area"].groupby("budget")["residual_region_fnr"].mean()
        rnd = sub[sub.ranking == "random"].groupby("budget")["residual_region_fnr"].mean()
        orc = sub[sub.ranking == "oracle"].groupby("budget")["residual_region_fnr"].mean()
        R0 = float(a.loc[0.0])
        beta = a.index.values
        analytic = (1 - beta) * R0
        dev = float(np.max(np.abs(rnd.values - analytic)) / R0)
        note(219, f"{s}: seeds={nseed}, R0={R0:.4f}",
             f"max |random - E[random]| = {100*dev:.1f}% of R0")
        gap = (a.values - analytic) / R0
        above = beta[(gap > 1e-12) & (beta > 0)]
        note(222, f"{s}: area ranking ABOVE the exact floor at budgets",
             str([round(float(b), 2) for b in above]) if above.size else "none")
        z = orc[orc.values <= 1e-12]
        note(229, f"{s}: oracle reaches zero at beta",
             f"{z.index.min():.2f}" if len(z) else "never")
        note(220, f"{s}: reduction at beta=0.5",
             f"area {100*(a.loc[0.5]/R0-1):+.0f}%  measured-random "
             f"{100*(rnd.loc[0.5]/R0-1):+.0f}%  exact floor -50%")

    # --- the numbers Section V-F quotes, asserted rather than printed ----
    curves = {}
    for name in settings:
        sub = sel(tr, setting=name)
        if sub.empty:
            continue
        a = sub[sub.ranking == "area"].groupby("budget")["residual_region_fnr"].mean()
        R0 = float(a.loc[0.0])
        beta = a.index.values
        curves[name] = (a, R0, (a.values - (1 - beta) * R0) / R0, beta)

    quoted = {"official": (0.070, 0.036, 48.6), "region_16PCC": (0.025, 0.006, 77),
              "region_16PDC": (0.029, 0.021, 29), "region_16PEC": (0.053, 0.021, 61),
              "season_spring": (0.148, 0.060, 60), "night_tierA50": (0.059, 0.038, 34.5)}
    for name, (r0, r50, pct) in quoted.items():
        if name not in curves:
            note(220, f"{name}: triage curve missing"); continue
        a, R0, _, _ = curves[name]
        near(220, f"{name}: no-review rate", r0, R0, 0.001)
        near(220, f"{name}: residual at beta=0.5", r50, float(a.loc[0.5]), 0.001)
        near(220, f"{name}: reduction at beta=0.5 in percent", pct,
             100 * (1 - float(a.loc[0.5]) / R0), 0.5)

    # the budget at which the score first drops below the exact random floor
    first_below = {"official": 0.05, "region_16PCC": 0.10, "region_16PEC": 0.25,
                   "season_spring": 0.35}
    for name, want in first_below.items():
        if name not in curves:
            continue
        _, _, gap, beta = curves[name]
        below = [float(b) for b, g in zip(beta, gap) if b > 0 and g < -1e-12]
        got = min(below) if below else None
        truth(222, f"{name}: score first beats the floor at beta={want}",
              got is not None and abs(got - want) < 1e-9, f"measured {got}")

    for name in ("region_16PDC", "night_tierA50"):
        if name not in curves:
            continue
        _, _, gap, beta = curves[name]
        pos = [g for b, g in zip(beta, gap) if b > 0]
        truth(222, f"{name}: score stays above the floor at every budget",
              all(g > 1e-12 for g in pos), f"min gap {min(pos):+.3f}")

    for name, want in (("region_16PDC", 0.21), ("official", 0.014)):
        if name not in curves:
            continue
        _, _, gap, _ = curves[name]
        near(222, f"{name}: gap to the floor at beta=0.5", want, float(gap[-1]), 0.005)

    zero_by = []
    for name in settings:
        sub = sel(tr, setting=name)
        if sub.empty:
            continue
        orc = sub[sub.ranking == "oracle"].groupby("budget")["residual_region_fnr"].mean()
        z = orc[orc.values <= 1e-12]
        zero_by.append((name, float(z.index.min()) if len(z) else None))
    truth(229, "an oracle ranking reaches zero residual risk by beta=0.3 everywhere",
          all(b is not None and b <= 0.3 + 1e-9 for _, b in zero_by), str(zero_by))

    _permutation_band_claims()


def _permutation_band_claims():
    """Section IV-G: the marked-area ranking against the permutation band.

    Every number the subsection prints is checked here. The claim it supports
    is negative -- the ranking is not separable from a random order at these
    sample sizes -- so these checks matter more than usual: a silent change
    in the band would turn a reported negative result into an unreported
    positive one.
    """
    path = results_dir("experiments") / "x12_triage_permutation_band.csv"
    if not path.exists():
        note(230, "permutation band not computed",
             "run scripts/27_triage_permutation_band.py")
        return
    band = pd.read_csv(path)
    band = band[(band["class_name"] != "all_classes")
                & np.isclose(band["alpha"].astype(float), 0.20)
                & np.isclose(band["rho"].astype(float), 0.5)]
    band = band.drop_duplicates(["setting", "class_name", "budget"])
    w = band[band["budget"] > 0].copy()
    w["above"] = w["area_rel"] > w["q95_rel"]

    n, below, above = len(w), int(w["below_band"].sum()), int(w["above"].sum())
    inside = n - below - above
    truth(230, "80 comparisons: 12 below the band, 21 above, 47 inside",
          (n, below, above, inside) == (80, 12, 21, 47),
          f"{n} comparisons, {below} below, {above} above, {inside} inside")
    truth(230, "1000 permutations per setting",
          set(band["n_permutations"].unique()) == {1000},
          str(sorted(band["n_permutations"].unique())))
    truth(230, "six settings", band["setting"].nunique() == 6,
          str(sorted(band["setting"].unique())))

    # Every above-band outcome is in the one replicated setting.
    ab = sorted(w.loc[w["above"], "setting"].unique())
    off_ab = w[(w["setting"] == "official") & w["above"]]
    truth(231, "20 of the 21 above-band outcomes arise in the replicated night setting; "
               "the other is the official split at beta=0.25",
          ab == ["night_tierA50", "official"] and len(off_ab) == 1
          and np.isclose(float(off_ab["budget"].iloc[0]), 0.25),
          f"above-band settings: {ab}; official above at {off_ab['budget'].tolist()}")
    ni = w[w["setting"] == "night_tierA50"]
    truth(231, "night, tier A: above the band in 20 of its 30 comparisons",
          (int(ni["above"].sum()), len(ni)) == (20, 30),
          f"{int(ni['above'].sum())} of {len(ni)}")
    nic = ni.groupby("class_name").agg(above=("above", "sum"), below=("below_band", "sum"))
    truth(231, "night: person and rider above the band at every budget, bicycle below at five",
          int(nic.loc["person", "above"]) == 10 and int(nic.loc["rider", "above"]) == 10
          and int(nic.loc["bicycle", "above"]) == 0 and int(nic.loc["bicycle", "below"]) == 5,
          str(nic.to_dict()))

    # The five single-partition settings, and 16PCC as the single exception.
    per = w.groupby("setting")["below_band"].sum()
    never = sorted(per[per == 0].index)
    truth(231, "the score never falls below the band in four of the five single "
               "MARIDA partitions and never leaves it in three",
          set(never) == {"official", "region_16PDC", "region_16PEC",
                         "season_spring"}
          and set(w[w["setting"].isin(never) & w["above"]]["setting"]) == {"official"},
          f"never below: {never}")
    pcc_all = w[w["setting"] == "region_16PCC"]
    pcc = pcc_all[pcc_all["below_band"].astype(bool)]
    below_at = sorted(round(float(b), 2) for b in pcc["budget"])
    truth(231, "16PCC falls below the band at seven of its ten budgets, "
               "0.1, 0.2 and 0.3-0.5, with 0.15 and 0.25 inside",
          len(pcc_all) == 10
          and below_at == [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5],
          f"{len(pcc)} of {len(pcc_all)} budgets below, at {below_at}")

    # The two quoted bands at beta = 0.5.
    def cell(setting):
        g = w[(w["setting"] == setting) & np.isclose(w["budget"], 0.5)]
        return (float(g["area_rel"].mean()), float(g["q05_rel"].mean()),
                float(g["q50_rel"].mean()), float(g["q95_rel"].mean()))

    a, lo, med, hi = cell("official")
    near(232, "official at beta=0.5: score", 0.514, a, tol=5e-3)
    near(232, "official at beta=0.5: band lower end", 0.36, lo, tol=5e-3)
    near(232, "official at beta=0.5: band upper end", 0.65, hi, tol=5e-3)
    near(232, "official at beta=0.5: band median", 0.51, med, tol=5e-3)
    # The width of the official band against the margin by which the score
    # misses the closed-form line, which is what "cannot resolve an effect of
    # the size at issue" means.
    near(232, "the official band is 0.28 wide", 0.28, hi - lo, tol=5e-3)
    closed = 0.50
    near(232, "the score misses the closed-form line by 0.014 (worse than random)",
         0.014, a - closed, tol=5e-4)
    margin = abs(a - closed)
    truth(232, "the official band is more than ten times that margin",
          margin > 0 and (hi - lo) > 10 * margin,
          f"width {hi - lo:.3f} against margin {margin:.4f}, a factor of {(hi - lo) / margin:.1f}")

    a, lo, med, hi = cell("night_tierA50")
    near(232, "night at beta=0.5: score", 0.65, a, tol=5e-3)
    near(232, "night at beta=0.5: band lower end", 0.485, lo, tol=5e-3)
    near(232, "night at beta=0.5: band upper end", 0.515, hi, tol=5e-3)
    truth(232, "the night band is narrower than the official one",
          (cell("night_tierA50")[3] - cell("night_tierA50")[1])
          < (cell("official")[3] - cell("official")[1]),
          f"night width {cell('night_tierA50')[3]-cell('night_tierA50')[1]:.3f} "
          f"vs official {cell('official')[3]-cell('official')[1]:.3f}")


# --------------------------------------------------------------------------
# cross-cutting
# --------------------------------------------------------------------------
def section_cross():
    head("X  Cross-cutting consistency")
    e2 = sel(load("e2_break__*"), method="region_crc", rho=0.5)
    parts = {"ACDC": cellmean(e2, ["model", "experiment", "alpha"])}
    try:
        l2 = sel(load("l2_break__*"), method="region_crc", rho=0.5)
        parts["LoveDA"] = cellmean(l2, ["experiment", "alpha"])
    except FileNotFoundError:
        pass
    try:
        h23 = sel(load("h[23]_*"), method="region_crc", rho=0.5)
        parts["MARIDA"] = cellmean(h23, ["experiment", "alpha"])
    except FileNotFoundError:
        pass
    worst = {}
    for k, c in parts.items():
        lev = c.index.get_level_values("alpha").values
        worst[k] = float((c.values / lev).max())
    truth("X1", "worst violation ratio per axis (class-averaged): "
                "2.7x weather, 3.5x geographic, 4.3x seasonal",
          abs(worst.get("ACDC", 0) - 2.74) < 0.01
          and abs(worst.get("LoveDA", 0) - 3.47) < 0.01
          and 4.3 <= worst.get("MARIDA", 0) <= 4.4,
          "worst ratio per axis: " + str({k: round(v, 2) for k, v in worst.items()}))


# --------------------------------------------------------------------------
# N. experimental design: no test patch may have been seen during model fitting
# --------------------------------------------------------------------------
# Schemes whose whole purpose is an exchangeable calibration/test pair drawn
# from the same pool. They carry no held-out claim, so the fitting-set check
# below does not apply to them.
POSITIVE_CONTROL_SCHEMES = {"marida_random_half"}


def section_holdout():
    """Cross-check every experiment's test group against the data its model saw.

    Numerical consistency between a paper and its CSVs says nothing about
    whether the CSVs measure what they are taken to measure. This section closes that
    gap for the one property the MARIDA axes rest on: a held-out tile or
    season must be absent from the training set, from the checkpoint-selection
    set, and from the calibration set of the model that is evaluated on it.

    The check reads each checkpoint's ``config.json``, which records the scheme
    it was fitted from together with digests of its train and monitor id lists,
    and compares those lists against the scheme groups the experiment consumed.
    """
    import json
    from record.models import load_model_registry
    from record.paths import repo_root, splits_dir
    from record.splits import load_scheme

    head("N  Experimental design: fitting sets versus test sets")

    ckpt_root = repo_root() / "checkpoints"
    metas = sorted(glob.glob(str(results_dir("experiments") / "*.meta.json")))
    if not metas:
        note(900, "no experiment sidecars found", str(results_dir("experiments")))
        return

    checked = 0
    for mpath in metas:
        meta = json.load(open(mpath))
        argv = meta.get("argv", {})
        model, scheme_name = argv.get("model"), argv.get("scheme")
        name = os.path.basename(mpath)[: -len(".meta.json")]
        if not model or not scheme_name or not model.startswith("marida"):
            continue
        # The in-distribution positive control is meant to overlap. It splits
        # the pooled patches at random so that calibration and test are
        # exchangeable by construction, which is the only condition the
        # guarantee needs; whether the model saw those patches is beside the
        # point, and a violation there would indict the implementation rather
        # than the protocol. Recorded rather than skipped silently.
        if scheme_name in POSITIVE_CONTROL_SCHEMES:
            note(917, f"{name}: positive control on {scheme_name}, "
                      "overlap with the fitting sets is by design",
                 "exchangeability of calibration and test is what it tests")
            continue
        # A deep ensemble has no checkpoint of its own; the registry lists the
        # member directories, and every member's fitting sets count as seen.
        spec = load_model_registry()["segmentation_models"].get(model, {})
        entries = spec.get("members") or [spec.get("checkpoint") or f"checkpoints/{model}"]
        members = [repo_root() / e / "config.json" for e in entries]
        members = [m for m in members if m.exists()]
        if not members:
            note(901, f"{name}: no checkpoint config for model {model}",
                 "cannot verify what this model was fitted on")
            continue

        scheme_path = splits_dir() / f"{scheme_name}.json"
        if not scheme_path.exists():
            note(903, f"{name}: scheme file {scheme_name}.json is absent",
                 "cannot verify this experiment's groups")
            continue
        entry = load_scheme(scheme_path)["seeds"]["0"]
        test_key = argv.get("test_split") or "test"
        cal_key = argv.get("cal_split") or "calibration"
        test_ids = set(entry.get(test_key, []))
        cal_ids = set(entry.get(cal_key, []))
        if not test_ids:
            continue

        seen = set()
        provenance_known = True
        for member in members:
            cfg = json.load(open(member))
            fitted = cfg.get("scheme")
            if fitted is None:
                # A checkpoint that records no scheme was fitted on the official
                # train split with the official val split for checkpoint
                # selection. The split assignment is read from the committed
                # metadata rather than from the imagery, so the audit runs from
                # a clean clone.
                meta_csv = pd.read_csv(splits_dir() / "marida_meta.csv")
                seen |= set(meta_csv.loc[
                    meta_csv.official_split.isin(["train", "val"]), "patch_id"])
            else:
                fit_path = splits_dir() / f"{fitted}.json"
                if not fit_path.exists():
                    provenance_known = False
                    continue
                fit_entry = load_scheme(fit_path)["seeds"]["0"]
                seen |= set(fit_entry.get("train", []))
                seen |= set(fit_entry.get("monitor", []))
            if "train_ids_sha256" not in cfg:
                provenance_known = False

        checked += 1
        leak_test = test_ids & seen
        leak_cal = cal_ids & seen
        pct = 100.0 * len(leak_test) / len(test_ids)
        truth(900 + checked,
              f"{name}: test group unseen during model fitting",
              not leak_test,
              f"{len(leak_test)}/{len(test_ids)} test patches ({pct:.1f}%) "
              f"were in the fitting sets of {model}")
        if leak_cal:
            fail(900 + checked,
                 f"{name}: calibration group unseen during model fitting",
                 f"{len(leak_cal)}/{len(cal_ids)} calibration patches were in "
                 f"the fitting sets of {model}")
        if not provenance_known:
            note(900 + checked,
                 f"{name}: checkpoint of {model} predates provenance recording",
                 "fitting sets inferred from the official split convention")

    if not checked:
        note(902, "no MARIDA experiment could be checked", "")


def section_revision():
    """Quantities quoted in the revised text: the certificate firing ratio,
    the LoveDA infeasibility corner, the MARIDA calibration sizes and the
    in-distribution positive control."""
    head("R  Revised-text quantities")

    # Certificate firing ratio under weight collapse: p_test = r/(n+r) > alpha
    # is equivalent to r > n*alpha/(B-alpha).
    n_cal = 167
    for alpha, want in [(0.05, 8.79), (0.10, 18.56), (0.20, 41.75)]:
        near(203, f"certificate fires above clip ratio {want} at alpha={alpha}",
             want, n_cal * alpha / (1 - alpha), 0.01)
    near(203, "collapse prediction at ratio 40 leaves the certificate silent "
              "at alpha=0.2", 0.193, 40 / (n_cal + 40), 0.001)

    # Section IV-B quotes the area-matched frontier from the released grid.
    pareto = (results_dir("tables") / "pareto_area_matched.tex")
    if pareto.exists():
        deltas = []
        for line in pareto.read_text().splitlines():
            parts = [c.strip() for c in line.split("&")]
            if len(parts) == 6 and parts[0].startswith("SegFormer"):
                deltas.append(float(parts[4]))
        near(70, "region CRC leads by at most 0.006 on the released grid",
             0.006, max(deltas), 0.0005)
        near(70, "region CRC trails by at most 0.011 on the released grid",
             -0.011, min(deltas), 0.0005)
        truth(70, "region CRC is the lower of the two at four of the six matched areas per model",
              len(deltas) == 12 and sum(d_ > 0 for d_ in deltas[:6]) == 4
              and sum(d_ > 0 for d_ in deltas[6:]) == 4,
              f"positive deltas: B2 {sum(d_ > 0 for d_ in deltas[:6])}, B5 {sum(d_ > 0 for d_ in deltas[6:])}")

    # The prose compares the two frontiers at every matched area, not only at
    # the six tabulated points, so the dense interpolation is checked too.
    try:
        dense = load("p1_pareto__*")
    except FileNotFoundError:
        note(70, "dense-alpha pareto runs missing", "skipped")
        dense = None
    if dense is not None:
        want = {"segformer_b2_cityscapes": (0.0082, -0.0060),
                "segformer_b5_cityscapes": (0.0077, -0.0106)}
        for model, (lead, trail) in want.items():
            sub = dense[(dense["model"] == model) & np.isclose(dense["rho"], 0.5)]
            cur = {}
            for meth in ("region_crc", "pixel_crc"):
                m = (sub[sub["method"] == meth]
                     .groupby("alpha")[["marked_area_fraction", "region_fnr"]]
                     .mean().sort_values("marked_area_fraction"))
                cur[meth] = m
            lo = max(float(c["marked_area_fraction"].min()) for c in cur.values())
            hi = min(float(c["marked_area_fraction"].max()) for c in cur.values())
            grid = np.geomspace(lo, hi, 2001)
            d = (np.interp(grid, cur["pixel_crc"]["marked_area_fraction"],
                           cur["pixel_crc"]["region_fnr"])
                 - np.interp(grid, cur["region_crc"]["marked_area_fraction"],
                             cur["region_crc"]["region_fnr"]))
            near(70, f"{model}: largest region-CRC lead over the whole band",
                 lead, float(d.max()), 0.0006)
            near(70, f"{model}: largest pixel-CRC lead over the whole band",
                 trail, float(d.min()), 0.0006)

    # Section IV-F: the two Tier-B estimators fail for different reasons.
    try:
        knn = sel(load("e4_tierB_knn__*"), method="weighted_crc")
        ratio = (knn["weight_ess"] / knn["n_cal_images"]).dropna()
        truth(157, "kNN weights are concentrated, not uniform",
              float(ratio.min()) < 0.5, f"ess/n min {float(ratio.min()):.3f}")
        rng_claim(157, "kNN effective sample size spans 1.8-134.3",
                  1.8, 134.3, list(knn["weight_ess"].dropna()), tol=0.1)
    except FileNotFoundError:
        note(157, "kNN tier-B runs missing", "skipped")

    # Section IV-G: LoveDA in-domain validity is a rho=0.5 statement.
    lv = sel(load("l1_indist__*"), method="region_crc")
    cells = cellmean(lv, ["scheme", "rho", "alpha"])
    over = [(k, v) for k, v in cells.items() if v > k[2] + 1e-12]
    truth(178, "the only class-AVERAGED LoveDA in-domain cell above its level "
               "is urban at "
               "rho=0.1, alpha=0.2",
          len(over) == 1 and abs(over[0][1] - 0.2003) < 5e-4, str(over))

    # LoveDA tier A: infeasibility is not negligible at the tight level.
    lv = sel(load("l3_tierA*"), method="region_crc", rho=0.5)
    sub = lv[(lv.scheme == "loveda_rural_targetcal25")
             & (lv.class_name == "building") & (np.isclose(lv.alpha, 0.05))]
    inf25 = 1 - float(sub["feasible"].astype(bool).mean())
    near(178, "LoveDA building at n_t=25, alpha=0.05 is infeasible in 98% "
              "of draws", 0.98, inf25, 0.01)

    # MARIDA calibration group sizes, which set the feasibility floor.
    mar = sel(load("h[123]_*"), method="region_crc", rho=0.5)
    sizes = mar.groupby("experiment")["n_cal_images"].first()
    off = [v for k, v in sizes.items() if k.startswith("h1_")]
    truth(198, "MARIDA official calibration group holds 82 debris patches",
          all(int(v) == 82 for v in off), str(sorted(set(int(v) for v in off))))
    rng_claim(198, "MARIDA calibration group sizes span 26-86",
              26, 86, [float(v) for v in sizes.values], tol=0.5)
    near(198, "smallest MARIDA calibration group puts the floor at 0.037",
         0.037, 1.0 / (float(min(sizes.values)) + 1), 0.001)

    # Section IV-H quotes the pooled component rate on spring alongside the
    # image-averaged one; the two aggregations must both reproduce.
    mar_all = sel(load("h[123]_*"), method="region_crc", rho=0.5)
    spring = mar_all[mar_all["experiment"].str.contains("spring")]
    for alpha, want in [(0.05, 0.155), (0.10, 0.245), (0.20, 0.300)]:
        cell = spring[np.isclose(spring["alpha"], alpha)]
        pooled = (float(cell["n_missed_components"].iloc[0])
                  / float(cell["n_test_components"].iloc[0]))
        near(200, f"spring pooled component rate at alpha={alpha}", want,
             pooled, 0.001)

    # Section IV-B: the tempered rows of Table II use the seed-0 scalar. The
    # prose bounds the dependence that reuse leaves behind by the move to the
    # per-draw refit, so the two runs are differenced cell by cell.
    cells = [(m, meth, rho)
             for m in ("segformer_b2_cityscapes", "segformer_b5_cityscapes")
             for meth in ("heuristic", "region_crc")
             for rho in (0.1, 0.5)]
    worst_fnr = worst_area = 0.0
    try:
        fixed = load("x5_temp__*")
        free = load("x6_temp_leakfree__*")
    except FileNotFoundError:
        fail(99, "the leakage-free temperature runs are absent",
             "run scripts/21_temperature_leakfree.py (run_all.sh phase 7b); "
             "the temperature-scaling comparison is reported from them")
        fixed = free = None
    if free is not None:
        # stage 4b tags the fixed-temperature run with a "_tempscaled" model key
        def _cell(frame, model, method, rho):
            keys = [k for k in frame["model"].unique() if k.startswith(model)]
            c = frame[frame["model"].isin(keys) & (frame["method"] == method)
                      & np.isclose(frame["rho"], rho)
                      & np.isclose(frame["alpha"], 0.2)]
            if c.empty:
                raise AssertionError(f"no rows for {model}/{method}/rho={rho}")
            return (float(c["region_fnr"].mean()),
                    float(c["marked_area_fraction"].mean()))
        for model, method, rho in cells:
            a = _cell(fixed, model, method, rho)
            b = _cell(free, model, method, rho)
            worst_fnr = max(worst_fnr, abs(a[0] - b[0]))
            worst_area = max(worst_area, abs(a[1] - b[1]))
        truth(99, "refitting the temperature per draw moves the tempered cells "
                  "by at most 0.0007 in FNR and 0.0002 in area",
              worst_fnr <= 0.0007 + 1e-9 and worst_area <= 0.0002 + 1e-9,
              f"largest FNR move {worst_fnr:.5f}, area move {worst_area:.5f}")
        for model, lo, hi, k in (("segformer_b2_cityscapes", 1.45, 1.65, 5),
                                 ("segformer_b5_cityscapes", 1.90, 2.15, 6)):
            temps = sorted(free[free["model"] == model]["temperature"].unique())
            truth(99, f"{model}: {k} distinct refitted temperatures in "
                      f"[{lo}, {hi}]",
                  len(temps) == k and abs(min(temps) - lo) < 1e-9
                  and abs(max(temps) - hi) < 1e-9, str(temps))

    # Section IV-B reports the measured union of the critical-class masks at
    # the selected thresholds, excluding draws where one class marks the whole
    # image. The per-draw bounds max_c <= union <= min(1, sum_c) must hold.
    try:
        uni = load("x7_union_area__*")
    except FileNotFoundError:
        fail(98, "the union-area runs are absent",
             "run scripts/22_union_marked_area.py (run_all.sh phase 7b); "
             "Section IV-B reports the measured union from them")
        uni = None
    if uni is not None:
        uni = uni[np.isclose(uni["alpha"], 0.2)]
        bad = int(((uni["union_area"] < uni["max_class_area"] - 1e-9)
                   | (uni["union_area"] > uni["sum_class_area"] + 1e-9)).sum())
        truth(98, "every draw satisfies max_c <= union <= min(1, sum_c)",
              bad == 0, f"{bad} violations in {len(uni)} draws")
        good = uni[uni["union_area"] <= 0.99]
        per = good.groupby("model")[["class_mean_area", "union_area"]].mean()
        rng_claim(98, "measured union covers 3.4-4.9% of the image",
                  0.0339, 0.0488, list(per["union_area"]), tol=0.0005)
        rng_claim(98, "class mean on the same draws is 1.4-2.4%",
                  0.0138, 0.0239, list(per["class_mean_area"]), tol=0.0005)
        ratios = list(per["union_area"] / per["class_mean_area"])
        rng_claim(98, "the union is 2.0-2.5 times the class mean",
                  2.04, 2.46, ratios, tol=0.01)
        n_degen = int((uni["union_area"] > 0.99).sum())
        truth(98, "no degenerate draw on the log-tail grid",
              n_degen == 0, f"{n_degen} across {uni['model'].nunique()} models")

    # Section IV-A: a tile becomes a hold-out scheme only at >= 50 patches.
    meta_path = paths_root() / "splits" / "marida_meta.csv"
    if meta_path.exists():
        counts = pd.read_csv(meta_path)["tile"].value_counts()
        eligible = sorted(counts[counts >= 50].index)
        truth(201, "exactly five MARIDA tiles carry at least 50 patches",
              eligible == ["16PCC", "16PDC", "16PEC", "18QYF", "48PZC"],
              str(eligible))
        truth(201, "MARIDA spans seventeen tiles", len(counts) == 17,
              f"{len(counts)} tiles")

    # In-distribution positive control on a random patch-level split.
    try:
        ctrl = sel(load("p5_marida_indist__*"), method="region_crc", rho=0.5)
    except FileNotFoundError:
        note(199, "MARIDA in-distribution control missing", "run p5")
        return
    quoted = {("marida_unet_official_holdout", 0.05): 0.048,
              ("marida_unet_official_holdout", 0.10): 0.097,
              ("marida_unet_official_holdout", 0.20): 0.201,
              ("marida_unet_official_holdout_ens5", 0.05): 0.048,
              ("marida_unet_official_holdout_ens5", 0.10): 0.097,
              ("marida_unet_official_holdout_ens5", 0.20): 0.198}
    for (model, alpha), want in quoted.items():
        cell = ctrl[(ctrl.model == model) & np.isclose(ctrl.alpha, alpha)]
        near(199, f"positive control {model} at alpha={alpha}", want,
             mean(cell), 0.001)
        x = feas(cell)["region_fnr"].dropna()
        lo = float(x.mean() - 1.959963984540054 * x.std(ddof=1) / np.sqrt(len(x)))
        truth(199, f"positive control {model} at alpha={alpha} is not "
                   "separably above the level", lo <= alpha, f"lower end {lo:.4f}")



def section_review():
    """Quantities added by the 2026-08-23 consistency review: the per-class
    LoveDA in-domain cells, the MARIDA official cells above the diagonal, the
    macro-versus-pooled spring rates, the estimator-dependent weight collapse,
    and the measured mask union against the sum of the class masks."""
    head("S  Consistency-review quantities")

    # Figure 4 and Section V-G report per-class cells, as Cityscapes does.
    lov = sel(load("l1_indist__*"), method="region_crc")
    if "feasible" in lov.columns:
        lov = lov[lov["feasible"].astype(bool)]
    g = lov.groupby(["model", "class_name", "alpha", "rho"])["region_fnr"]
    cells = g.mean().reset_index()
    spread = g.agg(["std", "size"]).reset_index()
    cells = cells.merge(spread, on=["model", "class_name", "alpha", "rho"])
    over = cells[cells.region_fnr > cells.alpha]
    truth(920, "seven of the 24 LoveDA in-domain per-class cells exceed the "
               "level", len(cells) == 24 and len(over) == 7,
          f"{len(over)} of {len(cells)}")
    truth(921, "every LoveDA exceedance is water: four urban (a=0.1 and 0.2 at both rho) "
               "and three rural",
          bool((over.class_name == "water").all())
          and int(over.model.str.endswith("urban").sum()) == 4
          and int(over.model.str.endswith("rural").sum()) == 3,
          str(sorted(zip(over.model.str[-5:], over.alpha, over.rho))))
    near(922, "the largest LoveDA in-domain exceedance is 0.003",
         0.003, float((over.region_fnr - over.alpha).max()), 0.0005)
    hi = over.region_fnr + 1.96 * over["std"] / np.sqrt(over["size"])
    lo = over.region_fnr - 1.96 * over["std"] / np.sqrt(over["size"])
    truth(923, "the level is inside the draw interval for all seven",
          bool(((lo <= over.alpha) & (over.alpha <= hi)).all()),
          f"lower ends {sorted(lo.round(4))}")
    truth(924, "one of the three sits at rho=0.5, so the in-domain claim is "
               "not scoped to rho=0.1 alone",
          bool(np.isclose(over.rho, 0.5).any()), str(sorted(over.rho)))

    off = sel(load("h1_official__*"), method="region_crc")
    ocells = off.groupby(["experiment", "class_name", "alpha", "rho"])[
        "region_fnr"].mean().reset_index()
    o_over = ocells[ocells.region_fnr > ocells.alpha]
    truth(925, "nine of the twelve MARIDA official cells lie above the level",
          len(ocells) == 12 and len(o_over) == 9,
          f"{len(o_over)} of {len(ocells)}")

    # Section V-H: the spring numbers quoted per image and per component.
    spring = sel(load("h3_season__spring*"), rho=0.5, alpha=0.2)
    arg = spring[spring.method == "argmax"]
    crc = spring[spring.method == "region_crc"]
    near(926, "argmax misses 45% of spring images' components on average",
         0.45, float(arg.region_fnr.mean()), 0.005)
    near(927, "the calibrated mask misses 43% on the same average",
         0.43, float(crc.region_fnr.mean()), 0.005)
    near(928, "pooling the components gives 32% for argmax",
         0.32, float(arg.region_fnr_component_avg.mean()), 0.005)
    near(929, "pooling the components gives 30% for region CRC",
         0.30, float(crc.region_fnr_component_avg.mean()), 0.005)

    # The abstract scopes the weight collapse to the logistic estimator.
    tb = sel(load("e4_tierB__*"), rho=0.5)
    knn = sel(load("e4_tierB_knn__*"), rho=0.5)
    for cid, name, frame, want_floor in ((930, "logistic", tb, True),
                                         (931, "kNN", knn, False)):
        ess = (frame.weight_ess / frame.n_cal_images).dropna()
        at_floor = bool(np.allclose(ess, 1.0, atol=1e-3))
        truth(cid, f"the {name} estimator's ESS/n is "
                   + ("exactly 1 (weights at the floor)" if want_floor
                      else "well below 1 (weights concentrated, not at floor)"),
              at_floor == want_floor,
              f"ESS/n in [{ess.min():.4f}, {ess.max():.4f}]")

    # Section V-A: the union of the three critical masks against their sum.
    exp_dir = results_dir("experiments")
    paths = sorted(glob.glob(str(exp_dir / "x7_union_area__*.csv")))
    if not paths:
        fail(932, "the union-area runs are absent",
             "run scripts/22_union_marked_area.py (run_all.sh phase 7b); "
             "Section V-A reports the union against the sum from them")
        return
    ratios_mean, ratios_sum = [], []
    for p in paths:
        u = pd.read_csv(p)
        keep = u[~u["degenerate"].astype(bool)] if "degenerate" in u else u
        cls = [c for c in u.columns if c.startswith("area__")]
        per = keep[cls].to_numpy()
        ratios_mean.append(float(keep.union_area.mean()
                                 / per.mean(axis=1).mean()))
        ratios_sum.append(float(keep.union_area.mean()
                                / per.sum(axis=1).mean()))
    rng_claim(933, "the union is 2.0-2.5x the class mean",
              2.04, 2.46, ratios_mean, 0.01)
    rng_claim(934, "the union is 68-82% of the sum of the three masks",
              0.680, 0.820, ratios_sum, 0.005)

    # Section IV-B now prints the union with the degenerate draws kept as well
    # as removed: they are the most expensive operating points, not invalid
    # observations, and on B5 three draws in twenty-five carry the figure from
    # 3.4% to 15.0%.
    cond, allw, degen = [], [], []
    for p_ in paths:
        u = pd.read_csv(p_)
        u = u[np.isclose(u["alpha"], 0.2)]
        keep = u[~u["degenerate"].astype(bool)] if "degenerate" in u else u
        cond.append(float(keep["union_area"].mean()))
        allw.append(float(u["union_area"].mean()))
        degen.append(int(u["degenerate"].sum()) if "degenerate" in u else 0)
    # Section V-B: the lowest grid index any model selects, and Mask2Former's
    # own minimum. The earlier text claimed every threshold sat in the top
    # twenty points, which the matrix does not support.
    e1_ = sel(load("e1_indist__*"), method="region_crc")
    e1_ = e1_[e1_["feasible"].astype(bool)]
    near(942, "the lowest selected grid index over all models is 155 (B5)",
         155, float(e1_["lam_index"].min()), 0.5)
    m2f_ = e1_[e1_["model"] == "mask2former_swinb_cityscapes"]
    truth(942, "Mask2Former selects grid indices 410-981 (cutoffs 1.3e-6 to 3.5e-3)",
          int(m2f_["lam_index"].min()) == 410 and int(m2f_["lam_index"].max()) == 981,
          f"[{int(m2f_['lam_index'].min())}, {int(m2f_['lam_index'].max())}]")

    # Section V-C: the degenerate-draw fraction with and without Mask2Former,
    # and the argmax night range separated by capture level.
    e2_ = sel(load("e2_break__*"), method="region_crc", rho=0.5)
    truth(942, "four SegFormer draws under shift are degenerate, and no Mask2Former draw",
          int((e2_[e2_["model"].isin(SEGF)]["lam_index"] == LAM_MAX).sum()) == 4
          and int((e2_["lam_index"] == LAM_MAX).sum()) == 4,
          f"{int((e2_[e2_['model'].isin(SEGF)]['lam_index'] == LAM_MAX).sum())} SegFormer, "
          f"{int((e2_['lam_index'] == LAM_MAX).sum())} in all of {len(e2_)}")
    an_ = sel(load("e2_break__*"), method="argmax", alpha=0.20)
    an_ = an_[an_["experiment"].str.contains("__night__")]
    for rho_, lo_, hi_ in ((0.5, 0.757, 0.835), (0.1, 0.679, 0.759)):
        v = an_[np.isclose(an_["rho"], rho_)].groupby("model")["region_fnr"].mean()
        rng_claim(942, f"argmax misses {lo_:.0%}-{hi_:.0%} of night regions at "
                       f"rho={rho_}", lo_, hi_, v.values, 0.001)

    # Section V-B: the size strata quoted in the text are B2's, and the
    # three-variant average is a different pair.
    st = sel(load("e1_indist__*"), method="region_crc", alpha=0.20, rho=0.5)
    st = st[st["feasible"].astype(bool)]
    b2 = st[st["model"] == "segformer_b2_cityscapes"]
    near(943, "B2 smallest-stratum FNR is 0.26", 0.26,
         float(b2["fnr_stratum0"].mean()), 0.005)
    near(943, "B2 largest-stratum FNR is 0.02", 0.02,
         float(b2["fnr_stratum2"].mean()), 0.005)
    sg = st[st["model"].isin(SEGF)]
    near(943, "the three SegFormer variants average 0.27 in the smallest stratum",
         0.27, float(sg.groupby("model")["fnr_stratum0"].mean().mean()), 0.005)
    near(943, "and 0.016 in the largest", 0.016,
         float(sg.groupby("model")["fnr_stratum2"].mean().mean()), 0.002)

    # Abstract and conclusion now quote per-class violation ratios, since the
    # guarantee is stated per class. The class-averaged figures stay in X1.
    br = sel(load("e2_break__*"), method="region_crc", rho=0.5)
    br = br[br["feasible"].astype(bool)]
    cc_ = br.groupby(["model", "experiment", "class_name", "alpha"])["region_fnr"].mean()
    ratio_acdc = float((cc_ / cc_.index.get_level_values("alpha")).max())
    near(944, "the worst ACDC per-class violation is 4.23x at rho=0.5 (the abstract's 4.2x)", 4.23, ratio_acdc, 0.01)
    truth(944, "the rho=0.5 worst cell is MC-dropout, rider, fog, alpha=0.05",
          (cc_ / cc_.index.get_level_values("alpha")).idxmax() ==
          ("segformer_b2_cityscapes_mcdrop8", "e2_break__fog__segformer_b2_cityscapes_mcdrop8", "rider", 0.05),
          str((cc_ / cc_.index.get_level_values("alpha")).idxmax()))
    near(944, "its value is 0.211", 0.211, float(cc_[(cc_ / cc_.index.get_level_values("alpha")).idxmax()]), 0.001)
    # The abstract quotes the larger of the two capture levels (fourth panel,
    # A1): at rho=0.1 the worst class cell is B5, bicycle, night, alpha=0.1.
    br1 = sel(load("e2_break__*"), method="region_crc", rho=0.1)
    br1 = br1[br1["feasible"].astype(bool)]
    cc1 = br1.groupby(["model", "experiment", "class_name", "alpha"])["region_fnr"].mean()
    ratio_acdc1 = float((cc1 / cc1.index.get_level_values("alpha")).max())
    near(944, "the worst ACDC per-class violation is 3.88x at rho=0.1",
         3.88, ratio_acdc1, 0.01)
    truth(944, "the rho=0.1 worst cell is the same one: MC-dropout, rider, fog, alpha=0.05",
          (cc1 / cc1.index.get_level_values("alpha")).idxmax() ==
          ("segformer_b2_cityscapes_mcdrop8", "e2_break__fog__segformer_b2_cityscapes_mcdrop8", "rider", 0.05),
          str((cc1 / cc1.index.get_level_values("alpha")).idxmax()))
    lo_ = sel(load("l2_break__*"), method="region_crc", rho=0.5)
    lo_ = lo_[lo_["feasible"].astype(bool)]
    cl_ = lo_.groupby(["experiment", "class_name", "alpha"])["region_fnr"].mean()
    ratio_love = float((cl_ / cl_.index.get_level_values("alpha")).max())
    near(944, "the worst LoveDA per-class violation is 4.46x (the conclusion's 4.5x)", 4.46, ratio_love, 0.01)
    truth(944, "both exceed the class-averaged figures the conclusion quotes (2.7x / 3.5x)",
          ratio_acdc > 2.74 and ratio_love > 3.47,
          f"per-class {ratio_acdc:.2f}/{ratio_love:.2f} against "
          "class-averaged 2.74/3.47")

    rng_claim(941, "the union over non-degenerate draws is 3.4-4.9% at a=0.2",
              0.0339, 0.0488, cond, 0.0006)
    rng_claim(941, "the union over all draws is the same (no degenerate draw)",
              0.0339, 0.0488, allw, 0.0006)
    truth(941, "no draw is degenerate on the log-tail grid",
          max(degen) == 0, f"degenerate counts {sorted(degen)}")

    # Section V-E: the class-conditional target pool does not lift the collapse.
    pool_paths = sorted(glob.glob(str(exp_dir / "x8_tierb_pool__*.csv")))
    if not pool_paths:
        fail(935, "the tier-B target-pool run is absent",
             "run scripts/23_tierb_target_pool.py (run_all.sh phase 7b); "
             "Section V-E reports the pool diagnostics from it")
        return
    x8 = pd.concat([pd.read_csv(p) for p in pool_paths], ignore_index=True)
    truth(935, "every weight clips to the floor under BOTH target pools",
          bool(np.allclose(x8.frac_at_floor, 1.0)),
          f"frac_at_floor in [{x8.frac_at_floor.min():.4f}, "
          f"{x8.frac_at_floor.max():.4f}] over {len(x8)} rows")
    truth(936, "no single unclipped weight reaches the floor 0.05",
          float(x8.raw_max.max()) < 0.05, f"largest raw weight {x8.raw_max.max():.4f}")
    w = x8.drop_duplicates(["condition", "class_name", "seed", "pool"])
    piv = w.pivot_table(index=["condition", "class_name"], columns="pool",
                        values=["raw_median", "n_pool"])
    lift = piv[("raw_median", "class")] / piv[("raw_median", "all")]
    inv_q = piv[("n_pool", "all")] / piv[("n_pool", "class")]
    rng_claim(937, "the class-conditional pool raises the median ratio by "
                   "3.1-21.4x", 3.09, 21.43, lift.values, 0.02)
    rng_claim(938, "the reciprocal class frequency spans 1.7-17.9",
              1.68, 17.86, inv_q.values, 0.02)
    near(939, "the median ratio under the class pool stays near 6e-4",
         6e-4, float(piv[("raw_median", "class")].max()), 5e-5)
    # The 'all' arm has to reproduce the published tier-B diagnostics exactly.
    try:
        pub = sel(load("e4_tierB__*"), rho=0.5, alpha=0.2, method="weighted_crc")
    except FileNotFoundError:
        note(940, "published tier-B runs absent; skipping the reproduction check")
        return
    pub = pub.assign(condition=pub.experiment.str.extract(r"e4_tierB__(\w+?)__")[0])
    a = x8[(x8["pool"] == "all") & np.isclose(x8.kappa, 20.0)].drop_duplicates(
        ["condition", "class_name", "seed"])
    j = a.merge(pub[["condition", "class_name", "seed", "weight_p_test",
                     "weight_ess"]],
                on=["condition", "class_name", "seed"], suffixes=("_23", "_5"))
    d = max(float((j.weight_p_test_23 - j.weight_p_test_5).abs().max()),
            float((j.weight_ess_23 - j.weight_ess_5).abs().max()))
    truth(940, "stage 23 reproduces the published tier-B weights exactly on "
               "the shared pool", len(j) > 0 and d == 0.0,
          f"{len(j)} matched records, max abs diff {d:.3e}")

    # Section VI(a): the expected number of missed regions per scene is
    # E[N L], not E[N] E[L]. The bound constrains L alone, so the product form
    # is exact only under independence, and the discussion quotes the gap
    # measured here rather than asserting the factorization.
    comp = results_dir("experiments") / "x11_marida_confidence__components.csv"
    if not comp.exists():
        fail(941, "the MARIDA component table is absent",
             "run scripts/26_marida_confidence.py; Section VI(a) quotes the "
             "product-form gap measured from it")
    else:
        cdf = pd.read_csv(comp)
        cdf = cdf[cdf["official_group"] == "test"]
        got = {}
        for a_ in (0.05, 0.2):
            col = f"missed__alpha{a_:g}__rho0.5"
            per = cdf.groupby("image_id")[col].agg(["size", "sum"])
            n_, l_ = per["size"], per["sum"] / per["size"]
            got[a_] = (float(n_.mean()) * float(l_.mean()), float((n_ * l_).mean()))
        near(941, "product form at alpha=0.2 gives 0.44", 0.44, got[0.2][0], 0.005)
        near(941, "the measured count at alpha=0.2 is 0.37", 0.37, got[0.2][1], 0.005)
        truth(941, "the product form overstates the count by about a fifth at "
                   "alpha=0.2",
              1.15 <= got[0.2][0] / got[0.2][1] <= 1.25,
              f"ratio {got[0.2][0] / got[0.2][1]:.3f}")
        truth(941, "the product form understates it at alpha=0.05, so the sign "
                   "of the error is not fixed",
              got[0.05][0] < got[0.05][1],
              f"product {got[0.05][0]:.4f} against measured {got[0.05][1]:.4f}")

def section_capture_rule():
    """The capture rule as the source actually implements it.

    Every other section here re-checks committed CSVs against each other, which
    makes this audit structurally blind to a change in how risk is computed: a
    mutation of the capture comparison leaves all 550 checks green. That blind
    spot is not hypothetical -- two stage-5 drivers widened the cached float16
    curves before the comparison, which at rho = 0.1 reclassified every
    exactly-captured component as missed, and neither this audit nor the test
    suite saw it. These checks read the source and exercise the rule directly.
    """
    from record.losses import capture_threshold, component_miss_matrix
    from record.paths import repo_root
    head("P  Capture rule as implemented in src/record")

    losses_src = (repo_root() / "src" / "record" / "losses.py").read_text()
    eval_src = (repo_root() / "src" / "record" / "evaluation.py").read_text()

    truth(950, "component_miss_matrix compares against the rounded threshold",
          "< capture_threshold(rho)" in losses_src,
          "a bare '< rho' here is the rho = 0.1 defect")
    truth(951, "stratified_region_fnr uses the same rounded threshold",
          "< capture_threshold(rho)" in eval_src,
          "the one-column diagnostic must agree with the matrix")

    # The rule is live, not merely named.
    truth(952, "rho = 0.1 rounds strictly below itself on the storage grid",
          capture_threshold(0.1) < 0.1,
          f"{capture_threshold(0.1):.10f}")
    truth(953, "rho = 0.5 is exact on the storage grid and is left alone",
          capture_threshold(0.5) == 0.5, f"{capture_threshold(0.5)}")

    # The property the drivers depend on: the answer cannot move with dtype.
    stored = np.array([[1.0 / 10.0, 0.0999146], [0.5, 0.49]], dtype=np.float16)
    outs = [component_miss_matrix(stored.astype(d), r)
            for d in (np.float16, np.float32, np.float64) for r in (0.1, 0.5)]
    ref = [component_miss_matrix(stored, r) for r in (0.1, 0.5)] * 3
    truth(954, "the miss matrix is identical whatever dtype the curves arrive in",
          all(np.array_equal(a, b) for a, b in zip(outs, ref)),
          "an astype upstream can no longer move a rho = 0.1 number")
    truth(955, "a component covered by exactly rho is captured, not missed",
          component_miss_matrix(stored.astype(np.float32), 0.1)[0, 0] == 0.0,
          "0.1 stored as float16 is 0.0999756")

    # The drivers, not only the library. The fourth panel (H2 #1) found three
    # bare comparisons in 05_run_experiments.py and one in
    # 26_marida_confidence.py that the two source checks above could not see,
    # because they only read src/record. Every "< rho" outside
    # capture_threshold's own definition is a repeat of the defect.
    import re
    bare = []
    for path in sorted((repo_root() / "scripts").glob("*.py")):
        if path.name == "18_audit.py":
            continue  # this file carries the pattern itself
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"[<>]=?\s*rho\b", line) and "capture_threshold" not in line:
                bare.append(f"{path.name}:{lineno}")
    truth(960, "no script under scripts/ compares a coverage against a bare rho",
          not bare, ", ".join(bare) if bare else "all comparisons go through capture_threshold")




def section_loggrid():
    """The 2026-09-08 grid change: every quantity the text derives from the
    threshold grid itself, plus the abstract's headline numbers."""
    head("Z  Threshold grid and headline numbers (log-tail grid, 2026-09-09)")
    from record.grid import GRID_KIND, N_POINTS
    k = np.arange(N_POINTS)
    want = 1.0 - 10.0 ** (-6.0 * k / (N_POINTS - 1)); want[-1] = 1.0
    truth(970, "the default grid is the log-tail grid of Section IV-A",
          GRID_KIND == "logtail", f"GRID_KIND={GRID_KIND}")
    truth(970, "lambda_k = 1 - 10^(-6k/1000) for k < 1000 and lambda_1000 = 1",
          N_POINTS == 1001 and bool(np.allclose(LAMBDA_GRID, want, atol=0, rtol=1e-12))
          and LAMBDA_GRID[-1] == 1.0,
          f"{N_POINTS} points, max abs deviation {float(np.abs(LAMBDA_GRID - want).max()):.2e}")
    truth(970, "the smallest nonzero cutoff is 1e-6 and the coarsest step is below 0.014",
          abs((1 - LAMBDA_GRID[-2]) - 1e-6 * 10 ** (6 / 1000)) < 1e-9
          and float(np.diff(LAMBDA_GRID).max()) < 0.014,
          f"1-lambda_999 = {1 - LAMBDA_GRID[-2]:.3e}, max step {float(np.diff(LAMBDA_GRID).max()):.4f}")
    tex = results_dir("tables") / "tier_a.tex"
    if tex.exists():
        t = tex.read_text()
        truth(971, "Table VI is over all four models again (no 'SegFormer variants' in its caption) "
                   "and its n_t=25, alpha=0.05 cell rests on 16 draws",
              "SegFormer variants" not in t and "(16)" in t, "")
    e1 = sel(load("e1_indist__*"), method="region_crc")
    near(972, "abstract: Mask2Former marks 5.3% at alpha=0.2", 0.0535,
         mean(sel(e1, model="mask2former_swinb_cityscapes", alpha=0.20, rho=0.5),
              "marked_area_fraction"), 5e-4)
    mc_b2 = [abs(mean(sel(e1, model="segformer_b2_cityscapes_mcdrop8", alpha=a, rho=0.5), "marked_area_fraction")
                 - mean(sel(e1, model="segformer_b2_cityscapes", alpha=a, rho=0.5), "marked_area_fraction"))
             for a in ALPHAS]
    truth(972, "MC-dropout and B2 cost the same area to within a tenth of a point at every level",
          max(mc_b2) <= 0.001, str([round(x, 4) for x in mc_b2]))
    e2 = sel(load("e2_break__*"), method="region_crc", rho=0.5)
    cells = cellmean(e2, ["model", "experiment", "alpha"])
    n_viol = {a: int((cells.xs(a, level="alpha").values > a).sum()) for a in ALPHAS}
    truth(973, "14 of the 16 model-condition cells violate at alpha=0.2 and 0.1, 13 at 0.05",
          n_viol == {0.05: 13, 0.10: 14, 0.20: 14}, str(n_viol))
    # The uniform-grid figures the text quotes for comparison come from the last
    # uniform-grid table in the git history (commit 8e43d21).
    import subprocess
    try:
        old = subprocess.run(["git", "-C", str(paths_root()), "show",
                              "8e43d21:results/tables/validity_cityscapes.tex"],
                             capture_output=True, text=True, check=True).stdout
        vals = {}
        for line in old.splitlines():
            parts = [c.strip() for c in line.split("&")]
            if len(parts) == 7 and parts[0] in ("SegFormer-B2", "SegFormer-B5", "Mask2Former",
                                                "SegFormer-B2 (MC-dropout)"):
                vals[parts[0]] = [float(parts[2]), float(parts[4]), float(parts[6].rstrip("\\ "))]
        seg = [v[i] for m, v in vals.items() if m != "Mask2Former" for i in (0, 1)]
        truth(975, "on the uniform grid the SegFormer variants marked 36-100% at alpha<=0.1 "
                   "and Mask2Former 100% everywhere (git 8e43d21)",
              abs(min(seg) - 0.358) < 5e-4 and max(seg) == 1.0
              and all(x == 1.0 for x in vals["Mask2Former"]),
              f"SegFormer alpha<=0.1 areas {sorted(seg)}; Mask2Former {vals['Mask2Former']}")
    except Exception as exc:  # no git history here (e.g. the server tree)
        note(975, "uniform-grid comparison table not reachable through git",
             f"{type(exc).__name__}; the text's 36-100% figure rests on commit 8e43d21")


SECTIONS = {
    "D": section_indist, "F": section_baselines, "G": section_ablations,
    "F2": section_lac_detail,
    "H": section_breakdown, "I": section_tier_a, "J": section_tier_b,
    "K": section_loveda, "L": section_marida, "M": section_triage,
    "N": section_holdout, "R": section_revision, "S": section_review, "P": section_capture_rule, "X": section_cross,
    "Z": section_loggrid,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None,
                    help="comma-separated section keys: " + ",".join(SECTIONS))
    ap.add_argument("--section", default=None,
                    help="run a single named section (alias for --only); "
                         "'holdout' selects the experimental-design check")
    args = ap.parse_args()
    named = {"holdout": "N"}
    if args.section:
        keys = [named.get(args.section, args.section)]
    else:
        keys = args.only.split(",") if args.only else list(SECTIONS)
    for k in keys:
        try:
            SECTIONS[k]()
        except Exception as exc:  # a failing check must not hide the rest
            global N_FAIL
            N_FAIL += 1
            print(f"[ERROR] section {k}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n=== SUMMARY: {N_OK} ok, {N_FAIL} FAIL, {N_NOTE} notes ===")
    return 1 if N_FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
