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

from record.grid import FP_SUBGRID_INDICES, fp_subgrid_slot
from record.paths import results_dir

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
    truth(53, "26 of the 72 cells mark more than 99% of the image",
          n_sat == 26, f"{n_sat} cells")
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
    truth(52, "at most eight cells, all at alpha=0.2, sit within 0.004 of the "
              "level at the upper end of the 95% interval",
          len(over) <= 8 and all(np.isclose(k[2], 0.2) for k in over.index)
          and float((over.values - 0.2).max()) <= 0.004,
          f"{len(over)} cells, max excess {float((over.values - 0.2).max()):.4f}"
          if len(over) else "none")
    lev2 = lo_.index.get_level_values("alpha").values
    truth("52b", "no cell has a 95% LOWER end above the level (violation test)",
          bool((lo_.values <= lev2 + 1e-12).all()),
          "max lower end - alpha = %.4f" % float((lo_.values - lev2).max()))

    # --- 54 / 2 / 231: marked area 2.5-9% at alpha=0.2 ---
    a = [mean(sel(r, model=mo, alpha=0.20, rho=0.5), "marked_area_fraction")
         for mo in SEGF]
    rng_claim(54, "SegFormer marked area 2.5-9% at alpha=0.2", 0.025, 0.090, a)

    # --- 55: alpha=0.05 marks most of the image ---
    a05 = {mo: mean(sel(r, model=mo, alpha=0.05, rho=0.5), "marked_area_fraction")
           for mo in CS_MODELS}
    truth(55, "alpha=0.05 requires marking most of the image (>0.8 everywhere)",
          min(a05.values()) > 0.8, str({k: round(v, 3) for k, v in a05.items()}))

    # --- 57: MC-dropout 36% vs B2 63% at alpha=0.1 ---
    near(57, "MC-dropout area at alpha=0.1 is 0.358",
         0.358, mean(sel(r, model="segformer_b2_cityscapes_mcdrop8", alpha=0.10,
                         rho=0.5), "marked_area_fraction"))
    near(57, "B2 area at alpha=0.1 is 0.625",
         0.625, mean(sel(r, model="segformer_b2_cityscapes", alpha=0.10,
                         rho=0.5), "marked_area_fraction"))

    # --- 59: Mask2Former degenerate at rho=0.5 ---
    m2f = sel(r, model="mask2former_swinb_cityscapes", rho=0.5)
    truth(59, "Mask2Former marks everything at rho=0.5, every level",
          bool(np.allclose(m2f.marked_area_fraction, 1.0)) and
          bool(np.allclose(m2f.lam, m2f.lam.max())),
          f"area min={m2f.marked_area_fraction.min():.3f}, lam unique={sorted(m2f.lam.unique())[:3]}")

    # --- 61-64: in-distribution Cityscapes cells ---
    tab2 = {
        "segformer_b2_cityscapes": [(0.003, 0.956), (0.036, 0.625), (0.182, 0.034)],
        "segformer_b5_cityscapes": [(0.000, 1.000), (0.052, 0.459), (0.180, 0.090)],
        "mask2former_swinb_cityscapes": [(0.000, 1.000), (0.000, 1.000), (0.000, 1.000)],
        "segformer_b2_cityscapes_mcdrop8": [(0.011, 0.803), (0.058, 0.358), (0.183, 0.025)],
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
    near(67, "pixel CRC worst region FNR at alpha=0.2, rho=0.5 is 0.257",
         0.257, max(pix.values()))
    truth(4, "pixel CRC violates the region target in distribution",
          max(pix.values()) > 0.20, str({k: round(v, 3) for k, v in pix.items()}))

    # 68: paired (FNR, area) on B5
    b5 = sel(e1, model="segformer_b5_cityscapes", alpha=0.20, rho=0.5)
    near(68, "B5 pixel CRC area 1.8%", 0.018,
         mean(sel(b5, method="pixel_crc"), "marked_area_fraction"))
    near(68, "B5 region CRC 0.180 at 9.0% area", 0.180,
         mean(sel(b5, method="region_crc"), "region_fnr"))
    near(68, "B5 region CRC area 9.0%", 0.090,
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
    truth(77, "B5 marks a larger area than B2 (9.0% against 3.4%)",
          a_b5 > a_b2 and abs(a_b5 - 0.090) < 5e-4 and abs(a_b2 - 0.034) < 5e-4,
          f"B5 area={a_b5:.3f} vs B2 area={a_b2:.3f} at alpha=0.2, rho=0.5")

    # 79: the new per-class LoveDA and breakdown conventions
    lov = sel(load("l1_*"), method="region_crc")
    lcells = lov[lov.feasible.astype(bool)].groupby(
        ["model", "class_name", "alpha", "rho"]).agg(
        fnr=("region_fnr", "mean"), area=("marked_area_fraction", "mean"))
    truth(79, "LoveDA per-class table covers 24 in-domain cells",
          len(lcells) == 24, f"{len(lcells)} cells")
    vac = lcells[lcells.area > 0.99]
    truth(79, "exactly two LoveDA in-domain cells mark >99% of the image",
          len(vac) == 2, str(sorted(vac.index.tolist())))
    truth(79, "both vacuous cells are rural water at rho=0.5, a<=0.1",
          all(m.endswith("rural") and c == "water" and np.isclose(rh, 0.5)
              and al <= 0.1 for m, c, al, rh in vac.index),
          str(sorted(vac.index.tolist())))
    over = lcells[lcells.fnr > lcells.index.get_level_values("alpha")]
    truth(79, "exactly three LoveDA in-domain cells exceed their level",
          len(over) == 3, str(sorted(over.index.tolist())))

    brk = sel(load("e2_break__*"), method="region_crc", rho=0.5)
    bcells = brk[brk.feasible.astype(bool)].groupby(
        ["model", "experiment", "alpha"]).agg(
        area=("marked_area_fraction", "mean"))
    star05 = bcells[(bcells.index.get_level_values("alpha") == 0.05)
                    & (bcells.area > 0.99)]
    models05 = sorted({m for m, _, _ in star05.index})
    truth(79, "at alpha=0.05 the vacuous breakdown cells are B5 and Mask2Former",
          models05 == ["mask2former_swinb_cityscapes",
                       "segformer_b5_cityscapes"],
          f"{len(star05)} starred cells over models {models05}")

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
        rng_claim(80, "LAC misses 96% at alpha=0.2", 0.958, 0.960, v02, 0.002)
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

    # 85: B5 area 9.0% -> 1.5%
    b5t = [mo for mo in tr.model.unique() if "b5" in mo]
    near(85, "tempered B5 area at alpha=0.2, rho=0.5 is 0.015", 0.015,
         mean(sel(tr, model=b5t, alpha=0.20, rho=0.5), "marked_area_fraction"))

    # 90-111: every cell of the method-baseline comparison
    table3 = {
        ("segformer_b2_cityscapes", "argmax"): [(0.586, 0.007), (0.676, 0.007)],
        ("segformer_b2_cityscapes", "heuristic"): [(0.192, 0.020), (0.187, 0.030)],
        ("segformer_b2_cityscapes", "pixel_crc"): [(0.166, 0.025), (0.230, 0.025)],
        ("segformer_b2_cityscapes", "region_crc"): [(0.187, 0.020), (0.182, 0.034)],
        ("segformer_b5_cityscapes", "argmax"): [(0.376, 0.007), (0.470, 0.007)],
        ("segformer_b5_cityscapes", "heuristic"): [(0.197, 0.011), (0.189, 0.060)],
        ("segformer_b5_cityscapes", "pixel_crc"): [(0.199, 0.018), (0.257, 0.018)],
        ("segformer_b5_cityscapes", "region_crc"): [(0.192, 0.015), (0.180, 0.090)],
        ("mask2former_swinb_cityscapes", "argmax"): [(0.548, 0.007), (0.633, 0.007)],
        ("mask2former_swinb_cityscapes", "heuristic"): [(0.062, 0.691), (0.000, 1.000)],
        ("mask2former_swinb_cityscapes", "pixel_crc"): [(0.081, 0.635), (0.105, 0.635)],
        ("mask2former_swinb_cityscapes", "region_crc"): [(0.058, 0.710), (0.000, 1.000)],
        ("segformer_b2_cityscapes_mcdrop8", "argmax"): [(0.592, 0.007), (0.684, 0.007)],
        ("segformer_b2_cityscapes_mcdrop8", "heuristic"): [(0.195, 0.020), (0.188, 0.024)],
        ("segformer_b2_cityscapes_mcdrop8", "pixel_crc"): [(0.166, 0.021), (0.236, 0.021)],
        ("segformer_b2_cityscapes_mcdrop8", "region_crc"): [(0.190, 0.020), (0.183, 0.025)],
    }
    for (mo, me), rows in table3.items():
        for rh, (f_, a_) in zip((0.1, 0.5), rows):
            d = sel(e1, model=mo, method=me, alpha=0.20, rho=rh)
            near(90, f"baseline {mo}/{me} rho={rh} FNR", f_, mean(d, "region_fnr"))
            near(90, f"baseline {mo}/{me} rho={rh} area", a_,
                 mean(d, "marked_area_fraction"))

    # 94/101: LAC rows; 95/96/102/103: tempered-posterior rows
    lacm = [m for m in lac.method.unique() if "lac" in m]
    lac_rows = {"segformer_b2": [(0.871, 0.003), (0.958, 0.003)],
                "segformer_b5": [(0.807, 0.003), (0.960, 0.003)]}
    for key, rows in lac_rows.items():
        mos = [m for m in lac.model.unique() if m.startswith(key)]
        for rh, (f_, a_) in zip((0.1, 0.5), rows):
            d = sel(lac, method=lacm[0], model=mos, alpha=0.20, rho=rh)
            near(94, f"baseline {key} LAC rho={rh} FNR", f_, mean(d))
            near(94, f"baseline {key} LAC rho={rh} area", a_,
                 mean(d, "marked_area_fraction"))
    tmp_rows = {("segformer_b2", "heuristic"): [(0.197, 0.020), (0.196, 0.024)],
                ("segformer_b2", "region_crc"): [(0.192, 0.020), (0.191, 0.025)],
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
    for mo, area_q, exp_r, exp_p in [("segformer_b2_cityscapes", 0.068, 0.171, 0.209),
                                     ("segformer_b5_cityscapes", 0.138, 0.163, 0.206)]:
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
        near(71, f"{mo} region CRC at {area_q:.1%} area", exp_r, rr, 0.002)
        near(71, f"{mo} pixel CRC at {area_q:.1%} area", exp_p, pp, 0.002)
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
            (0.5, (0.958, 0.960), (0.510, 0.612)),
            (0.1, (0.807, 0.871), (0.4125, 0.5320))]:
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
    rng_claim(97, "LAC marked area at alpha=0.2 is 0.3% of the image",
              0.0026, 0.0027, a20)
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

    Both are produced by stage 5 only when asked for, so this check has two
    jobs. If a run recorded that it asked for them, they must be present --
    a run that requested the arm and shipped without it is a silent hole in
    the table, which is exactly the failure this check exists to catch. If no
    run asked for them, that is reported as an outstanding item, not passed
    over.
    """
    raw = load("x4_lac__*")
    argv = _x4_argv()
    asked_cc = any("class_conditional" in (a.get("lac_variants") or []) for a in argv)
    asked_px = any(a.get("measure_pixel_fnr") for a in argv)

    cc = raw[raw["method"] == "lac_classcond"]
    if asked_cc:
        truth(98, "class-conditional LAC rows present in x4_lac__*",
              not cc.empty,
              f"{len(cc)} rows" if not cc.empty
              else "a run requested --lac-variants class_conditional but wrote no lac_classcond row")
    elif not cc.empty:
        truth(98, "class-conditional LAC rows present in x4_lac__*", True,
              f"{len(cc)} rows")
    else:
        note(98, "class-conditional LAC arm not yet run",
             "rerun stage 5 with --lac-variants marginal class_conditional; "
             "the table row is written the moment the rows exist")

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
                 0.002, float(dreg.max()), tol=5e-4)
            near(98, "largest class-conditional / pixel-CRC marked-area gap",
                 0.028, float(darea.max()), tol=5e-4)

    has_px = "realized_pixel_fnr" in raw.columns
    if asked_px:
        truth(99, "realized_pixel_fnr column present in x4_lac__*", has_px,
              "a run requested --measure-pixel-fnr but wrote no column"
              if not has_px else "")
    elif not has_px:
        note(99, "realized pixel FNR not yet measured",
             "rerun stage 5 with --measure-pixel-fnr")
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
MARGINAL_REGION_SEPARATION = 0.40


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
    for alpha, want in ((0.05, 0.46), (0.10, 0.67), (0.20, 0.88)):
        near(100, f"marginal LAC pixel FNR on the critical class at alpha={alpha:g}",
             want, float(by_level[("lac_global", alpha)]), tol=5e-3)
    excess = [float(by_level[("lac_global", a)] - a) for a in (0.05, 0.10, 0.20)]
    rng_claim(100, "the marginal excess over the level spans 0.41-0.68",
              0.41, 0.68, excess, tol=5e-3)
    for alpha, want in ((0.05, 0.005), (0.10, 0.060), (0.20, 0.188)):
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
    rng_claim(113, "per-class area 36-63% at alpha=0.1", 0.358, 0.625, per)
    truth(113, "shared area is 100% at alpha=0.1",
          bool(np.allclose(shd, 1.0, atol=5e-3)), str([round(x, 3) for x in shd]))

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
    truth(116, "caption: shared threshold inflates the marked area by up to 3.0x",
          2.5 <= worst <= 3.0, f"largest measured ratio = {worst:.2f}x "
                               f"({max(ratios, key=ratios.get)})")

    # 114: size-weighted loss
    d = sel(sw, alpha=0.20, rho=0.5)
    cr = mean(d, "controlled_risk")
    fnrs = [mean(sel(d, model=mo), "region_fnr") for mo in
            ["segformer_b2_cityscapes", "segformer_b5_cityscapes"]]
    near(114, "size-weighted controlled risk at alpha=0.2", 0.19, cr, 0.006)
    rng_claim(114, "size-weighted unweighted region FNR 0.26-0.30",
              0.255, 0.296, fnrs, 0.006)


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
    near(124, "worst violation ratio at alpha=0.2 is 2.03x", 2.03, float(worst), 0.01)
    note(124, "worst cell", f"{c.idxmax()} = {c.max():.4f}")

    b5fog = mean(sel(r, model="segformer_b5_cityscapes", alpha=0.20, rho=0.5,
                     experiment="e2_break__fog__segformer_b5_cityscapes"))
    truth(125, "B5 under fog exceeds the level only in the fourth decimal",
          0.20 < b5fog <= 0.2005, f"{b5fog:.4f}")

    c1 = cellmean(sel(r, model=SEGF, alpha=0.10, rho=0.5), ["experiment"])
    near(126, "worst violation ratio at alpha=0.1 is 1.5x", 1.48,
         float(c1.max() / 0.10), 0.02)

    mc = cellmean(sel(r, model="segformer_b2_cityscapes_mcdrop8", alpha=0.10,
                      rho=0.5), ["experiment"])
    mc = {k.split("__")[1]: v for k, v in mc.items()}
    truth(127, "MC-dropout breaks under fog/rain/snow but not at night at alpha=0.1",
          all(mc[k] > 0.10 for k in ("fog", "rain", "snow")) and mc["night"] <= 0.10,
          str({k: round(v, 3) for k, v in mc.items()}))

    c05 = cellmean(sel(r, alpha=0.05, rho=0.5), ["model", "experiment"])
    truth(128, "no cell violates at alpha=0.05",
          bool((c05.values <= 0.05 + 1e-12).all()),
          f"max={c05.max():.4f}")

    cls = cellmean(sel(r, alpha=0.20, rho=0.5), ["model", "experiment", "class_name"])
    near(129, "worst class-level cell is 0.487", 0.487, float(cls.max()))
    note(129, "worst class-level cell identity", str(cls.idxmax()))

    an = [mean(sel(e2, method="argmax", model=mo, alpha=0.20, rho=rh,
                   experiment=f"e2_break__night__{mo}"))
          for mo in CS_MODELS for rh in (0.1, 0.5)]
    rng_claim(130, "argmax misses 68-83% of regions at night", 0.679, 0.835, an)

    m2f = sel(r, model="mask2former_swinb_cityscapes", rho=0.5)
    truth(131, "Mask2Former under shift: FNR 0 and area 1 everywhere",
          bool(np.allclose(m2f.region_fnr, 0.0)) and
          bool(np.allclose(m2f.marked_area_fraction, 1.0)),
          f"max FNR={m2f.region_fnr.max():.4f}, min area={m2f.marked_area_fraction.min():.4f}")

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
    near(132, "46.8% of region-CRC draws at rho=0.5 are degenerate", 0.468,
         float((mon.lam_index == LAM_MAX).mean()), 0.002)
    usable = mon[mon.lam_index < LAM_MAX]
    fl = usable.groupby("experiment")["monitor_flag"].mean()
    rng_claim(132, "monitor fires in 64-100% of the non-degenerate region-CRC "
                   "(draw, class, level) configurations at rho=0.5",
              0.64, 1.00, fl.values, 0.006)
    ind = sel(e1, method="region_crc", rho=0.5)
    ind = ind[ind.model.isin(SEGF)]
    ind = ind[ind.lam_index < LAM_MAX]
    fa = ind.groupby("model")["monitor_flag"].mean()
    rng_claim(132, "in-distribution false-alarm rate 0.5-1.9% on the same subset",
              0.005, 0.019, fa.values, 0.0006)
    note(132, "nominal KS level is 1%; the realized in-distribution rate is",
         f"{float(ind.monitor_flag.mean()):.4f} pooled over the three variants")

    by_cond = {}
    for c_ in CONDS:
        sub = sel(r, rho=0.5)
        sub = sub[sub.model.isin(SEGF) & sub.experiment.str.contains(f"__{c_}__")]
        cellsc = cellmean(sel(sub, alpha=0.20), ["model"])
        by_cond[c_] = (float(cellsc.mean()), float(cellsc.max()),
                       float(sub[sub.lam_index < LAM_MAX].monitor_flag.mean()))
    for c_, want in (("night", 0.729), ("rain", 0.839),
                     ("snow", 0.982), ("fog", 1.000)):
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
        "segformer_b2_cityscapes": [0.007, 0.003, 0.003, 0.004, 0.066, 0.036,
                                    0.051, 0.057, 0.325, 0.298, 0.387, 0.345],
        "segformer_b5_cityscapes": [0.000, 0.000, 0.000, 0.000, 0.062, 0.148,
                                    0.081, 0.087, 0.200, 0.406, 0.309, 0.225],
        "mask2former_swinb_cityscapes": [0.0] * 12,
        "segformer_b2_cityscapes_mcdrop8": [0.025, 0.008, 0.012, 0.011, 0.104,
                                            0.065, 0.119, 0.134, 0.320, 0.291,
                                            0.388, 0.333],
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
    truth(140, "tier A restores the guarantee in 644 of the 648 feasible cells, "
               "the four exceptions resting on ten or fewer feasible draws",
          len(res) == 648 and len(bad) == 4 and int(bad["size"].max()) <= 10
          and abs(float(bad["mean"].max()) - 0.265) < 5e-4,
          f"{len(bad)} of {len(res)} cells above the level; feasible draws behind "
          f"them: {bad['size'].tolist()}; worst "
          f"{bad['mean'].max():.3f} at {bad['mean'].idxmax()}" if len(bad) else
          f"{len(res)} cells all at or below the level")

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
    def _balanced(n, alpha, col="region_fnr"):
        d = sel(r, n_target=n, alpha=alpha, rho=0.5)
        d = d[d.feasible.astype(bool)]
        return float(d.groupby("class_name")[col].mean().mean())

    ar = [_balanced(n, 0.20, "marked_area_fraction") for n in (25, 50, 100)]
    rng_claim(142, "tier-A marked area 67-73% at alpha=0.2 (class-balanced)",
              0.668, 0.726, ar, 0.006)
    truth(142, "area does not decrease from n_t=25 to 100",
          not (ar[0] > ar[1] > ar[2]), str([round(x, 3) for x in ar]))

    inf25 = 1 - sel(r, n_target=25, alpha=0.05).feasible.astype(bool).mean()
    inf100 = 1 - sel(r, n_target=100, alpha=0.05).feasible.astype(bool).mean()
    truth(145, "alpha=0.05 unattainable in >99% of n_t=25 draws",
          inf25 > 0.99, f"{inf25:.4f}")
    near(145, "infeasible fraction at n_t=100, alpha=0.05 is 0.47", 0.47, inf100, 0.006)

    table6 = {25: [(0.005, 0.940, 0.99), (0.011, 0.912, 0.74), (0.065, 0.726, 0.38)],
              50: [(0.001, 0.980, 0.80), (0.015, 0.887, 0.45), (0.078, 0.668, 0.19)],
              100: [(0.002, 0.972, 0.47), (0.014, 0.876, 0.27), (0.071, 0.670, 0.03)]}
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
    lam_max = w.lam.max()
    truth(153, "tier B returns lambda_max in EVERY configuration of the main matrix",
          bool((w.lam >= lam_max - 1e-12).all()),
          f"{int((w.lam < lam_max).sum())} of {len(w)} rows informative")
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
              bool((cv.lam >= cv.lam.max() - 1e-12).all()),
              f"{int((cv.lam < cv.lam.max()).sum())} informative")
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

    for al, targets in [(0.20, {20: (0.50, 0.093, 0.51), 10: (0.73, 0.174, 0.28),
                                8: (0.80, 0.211, None), 4: (0.93, 0.282, None)})]:
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

    big = p3[(p3.ratio >= 50) & np.isclose(p3.alpha, 0.20)]
    truth(162, "at alpha=0.2 ratios of 50 and above leave nothing informative",
          bool((big.lam >= big.lam.max() - 1e-12).all()) if len(big) else False,
          f"{int((big.lam < big.lam.max()).sum())} informative of {len(big)}")

    a10 = p3[np.isclose(p3.alpha, 0.10)]
    inf_by_ratio = a10.assign(inf=(a10.lam < a10.lam.max() - 1e-12)).groupby(
        "ratio")["inf"].mean()
    risk_by_ratio = a10.groupby("ratio")["region_fnr"].mean()
    active = inf_by_ratio[inf_by_ratio > 0]
    truth(167, "at alpha=0.1 only ratios <=10 return anything",
          bool((active.index <= 10).all()) if len(active) else False,
          str({round(k, 1): round(v, 3) for k, v in active.items()}))
    truth(167, "and every such ratio is valid at alpha=0.1",
          bool((risk_by_ratio[active.index] <= 0.10 + 1e-12).all()) if len(active) else False,
          str({round(k, 1): round(risk_by_ratio[k], 4) for k in active.index}))

    a05 = p3[np.isclose(p3.alpha, 0.05) & np.isclose(p3.rho, 0.5)]
    truth(168, "at alpha=0.05 nothing is informative at any ratio (rho=0.5)",
          bool((a05.lam >= a05.lam.max() - 1e-12).all()),
          f"{int((a05.lam < a05.lam.max()).sum())} informative of {len(a05)}")
    a05b = p3[np.isclose(p3.alpha, 0.05) & np.isclose(p3.rho, 0.1)]
    n_inf = int((a05b.lam < a05b.lam.max() - 1e-12).sum())
    note(168, "same claim at rho=0.1",
         f"{n_inf} informative draws of {len(a05b)}" +
         ("" if not n_inf else " -- the claim is false at rho=0.1"))

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
    over = int((j["region_fnr_sq"].values > lev).sum())
    truth(245, "no sequence-disjoint cell exceeds its level", over == 0,
          f"{over} of {len(j)} cells above the level; worst "
          f"{float(j['region_fnr_sq'].max()):.4f}")
    near(246, "largest published tier-A cell at n_t=25", 0.113,
         float(j["region_fnr_pub"].max()), tol=5e-4)
    near(246, "largest sequence-disjoint cell at n_t=25", 0.127,
         float(j["region_fnr_sq"].max()), tol=5e-4)
    move = j["region_fnr_sq"] - j["region_fnr_pub"]
    near(246, "largest single movement in region FNR", 0.055,
         float(move.max()), tol=5e-4)
    truth(246, "the largest movement is on snow at alpha=0.2, rho=0.5",
          move.idxmax() == ("snow", 0.20, 0.5), str(move.idxmax()))
    area = j["marked_area_fraction_sq"] - j["marked_area_fraction_pub"]
    truth(247, "marked area is smaller in fourteen of the eighteen cells",
          int((area < 0).sum()) == 14, f"{int((area < 0).sum())} of {len(area)}")
    truth(247, "the largest area reduction is at most 0.17",
          float(-area.min()) <= 0.175,
          f"largest reduction {float(-area.min()):.4f} at {area.idxmin()}")


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
        near(251, "ensemble at alpha=0.2: no-High", 0.247,
             float(er["miss_rate_component_avg_no_high"]), tol=5e-4)
    else:
        note(251, "the ensemble confidence run is missing",
             "rerun stage 26 with --out-stem x11_marida_confidence__ens5")

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
    near(177, "tightest urban cell 0.199", 0.199,
         mean(sel(r, experiment="l1_indist__urban", alpha=0.20)))
    near(177, "tightest rural cell 0.195", 0.195,
         mean(sel(r, experiment="l1_indist__rural", alpha=0.20)))
    near(178, "LoveDA urban pixel CRC region FNR 0.219", 0.219,
         mean(sel(i, method="pixel_crc", experiment="l1_indist__urban",
                  alpha=0.20, rho=0.5)))

    u2r = cellmean(sel(b, method="region_crc", experiment="l2_break__urban2rural",
                       rho=0.5), ["alpha"])
    ratios = [u2r[a_] / a_ for a_ in ALPHAS if a_ in u2r.index]
    rng_claim(180, "urban->rural violates by 2.5-3.0x", 2.46, 3.04, ratios, 0.01)
    r2u = cellmean(sel(b, method="region_crc", experiment="l2_break__rural2urban",
                       rho=0.5), ["alpha"])
    truth(181, "rural->urban transfers almost intact (FNR <= alpha everywhere)",
          bool(all(r2u[a_] <= a_ + 1e-12 for a_ in ALPHAS if a_ in r2u.index)),
          str({k: round(v, 3) for k, v in r2u.items()}))

    try:
        t = load("l3_tierA*")
        t["n_target"] = pd.to_numeric(t.experiment.str.extract(r"tierA(\d+)")[0])
        tr = sel(t, method="region_crc", rho=0.5)
        ar = list(cellmean(sel(tr, alpha=0.20), ["n_target", "experiment"],
                           "marked_area_fraction").values)
        rng_claim(183, "LoveDA tier-A area 21-39% at alpha=0.2", 0.211, 0.391, ar, 0.006)
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
        "h1_official__marida_unet_official_holdout_ens5": [0.069, 0.098, 0.265],
        "h2_region__16PCC__marida_unet_holdout_region_16PCC": [0.000, 0.027, 0.094],
        "h2_region__16PDC__marida_unet_holdout_region_16PDC": [0.009, 0.027, 0.142],
        "h2_region__16PEC__marida_unet_holdout_region_16PEC": [0.000, 0.159, 0.333],
        "h2_region__18QYF__marida_unet_holdout_region_18QYF": [0.001, 0.096, 0.267],
        "h2_region__48PZC__marida_unet_holdout_region_48PZC": [0.000, 0.250, 0.375],
        "h3_season__winter__marida_unet_holdout_season_winter": [0.030, 0.041, 0.108],
        "h3_season__spring__marida_unet_holdout_season_spring": [0.217, 0.357, 0.430],
        "h3_season__summer__marida_unet_holdout_season_summer": [0.000, 0.005, 0.044],
        "h3_season__autumn__marida_unet_holdout_season_autumn": [0.000, 0.002, 0.054],
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
    for al, ratio, lo in zip(ALPHAS, [4.35, 3.57, 2.15], [0.063, 0.165, 0.231]):
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
        note(190, "component summary and tables unreadable; debris-patch "
                  "count not checked")
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
        note(202, "component summary and tables absent; size statistics "
                  "not checked")

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

    quoted = {"official": (0.075, 0.036, 51.5), "region_16PCC": (0.025, 0.006, 77),
              "region_16PDC": (0.029, 0.021, 29), "region_16PEC": (0.053, 0.021, 61),
              "season_spring": (0.148, 0.060, 60), "night_tierA50": (0.036, 0.022, 38)}
    for name, (r0, r50, pct) in quoted.items():
        if name not in curves:
            note(220, f"{name}: triage curve missing"); continue
        a, R0, _, _ = curves[name]
        near(220, f"{name}: no-review rate", r0, R0, 0.001)
        near(220, f"{name}: residual at beta=0.5", r50, float(a.loc[0.5]), 0.001)
        near(220, f"{name}: reduction at beta=0.5 in percent", pct,
             100 * (1 - float(a.loc[0.5]) / R0), 0.5)

    # the budget at which the score first drops below the exact random floor
    first_below = {"official": 0.50, "region_16PCC": 0.10, "region_16PEC": 0.25,
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

    for name, want in (("region_16PDC", 0.21), ("official", -0.015)):
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
    truth(230, "80 comparisons: 9 below the band, 18 above, 53 inside",
          (n, below, above, inside) == (80, 9, 18, 53),
          f"{n} comparisons, {below} below, {above} above, {inside} inside")
    truth(230, "1000 permutations per setting",
          set(band["n_permutations"].unique()) == {1000},
          str(sorted(band["n_permutations"].unique())))
    truth(230, "six settings", band["setting"].nunique() == 6,
          str(sorted(band["setting"].unique())))

    # Every above-band outcome is in the one replicated setting.
    ab = sorted(w.loc[w["above"], "setting"].unique())
    truth(231, "every above-band outcome arises in the replicated night setting",
          ab == ["night_tierA50"], f"above-band settings: {ab}")
    ni = w[w["setting"] == "night_tierA50"]
    truth(231, "night, tier A: above the band in 18 of its 30 comparisons",
          (int(ni["above"].sum()), len(ni)) == (18, 30),
          f"{int(ni['above'].sum())} of {len(ni)}")

    # The five single-partition settings, and 16PCC as the single exception.
    per = w.groupby("setting")["below_band"].sum()
    never = sorted(per[per == 0].index)
    truth(231, "the score never leaves the band in four of the five single "
               "MARIDA partitions",
          set(never) == {"official", "region_16PDC", "region_16PEC",
                         "season_spring"},
          f"never below: {never}")
    pcc = w[(w["setting"] == "region_16PCC") & w["below_band"]]
    truth(231, "16PCC falls below the band from beta=0.1 upward",
          not pcc.empty and abs(float(pcc["budget"].min()) - 0.10) < 1e-9,
          f"first separation at beta={float(pcc['budget'].min()):g}"
          if not pcc.empty else "never separates")

    # The two quoted bands at beta = 0.5.
    def cell(setting):
        g = w[(w["setting"] == setting) & np.isclose(w["budget"], 0.5)]
        return (float(g["area_rel"].mean()), float(g["q05_rel"].mean()),
                float(g["q50_rel"].mean()), float(g["q95_rel"].mean()))

    a, lo, med, hi = cell("official")
    near(232, "official at beta=0.5: score", 0.49, a, tol=5e-3)
    near(232, "official at beta=0.5: band lower end", 0.37, lo, tol=5e-3)
    near(232, "official at beta=0.5: band upper end", 0.64, hi, tol=5e-3)
    near(232, "official at beta=0.5: band median", 0.51, med, tol=5e-3)
    a, lo, med, hi = cell("night_tierA50")
    near(232, "night at beta=0.5: score", 0.63, a, tol=5e-3)
    near(232, "night at beta=0.5: band lower end", 0.46, lo, tol=5e-3)
    near(232, "night at beta=0.5: band upper end", 0.54, hi, tol=5e-3)
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
    truth("X1", "worst violation ratio per axis: "
                "2.0x weather, 3.0x geographic, 4.3x seasonal",
          abs(worst.get("ACDC", 0) - 2.03) < 0.01
          and abs(worst.get("LoveDA", 0) - 3.04) < 0.01
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
        near(70, "region CRC leads by at most 0.043 on the released grid",
             0.043, max(deltas), 0.0005)
        near(70, "region CRC trails by at most 0.024 on the released grid",
             -0.024, min(deltas), 0.0005)

    # The prose compares the two frontiers at every matched area, not only at
    # the six tabulated points, so the dense interpolation is checked too.
    try:
        dense = load("p1_pareto__*")
    except FileNotFoundError:
        note(70, "dense-alpha pareto runs missing", "skipped")
        dense = None
    if dense is not None:
        want = {"segformer_b2_cityscapes": (0.044, -0.016),
                "segformer_b5_cityscapes": (0.064, -0.025)}
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
          len(over) == 1 and abs(over[0][1] - 0.2006) < 5e-4, str(over))

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
    for alpha, want in [(0.05, 0.155), (0.10, 0.250), (0.20, 0.300)]:
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
        note(99, "temperature runs missing", "skipped")
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
        note(98, "union-area runs missing", "skipped")
        uni = None
    if uni is not None:
        uni = uni[np.isclose(uni["alpha"], 0.2)]
        bad = int(((uni["union_area"] < uni["max_class_area"] - 1e-9)
                   | (uni["union_area"] > uni["sum_class_area"] + 1e-9)).sum())
        truth(98, "every draw satisfies max_c <= union <= min(1, sum_c)",
              bad == 0, f"{bad} violations in {len(uni)} draws")
        good = uni[uni["union_area"] <= 0.99]
        per = good.groupby("model")[["class_mean_area", "union_area"]].mean()
        rng_claim(98, "measured union covers 3.4-5.0% of the image",
                  0.0341, 0.0495, list(per["union_area"]), tol=0.0005)
        rng_claim(98, "class mean on the same draws is 1.4-2.5%",
                  0.0140, 0.0246, list(per["class_mean_area"]), tol=0.0005)
        ratios = list(per["union_area"] / per["class_mean_area"])
        rng_claim(98, "the union is 2.0-2.5 times the class mean",
                  2.01, 2.45, ratios, tol=0.01)
        n_degen = int((uni["union_area"] > 0.99).sum())
        truth(98, "none to three degenerate draws per model",
              0 <= n_degen <= 3 * uni["model"].nunique(),
              f"{n_degen} across {uni['model'].nunique()} models")

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
    quoted = {("marida_unet_official_holdout", 0.05): 0.047,
              ("marida_unet_official_holdout", 0.10): 0.097,
              ("marida_unet_official_holdout", 0.20): 0.202,
              ("marida_unet_official_holdout_ens5", 0.05): 0.049,
              ("marida_unet_official_holdout_ens5", 0.10): 0.098,
              ("marida_unet_official_holdout_ens5", 0.20): 0.199}
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
    truth(920, "three of the 24 LoveDA in-domain per-class cells exceed the "
               "level", len(cells) == 24 and len(over) == 3,
          f"{len(over)} of {len(cells)}")
    truth(921, "every LoveDA exceedance is water at alpha=0.2",
          bool((over.class_name == "water").all()
               and np.allclose(over.alpha, 0.2)),
          str(sorted(zip(over.class_name, over.alpha))))
    near(922, "the largest LoveDA in-domain exceedance is 0.003",
         0.003, float((over.region_fnr - over.alpha).max()), 0.0005)
    hi = over.region_fnr + 1.96 * over["std"] / np.sqrt(over["size"])
    lo = over.region_fnr - 1.96 * over["std"] / np.sqrt(over["size"])
    truth(923, "the level is inside the draw interval for all three",
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
        note(932, "union-area runs absent; skipping the union checks")
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
              2.01, 2.45, ratios_mean, 0.01)
    rng_claim(934, "the union is 67-82% of the sum of the three masks",
              0.669, 0.816, ratios_sum, 0.005)

    # Section V-E: the class-conditional target pool does not lift the collapse.
    pool_paths = sorted(glob.glob(str(exp_dir / "x8_tierb_pool__*.csv")))
    if not pool_paths:
        note(935, "tier-B target-pool run absent; skipping the pool checks")
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

SECTIONS = {
    "D": section_indist, "F": section_baselines, "G": section_ablations,
    "F2": section_lac_detail,
    "H": section_breakdown, "I": section_tier_a, "J": section_tier_b,
    "K": section_loveda, "L": section_marida, "M": section_triage,
    "N": section_holdout, "R": section_revision, "S": section_review, "X": section_cross,
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
