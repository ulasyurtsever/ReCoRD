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
import os
import re

import numpy as np
import pandas as pd

from record.paths import results_dir


def paths_root():
    return results_dir('tables').parent.parent

TOL = 5e-4          # a printed three-decimal value must agree to half a unit
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

    # 76 / 77: false positives, and the "smaller area" claim
    fp_b2 = mean(sel(e1, method="region_crc", model="segformer_b2_cityscapes",
                     alpha=0.20, rho=0.5), "fp_components_per_image")
    fp_b5 = mean(sel(e1, method="region_crc", model="segformer_b5_cityscapes",
                     alpha=0.20, rho=0.5), "fp_components_per_image")
    near(76, "false-positive components per image, B2", 0.7, fp_b2, 0.05)
    near(76, "false-positive components per image, B5", 5.6, fp_b5, 0.05)
    a_b2 = mean(sel(e1, method="region_crc", model="segformer_b2_cityscapes",
                    alpha=0.20, rho=0.5), "marked_area_fraction")
    a_b5 = mean(sel(e1, method="region_crc", model="segformer_b5_cityscapes",
                    alpha=0.20, rho=0.5), "marked_area_fraction")
    truth(77, "text: B5's LARGER mask (9.0% against 3.4%) is the more fragmented",
          a_b5 > a_b2 and abs(a_b5 - 0.090) < 5e-4 and abs(a_b2 - 0.034) < 5e-4,
          f"B5 area={a_b5:.3f} vs B2 area={a_b2:.3f} at alpha=0.2, rho=0.5")

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

    mon = sel(r, rho=0.5)
    mon = mon[mon.model.isin(SEGF)]
    fl = mon.groupby("experiment")["monitor_flag"].mean()
    rng_claim(132, "monitor fires in 38-63% of the region-CRC (draw, class, "
                   "level) configurations at rho=0.5, per model x condition",
              0.38, 0.63, fl.values, 0.006)
    ind = sel(e1, method="region_crc", rho=0.5)
    ind = ind[ind.model.isin(SEGF)]
    fa = ind.groupby("model")["monitor_flag"].mean()
    rng_claim(132, "in-distribution false-alarm rate 0.2-1.0%", 0.002, 0.010,
              fa.values, 0.0006)

    by_cond = {}
    for c_ in CONDS:
        sub = sel(r, rho=0.5)
        sub = sub[sub.model.isin(SEGF) & sub.experiment.str.contains(f"__{c_}__")]
        cellsc = cellmean(sel(sub, alpha=0.20), ["model"])
        by_cond[c_] = (float(cellsc.mean()), float(cellsc.max()),
                       float(sub.monitor_flag.mean()))
    worst_mean = max(by_cond, key=lambda k: by_cond[k][0])
    worst_cell = max(by_cond, key=lambda k: by_cond[k][1])
    least_flag = min(by_cond, key=lambda k: by_cond[k][2])
    truth(133, "night is the LEAST FLAGGED condition",
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
def section_tier_a():
    head("I  Tier A recalibration")
    a = load("e3_tierA*")
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

    ar = [mean(sel(r, n_target=n, alpha=0.20, rho=0.5), "marked_area_fraction")
          for n in (25, 50, 100)]
    rng_claim(142, "tier-A marked area 62-66% at alpha=0.2", 0.623, 0.662, ar)
    truth(142, "area does not decrease from n_t=25 to 100",
          not (ar[0] > ar[1] > ar[2]), str([round(x, 3) for x in ar]))

    inf25 = 1 - sel(r, n_target=25, alpha=0.05).feasible.astype(bool).mean()
    inf100 = 1 - sel(r, n_target=100, alpha=0.05).feasible.astype(bool).mean()
    truth(145, "alpha=0.05 unattainable in >99% of n_t=25 draws",
          inf25 > 0.99, f"{inf25:.4f}")
    near(145, "infeasible fraction at n_t=100, alpha=0.05 is 0.47", 0.47, inf100, 0.006)

    table6 = {25: [(0.005, 0.940, 0.99), (0.022, 0.807, 0.74), (0.075, 0.648, 0.38)],
              50: [(0.004, 0.944, 0.80), (0.022, 0.818, 0.45), (0.085, 0.623, 0.19)],
              100: [(0.004, 0.949, 0.47), (0.018, 0.835, 0.27), (0.072, 0.662, 0.03)]}
    for n, rows in table6.items():
        for al, (f_, ar_, inf_) in zip(ALPHAS, rows):
            d = sel(r, n_target=n, alpha=al, rho=0.5)
            near(149, f"tier A n_t={n} a={al} FNR", f_, mean(d))
            near(149, f"tier A n_t={n} a={al} area", ar_,
                 mean(d, "marked_area_fraction"))
            got = 1 - d.feasible.astype(bool).mean()
            if n == 25 and al == 0.05:
                truth(149, f"tier A n_t=25 a=0.05 infeasible >0.99", got > 0.99, f"{got:.4f}")
            else:
                near(149, f"tier A n_t={n} a={al} infeasible", inf_, got, 0.006)


# --------------------------------------------------------------------------
# J. tier B
# --------------------------------------------------------------------------
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
            print(f"[ERROR] section {k}: {type(exc).__name__}: {exc}")
    print(f"\n=== SUMMARY: {N_OK} ok, {N_FAIL} FAIL, {N_NOTE} notes ===")
    return 1 if N_FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
