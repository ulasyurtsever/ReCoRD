"""Shared loading and aggregation for stage-6 tables and figures.

Consumes the stage-5 experiment CSVs under ``results/experiments/`` and
exposes tidy frames keyed by experiment block. All downstream tables and
figures are deterministic arithmetic over these frames.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from record.paths import results_dir

MODEL_LABELS = {
    "segformer_b2_cityscapes": "SegFormer-B2",
    "segformer_b5_cityscapes": "SegFormer-B5",
    "mask2former_swinb_cityscapes": "Mask2Former",
    "segformer_b2_cityscapes_mcdrop8": "SegFormer-B2 (MC-dropout)",
    "segformer_b2_loveda_urban": "SegFormer-B2 (Urban)",
    "segformer_b2_loveda_rural": "SegFormer-B2 (Rural)",
    # One MARIDA model per held-out axis. Only the official axis carries both
    # a single model and a deep ensemble, so only those two need labels.
    "marida_unet_official_holdout": "U-Net (single)",
    "marida_unet_official_holdout_ens5": "U-Net (ensemble-5)",
}

METHOD_LABELS = {
    "region_crc": "Region CRC (ours)",
    "weighted_crc": "Weighted CRC",
    "pixel_crc": "Pixel CRC",
    "heuristic": "Uncorrected threshold",
    "argmax": "Argmax",
}


def load_experiments(pattern: str = "*") -> pd.DataFrame:
    """Load experiment CSVs into one frame with an ``experiment`` column."""
    exp_dir = results_dir("experiments")
    paths = sorted(glob.glob(str(exp_dir / f"{pattern}.csv")))
    if not paths:
        raise FileNotFoundError(f"no experiment CSVs matching '{pattern}' under {exp_dir}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["experiment"] = os.path.basename(path)[:-4]
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["alpha"] = out["alpha"].astype(float)
    out["rho"] = out["rho"].astype(float)
    return out


def parse_axis_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive block, domain/condition, and sweep fields from experiment names."""
    frame = frame.copy()
    name = frame["experiment"]
    frame["block"] = name.str.extract(r"^([a-z]\d[a-z]?)_")[0]
    frame["condition"] = name.str.extract(
        r"__(fog|night|rain|snow|urban2rural|rural2urban|urban|rural)(?:__|$)")[0]
    frame["n_target"] = pd.to_numeric(
        name.str.extract(r"tierA(\d+)")[0], errors="coerce")
    frame["clip_max"] = pd.to_numeric(
        name.str.extract(r"clip(\d+)__")[0], errors="coerce")
    frame["embedding"] = name.str.extract(r"__(dinov2_vitb14|clip_vitb16)$")[0]
    frame["region"] = name.str.extract(r"h2_region__(\w+?)__")[0]
    frame["season"] = name.str.extract(r"h3_season__(\w+?)__")[0]
    return frame


def load_triage(pattern: str = "*_triage") -> pd.DataFrame:
    """Load triage sidecar CSVs into one frame with an ``experiment`` column."""
    exp_dir = results_dir("experiments")
    paths = sorted(glob.glob(str(exp_dir / f"{pattern}.csv")))
    if not paths:
        raise FileNotFoundError(f"no triage CSVs matching '{pattern}' under {exp_dir}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["experiment"] = os.path.basename(path)[: -len("_triage.csv")]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cell_means(frame: pd.DataFrame, by: list[str],
               feasible_only: bool = True) -> pd.DataFrame:
    """Mean metrics per cell, averaging over seeds (and unlisted axes)."""
    sub = frame
    if feasible_only and "feasible" in frame.columns:
        sub = frame[frame["feasible"].astype(bool) | (frame["method"] == "argmax")]
    agg = sub.groupby(by, dropna=False).agg(
        region_fnr=("region_fnr", "mean"),
        marked_area=("marked_area_fraction", "mean"),
        lam=("lam", "mean"),
        fp_per_image=("fp_components_per_image", "mean"),
        monitor_flag=("monitor_flag", "mean"),
        weight_ess=("weight_ess", "mean"),
        weight_p_test=("weight_p_test", "mean"),
        n_rows=("region_fnr", "size"),
    ).reset_index()
    return agg


def infeasibility(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Fraction of infeasible and undefined (empty-calibration) draws."""
    sub = frame[frame["method"] != "argmax"]
    return sub.groupby(by, dropna=False).apply(
        lambda x: pd.Series({
            "infeasible_rate": 1.0 - x["feasible"].astype(bool).mean(),
            "undefined_rate": x["lam"].isna().mean(),
        }), include_groups=False).reset_index()


def fmt(value: float, digits: int = 3) -> str:
    """Format a number for LaTeX tables; NaN renders as an em-dash."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def latex_table(body: str, caption: str, label: str, colspec: str,
                header: str, star: bool = False, size: str = "footnotesize",
                colsep: str = "3pt") -> str:
    """Wrap a tabular body in a booktabs table environment.

    ``star=True`` emits ``table*`` for tables wider than one column. ``size``
    and ``colsep`` control the font size and inter-column padding; the
    defaults keep single-column tables inside the IEEE column width.
    """
    env = "table*" if star else "table"
    return "\n".join([
        rf"\begin{{{env}}}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\{size}",
        rf"\setlength{{\tabcolsep}}{{{colsep}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        body,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\end{{{env}}}",
        "",
    ])


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
