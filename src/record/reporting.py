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
    # The two LAC variants are labelled apart because they answer different
    # questions: the marginal one sets a single threshold from the all-class
    # pixel pool, the class-conditional one gets the same class information
    # the method under test uses. See run_lac in scripts/05_run_experiments.py.
    "lac_global": "LAC (marginal)",
    "lac_classcond": "LAC (class-conditional)",
    "heuristic": "Uncorrected threshold",
    "argmax": "Argmax",
}


# CSVs that stages other than 05 write into results/experiments. They share
# the directory and the extension with the stage-5 tables but not the schema,
# so a wildcard load must not sweep them in: one of them (x10) even carries
# alpha and rho columns, which is enough to survive the cast below and quietly
# contribute rows to a table that means something else.
NON_STAGE5_PREFIXES = (
    "x7_union_area",                # stage 22: one row per image, no method
    "x8_tierb_pool",                # stage 23: one row per candidate pool
    "x10_tierb_test_charge",        # stage 25
    "x11_marida_confidence",        # stage 26
    "x12_triage_permutation_band",  # stage 27
)
NON_STAGE5_SUFFIXES = ("_triage",)  # stage-5 triage sidecars, a different row type

# The columns every stage-5 experiment table has. A CSV in the directory that
# lacks them is not a stage-5 table, whatever it is named.
STAGE5_COLUMNS = frozenset({
    "method", "alpha", "rho", "seed", "class_name", "model",
    "region_fnr", "marked_area_fraction",
})


def is_stage5_experiment(name: str) -> bool:
    """True if ``name`` (a CSV stem) is a stage-5 experiment table."""
    return not (name.startswith(NON_STAGE5_PREFIXES)
                or name.endswith(NON_STAGE5_SUFFIXES))


def load_experiments(pattern: str = "*") -> pd.DataFrame:
    """Load stage-5 experiment CSVs into one frame with an ``experiment`` column.

    Files written by the supplementary stages are skipped when ``pattern`` is
    a wildcard that happened to catch them, and rejected loudly when they were
    asked for by name. A CSV that is neither a known non-stage-5 output nor a
    valid stage-5 table raises: a new stage that starts writing into this
    directory has to be registered above rather than silently join the tables.
    """
    exp_dir = results_dir("experiments")
    paths = sorted(glob.glob(str(exp_dir / f"{pattern}.csv")))
    if not paths:
        raise FileNotFoundError(f"no experiment CSVs matching '{pattern}' under {exp_dir}")
    frames, skipped = [], []
    for path in paths:
        name = os.path.basename(path)[:-4]
        if not is_stage5_experiment(name):
            skipped.append(name)
            continue
        frame = pd.read_csv(path)
        missing = STAGE5_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path} is in results/experiments but is not a stage-5 "
                f"experiment table (missing columns {sorted(missing)}). If a "
                "new stage writes it, add its prefix to "
                "record.reporting.NON_STAGE5_PREFIXES.")
        frame["experiment"] = name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"no stage-5 experiment CSVs matching '{pattern}' under {exp_dir}"
            + (f" (skipped non-stage-5: {', '.join(skipped)})" if skipped else ""))
    out = pd.concat(frames, ignore_index=True)
    out["alpha"] = out["alpha"].astype(float)
    out["rho"] = out["rho"].astype(float)
    return out


def parse_axis_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive block, domain/condition, and sweep fields from experiment names."""
    frame = frame.copy()
    name = frame["experiment"]
    # Two digits, not one: the supplementary blocks reached x10 and beyond,
    # and a one-digit pattern silently parsed x10/x11/x12 as no block at all.
    frame["block"] = name.str.extract(r"^([a-z]\d{1,2}[a-z]?)_")[0]
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
