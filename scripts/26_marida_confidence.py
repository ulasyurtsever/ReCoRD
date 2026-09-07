#!/usr/bin/env python
"""Stage 26: MARIDA annotation confidence behind the region-level guarantee.

MARIDA ships an annotation-confidence layer next to every class mask, and the
pipeline has never read it: :func:`record.marida._patch_index` skips any file
whose stem ends in ``_cl`` or ``_conf``, so ``*_conf.tif`` is excluded from the
patch index by construction and no stage downstream ever opens it. Every number
the paper reports on MARIDA therefore treats a Low-confidence annotation and a
High-confidence one as the same ground truth.

That is defensible for a risk statement -- the guarantee is made against the
released labels, whatever their provenance -- but it is not something the paper
can leave implicit. Its selling point is an auditable commitment, and a
commitment is only auditable if the reader can see what it was made against. If
a large share of the debris components the guarantee is spent on are annotations
MARIDA itself flags as uncertain, and if those are systematically the smallest
components, then the residual risk after calibration sits precisely where the
label is weakest. This stage measures that instead of asserting it.

Where the confidence lives, and what it is a property of
-------------------------------------------------------
It is a property of the **annotation polygon**, distributed as a **per-pixel
raster**. The released shapefiles (``shapefiles/S2_<date>_<tile>.dbf``) carry
one ``conf`` attribute per digitized polygon, alongside ``id`` (the class) and
``report``; ``patches/**/<patch>_conf.tif`` is the rasterization of that single
attribute onto the polygon's pixels. So the layer has per-pixel *granularity*
but only per-polygon *information*: within one polygon every pixel necessarily
carries the same code, and a pixel is annotated at exactly one confidence.

Consequently a ground-truth component is single-confidence unless it merges
pixels from two or more polygons. That happens: components are extracted with
8-connectivity from the class mask (:mod:`record.components`), which fuses
touching or diagonally adjacent polygons of the same class into one component,
and the fused polygons may have been digitized at different confidence.

What is done with mixed components
----------------------------------
Nothing is split or reassigned. A component is the unit the loss is defined
over, so it stays one unit here too. Each component is reported with the raw
counts of its High / Moderate / Low pixels, and is classified as
``all_high`` / ``all_moderate`` / ``all_low`` / ``mixed`` -- ``mixed`` is its
own category, never folded into a majority vote, because a majority vote would
manufacture a certainty the annotation does not carry. The restricted miss rate
in part (3) is instead driven by the ``any_high`` flag (the component contains
at least one High-confidence pixel), which is the decision-relevant reading: a
mixed component has at least one pixel an annotator was willing to stand behind.
Both the ``any_high`` subset and its complement are reported, since the
complement -- components with no High-confidence pixel at all -- is where the
referee's concern actually lives.

Encoding assumption
-------------------
The raster is float32 with values ``{0, 1, 2, 3}``: 0 exactly on the unlabeled
pixels of the class mask, and a nonzero code on every labeled pixel. The
code-to-name mapping follows the MARIDA reference implementation
(``conf_mapping`` in marine-debris/marine-debris.github.io, ``utils/assets.py``;
Kikaki et al., 2022): 1 = High, 2 = Moderate, 3 = Low. The dataset ships no
id-to-name file for it, exactly as for the class ids, so the mapping is fixed
here rather than discovered; ``--confidence-codes`` overrides it if the
reference is ever read the other way round. The per-code pixel totals are
written to the JSON so a reversed mapping is visible rather than silent.

What it costs
-------------
Two small GeoTIFF reads per debris-bearing patch (the class mask and the
confidence mask, 256x256 each) and one connected-component labeling per patch,
i.e. seconds on a CPU. No model, no posterior cache, no GPU. The miss rates in
part (3) are re-tabulations of the stage-4 coverage curves at thresholds
recalibrated with the same CRC rule stage 5 uses; inference is never re-run.

Joining confidence onto the cached component rows
-------------------------------------------------
``components.parquet`` (stage 4) stores ``component_id``, the label id that
``scipy.ndimage.label`` assigned under the fixed 8-connectivity structure, so
re-extracting components from the same class mask with the same
``min_component_px`` reproduces the same ids. ``(image_id, component_id)`` is
therefore the join key, and ``size_px`` and the bounding box are re-checked on
every row so a silent misalignment is impossible. The coverage curves are joined
through ``curve_row`` (a row index into ``coverage_curves.npy``, offset per
dataset key exactly as stage 5's ``TableStore`` does), and components are mapped
to images through ``image_row`` / ``component_image_rows``.

Writes ``results/experiments/<stem>.json`` (committed; every number the
article would quote) and ``results/experiments/<stem>__components.csv`` (one
row per component, so the whole analysis is re-derivable without the
rasters), where ``<stem>`` is ``--out-stem``, by default
``x11_marida_confidence``.

Pass ``--out-stem`` whenever this stage is run for a second model. The stem
does not carry the model name on its own, so two runs left at the default
would write the same two files and the second would overwrite the first
without saying so.

Example
-------
    python scripts/26_marida_confidence.py
    python scripts/26_marida_confidence.py \
        --model marida_unet_official_holdout_ens5 \
        --experiment h1_official__marida_unet_official_holdout_ens5 \
        --out-stem x11_marida_confidence__ens5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from record.components import extract_components
from record.crc import crc_threshold
from record.evaluation import index_rows
from record.grid import LAMBDA_GRID
from record.labelmaps import critical_classes_for
from record.losses import capture_threshold, enforce_nonincreasing, image_loss_curves
from record.marida import load_class_mask, patch_paths
from record.paths import results_dir, splits_dir
from record.splits import load_scheme

SPLITS = ("marida_train", "marida_val", "marida_test")

# Confidence level names in code order (see "Encoding assumption" above).
CONFIDENCE_LEVELS = ("high", "moderate", "low")

# Composition categories a component can fall into. "mixed" is deliberately
# last and deliberately separate: it is not a level, it is the absence of one.
COMPOSITION_CLASSES = ("all_high", "all_moderate", "all_low", "mixed")


# --------------------------------------------------------------------------
# Confidence rasters
# --------------------------------------------------------------------------

def confidence_path(patch_id: str) -> Path:
    """Path of a patch's ``*_conf.tif``.

    Derived from the image path rather than searched for, because the patch
    index in :mod:`record.marida` filters ``_conf`` files out and so cannot be
    asked for one directly.
    """
    image_path, _ = patch_paths(patch_id)
    path = image_path.with_name(image_path.stem + "_conf" + image_path.suffix)
    if not path.exists():
        raise FileNotFoundError(f"confidence mask missing for {patch_id}: {path}")
    return path


def load_confidence_mask(patch_id: str) -> np.ndarray:
    """Load a patch's confidence mask as a 2-D integer array.

    The raster is stored as float32 holding small integers; it is cast only
    after checking that every value is integral, so a file that turns out to
    hold continuous scores fails here instead of being silently truncated.
    """
    import rasterio

    with rasterio.open(confidence_path(patch_id)) as src:
        conf = src.read(1).astype(np.float64)
    conf = np.nan_to_num(conf, nan=0.0)
    if not np.all(conf == np.rint(conf)):
        raise ValueError(
            f"{patch_id}: confidence raster holds non-integer values; the "
            "released layer is expected to be the coded levels {0, 1, 2, 3}")
    return np.rint(conf).astype(np.int32)


# --------------------------------------------------------------------------
# Cached stage-4 tables
# --------------------------------------------------------------------------

def load_pooled_tables(model_key: str, dataset_keys: tuple[str, ...]):
    """Concatenate the stage-4 tables of several dataset keys.

    Reproduces the offsetting that stage 5's ``TableStore`` performs, so that
    ``image_row`` indexes the pooled image-id list and ``curve_row`` indexes
    the stacked coverage-curve array.
    """
    frames, curve_blocks, image_ids = [], [], []
    row_offset = curve_offset = 0
    for key in dataset_keys:
        table_dir = results_dir("raw") / model_key / key
        comp_path = table_dir / "components.parquet"
        if not comp_path.exists():
            raise FileNotFoundError(f"component table missing: {comp_path}")
        frame = pd.read_parquet(comp_path)
        curves = np.load(table_dir / "coverage_curves.npy").astype(np.float32)
        stats = np.load(table_dir / "image_stats.npz", allow_pickle=False)

        frame["split"] = key
        frame["image_row"] += row_offset
        frame["curve_row"] += curve_offset
        frames.append(frame)
        curve_blocks.append(curves)
        image_ids.extend(str(i) for i in stats["image_ids"])
        row_offset = len(image_ids)
        curve_offset += curves.shape[0]

    return (pd.concat(frames, ignore_index=True),
            np.vstack(curve_blocks), image_ids)


def min_component_px(model_key: str, dataset_keys: tuple[str, ...]) -> int:
    """Minimum component size the cached tables were built with.

    Read from the stage-4 manifests rather than assumed, because components
    must be re-extracted under exactly the rule that produced the stored ids.
    """
    values = set()
    for key in dataset_keys:
        manifest = results_dir("raw") / model_key / key / "MANIFEST.json"
        if not manifest.exists():
            raise FileNotFoundError(f"stage-4 manifest missing: {manifest}")
        values.add(int(json.loads(manifest.read_text())["min_component_px"]))
    if len(values) != 1:
        raise ValueError(f"inconsistent min_component_px across splits: {values}")
    return values.pop()


# --------------------------------------------------------------------------
# Per-component confidence composition
# --------------------------------------------------------------------------

def component_confidence(patch_id: str, gt_value: int, min_size_px: int,
                         codes: dict[str, int]) -> dict[int, dict]:
    """Confidence composition of every debris component of one patch.

    Returns ``{component_id: {"n_high": .., "n_moderate": .., "n_low": ..,
    "size_px": .., "bbox": (..)}}``. Component ids are the ``ndimage.label``
    ids, which is what stage 4 stored.
    """
    mask = load_class_mask(patch_id)
    conf = load_confidence_mask(patch_id)
    if conf.shape != mask.shape:
        raise ValueError(f"{patch_id}: confidence raster {conf.shape} does not "
                         f"match class mask {mask.shape}")

    labels, components = extract_components(mask == gt_value, min_size_px=min_size_px)
    out: dict[int, dict] = {}
    for comp in components:
        r0, c0, r1, c1 = comp.bbox
        # Same windowed indexing as record.coverage: restrict to the bounding
        # box, then to the pixels actually carrying this component's label.
        inside = labels[r0:r1, c0:c1] == comp.component_id
        values = conf[r0:r1, c0:c1][inside]
        if values.min() < 1 or values.max() > 3:
            raise ValueError(
                f"{patch_id} component {comp.component_id}: confidence codes "
                f"{sorted(np.unique(values).tolist())} outside {{1, 2, 3}}; a "
                "labeled pixel with code 0 means the confidence raster and the "
                "class mask disagree about which pixels are annotated")
        counts = np.bincount(values, minlength=4)
        out[comp.component_id] = {
            "size_px": comp.size_px,
            "bbox": comp.bbox,
            **{f"n_{name}": int(counts[codes[name]]) for name in CONFIDENCE_LEVELS},
        }
    return out


def classify(n_high: int, n_moderate: int, n_low: int) -> str:
    """Composition category of a component from its per-level pixel counts."""
    present = [name for name, n in zip(CONFIDENCE_LEVELS, (n_high, n_moderate, n_low)) if n > 0]
    if len(present) == 1:
        return f"all_{present[0]}"
    return "mixed"


def attach_confidence(comp: pd.DataFrame, gt_value: int, min_size_px: int,
                      codes: dict[str, int]) -> pd.DataFrame:
    """Add the per-component confidence columns to the cached component rows.

    Every joined row is re-checked against the cached ``size_px`` and bounding
    box: the join key is a label id, and a label id is only meaningful if the
    same mask and the same connectivity produced it.
    """
    records = []
    for patch_id in tqdm(sorted(comp["image_id"].unique()),
                         desc="confidence", unit="patch"):
        per_component = component_confidence(patch_id, gt_value, min_size_px, codes)
        for cid, info in per_component.items():
            records.append({"image_id": patch_id, "component_id": cid, **info})
    conf_frame = pd.DataFrame(records)

    merged = comp.merge(conf_frame, on=["image_id", "component_id"],
                        how="left", validate="one_to_one", suffixes=("", "_recomputed"))
    orphan = merged["size_px_recomputed"].isna()
    if orphan.any():
        raise ValueError(
            f"{int(orphan.sum())} cached component rows have no re-extracted "
            "counterpart; the class masks or the connectivity have changed "
            "since stage 4 built the tables")
    bad_size = merged["size_px"] != merged["size_px_recomputed"]
    if bad_size.any():
        example = merged.loc[bad_size].iloc[0]
        raise ValueError(
            f"size mismatch on {int(bad_size.sum())} components, e.g. "
            f"{example['image_id']}#{example['component_id']}: cached "
            f"{example['size_px']} px vs re-extracted "
            f"{example['size_px_recomputed']} px")
    bbox = np.array(merged["bbox"].tolist())
    cached_bbox = merged[["bbox_r0", "bbox_c0", "bbox_r1", "bbox_c1"]].to_numpy()
    if not np.array_equal(bbox, cached_bbox):
        raise ValueError("bounding boxes disagree between cached and re-extracted "
                         "components; the component ids are not aligned")

    merged = merged.drop(columns=["size_px_recomputed", "bbox"])
    counts = merged[[f"n_{name}" for name in CONFIDENCE_LEVELS]].to_numpy()
    if not np.array_equal(counts.sum(axis=1), merged["size_px"].to_numpy()):
        raise ValueError("per-level pixel counts do not sum to the component size")
    merged["confidence_class"] = [classify(*row) for row in counts]
    merged["any_high"] = merged["n_high"] > 0
    return merged


# --------------------------------------------------------------------------
# Size distributions
# --------------------------------------------------------------------------

def size_summary(size: pd.Series, n_total: int) -> dict:
    """Size distribution of one subset of components, in pixels."""
    if len(size) == 0:
        return {"n": 0, "frac_of_components": 0.0}
    return {
        "n": int(len(size)),
        "frac_of_components": float(len(size) / n_total),
        "size_px_median": float(size.median()),
        "size_px_mean": float(size.mean()),
        "size_px_q25": float(size.quantile(0.25)),
        "size_px_q75": float(size.quantile(0.75)),
        "size_px_min": int(size.min()),
        "size_px_max": int(size.max()),
        "frac_size_eq_1": float((size == 1).mean()),
        "frac_size_le_3": float((size <= 3).mean()),
    }


def size_comparison(frame: pd.DataFrame) -> dict:
    """Test the 'the uncertain annotations are the smallest' claim.

    Reported as a rank test rather than a mean difference: component sizes are
    heavily right-skewed (a handful of large slicks against a mass of one- and
    two-pixel detections), so a t-test would be answering a question about the
    tail rather than about the typical component.
    """
    from scipy import stats

    any_high = frame.loc[frame["any_high"], "size_px"]
    no_high = frame.loc[~frame["any_high"], "size_px"]
    out = {
        "n_any_high": int(len(any_high)),
        "n_no_high": int(len(no_high)),
        "median_any_high": float(any_high.median()) if len(any_high) else float("nan"),
        "median_no_high": float(no_high.median()) if len(no_high) else float("nan"),
    }
    if len(any_high) and len(no_high):
        result = stats.mannwhitneyu(any_high, no_high, alternative="two-sided")
        # Probability that a randomly drawn any-High component is larger than a
        # randomly drawn no-High one; 0.5 means the claim has no support.
        out["mannwhitney_u"] = float(result.statistic)
        out["mannwhitney_p"] = float(result.pvalue)
        out["prob_any_high_larger"] = float(result.statistic / (len(any_high) * len(no_high)))
    return out


# --------------------------------------------------------------------------
# Restricted region miss rate on the official split
# --------------------------------------------------------------------------

def image_averaged_miss(miss: np.ndarray, image_rows: np.ndarray) -> float:
    """Mean over images of the within-image missed-component fraction.

    This is the estimand stage 5 controls (Eq. 3): images are weighted equally,
    not components. Images contributing no component to the subset are dropped
    rather than counted as zero loss, which is the same convention the loss
    curves use for images without components of the class.
    """
    if miss.size == 0:
        return float("nan")
    per_image = [miss[image_rows == r].mean() for r in np.unique(image_rows)]
    return float(np.mean(per_image))


def official_split_rates(sel: pd.DataFrame, curves: np.ndarray, image_ids: list[str],
                         scheme: dict, cal_key: str, test_key: str,
                         alphas: list[float], rhos: list[float]) -> tuple[list[dict], dict]:
    """Region miss rates at the paper's operating points, restricted and not.

    The threshold is *not* re-chosen to suit the restricted subset: it is the
    threshold the deployed system commits to, selected by CRC on the calibration
    group over all components exactly as stage 5 does. Only the set of test
    components the realized miss rate is measured over changes. Re-calibrating
    on the restricted subset would answer a different, easier question.
    """
    entry = scheme["seeds"]["0"]
    for key in (cal_key, test_key):
        if key not in entry:
            raise KeyError(f"scheme entry has keys {sorted(entry)}; requested '{key}'")
    cal_rows = index_rows(image_ids, entry[cal_key])
    test_rows = index_rows(image_ids, entry[test_key])

    comp_curves = curves[sel["curve_row"].to_numpy()]
    comp_image_rows = sel["image_row"].to_numpy()
    any_high = sel["any_high"].to_numpy()
    in_test = np.isin(comp_image_rows, test_rows)

    rows: list[dict] = []
    per_component_flags: dict[tuple[float, float], np.ndarray] = {}
    for rho in rhos:
        raw_losses, _ = image_loss_curves(comp_curves, comp_image_rows,
                                          len(image_ids), rho=rho)
        # Monotone by construction; the envelope is the identity here and is
        # applied only to match stage 5's calibration input exactly.
        losses = enforce_nonincreasing(raw_losses)
        cal_defined = cal_rows[~np.isnan(losses[cal_rows][:, 0])]

        for alpha in alphas:
            selection = crc_threshold(losses[cal_defined], alpha, LAMBDA_GRID)
            col = selection.lam_index
            miss = comp_curves[in_test][:, col] < capture_threshold(rho)
            rows_of_comp = comp_image_rows[in_test]
            high = any_high[in_test]

            flags = np.full(len(sel), np.nan)
            flags[in_test] = miss.astype(float)
            per_component_flags[(alpha, rho)] = flags

            rows.append({
                "alpha": float(alpha),
                "rho": float(rho),
                "lam": float(selection.lam),
                "lam_index": int(col),
                "feasible": bool(selection.feasible),
                "calibration_risk": float(selection.calibration_risk),
                "n_cal_images": int(len(cal_defined)),
                # Unrestricted: every test component, as the paper reports it.
                "n_test_components": int(miss.size),
                "n_missed": int(miss.sum()),
                "miss_rate_component_avg": float(miss.mean()) if miss.size else float("nan"),
                "miss_rate_image_avg": image_averaged_miss(miss, rows_of_comp),
                # Restricted to components carrying at least one High pixel.
                "n_test_components_any_high": int(high.sum()),
                "n_missed_any_high": int(miss[high].sum()),
                "miss_rate_component_avg_any_high":
                    float(miss[high].mean()) if high.any() else float("nan"),
                "miss_rate_image_avg_any_high":
                    image_averaged_miss(miss[high], rows_of_comp[high]),
                # The complement: where the residual risk actually sits.
                "n_test_components_no_high": int((~high).sum()),
                "n_missed_no_high": int(miss[~high].sum()),
                "miss_rate_component_avg_no_high":
                    float(miss[~high].mean()) if (~high).any() else float("nan"),
                "miss_rate_image_avg_no_high":
                    image_averaged_miss(miss[~high], rows_of_comp[~high]),
            })
    return rows, per_component_flags


def crosscheck_thresholds(path: Path, rows: list[dict]) -> dict:
    """Compare the recomputed thresholds with the committed experiment CSV.

    The thresholds above are recomputed from the cached curves rather than read
    off stage 5's output, so this check is what ties them to the numbers the
    paper actually prints. A disagreement means this stage and the reported
    experiment are not describing the same operating point.
    """
    if not path.exists():
        return {"experiment_csv": str(path), "available": False}
    frame = pd.read_csv(path)
    frame = frame[(frame["method"] == "region_crc") & (frame["seed"].astype(str) == "0")]
    mismatches, unmatched, n_matched = [], [], 0
    for row in rows:
        match = frame[np.isclose(frame["alpha"], row["alpha"])
                      & np.isclose(frame["rho"], row["rho"])]
        if match.empty:
            # Not a benign skip: an operating point the paper reports has no
            # counterpart in the committed CSV, so it is unchecked. Record it
            # rather than quietly shrinking the comparison.
            unmatched.append({"alpha": row["alpha"], "rho": row["rho"]})
            continue
        n_matched += 1
        reported = int(match["lam_index"].iloc[0])
        if reported != row["lam_index"]:
            mismatches.append({"alpha": row["alpha"], "rho": row["rho"],
                               "reported_lam_index": reported,
                               "recomputed_lam_index": row["lam_index"]})
    return {"experiment_csv": str(path), "available": True,
            "n_requested": int(len(rows)),
            "n_compared": int(n_matched),
            "unmatched": unmatched,
            "mismatches": mismatches}


# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="marida_unet_official_holdout",
                        help="model key whose cached component tables are read")
    parser.add_argument("--scheme", default="marida_official_holdout",
                        help="split scheme providing the official calibration "
                             "and test groups")
    parser.add_argument("--cal-split", default="calibration",
                        help="scheme entry key providing calibration ids")
    parser.add_argument("--test-split", default="test",
                        help="scheme entry key providing test ids")
    parser.add_argument("--experiment", default=None,
                        help="experiment CSV to cross-check the recomputed "
                             "thresholds against (default: h1_official__<model>)")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20],
                        help="risk levels the paper reports")
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.1, 0.5],
                        help="capture levels the paper reports")
    parser.add_argument("--out-stem", default="x11_marida_confidence",
                        help="output file stem under results/experiments; the "
                             "default does not carry the model name, so a "
                             "second model must be given its own stem or it "
                             "overwrites the first run's two files")
    parser.add_argument("--confidence-codes", default="1,2,3",
                        help="raster codes for High,Moderate,Low in that order; "
                             "the default follows the MARIDA reference "
                             "implementation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    code_values = [int(v) for v in args.confidence_codes.split(",")]
    if sorted(code_values) != [1, 2, 3]:
        print(f"ERROR: --confidence-codes must be a permutation of 1,2,3; "
              f"got {args.confidence_codes}")
        return 1
    codes = dict(zip(CONFIDENCE_LEVELS, code_values))

    class_name, (gt_value, _) = next(iter(critical_classes_for("marida").items()))
    comp, curves, image_ids = load_pooled_tables(args.model, SPLITS)
    comp = comp[comp["class_name"] == class_name].reset_index(drop=True)
    if comp.empty:
        print(f"ERROR: no components of class '{class_name}' in the cached tables")
        return 1

    frame = attach_confidence(comp, gt_value, min_component_px(args.model, SPLITS), codes)

    n_total = len(frame)
    pixels = {name: int(frame[f"n_{name}"].sum()) for name in CONFIDENCE_LEVELS}
    composition = {
        key: size_summary(frame.loc[frame["confidence_class"] == key, "size_px"], n_total)
        for key in COMPOSITION_CLASSES
    }
    by_any_high = {
        "any_high": size_summary(frame.loc[frame["any_high"], "size_px"], n_total),
        "no_high": size_summary(frame.loc[~frame["any_high"], "size_px"], n_total),
    }

    scheme = load_scheme(splits_dir() / f"{args.scheme}.json")
    rates, flags = official_split_rates(
        frame, curves, image_ids, scheme, args.cal_split, args.test_split,
        args.alphas, args.rhos)
    experiment = args.experiment or f"h1_official__{args.model}"
    crosscheck = crosscheck_thresholds(
        results_dir("experiments") / f"{experiment}.csv", rates)

    # Per-component miss flags at each reported operating point, so the rates
    # above can be re-derived from the CSV alone.
    for (alpha, rho), values in flags.items():
        frame[f"missed__alpha{alpha:g}__rho{rho:g}"] = values

    # Which official group each patch belongs to, for the same reason.
    group_of = {pid: name for name, ids in scheme["seeds"]["0"].items() for pid in ids}
    frame["official_group"] = frame["image_id"].map(group_of)

    summary = {
        "model_key": args.model,
        "splits": list(SPLITS),
        "scheme": args.scheme,
        "cal_split": args.cal_split,
        "test_split": args.test_split,
        "class_name": class_name,
        "confidence_codes": codes,
        "confidence_source": "per-polygon 'conf' attribute of the MARIDA "
                             "shapefiles, distributed rasterized as *_conf.tif",
        "model": args.model,
        "scheme": args.scheme,
        "out_stem": args.out_stem,
        "n_components": n_total,
        "n_debris_patches": int(frame["image_id"].nunique()),
        "pixels_by_confidence": pixels,
        "n_debris_pixels": int(frame["size_px"].sum()),
        "composition_counts": {key: int((frame["confidence_class"] == key).sum())
                               for key in COMPOSITION_CLASSES},
        "composition_fractions": {key: float((frame["confidence_class"] == key).mean())
                                  for key in COMPOSITION_CLASSES},
        "frac_any_high": float(frame["any_high"].mean()),
        "size_by_composition": composition,
        "size_by_any_high": by_any_high,
        "size_comparison": size_comparison(frame),
        "official_split_operating_points": rates,
        "threshold_crosscheck": crosscheck,
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_dir = results_dir("experiments")
    json_path = out_dir / f"{args.out_stem}.json"
    csv_path = out_dir / f"{args.out_stem}__components.csv"
    json_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    frame.to_csv(csv_path, index=False)

    print(f"{n_total} debris components over {summary['n_debris_patches']} patches: "
          + ", ".join(f"{k} {summary['composition_counts'][k]}"
                      for k in COMPOSITION_CLASSES))
    print(f"any High-confidence pixel: {summary['frac_any_high']:.3f} of components; "
          f"median size {by_any_high['any_high'].get('size_px_median', float('nan')):.0f} px "
          f"vs {by_any_high['no_high'].get('size_px_median', float('nan')):.0f} px without")
    for row in rates:
        print(f"  alpha={row['alpha']:.2f} rho={row['rho']:.1f} "
              f"lam={row['lam']:.3f}: region miss "
              f"{row['miss_rate_component_avg']:.3f} overall, "
              f"{row['miss_rate_component_avg_any_high']:.3f} on any-High, "
              f"{row['miss_rate_component_avg_no_high']:.3f} on no-High "
              f"({row['n_test_components_no_high']}/{row['n_test_components']} components)")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")

    if not crosscheck.get("available"):
        print(f"\nRESULT: FAIL (no experiment CSV at "
              f"{crosscheck.get('experiment_csv')}; the recomputed thresholds "
              "were never tied to the reported ones)")
        return 1
    if crosscheck.get("unmatched"):
        for row in crosscheck["unmatched"]:
            print(f"UNMATCHED: alpha={row['alpha']} rho={row['rho']} has no "
                  f"row in {crosscheck['experiment_csv']}")
        print(f"\nRESULT: FAIL ({len(crosscheck['unmatched'])} of "
              f"{crosscheck['n_requested']} operating points unchecked)")
        return 1
    if crosscheck.get("mismatches"):
        for row in crosscheck["mismatches"]:
            print(f"MISMATCH: alpha={row['alpha']} rho={row['rho']}: "
                  f"reported lam_index {row['reported_lam_index']} vs "
                  f"recomputed {row['recomputed_lam_index']}")
        print(f"\nRESULT: FAIL ({len(crosscheck['mismatches'])} recomputed "
              f"thresholds disagree with {crosscheck['experiment_csv']}; this "
              "stage and the reported experiment are not describing the same "
              "operating point)")
        return 1
    print(f"\nRESULT: PASS ({crosscheck['n_compared']} operating points "
          f"agree with {crosscheck['experiment_csv']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
