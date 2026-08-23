#!/usr/bin/env python
"""Pixel-level marine-debris F1 on the official MARIDA test split.

Reads the cached argmax predictions (stage 3) and the ground-truth class
masks, and reports precision, recall, and F1 for the marine-debris class
over all annotated pixels of the official test scenes. This places the
underlying segmentation models on the same scale as published pixel-level
MARIDA baselines; it involves no calibration and no thresholds.

Unlabeled pixels (class 0) are excluded from both the prediction and the
reference, matching the standard MARIDA evaluation protocol.

Outputs ``results/experiments/marida_test_f1__<model>.json`` per model and
prints a summary line.

Example
-------
    python scripts/10_marida_test_f1.py
    python scripts/10_marida_test_f1.py --models marida_unet_official_holdout
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from record.cache import image_cache_path, is_complete
from record.gt import list_ids, load_label_array
from record.labelmaps import critical_classes_for
from record.paths import results_dir

DATASET_KEY = "marida_test"
DEFAULT_MODELS = ["marida_unet_official_holdout",
                  "marida_unet_official_holdout_ens5"]


def binary_counts(pred: np.ndarray, ref: np.ndarray,
                  valid: np.ndarray) -> tuple[int, int, int]:
    """True/false positive and false negative pixel counts on a valid mask."""
    pred = pred & valid
    ref = ref & valid
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, and F1 from pixel counts (0.0 when undefined)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return precision, recall, f1


def evaluate_model(model_key: str, image_ids: list[str],
                   gt_value: int, channel: int) -> dict:
    """Aggregate debris pixel counts over the test split for one model."""
    tp = fp = fn = support = labeled = 0
    missing = []
    for image_id in image_ids:
        path = image_cache_path(model_key, DATASET_KEY, image_id)
        if not is_complete(path):
            missing.append(image_id)
            continue
        with np.load(path) as arrays:
            argmax = arrays["argmax"]
        gt = load_label_array(DATASET_KEY, image_id)
        valid = gt > 0
        t, p, n = binary_counts(argmax == channel, gt == gt_value, valid)
        tp += t
        fp += p
        fn += n
        support += int(np.count_nonzero((gt == gt_value) & valid))
        labeled += int(np.count_nonzero(valid))
    if missing:
        raise FileNotFoundError(
            f"{model_key}/{DATASET_KEY}: {len(missing)} cached predictions "
            f"missing (first: {missing[0]}); run stage 3 for this model first.")
    precision, recall, f1 = prf(tp, fp, fn)
    return {
        "model_key": model_key,
        "dataset_key": DATASET_KEY,
        "class_name": "marine_debris",
        "gt_value": gt_value,
        "model_channel": channel,
        "n_images": len(image_ids),
        "n_labeled_pixels": labeled,
        "n_debris_pixels": support,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="model keys with stage-3 caches on marida_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_value, channel = critical_classes_for("marida")["marine_debris"]
    image_ids = list_ids(DATASET_KEY)
    out_dir = results_dir("experiments")
    for model_key in args.models:
        payload = evaluate_model(model_key, image_ids, gt_value, channel)
        out_path = out_dir / f"marida_test_f1__{model_key}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"{model_key}: debris F1={payload['f1']:.3f} "
              f"(precision={payload['precision']:.3f}, "
              f"recall={payload['recall']:.3f}, "
              f"support={payload['n_debris_pixels']} px, "
              f"{payload['n_images']} images) -> {out_path}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
