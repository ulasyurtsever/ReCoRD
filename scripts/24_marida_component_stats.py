#!/usr/bin/env python
"""Stage 24: descriptive statistics of the MARIDA ground-truth components.

Two numbers quoted in the MARIDA section are properties of the benchmark rather than
of any calibration run: the size distribution of the debris components and the
number of MARIDA patches that carry at least one of them. Both are read from
the cached component tables that stage 4 writes under ``results/raw``. Those
tables are rebuilt from the public imagery by the released pipeline but are not
themselves redistributed, and opening them needs a parquet engine that a bare
checkout may lack.

This stage reduces them once to a small JSON summary that is committed with the
repository, so the two numbers and the audit checks that guard them can be
verified without the raw tables and without a parquet engine.

Writes ``results/experiments/x9_marida_component_stats.json``.

Example
-------
    python scripts/24_marida_component_stats.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from record.paths import results_dir

SPLITS = ("marida_train", "marida_val", "marida_test")


def pooled_components(model_key: str) -> pd.DataFrame:
    """Every ground-truth debris component of the pooled MARIDA splits."""
    frames = []
    for split in SPLITS:
        path = results_dir("raw") / model_key / split / "components.parquet"
        if not path.exists():
            raise FileNotFoundError(f"component table missing: {path}")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def summarize(comp: pd.DataFrame, model_key: str) -> dict:
    size = comp["size_px"]
    return {
        "model_key": model_key,
        "splits": list(SPLITS),
        "n_components": int(len(size)),
        "n_debris_patches": int(comp["image_id"].nunique()),
        "size_px_median": float(size.median()),
        "size_px_mean": float(size.mean()),
        "size_px_min": int(size.min()),
        "size_px_max": int(size.max()),
        "frac_size_le_3": float((size <= 3).mean()),
        "frac_size_le_2": float((size <= 2).mean()),
        "frac_size_eq_1": float((size == 1).mean()),
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="marida_unet_official_holdout",
                        help="model key whose cached component tables are read")
    args = parser.parse_args()

    summary = summarize(pooled_components(args.model), args.model)
    out = results_dir("experiments") / "x9_marida_component_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    print(f"{summary['n_components']} components over "
          f"{summary['n_debris_patches']} debris-bearing patches; "
          f"median {summary['size_px_median']:.0f} px, "
          f"{summary['frac_size_le_3']:.3f} at 3 px or smaller")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
