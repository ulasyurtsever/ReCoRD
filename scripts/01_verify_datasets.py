#!/usr/bin/env python
"""Stage 1: verify dataset layout and inventories.

Enumerates every dataset under the data root, compares file counts against
the expected values in ``configs/datasets.yaml``, writes the outcome to
``results/data_verification.csv``, and exits non-zero on any mismatch.
"""

from __future__ import annotations

import pandas as pd

from record.datasets import verify_all
from record.paths import data_root, results_dir


def main() -> int:
    print(f"data root: {data_root()}")
    checks = verify_all()

    frame = pd.DataFrame(
        {
            "dataset": c.dataset,
            "item": c.item,
            "expected": c.expected,
            "found": c.found,
            "status": "PASS" if c.passed else "FAIL",
        }
        for c in checks
    )
    out = results_dir() / "data_verification.csv"
    frame.to_csv(out, index=False)

    print(frame.to_string(index=False))
    print(f"\ncsv written: {out}")

    failed = frame[frame["status"] == "FAIL"]
    if not failed.empty:
        print(f"\nRESULT: FAIL ({len(failed)} mismatching checks)")
        return 1
    print(f"\nRESULT: PASS ({len(frame)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
