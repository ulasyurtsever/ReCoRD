#!/usr/bin/env python
"""Stage 0: verify the Python environment and compute backend.

Checks interpreter version, required packages, and PyTorch backend
availability. Writes a plain-text report to ``results/environment_report.txt``
and exits non-zero if any required component is missing.
"""

from __future__ import annotations

import importlib
import platform
import sys
from datetime import datetime, timezone

REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "sklearn",
    "skimage",
    "torch",
    "torchvision",
    "transformers",
    "timm",
    "open_clip",
    "rasterio",
    "matplotlib",
    "yaml",
    "tqdm",
]


def main() -> int:
    lines: list[str] = []
    ok = True

    lines.append(f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"platform: {platform.platform()} ({platform.machine()})")
    lines.append(f"python: {sys.version.split()[0]}")

    if sys.version_info < (3, 11):
        lines.append("ERROR: Python >= 3.11 required")
        ok = False

    for name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "unknown")
            lines.append(f"package {name}: {version}")
        except ImportError as exc:
            lines.append(f"ERROR: package {name} not importable ({exc})")
            ok = False

    try:
        import torch

        backend = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        lines.append(f"torch backend: {backend}")
        if backend == "cpu":
            lines.append("WARNING: no GPU backend available; inference stages will be slow")
    except ImportError:
        pass

    try:
        from record.paths import data_root, repo_root

        lines.append(f"repo_root: {repo_root()}")
        lines.append(f"data_root: {data_root()} (exists: {data_root().is_dir()})")
    except ImportError:
        lines.append("ERROR: package 'record' not installed (run: pip install -e .)")
        ok = False

    report = "\n".join(lines)
    print(report)

    try:
        from record.paths import results_dir

        out = results_dir() / "environment_report.txt"
        out.write_text(report + "\n")
        print(f"\nreport written: {out}")
    except ImportError:
        pass

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
