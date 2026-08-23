"""Path resolution for data, configuration, and outputs.

The repository is self-contained: every path is resolved relative to the
repository root, except the data root, which defaults to ``../../00_datasets``
and can be overridden with the ``RECORD_DATA_ROOT`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """Return the repository root directory."""
    return _REPO_ROOT


def data_root() -> Path:
    """Return the dataset root directory.

    Resolution order:
    1. ``RECORD_DATA_ROOT`` environment variable, if set.
    2. ``../../00_datasets`` relative to the repository root.
    """
    env = os.environ.get("RECORD_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (_REPO_ROOT.parent.parent / "00_datasets").resolve()


def config_path(name: str = "datasets.yaml") -> Path:
    """Return the path of a configuration file under ``configs/``."""
    return _REPO_ROOT / "configs" / name


def splits_dir() -> Path:
    """Return the directory holding committed split definitions.

    Overridable with ``RECORD_SPLITS_ROOT`` (used by integration tests).
    """
    env = os.environ.get("RECORD_SPLITS_ROOT")
    path = Path(env).expanduser().resolve() if env else _REPO_ROOT / "splits"
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(subdir: str | None = None) -> Path:
    """Return (and create) the results directory or one of its subdirectories.

    Overridable with ``RECORD_RESULTS_ROOT`` (used by integration tests).
    """
    env = os.environ.get("RECORD_RESULTS_ROOT")
    path = Path(env).expanduser().resolve() if env else _REPO_ROOT / "results"
    if subdir:
        path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path
