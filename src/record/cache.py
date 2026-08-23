"""Inference cache layout and atomic I/O.

Cache layout (under ``<repo>/cache/`` by default, override with
``RECORD_CACHE_ROOT``):

.. code-block:: text

    cache/<model_key>/<dataset_key>/probs/<image_id>.npz
    cache/<embedding_key>/<dataset_key>/embeddings.npz

Each per-image archive contains:

- ``critical_probs``: float16, shape ``(n_critical, H, W)``
- ``argmax``: uint8, shape ``(H, W)``, model output channel indices
- ``strided_probs``: float16, shape ``(n_classes, H//s, W//s)`` full class
  posterior at stride ``s`` (retained for post-hoc baselines such as
  temperature scaling)
- ``critical_channels``: int16, model output channels of the critical classes

Writes are atomic (temp file + rename) so interrupted runs can resume by
skipping existing files.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from record.paths import repo_root


def cache_root() -> Path:
    env = os.environ.get("RECORD_CACHE_ROOT")
    root = Path(env).expanduser().resolve() if env else repo_root() / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def probs_dir(model_key: str, dataset_key: str) -> Path:
    path = cache_root() / model_key / dataset_key / "probs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_cache_path(model_key: str, dataset_key: str, image_id: str) -> Path:
    safe_id = image_id.replace("/", "__")
    return probs_dir(model_key, dataset_key) / f"{safe_id}.npz"


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    """Write an ``.npz`` atomically.

    The archive is written through an open file handle: passing a *path*
    to ``np.savez_compressed`` is unsafe here because NumPy appends an
    ``.npz`` extension to names lacking it, which would divorce the written
    data from the temporary name being renamed.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npz")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def is_complete(path: Path) -> bool:
    """Return True if a cache file exists and is non-empty.

    Zero-byte files are treated as missing so that interrupted or faulty
    writes are recomputed rather than silently skipped by resumable stages.
    """
    return path.exists() and path.stat().st_size > 0


def write_manifest(directory: Path, payload: dict) -> Path:
    path = directory / "MANIFEST.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    return path


def embeddings_path(embedding_key: str, dataset_key: str) -> Path:
    path = cache_root() / embedding_key / dataset_key
    path.mkdir(parents=True, exist_ok=True)
    return path / "embeddings.npz"
