"""Deterministic generation of calibration/test splits.

Every experiment references a committed split file; nothing is re-sampled at
experiment time. Determinism is guaranteed by seeding a PCG64 generator with
``(base_seed, scheme_key, seed_index)``, where ``scheme_key`` is derived from
the scheme name, so schemes are mutually independent and individually
reproducible.

Split files are JSON, one file per scheme, containing all seed replicates:

.. code-block:: json

    {
      "schema_version": 1,
      "scheme": "cityscapes_val_half",
      "base_seed": 20260806,
      "n_items": 500,
      "seeds": {"0": {"calibration": [...], "test": [...]}, ...}
    }
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

SCHEMA_VERSION = 1


def _generator(base_seed: int, scheme: str, seed_index: int) -> np.random.Generator:
    """Return an RNG keyed by base seed, scheme name, and seed index."""
    scheme_key = int.from_bytes(hashlib.sha256(scheme.encode()).digest()[:4], "big")
    return np.random.default_rng([base_seed, scheme_key, seed_index])


def _partition(ids: Sequence[str], rng: np.random.Generator, n_first: int) -> tuple[list[str], list[str]]:
    """Randomly partition ``ids`` into disjoint groups of size ``n_first`` and the rest."""
    ids = sorted(ids)
    if not 0 < n_first < len(ids):
        raise ValueError(f"n_first={n_first} out of range for {len(ids)} items")
    perm = rng.permutation(len(ids))
    first = sorted(ids[i] for i in perm[:n_first])
    rest = sorted(ids[i] for i in perm[n_first:])
    return first, rest


def make_two_way_scheme(
    scheme: str,
    ids: Sequence[str],
    n_cal: int,
    base_seed: int,
    n_seeds: int,
    cal_key: str = "calibration",
    test_key: str = "test",
) -> dict:
    """Build a scheme with ``n_seeds`` independent (calibration, test) partitions."""
    seeds = {}
    for k in range(n_seeds):
        cal, test = _partition(ids, _generator(base_seed, scheme, k), n_cal)
        seeds[str(k)] = {cal_key: cal, test_key: test}
    return {
        "schema_version": SCHEMA_VERSION,
        "scheme": scheme,
        "base_seed": base_seed,
        "n_items": len(ids),
        "n_calibration": n_cal,
        "seeds": seeds,
    }


def scene_disjoint_partition(
    pool: Sequence[str],
    scene_of: dict,
    fractions: dict[str, float],
    base_seed: int,
    scheme: str,
) -> dict[str, list[str]]:
    """Partition ``pool`` into named groups whose acquisition scenes are disjoint.

    The unit of assignment is the scene, not the patch, so no scene straddles
    two groups. On MARIDA this is what keeps the model-fitting groups genuinely
    separate from the calibration group: patches from one scene are neither
    independent of nor exchangeable with patches from another, so a patch-level
    partition would leave near-duplicates on both sides.

    ``fractions`` gives the target share of patches per group and must sum to
    one. Scenes are shuffled with a generator keyed by ``(base_seed, scheme)``
    and assigned greedily to whichever group is furthest below its target, so
    the result is deterministic and reproducible from the committed seed.
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {sum(fractions.values())}")
    by_scene: dict = {}
    for pid in sorted(pool):
        by_scene.setdefault(scene_of[pid], []).append(pid)
    scenes = sorted(by_scene)
    if len(scenes) < len(fractions):
        raise ValueError(
            f"{scheme}: {len(scenes)} scene(s) cannot fill {len(fractions)} groups")
    rng = _generator(base_seed, scheme, 0)
    order = [scenes[i] for i in rng.permutation(len(scenes))]
    total = len(pool)
    groups: dict[str, list[str]] = {k: [] for k in fractions}
    for scene in order:
        deficit = {k: fractions[k] * total - len(groups[k]) for k in fractions}
        target = max(sorted(deficit), key=lambda k: deficit[k])
        groups[target].extend(by_scene[scene])
    empty = [k for k, v in groups.items() if not v]
    if empty:
        raise ValueError(f"{scheme}: group(s) {empty} received no patches")
    return {k: sorted(v) for k, v in groups.items()}


def make_fixed_scheme(scheme: str, groups: dict[str, Sequence[str]]) -> dict:
    """Build a non-randomized scheme from fixed identifier groups.

    Used for official splits (e.g. MARIDA) so that all experiments consume
    splits through a single interface.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "scheme": scheme,
        "base_seed": None,
        "n_items": sum(len(v) for v in groups.values()),
        "seeds": {"0": {k: sorted(v) for k, v in groups.items()}},
    }


def write_scheme(scheme_dict: dict, out_dir: Path) -> Path:
    """Write a scheme to ``<out_dir>/<scheme>.json`` and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scheme_dict['scheme']}.json"
    with open(path, "w") as f:
        json.dump(scheme_dict, f, indent=1)
    return path


def load_scheme(path: Path) -> dict:
    """Load a scheme file, validating its schema version."""
    with open(path) as f:
        scheme = json.load(f)
    if scheme.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema version {scheme.get('schema_version')}")
    return scheme
