#!/usr/bin/env python
"""Stage 2b: MARIDA held-out tile and season schemes from patch metadata.

Patch identifiers encode the acquisition date and MGRS tile; the tile is the
region proxy for leave-region-out evaluation and the acquisition month defines
the season axis.

Each scheme partitions the labeled pool (official train+val+test union) into
four disjoint groups, so that a held-out axis is genuinely held out:

``test``
    every patch of the held-out tile or season;
``train``
    patches used for gradient updates;
``monitor``
    patches used only to select the training checkpoint;
``calibration``
    patches used only for conformal calibration, seen by neither the optimizer
    nor the checkpoint selection.

The three fitting/calibration groups are drawn from the remaining regions at
*scene* granularity: no acquisition scene straddles two groups. Patches from
one scene are neither independent of nor exchangeable with patches from
another, so a patch-level partition would leave near-duplicates on both sides
of the boundary.

The official-split scheme follows the same discipline: the monitor set is
carved out of the official train split, so the official validation split is
free to serve as the calibration set without having informed the checkpoint.

Consuming scripts: ``07_train_marida_unet.py --scheme`` reads ``train`` and
``monitor``; ``05_run_experiments.py --cal-split calibration --test-split
test`` reads the other two.

Outputs (committed to ``splits/``):

- ``marida_meta.csv``: patch id, official split, tile, day/month/year, scene
- ``marida_holdout_region_<tile>.json``: one per region with at least
  ``--min-region-patches`` patches
- ``marida_holdout_season_<name>.json``: one per season
- ``marida_official_holdout.json``: the official split under the same discipline

Tile-to-region and season groupings should be cross-checked against the MARIDA
datasheet before the corresponding results are reported.
"""

from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

from record.datasets import load_config, marida_official_split
from record.marida import parse_patch_id, verify_class_presence
from record.paths import splits_dir
from record.splits import make_fixed_scheme, scene_disjoint_partition, write_scheme

SEASONS = {
    "winter": (12, 1, 2), "spring": (3, 4, 5),
    "summer": (6, 7, 8), "autumn": (9, 10, 11),
}

# Share of the remaining-region pool given to each group. The binding constraint
# is the calibration share: the smallest controllable level is 1/(n_c+1) over the
# calibration patches that contain debris, and only 373 of 1381 patches do. Over
# the nine held-out schemes this split leaves 26-158 debris-bearing calibration
# patches (smallest controllable level 0.037), so alpha=0.05 stays attainable
# everywhere, while the training pool (492-950 patches) matches or exceeds the
# 694 of the official train split in most schemes. A 0.15 calibration share would
# drop the worst scheme to 18 debris patches, a smallest controllable level of
# 0.053, putting alpha=0.05 out of reach on that axis.
FRACTIONS = {"train": 0.70, "monitor": 0.10, "calibration": 0.20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-region-patches", type=int, default=50)
    parser.add_argument("--min-test-patches", type=int, default=20)
    return parser.parse_args()


def _summarise(name: str, groups: dict[str, list[str]], scene_of: dict) -> None:
    parts = " ".join(
        f"{k}={len(v)}/{len({scene_of[p] for p in v})}sc" for k, v in sorted(groups.items()))
    print(f"  {name:38s} {parts}")


def _assert_disjoint(name: str, groups: dict[str, list[str]], scene_of: dict) -> None:
    """Fail loudly if any patch or any scene appears in two groups."""
    keys = sorted(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = set(groups[a]) & set(groups[b])
            if shared:
                raise AssertionError(f"{name}: {a} and {b} share {len(shared)} patch(es)")
            scenes = {scene_of[p] for p in groups[a]} & {scene_of[p] for p in groups[b]}
            if scenes:
                raise AssertionError(f"{name}: {a} and {b} share {len(scenes)} scene(s)")


def main() -> int:
    args = parse_args()
    cfg = load_config()
    base_seed = cfg["splits"]["base_seed"]

    rows = []
    for split in ("train", "val", "test"):
        for patch_id in marida_official_split(split, cfg):
            meta = parse_patch_id(patch_id)
            rows.append({"patch_id": patch_id, "official_split": split,
                         "tile": meta.tile, "day": meta.day,
                         "month": meta.month, "year": meta.year,
                         "scene": f"{meta.tile}_{meta.year}-{meta.month}-{meta.day}"})
    frame = pd.DataFrame(rows)

    # Validate the canonical class-id convention against the released data on
    # a deterministic subsample spanning tiles and dates.
    sample = frame.patch_id.tolist()[::25]
    for patch_id in sample:
        verify_class_presence(patch_id)
    print(f"class-presence check: {len(sample)} patches consistent")

    out_dir = splits_dir()
    frame.to_csv(out_dir / "marida_meta.csv", index=False)
    scene_of = dict(zip(frame.patch_id, frame.scene))
    print(f"metadata: {len(frame)} patches, {frame.tile.nunique()} tiles, "
          f"{frame.scene.nunique()} scenes, months {sorted(frame.month.unique())}")

    all_ids = frame.patch_id.tolist()
    print("\nheld-out schemes (group=patches/scenes):")

    n_schemes = 0
    for tile, count in sorted(Counter(frame.tile).items()):
        if count < args.min_region_patches:
            continue
        held_out = frame[frame.tile == tile].patch_id.tolist()
        pool = [p for p in all_ids if p not in set(held_out)]
        name = f"marida_holdout_region_{tile}"
        groups = scene_disjoint_partition(pool, scene_of, FRACTIONS, base_seed, name)
        groups["test"] = sorted(held_out)
        _assert_disjoint(name, groups, scene_of)
        write_scheme(make_fixed_scheme(name, groups), out_dir)
        _summarise(name, groups, scene_of)
        n_schemes += 1
    if n_schemes == 0:
        print(f"WARNING: no tile reaches {args.min_region_patches} patches; "
              "lower --min-region-patches")

    for season, months in SEASONS.items():
        held_out = frame[frame.month.isin(months)].patch_id.tolist()
        if len(held_out) < args.min_test_patches:
            print(f"  marida_holdout_season_{season}: skipped (test={len(held_out)})")
            continue
        pool = [p for p in all_ids if p not in set(held_out)]
        name = f"marida_holdout_season_{season}"
        groups = scene_disjoint_partition(pool, scene_of, FRACTIONS, base_seed, name)
        groups["test"] = sorted(held_out)
        _assert_disjoint(name, groups, scene_of)
        write_scheme(make_fixed_scheme(name, groups), out_dir)
        _summarise(name, groups, scene_of)

    # Official split under the same discipline: the checkpoint is selected on a
    # monitor set carved out of the official train split, which frees the
    # official validation split to act as the calibration set.
    name = "marida_official_holdout"
    train_pool = frame[frame.official_split == "train"].patch_id.tolist()
    groups = scene_disjoint_partition(
        train_pool, scene_of, {"train": 0.85, "monitor": 0.15}, base_seed, name)
    groups["calibration"] = sorted(frame[frame.official_split == "val"].patch_id)
    groups["test"] = sorted(frame[frame.official_split == "test"].patch_id)
    _assert_disjoint(name, groups, scene_of)
    write_scheme(make_fixed_scheme(name, groups), out_dir)
    _summarise(name, groups, scene_of)

    print(f"\nwrote schemes to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
