#!/usr/bin/env python
"""Stage 2: generate all seeded calibration/test splits.

Produces one JSON file per scheme under ``splits/`` (committed to version
control) and a manifest summarizing every scheme. Schemes:

- ``cityscapes_val_half``: Cityscapes val (500) into 250 calibration / 250
  in-distribution test, per seed.
- ``loveda_urban_val_half`` / ``loveda_rural_val_half``: LoveDA val halves for
  in-distribution calibration and testing in each transfer direction.
- ``acdc_<condition>_targetcal<n>``: for each ACDC condition, ``n`` labeled
  target-calibration images drawn from the condition's train+val pool, with
  the remainder as shifted test (n in {25, 50, 100}).
- ``loveda_<domain>_targetcal<n>``: analogous target-calibration draws from
  each LoveDA val domain for the shifted direction.
- ``marida_official``: the released MARIDA train/val/test lists, wrapped in
  the common split format. Region/season re-splits are produced by a later
  stage after patch metadata extraction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from record.datasets import acdc_ids, cityscapes_ids, load_config, loveda_ids, marida_official_split
from record.paths import splits_dir
from record.splits import make_fixed_scheme, make_two_way_scheme, write_scheme


def main() -> int:
    cfg = load_config()
    sp = cfg["splits"]
    base_seed: int = sp["base_seed"]
    n_seeds: int = sp["n_seeds"]
    target_sizes: list[int] = sp["target_cal_sizes"]
    out_dir = splits_dir()
    manifest: list[dict] = []

    def emit(scheme_dict: dict) -> None:
        path = write_scheme(scheme_dict, out_dir)
        n_seed_entries = len(scheme_dict["seeds"])
        manifest.append(
            {
                "scheme": scheme_dict["scheme"],
                "file": path.name,
                "n_items": scheme_dict["n_items"],
                "n_seeds": n_seed_entries,
            }
        )
        print(f"wrote {path.name}: {scheme_dict['n_items']} items, {n_seed_entries} seeds")

    # Cityscapes: in-distribution calibration/test halves.
    cs_val = cityscapes_ids("val", cfg)
    emit(make_two_way_scheme(
        "cityscapes_val_half", cs_val, sp["cityscapes_val_cal_size"], base_seed, n_seeds))

    # LoveDA: in-distribution halves per domain.
    for domain in ("Urban", "Rural"):
        ids = loveda_ids("Val", domain, cfg)
        emit(make_two_way_scheme(
            f"loveda_{domain.lower()}_val_half", ids,
            int(len(ids) * sp["loveda_val_cal_fraction"]), base_seed, n_seeds))

    # ACDC: target-calibration draws per condition from the labeled pool.
    for cond in cfg["datasets"]["acdc"]["conditions"]:
        pool = acdc_ids(cond, "train", cfg) + acdc_ids(cond, "val", cfg)
        for n in target_sizes:
            emit(make_two_way_scheme(
                f"acdc_{cond}_targetcal{n}", pool, n, base_seed, n_seeds,
                cal_key="target_calibration"))

    # LoveDA: target-calibration draws per domain (shifted direction).
    for domain in ("Urban", "Rural"):
        ids = loveda_ids("Val", domain, cfg)
        for n in target_sizes:
            emit(make_two_way_scheme(
                f"loveda_{domain.lower()}_targetcal{n}", ids, n, base_seed, n_seeds,
                cal_key="target_calibration"))

    # MARIDA: official released splits in the common format.
    marida_groups = {name: marida_official_split(name, cfg)
                     for name in ("train", "val", "test")}
    emit(make_fixed_scheme("marida_official", marida_groups))

    # MARIDA: in-distribution positive control. The official split is
    # scene-disjoint, so the exchangeability the guarantee needs does not hold
    # there and the theorem is not expected to bind. Without a random
    # patch-level split there is no in-distribution check on the only remote
    # sensing benchmark in the study -- and on a random split the guarantee
    # must hold, so a violation here would mean the implementation is wrong,
    # not that the benchmark is hard.
    marida_all = sorted({pid for ids in marida_groups.values() for pid in ids})
    emit(make_two_way_scheme(
        "marida_random_half", marida_all, len(marida_all) // 2,
        base_seed, n_seeds))

    manifest_path = out_dir / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "base_seed": base_seed,
                "n_seeds": n_seeds,
                "schemes": manifest,
            },
            f,
            indent=1,
        )
    print(f"\nmanifest written: {manifest_path} ({len(manifest)} schemes)")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
