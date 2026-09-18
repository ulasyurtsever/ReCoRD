#!/usr/bin/env python
"""Stage 31: tier-A target-calibration schemes for the held-out MARIDA spring.

WHY. The experiments section reports tier A (recalibration on a few labeled target images)
on the weather and urban--rural axes only, while the one MARIDA partition whose
breakdown is separable from its own uncertainty, held-out spring, had no
remedy measured on it. This stage draws the
labeled target set from the spring test group of ``marida_holdout_season_spring``
(134 patches, 12 scenes, 46 of the patches carrying debris) and leaves the rest
of that group as the test side, in the stage-2 two-way format
(``target_calibration`` / ``test`` per seed), so stage 5 consumes it with
``--cal-split target_calibration --test-split test`` on the spring model.

Two variants, both over 100 seeded draws:

- ``marida_spring_targetcal<N>``: patches drawn uniformly from the spring
  group, the counterpart of the published ACDC tier-A schemes;
- ``marida_spring_targetcal<N>_scenedisjoint``: whole acquisition scenes are
  consumed until they hold at least N patches and the calibration list is
  truncated to N; no test patch comes from a scene that supplied a
  calibration patch (the counterpart of stage 28 on ACDC). With twelve scenes
  the calibration side rests on one to three scenes, and its debris count is
  what decides feasibility, so the draw-to-draw spread is the headline.

The scheme name enters the generator key, so these draws are independent of
every other scheme. The spring model itself never saw any spring patch
(stage 2b), so the target labels are the only spring information it receives.

Example
-------
    python scripts/31_marida_target_schemes.py --sizes 25 50
    python scripts/05_run_experiments.py \
        --name h4_tierA25__spring__marida_unet_holdout_season_spring \
        --model marida_unet_holdout_season_spring \
        --scheme marida_spring_targetcal25 \
        --cal-split target_calibration --test-split test \
        --cal-datasets marida_train marida_val marida_test \
        --test-datasets marida_train marida_val marida_test \
        --methods region_crc heuristic argmax
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from record.paths import splits_dir
from record.splits import load_scheme, make_two_way_scheme, write_scheme

SOURCE_SCHEME = "marida_holdout_season_spring"
BASE_SEED = 20260806


def _generator(base_seed: int, scheme: str, seed_index: int) -> np.random.Generator:
    scheme_key = int.from_bytes(hashlib.sha256(scheme.encode()).digest()[:4], "big")
    return np.random.default_rng([base_seed, scheme_key, seed_index])


def scene_disjoint_split(pool, scene_of, n_cal, base_seed, scheme, seed_index):
    """Consume whole scenes until >= n_cal patches, truncate to n_cal, and
    keep every patch of every unconsumed scene as the test side."""
    by_scene: dict[str, list[str]] = {}
    for pid in sorted(pool):
        by_scene.setdefault(scene_of[pid], []).append(pid)
    scenes = sorted(by_scene)
    rng = _generator(base_seed, scheme, seed_index)
    order = [scenes[i] for i in rng.permutation(len(scenes))]
    consumed, cal_pool = [], []
    for scene in order:
        if len(cal_pool) >= n_cal:
            break
        consumed.append(scene)
        cal_pool.extend(by_scene[scene])
    if len(consumed) == len(scenes):
        raise ValueError(f"{scheme} seed {seed_index}: every scene consumed, no test side")
    cal_pool = [cal_pool[i] for i in rng.permutation(len(cal_pool))]
    cal = sorted(cal_pool[:n_cal])
    test = sorted(p for s in scenes if s not in consumed for p in by_scene[s])
    return cal, test, consumed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[25, 50])
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    args = ap.parse_args()

    src = load_scheme(splits_dir() / f"{SOURCE_SCHEME}.json")
    pool = sorted(src["seeds"]["0"]["test"])
    meta = pd.read_csv(splits_dir() / "marida_meta.csv").set_index("patch_id")
    scene_of = meta.loc[pool, "scene"].to_dict()
    n_scenes = len(set(scene_of.values()))
    print(f"spring pool: {len(pool)} patches over {n_scenes} scenes")

    for n in args.sizes:
        name = f"marida_spring_targetcal{n}"
        scheme = make_two_way_scheme(name, pool, n, args.base_seed, args.n_seeds,
                                     cal_key="target_calibration", test_key="test")
        scheme["source_scheme"] = SOURCE_SCHEME
        path = write_scheme(scheme, splits_dir())
        print(f"wrote {path}: {args.n_seeds} seeds, {n} calibration / {len(pool) - n} test")

        name_sd = f"{name}_scenedisjoint"
        seeds, n_cal_scenes, n_test = {}, [], []
        for k in range(args.n_seeds):
            cal, test, consumed = scene_disjoint_split(pool, scene_of, n, args.base_seed, name_sd, k)
            seeds[str(k)] = {"target_calibration": cal, "test": test}
            n_cal_scenes.append(len(consumed)); n_test.append(len(test))
        scheme_sd = {"schema_version": 1, "scheme": name_sd, "base_seed": args.base_seed,
                     "n_items": len(pool), "n_calibration": n, "source_scheme": SOURCE_SCHEME,
                     "seeds": seeds}
        path_sd = write_scheme(scheme_sd, splits_dir())
        diag = {"scheme": name_sd, "n_seeds": args.n_seeds, "n_calibration": n,
                "calibration_scenes": {"min": int(min(n_cal_scenes)), "median": float(np.median(n_cal_scenes)),
                                       "max": int(max(n_cal_scenes))},
                "test_patches": {"min": int(min(n_test)), "median": float(np.median(n_test)),
                                 "max": int(max(n_test))},
                "pool_patches": len(pool), "pool_scenes": n_scenes,
                "generated_utc": datetime.now(timezone.utc).isoformat()}
        (splits_dir() / f"{name_sd}.meta.json").write_text(json.dumps(diag, indent=1))
        print(f"wrote {path_sd}: calibration scenes {diag['calibration_scenes']}, "
              f"test patches {diag['test_patches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
