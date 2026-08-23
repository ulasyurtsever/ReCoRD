#!/usr/bin/env python
"""Stage 3: one-time inference caching (GPU stage).

For every image of the requested datasets, runs the segmentation model once
and stores per-pixel statistics sufficient for all downstream analyses:
critical-class probabilities at full resolution, the argmax map, and the full
class posterior at a coarse stride (for post-hoc baselines).

Resumable: images with an existing cache file are skipped. Run with
``--smoke N`` first to validate checkpoints, memory, and output shapes on N
images per dataset before the full pass.

Examples
--------
Smoke run:
    python scripts/03_precompute_cache.py \
        --model segformer_b2_cityscapes --datasets cityscapes_val --smoke 10

Full pass over the Cityscapes->ACDC axis:
    python scripts/03_precompute_cache.py \
        --model segformer_b2_cityscapes \
        --datasets cityscapes_val acdc_fog_train acdc_fog_val \
                   acdc_night_train acdc_night_val acdc_rain_train \
                   acdc_rain_val acdc_snow_train acdc_snow_val
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from tqdm import tqdm

from record.cache import atomic_savez, image_cache_path, is_complete, probs_dir, write_manifest
from record.gt import list_ids, load_model_input, parse_dataset_key
from record.labelmaps import compatible_dataset_families, critical_classes_for
from record.models import (
    SegmentationModel,
    cache_arrays_for_posterior,
    load_model_registry,
    resolve_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model key from configs/models.yaml")
    parser.add_argument("--datasets", nargs="+", required=True, help="dataset keys")
    parser.add_argument("--smoke", type=int, default=0,
                        help="process only the first N images per dataset")
    parser.add_argument("--device", default=None, help="override device (cuda/mps/cpu)")
    parser.add_argument("--mc-dropout", type=int, default=0, metavar="T",
                        help="cache the mean posterior over T stochastic passes "
                             "(dropout + stochastic depth enabled); outputs are "
                             "stored under the cache key <model>_mcdrop<T>")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_model_registry()
    stride = registry["cache"]["strided_probs_stride"]
    spec = registry["segmentation_models"][args.model]

    if not resolve_checkpoint(spec):
        print(f"ERROR: checkpoint(s) not available for {args.model}: "
              f"{spec.get('members') or spec.get('checkpoint')}")
        print("Train the model or fix configs/models.yaml before running the full pass.")
        return 1

    model = SegmentationModel(args.model, device=args.device)
    cache_key = args.model
    if args.mc_dropout:
        counts = model.enable_mc_sampling()
        cache_key = f"{args.model}_mcdrop{args.mc_dropout}"
        print(f"MC sampling enabled: {counts} -> cache key {cache_key}")
    checkpoint_desc = spec.get("checkpoint") or f"ensemble[{len(spec['members'])} members]"
    print(f"model={args.model} checkpoint={checkpoint_desc} device={model.device}")

    for dataset_key in args.datasets:
        family, _ = parse_dataset_key(dataset_key)
        if family not in compatible_dataset_families(spec["family"]):
            print(f"ERROR: model family '{spec['family']}' cannot be evaluated on "
                  f"dataset '{dataset_key}' (family '{family}')")
            return 1
        channels = [ch for _, (_, ch) in critical_classes_for(family).items()]

        ids = list_ids(dataset_key)
        if args.smoke:
            ids = ids[: args.smoke]
        pending = [i for i in ids if not is_complete(image_cache_path(cache_key, dataset_key, i))]
        print(f"{dataset_key}: {len(ids)} images, {len(pending)} to compute")

        t0 = time.time()
        for image_id in tqdm(pending, desc=dataset_key, unit="img"):
            image = load_model_input(dataset_key, image_id)
            if args.mc_dropout:
                probs = model.posterior_mc_mean(image, args.mc_dropout)
            else:
                probs = model.posterior(image)
            arrays = cache_arrays_for_posterior(probs, channels, stride)
            atomic_savez(image_cache_path(cache_key, dataset_key, image_id), **arrays)

        elapsed = time.time() - t0
        write_manifest(
            probs_dir(cache_key, dataset_key).parent,
            {
                "model_key": args.model,
                "cache_key": cache_key,
                "mc_dropout_passes": args.mc_dropout or None,
                "checkpoint": spec.get("checkpoint") or spec.get("members"),
                "dataset_key": dataset_key,
                "n_images_total": len(ids),
                "n_images_computed_this_run": len(pending),
                "critical_channels": channels,
                "strided_probs_stride": stride,
                "device": model.device,
                "seconds_this_run": round(elapsed, 1),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "smoke": args.smoke,
            },
        )
        if pending:
            print(f"{dataset_key}: {elapsed / max(len(pending), 1):.2f} s/img")

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
