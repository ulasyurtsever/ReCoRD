#!/usr/bin/env python
"""Stage 3b: one-time image-embedding caching (GPU stage).

Computes a frozen embedding per image for the requested datasets and stores
one consolidated archive per (embedding model, dataset). Embeddings feed the
importance-weight estimators of the shift-aware calibration tier and the
embedding-choice ablation.

Resumable at dataset granularity: existing archives with a complete id set
are skipped.

Example
-------
    python scripts/03b_precompute_embeddings.py \
        --embedding dinov2_vitb14 \
        --datasets cityscapes_val acdc_fog_train acdc_fog_val
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

from record.cache import atomic_savez, embeddings_path, is_complete, write_manifest
from record.gt import list_ids, load_image_array
from record.models import EmbeddingModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", required=True,
                        help="embedding key from configs/models.yaml")
    parser.add_argument("--datasets", nargs="+", required=True, help="dataset keys")
    parser.add_argument("--smoke", type=int, default=0,
                        help="process only the first N images per dataset")
    parser.add_argument("--device", default=None, help="override device (cuda/mps/cpu)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = EmbeddingModel(args.embedding, device=args.device)
    print(f"embedding={args.embedding} device={model.device}")

    for dataset_key in args.datasets:
        ids = list_ids(dataset_key)
        if args.smoke:
            ids = ids[: args.smoke]

        out_path = embeddings_path(args.embedding, dataset_key)
        if is_complete(out_path):
            try:
                existing = np.load(out_path, allow_pickle=False)
                if list(existing["image_ids"]) == ids:
                    print(f"{dataset_key}: archive complete, skipping")
                    continue
            except (OSError, ValueError, KeyError):
                pass  # corrupt or partial archive: recompute below

        vectors = np.stack([
            model.embed(load_image_array(dataset_key, image_id))
            for image_id in tqdm(ids, desc=dataset_key, unit="img")
        ])
        atomic_savez(out_path, image_ids=np.array(ids), embeddings=vectors.astype(np.float32))
        write_manifest(
            out_path.parent,
            {
                "embedding_key": args.embedding,
                "dataset_key": dataset_key,
                "n_images": len(ids),
                "dim": int(vectors.shape[1]),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "smoke": args.smoke,
            },
        )
        print(f"{dataset_key}: wrote {vectors.shape} embeddings")

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
