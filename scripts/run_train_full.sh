#!/usr/bin/env bash
# LoveDA fine-tuning and the caching/table pipeline for the LoveDA axis.
#
# MARIDA is not built here. Every MARIDA model is tied to the axis it is held
# out from, so its training, caching and tables live in one place,
# scripts/run_marida_holdout.sh, which this script does not call.
#
# Steps:
#   1. LoveDA SegFormer-B2 fine-tune, one model per transfer direction
#   2. Split definitions (LoveDA and the MARIDA held-out schemes)
#   3. Posterior caching for the LoveDA Val domains
#   4. Image embeddings for the LoveDA shift axis (tier B)
#   5. Region statistic tables for every new cache
#
# Resumable: training scripts overwrite their checkpoints; caching and table
# stages skip completed work.
#
# Usage:
#   nohup bash scripts/run_train_full.sh > train_full.log 2>&1 &
#   tail -f train_full.log

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Step 1: LoveDA fine-tuning ==="
python scripts/06_train_loveda.py --domain urban
python scripts/06_train_loveda.py --domain rural

echo "=== Step 2: split definitions ==="
# Deterministic, and regenerated rather than assumed: the schemes depend on
# the canonical patch-id normalization applied here.
python scripts/02_generate_splits.py
python scripts/02b_generate_marida_splits.py

echo "=== Step 3: posterior caching ==="
LOVEDA_SETS=(loveda_Val_Urban loveda_Val_Rural)
for MODEL in segformer_b2_loveda_urban segformer_b2_loveda_rural; do
  python scripts/03_precompute_cache.py --model "$MODEL" --datasets "${LOVEDA_SETS[@]}"
done

echo "=== Step 4: LoveDA embeddings (tier B) ==="
for EMB in dinov2_vitb14 clip_vitb16; do
  python scripts/03b_precompute_embeddings.py --embedding "$EMB" --datasets "${LOVEDA_SETS[@]}"
done

echo "=== Step 5: region statistic tables ==="
for MODEL in segformer_b2_loveda_urban segformer_b2_loveda_rural; do
  python scripts/04_build_region_tables.py --model "$MODEL" --datasets "${LOVEDA_SETS[@]}"
done

echo "=== LOVEDA TRAINING PIPELINE DONE ==="
echo "MARIDA models, caches and tables: bash scripts/run_marida_holdout.sh"
