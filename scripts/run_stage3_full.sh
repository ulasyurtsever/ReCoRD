#!/usr/bin/env bash
# Full stage-3/4 pass over the Cityscapes->ACDC axis.
#
# Caches posteriors for three model families, image embeddings for two
# encoders, and the MC-dropout (T=8) variant of SegFormer-B2, then reduces
# every cache to region-level statistic tables. Every step is resumable:
# rerunning this script skips completed work.
#
# Usage:
#   nohup bash scripts/run_stage3_full.sh > full_run.log 2>&1 &
#   tail -f full_run.log

set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS=(cityscapes_val
  acdc_fog_train acdc_fog_val
  acdc_night_train acdc_night_val
  acdc_rain_train acdc_rain_val
  acdc_snow_train acdc_snow_val)

MODELS=(segformer_b2_cityscapes segformer_b5_cityscapes mask2former_swinb_cityscapes)

echo "=== Stage 3: posterior caching ==="
for MODEL in "${MODELS[@]}"; do
  python scripts/03_precompute_cache.py --model "$MODEL" --datasets "${DATASETS[@]}"
done

echo "=== Stage 3: MC-dropout variant (SegFormer-B2, T=8) ==="
python scripts/03_precompute_cache.py --model segformer_b2_cityscapes \
  --datasets "${DATASETS[@]}" --mc-dropout 8

echo "=== Stage 3b: image embeddings ==="
for EMB in dinov2_vitb14 clip_vitb16; do
  python scripts/03b_precompute_embeddings.py --embedding "$EMB" --datasets "${DATASETS[@]}"
done

echo "=== Stage 4: region statistic tables ==="
for KEY in "${MODELS[@]}" segformer_b2_cityscapes_mcdrop8; do
  python scripts/04_build_region_tables.py --model "$KEY" --datasets "${DATASETS[@]}"
done

echo "=== ALL DONE ==="
