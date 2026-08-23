#!/usr/bin/env bash
# Full stage-5 experiment matrix over the Cityscapes->ACDC axis (CPU only).
#
# Blocks, per cache key (3 model families + the MC-dropout variant):
#   E1  in-distribution validity/efficiency, 100 seeds
#   E2  shift breakdown: source-calibrated thresholds evaluated on each ACDC
#       condition (the motivation experiment)
#   E3  tier A: exact recalibration from 25/50/100 labeled target images
#   E4  tier B: label-free importance-weighted calibration (DINOv2 and CLIP
#       embeddings; kNN-estimator ablation on SegFormer-B2 with DINOv2)
#
# Every invocation writes results/experiments/<name>.csv (+ .meta.json) and is
# independent: rerunning the script overwrites CSVs deterministically.
#
# Usage:
#   nohup bash scripts/run_stage5_full.sh > stage5_run.log 2>&1 &
#   tail -f stage5_run.log

set -euo pipefail
cd "$(dirname "$0")/.."

KEYS=(segformer_b2_cityscapes segformer_b5_cityscapes
      mask2former_swinb_cityscapes segformer_b2_cityscapes_mcdrop8)
CONDS=(fog night rain snow)

run() { echo ">>> $*"; python scripts/05_run_experiments.py "$@"; }

for KEY in "${KEYS[@]}"; do
  echo "===== ${KEY} ====="

  run --name "e1_indist__${KEY}" --model "$KEY" \
      --scheme cityscapes_val_half \
      --cal-datasets cityscapes_val --test-datasets cityscapes_val \
      --methods region_crc pixel_crc heuristic argmax

  for C in "${CONDS[@]}"; do
    POOL=("acdc_${C}_train" "acdc_${C}_val")

    run --name "e2_break__${C}__${KEY}" --model "$KEY" \
        --scheme cityscapes_val_half \
        --cal-datasets cityscapes_val --test-datasets "${POOL[@]}" \
        --methods region_crc pixel_crc heuristic argmax

    for N in 25 50 100; do
      run --name "e3_tierA${N}__${C}__${KEY}" --model "$KEY" \
          --scheme "acdc_${C}_targetcal${N}" \
          --cal-datasets "${POOL[@]}" --test-datasets "${POOL[@]}" \
          --methods region_crc heuristic argmax
    done

    for EMB in dinov2_vitb14 clip_vitb16; do
      run --name "e4_tierB__${C}__${KEY}__${EMB}" --model "$KEY" \
          --scheme cityscapes_val_half \
          --cal-datasets cityscapes_val --test-datasets "${POOL[@]}" \
          --methods weighted_crc --embedding "$EMB"
    done
  done
done

# Weight-estimator ablation: kNN density ratio, representative model only.
for C in "${CONDS[@]}"; do
  run --name "e4_tierB_knn__${C}__segformer_b2_cityscapes__dinov2_vitb14" \
      --model segformer_b2_cityscapes --scheme cityscapes_val_half \
      --cal-datasets cityscapes_val \
      --test-datasets "acdc_${C}_train" "acdc_${C}_val" \
      --methods weighted_crc --embedding dinov2_vitb14 --weight-estimator knn
done

echo "=== STAGE 5 MATRIX DONE ==="
