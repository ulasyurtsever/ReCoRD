#!/usr/bin/env bash
# Stage-5 experiment matrix for the LoveDA and MARIDA axes, plus the tier-B
# weight-clip ablation on the Cityscapes->ACDC axis (CPU only).
#
# Blocks:
#   L1  LoveDA in-distribution validity per domain model
#   L2  LoveDA cross-domain breakdown (urban<->rural, source calibration)
#   L3  LoveDA tier A: exact recalibration from 25/50/100 labeled target images
#   L4  LoveDA tier B: importance-weighted calibration (DINOv2 and CLIP)
#   M1  MARIDA official split (calibrate on val, test on test)
#   M2  MARIDA leave-region-out (one held-out MGRS tile per scheme)
#   M3  MARIDA season splits
#   C1  Cityscapes->ACDC tier-B weight-clip ablation with weight diagnostics
#
# Every invocation writes results/experiments/<name>.csv (+ .meta.json) and is
# independent: rerunning the script overwrites CSVs deterministically.
#
# Usage:
#   nohup bash scripts/run_stage5_loveda_marida.sh > stage5_lm_run.log 2>&1 &
#   tail -f stage5_lm_run.log

set -euo pipefail
cd "$(dirname "$0")/.."

RUN="python scripts/05_run_experiments.py"

declare -A OTHER=( [urban]=rural [rural]=urban )
declare -A DSET=( [urban]=loveda_Val_Urban [rural]=loveda_Val_Rural )

echo "=== L1: LoveDA in-distribution ==="
for D in urban rural; do
  $RUN --name "l1_indist__${D}" \
    --model "segformer_b2_loveda_${D}" --scheme "loveda_${D}_val_half" \
    --cal-datasets "${DSET[$D]}" --test-datasets "${DSET[$D]}" \
    --methods region_crc pixel_crc heuristic argmax
done

echo "=== L2: LoveDA cross-domain breakdown ==="
for D in urban rural; do
  O=${OTHER[$D]}
  $RUN --name "l2_break__${D}2${O}" \
    --model "segformer_b2_loveda_${D}" --scheme "loveda_${D}_val_half" \
    --cal-datasets "${DSET[$D]}" --test-datasets "${DSET[$O]}" \
    --methods region_crc heuristic argmax
done

echo "=== L3: LoveDA tier A ==="
for D in urban rural; do
  O=${OTHER[$D]}
  for N in 25 50 100; do
    $RUN --name "l3_tierA${N}__${D}2${O}" \
      --model "segformer_b2_loveda_${D}" --scheme "loveda_${O}_targetcal${N}" \
      --cal-datasets "${DSET[$O]}" --test-datasets "${DSET[$O]}" \
      --methods region_crc argmax
  done
done

echo "=== L4: LoveDA tier B ==="
for D in urban rural; do
  O=${OTHER[$D]}
  for EMB in dinov2_vitb14 clip_vitb16; do
    $RUN --name "l4_tierB__${D}2${O}__${EMB}" \
      --model "segformer_b2_loveda_${D}" --scheme "loveda_${D}_val_half" \
      --cal-datasets "${DSET[$D]}" --test-datasets "${DSET[$O]}" \
      --methods weighted_crc --embedding "$EMB"
  done
done

# MARIDA blocks (official split, held-out regions, held-out seasons) are run by
# scripts/run_marida_holdout.sh. They need one model per held-out axis, fitted
# without that axis, so they cannot be expressed as a loop over two pretrained
# models the way the LoveDA blocks can.
echo "=== M1-M3: MARIDA held-out protocol ==="
bash scripts/run_marida_holdout.sh

echo "=== C1: Cityscapes->ACDC weight-clip ablation ==="
for CLIP in 2 5 10 20; do
  for COND in fog night rain snow; do
    $RUN --name "c1_clip${CLIP}__${COND}__segformer_b2_cityscapes__dinov2_vitb14" \
      --model segformer_b2_cityscapes --scheme cityscapes_val_half \
      --cal-datasets cityscapes_val \
      --test-datasets "acdc_${COND}_train" "acdc_${COND}_val" \
      --methods weighted_crc --embedding dinov2_vitb14 \
      --weight-clip-max "$CLIP"
  done
done

echo "=== STAGE 5 LOVEDA/MARIDA MATRIX DONE ==="
