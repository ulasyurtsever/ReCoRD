#!/usr/bin/env bash
# Remaining baseline experiments: LAC-style global pixel-coverage threshold
# and temperature scaling (CPU passes over the stage-3 cache; no GPU).
#
# Steps:
#   1. Stage 4b: fit a scalar temperature (seed-0 calibration list), build
#      region tables for the tempered posterior, and derive per-image LAC
#      miscoverage curves (SegFormer-B2 and B5, Cityscapes val)
#   2. Stage 5: LAC baseline (x4) and temperature-scaling baseline (x5)
#
# Usage:
#   nohup bash scripts/run_stage5_baselines.sh > stage5_baselines.log 2>&1 &
#   tail -f stage5_baselines.log

set -euo pipefail
cd "$(dirname "$0")/.."

RUN="python scripts/05_run_experiments.py"

echo "=== Step 1: stage-4b derivations ==="
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes; do
  python scripts/04b_build_baseline_tables.py \
    --model "$MODEL" --datasets cityscapes_val \
    --temperature-cal-scheme cityscapes_val_half
done

echo "=== Step 2: x4 LAC baseline ==="
# The published block carries four arms, not two. The marginal LAC threshold is
# fitted on the pooled pixels and then scored on three rare classes; the
# class-conditional variant is the fair version, and --measure-pixel-fnr
# records the quantity pixel CRC is calibrated on, so that the reduction of
# class-conditional LAC to the pixel baseline is measured rather than asserted.
# Table II prints both LAC rows and Section V quotes the two gaps, so a run
# that omits them silently removes published content: this invocation must stay
# identical to the one in run_referee_response.sh.
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes; do
  $RUN --name "x4_lac__${MODEL}" \
    --model "$MODEL" --scheme cityscapes_val_half \
    --cal-datasets cityscapes_val --test-datasets cityscapes_val \
    --methods lac_global region_crc pixel_crc \
    --lac-variants marginal class_conditional --measure-pixel-fnr
done

echo "=== Step 3: x5 temperature-scaling baseline ==="
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes; do
  $RUN --name "x5_temp__${MODEL}" \
    --model "${MODEL}_tempscaled" --scheme cityscapes_val_half \
    --cal-datasets cityscapes_val --test-datasets cityscapes_val \
    --methods region_crc heuristic pixel_crc argmax
done

echo "=== STAGE 5 BASELINES DONE ==="
