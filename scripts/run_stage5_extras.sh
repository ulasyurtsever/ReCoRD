#!/usr/bin/env bash
# Stage-5 extension experiments (CPU only): shared-threshold ablation,
# size-weighted-loss ablation, and human-review triage curves.
#
# Blocks:
#   X1  Shared threshold: one lambda controlling the max per-class region
#       loss, versus independent per-class thresholds (Cityscapes, LoveDA)
#   X2  Size-weighted region loss: calibrate on area-weighted miss fraction,
#       report both the controlled risk and the unweighted region FNR
#   X3  Triage curves: residual system-level miss rate versus review budget,
#       on the MARIDA axes and one adverse-weather tier-A setting
#
# Usage:
#   nohup bash scripts/run_stage5_extras.sh > stage5_extras.log 2>&1 &
#   tail -f stage5_extras.log

set -euo pipefail
cd "$(dirname "$0")/.."

RUN="python scripts/05_run_experiments.py"

echo "=== X1: shared-threshold ablation ==="
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes \
             mask2former_swinb_cityscapes segformer_b2_cityscapes_mcdrop8; do
  $RUN --name "x1_shared__${MODEL}" \
    --model "$MODEL" --scheme cityscapes_val_half \
    --cal-datasets cityscapes_val --test-datasets cityscapes_val \
    --methods region_crc shared_crc
done
for D in urban rural; do
  $RUN --name "x1_shared__loveda_${D}" \
    --model "segformer_b2_loveda_${D}" --scheme "loveda_${D}_val_half" \
    --cal-datasets "loveda_Val_$(echo ${D^})" --test-datasets "loveda_Val_$(echo ${D^})" \
    --methods region_crc shared_crc
done

echo "=== X2: size-weighted-loss ablation ==="
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes; do
  $RUN --name "x2_sizew__${MODEL}" \
    --model "$MODEL" --scheme cityscapes_val_half \
    --cal-datasets cityscapes_val --test-datasets cityscapes_val \
    --methods region_crc --loss size_weighted
done

echo "=== X3: triage curves ==="
MARIDA_SETS=(marida_train marida_val marida_test)
$RUN --name "x3_triage__official__marida_unet_official_holdout_ens5" \
  --model marida_unet_official_holdout_ens5 --scheme marida_official_holdout \
  --cal-split calibration --test-split test \
  --cal-datasets "${MARIDA_SETS[@]}" --test-datasets "${MARIDA_SETS[@]}" \
  --methods region_crc --triage --triage-rankings area random oracle
# Triage runs consume the held-out schemes and their per-axis models, so that
# the residual-risk curves are measured on data the model never saw.
for R in 16PCC 16PDC 16PEC; do
  $RUN --name "x3_triage__region_${R}__marida_unet_holdout_region_${R}" \
    --model "marida_unet_holdout_region_${R}" --scheme "marida_holdout_region_${R}" \
    --cal-split calibration --test-split test \
    --cal-datasets "${MARIDA_SETS[@]}" --test-datasets "${MARIDA_SETS[@]}" \
    --methods region_crc --triage --triage-rankings area random oracle
done
$RUN --name "x3_triage__season_spring__marida_unet_holdout_season_spring" \
  --model marida_unet_holdout_season_spring --scheme marida_holdout_season_spring \
  --cal-split calibration --test-split test \
  --cal-datasets "${MARIDA_SETS[@]}" --test-datasets "${MARIDA_SETS[@]}" \
  --methods region_crc --triage --triage-rankings area random oracle
$RUN --name "x3_triage__night_tierA50__segformer_b2_cityscapes" \
  --model segformer_b2_cityscapes --scheme acdc_night_targetcal50 \
  --cal-datasets acdc_night_train acdc_night_val \
  --test-datasets acdc_night_train acdc_night_val \
  --methods region_crc --triage --triage-rankings area random oracle

echo "=== STAGE 5 EXTRAS DONE ==="
