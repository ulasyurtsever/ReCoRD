#!/usr/bin/env bash
# Supplementary stage-5 blocks: the experiments consumed by the Pareto,
# clip-window, tier-B summary and positive-control reports.
#
# Blocks:
#   p1  dense-alpha sweep for the area-matched Pareto comparison
#   p3  joint (lower, upper) clip sweep behind the vacuity certificate
#   p4  cross-fitted importance weights (out-of-fold domain classifier)
#   p5  in-distribution positive control on a random patch-level split
#   p6  tier B with the target set split, weights fitted on one half
#   p7  tier B with both sides split, weights independent of calibration
#       and test alike
#
# All blocks read the stage-4 region tables; none requires a GPU.
#
# Usage:
#   bash scripts/run_stage5_supplementary.sh

set -euo pipefail
cd "$(dirname "$0")/.."

RUN="python scripts/05_run_experiments.py"
CONDS=(fog night rain snow)
CS_MODELS=(segformer_b2_cityscapes segformer_b5_cityscapes)
MARIDA_SETS=(marida_train marida_val marida_test)

echo "=== p1: dense-alpha sweep for the area-matched comparison ==="
for MODEL in "${CS_MODELS[@]}"; do
  $RUN --name "p1_pareto__${MODEL}" --model "$MODEL" \
       --scheme cityscapes_val_half \
       --cal-datasets cityscapes_val --test-datasets cityscapes_val \
       --methods region_crc pixel_crc \
       --alphas 0.02 0.03 0.05 0.07 0.10 0.15 0.20 0.30 0.40 0.50
done

echo "=== p3: joint clip sweep ==="
# The conservative test mass depends on the clip interval only through the
# ratio of its ends, so the grid varies both ends rather than the ceiling
# alone.
for C in fog night; do
  for LO in 0.05 0.10 0.25 0.50; do
    for HI in 2 5 20; do
      TAG=$(echo "$LO" | tr -d '.')
      $RUN --name "p3_lo${TAG}_clip${HI}__${C}__segformer_b2_cityscapes__dinov2_vitb14" \
           --model segformer_b2_cityscapes --scheme cityscapes_val_half \
           --cal-datasets cityscapes_val \
           --test-datasets "acdc_${C}_train" "acdc_${C}_val" \
           --methods weighted_crc --embedding dinov2_vitb14 \
           --weight-clip-min "$LO" --weight-clip-max "$HI"
    done
  done
done

echo "=== p4: cross-fitted importance weights ==="
# Separates a genuine loss of source-target overlap from the domain
# classifier separating its own training points.
for C in "${CONDS[@]}"; do
  $RUN --name "p4_tierB_cv__${C}__segformer_b2_cityscapes__dinov2_vitb14" \
       --model segformer_b2_cityscapes --scheme cityscapes_val_half \
       --cal-datasets cityscapes_val \
       --test-datasets "acdc_${C}_train" "acdc_${C}_val" \
       --methods weighted_crc --embedding dinov2_vitb14 \
       --weight-estimator logistic_cv
done

echo "=== p5: in-distribution positive control ==="
# Calibration and test are exchangeable by construction on a random
# patch-level split, so the guarantee must hold: a violation here would
# indict the implementation rather than the benchmark.
for MODEL in marida_unet_official_holdout marida_unet_official_holdout_ens5; do
  $RUN --name "p5_marida_indist__${MODEL}" --model "$MODEL" \
       --scheme marida_random_half \
       --cal-datasets "${MARIDA_SETS[@]}" --test-datasets "${MARIDA_SETS[@]}" \
       --methods region_crc pixel_crc heuristic argmax
done

echo "=== p6, p7: tier B outside the transductive protocol ==="
# Proposition 1 assumes a weight function that does not depend on which of
# the n+1 points is the test point. p6 splits the target set so the weights
# see one half and the risk is measured on the other; p7 splits the source
# side as well, leaving the weights independent of the calibration points
# and the test point alike.
for C in "${CONDS[@]}"; do
  $RUN --name "p6_tierB_holdout__${C}__segformer_b2_cityscapes__dinov2_vitb14" \
       --model segformer_b2_cityscapes --scheme cityscapes_val_half \
       --cal-datasets cityscapes_val \
       --test-datasets "acdc_${C}_train" "acdc_${C}_val" \
       --methods weighted_crc --embedding dinov2_vitb14 \
       --weight-holdout 0.5 --alphas 0.05 0.1 0.2 --rhos 0.1 0.5 \
       --test-split test
  $RUN --name "p7_tierB_bothholdout__${C}__segformer_b2_cityscapes__dinov2_vitb14" \
       --model segformer_b2_cityscapes --scheme cityscapes_val_half \
       --cal-datasets cityscapes_val \
       --test-datasets "acdc_${C}_train" "acdc_${C}_val" \
       --methods weighted_crc --embedding dinov2_vitb14 \
       --weight-holdout 0.5 --weight-source-holdout 0.5 \
       --alphas 0.05 0.1 0.2 --rhos 0.1 0.5 --test-split test
done

echo "=== STAGE 5 SUPPLEMENTARY BLOCKS DONE ==="
