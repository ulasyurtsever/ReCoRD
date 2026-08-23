#!/usr/bin/env bash
# Rebuild the region tables on the densified false-positive subgrid.
#
# WHY. The false-positive component count was read at the nearest point of a
# uniform 25-step subgrid. Every calibrated threshold sits in the top twenty
# grid points, so almost every draw was read at lambda_max, where the mask
# covers the whole image, the prediction is a single component, and the count
# degenerates to the class-absence indicator. The metric was withdrawn from the
# article until this rebuild lands.
#
# WHAT CHANGED. FP_SUBGRID_INDICES is now dense from index 960 upward (80 points
# instead of 41), and the driver takes the largest subgrid index at or below the
# calibrated one rather than the nearest. The cached per-image statistics carry
# the old 41-column array, so stage 4 must be rerun before stage 5 can read the
# column; stage 5 now refuses to run against a stale cache rather than reading
# the wrong slot.
#
# COST. Stage 4 reads the cached posteriors written by stage 3. There is no
# model inference here: it is a CPU pass over the existing cache, dominated by
# the connected-component labelling at 80 thresholds per image and class.
#
# AFTER THIS FINISHES, rerun the in-distribution experiments (stage 5) so the
# column is recomputed, then rerun the audit.
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">>> stage 4: Cityscapes region tables on the densified subgrid"
for KEY in segformer_b2_cityscapes segformer_b5_cityscapes \
           mask2former_swinb_cityscapes segformer_b2_cityscapes_mcdrop8; do
  python scripts/04_build_region_tables.py \
    --model "$KEY" --datasets cityscapes_val --force
done

echo ">>> stage 5: in-distribution runs (the only ones that reported the column)"
bash scripts/run_stage5_baselines.sh

echo ">>> audit"
python scripts/18_audit.py
