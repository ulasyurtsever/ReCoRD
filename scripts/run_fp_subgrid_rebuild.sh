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
# the old 41-column array. Stage 5 does not refuse such a cache -- it warns and
# writes NaN into fp_components_per_image -- so skipping this rebuild costs the
# column rather than stopping the run.
#
# COST. Stage 4 reads the cached posteriors written by stage 3. There is no
# model inference here: it is a CPU pass over the existing cache, dominated by
# the connected-component labelling at 80 thresholds per image and class.
#
# AFTER THIS FINISHES, rerun the in-distribution experiments (stage 5) so the
# column is recomputed, then rerun the audit.
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">>> stage 4: region tables on the densified subgrid"
# Every dataset any later step reads has to be rebuilt, not just the one the
# in-distribution block uses: the sequence-disjoint tier-A runs read the ACDC
# tables, and a table left on the old subgrid yields an all-NaN column and a
# warning rather than an error.
for KEY in segformer_b2_cityscapes segformer_b5_cityscapes \
           mask2former_swinb_cityscapes segformer_b2_cityscapes_mcdrop8; do
  python scripts/04_build_region_tables.py \
    --model "$KEY" --datasets cityscapes_val --force
done
for C in fog night rain snow; do
  python scripts/04_build_region_tables.py \
    --model segformer_b2_cityscapes \
    --datasets "acdc_${C}_train" "acdc_${C}_val" --force
done

echo ">>> stage 5: in-distribution runs (the only ones that reported the column)"
# e1_indist is where the false-positive column was read. Nothing else in the
# rebuilt tables changes: stage 4 is deterministic, and the densified subgrid
# only alters fp_component_counts, so the coverage curves, marked-area curves
# and component tables come back byte-identical. Rerunning e1 is therefore
# enough to refresh the column, and every other stage-5 CSV stays valid.
for KEY in segformer_b2_cityscapes segformer_b5_cityscapes \
           mask2former_swinb_cityscapes segformer_b2_cityscapes_mcdrop8; do
  python scripts/05_run_experiments.py --name "e1_indist__${KEY}" \
      --model "$KEY" --scheme cityscapes_val_half \
      --cal-datasets cityscapes_val --test-datasets cityscapes_val \
      --methods region_crc pixel_crc heuristic argmax
done

# The audit is run here when this script is invoked on its own, because then
# it is the only check there is. It is SKIPPED when the referee-response
# driver calls this as its first step: at that point steps 2 to 6 have not
# run, so the audit is looking at a half-updated results directory, and its
# non-zero exit would abort the driver before the work that fixes what it is
# complaining about. Step 8 runs the audit for real, over the finished set.
if [ "${SKIP_AUDIT:-0}" = "1" ]; then
  echo ">>> audit skipped (SKIP_AUDIT=1); the caller runs it over the finished set"
else
  echo ">>> audit"
  python scripts/18_audit.py
fi

cat <<'NOTE'

NOTE. Every stage-5 CSV other than e1_indist still carries a
fp_components_per_image column computed against the OLD 41-point subgrid.
Those values were never quoted in the article and the audit only reports them
as a note, but they are not comparable with the rebuilt e1 column. Treat the
column as void outside e1 until its run is repeated.
NOTE
