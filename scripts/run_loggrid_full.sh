#!/usr/bin/env bash
# Move every published number onto the log-tail threshold grid (2026-09-08).
#
# WHY. The fourth panel's x17 arm showed that Mask2Former's "mark everything"
# calibration was an artifact of the uniform 1001-point grid, whose single
# point within 1e-3 of lambda = 1 could not resolve that model's thresholds;
# on a grid that spaces 1 - lambda over six decades the same model calibrates
# at 2-6% marked area in distribution. The grid is a design choice, so the
# article now uses the log-tail grid for every model (record.grid default
# "logtail"). Every cached statistic is tabulated on the grid, so this driver
# rebuilds stage 4 for every (model, dataset) pair from the existing caches,
# then re-derives stage 5 and every arm the article quotes. No training, no
# inference.
#
# BACKUPS (nothing is deleted):
#   results/raw          -> results/raw_uniform          (stage-4 tables, old grid)
#   results/experiments  -> results/experiments_pre_loggrid
#   results/tables, results/figures are versioned in the git mirror.
#
# ORDER. Stage 4 for all models (from raw_uniform's directory list), stage 30
# (dilation tables), then stage 5 exactly as run_panel4.sh minus the x17 arm,
# which is now the main matrix; x17 files and results_loggrid/ are parked.
# The comparison against the backup runs WITHOUT --strict: every number is
# expected to move, and lambda-hat indices are not comparable across grids.
# The audit at the end will FAIL wherever the article's constants still carry
# uniform-grid values; that list is the work order for the text revision.
#
# Usage (repository root, conda env `record`):
#   nohup bash scripts/run_loggrid_full.sh > loggrid_full.log 2>&1 &
# Dry run: DRY_RUN=1 bash scripts/run_loggrid_full.sh

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
unset RECORD_GRID RECORD_RESULTS_ROOT      # the default grid is now logtail
DRY="${DRY_RUN:-0}"
RAW_BACKUP=results/raw_uniform
EXP_BACKUP=results/experiments_pre_loggrid

run() {
  echo ">>> $*"
  if [ "$DRY" = 1 ]; then return 0; fi
  "$@"
}
step() { echo; echo "############ $*"; date; }

step "0. preflight"
run python -c "from record.grid import GRID_KIND; assert GRID_KIND == 'logtail', GRID_KIND; print('grid:', GRID_KIND)"
run python -m pytest -q -p no:cacheprovider          # expected: 131 passed
if [ -d "$RAW_BACKUP" ]; then
  echo "raw backup $RAW_BACKUP exists; the uniform-grid tables are already parked"
else
  run mv results/raw "$RAW_BACKUP"
  run mkdir -p results/raw
fi
if [ -d "$EXP_BACKUP" ]; then
  echo "experiment backup $EXP_BACKUP exists; kept"
else
  run cp -a results/experiments "$EXP_BACKUP"
fi
mkdir -p _to_delete/loggrid
for f in results/experiments/x17_loggrid__*; do
  [ -e "$f" ] && run mv "$f" _to_delete/loggrid/
done
[ -d results_loggrid ] && run mv results_loggrid _to_delete/loggrid/results_loggrid_arm || true

step "1. stage 4 on the log-tail grid, every model and dataset that had tables"
# The (model, dataset) list is read from the parked uniform-grid tree, so
# nothing that was published is skipped. Derived model keys are rebuilt by
# their own stages: *_tempscaled by 04b (baselines step 1) and 21, __dilation
# by stage 30 below.
# In a dry run the backup does not exist yet, so the list is read from the
# live tree instead; the two hold the same directories.
SRC_LIST="$RAW_BACKUP"; [ -d "$SRC_LIST" ] || SRC_LIST=results/raw
for mdir in "$SRC_LIST"/*/; do
  [ -d "$mdir" ] || continue
  model=$(basename "$mdir")
  case "$model" in *_tempscaled*|*__dilation) continue ;; esac
  datasets=$(ls "$mdir" | tr '\n' ' ')
  run python scripts/04_build_region_tables.py --model "$model" --datasets $datasets
done

step "1b. stage 30: dilation tables (four Cityscapes models)"
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes \
             mask2former_swinb_cityscapes segformer_b2_cityscapes_mcdrop8; do
  run python scripts/30_build_dilation_tables.py --model "$MODEL" --datasets cityscapes_val --force
done

step "2. stage-5 matrices"
run bash scripts/run_stage5_full.sh
export STAGE=experiments
run bash scripts/run_stage5_loveda_marida.sh
unset STAGE
run bash scripts/run_stage5_extras.sh
run bash scripts/run_stage5_baselines.sh              # step 1 rebuilds the tempscaled and LAC tables
run bash scripts/run_stage5_supplementary.sh

step "3. referee-response arms x10-x13"
run python scripts/25_tierb_test_charge.py \
    --model segformer_b2_cityscapes --scheme cityscapes_val_half \
    --cal-datasets cityscapes_val --embedding dinov2_vitb14 \
    --conditions fog night rain snow --rhos 0.5 0.1 --max-seeds 25
run python scripts/25_tierb_test_charge.py \
    --model segformer_b2_loveda_urban --scheme loveda_urban_val_half \
    --cal-datasets loveda_Val_Urban --embedding dinov2_vitb14 \
    --conditions urban2rural --dataset-template loveda_Val_Rural \
    --classes building water --rhos 0.5 0.1 --max-seeds 25 \
    --out-name x10_tierb_test_charge__loveda_urban2rural__dinov2_vitb14
run python scripts/25_tierb_test_charge.py \
    --model segformer_b2_loveda_rural --scheme loveda_rural_val_half \
    --cal-datasets loveda_Val_Rural --embedding dinov2_vitb14 \
    --conditions rural2urban --dataset-template loveda_Val_Urban \
    --classes building water --rhos 0.5 0.1 --max-seeds 25 \
    --out-name x10_tierb_test_charge__loveda_rural2urban__dinov2_vitb14
run python scripts/26_marida_confidence.py
run python scripts/26_marida_confidence.py \
    --model marida_unet_official_holdout_ens5 \
    --experiment h1_official__marida_unet_official_holdout_ens5 \
    --out-stem x11_marida_confidence__ens5
run python scripts/27_triage_permutation_band.py
run python scripts/28_acdc_sequence_schemes.py
run python scripts/28_acdc_sequence_schemes.py --sizes 50 100
for N in 25 50 100; do
  for C in fog night rain snow; do
    run python scripts/05_run_experiments.py \
        --name "x13_tierA${N}_seqdisjoint__${C}__segformer_b2_cityscapes" \
        --model segformer_b2_cityscapes \
        --scheme "acdc_${C}_targetcal${N}_seqdisjoint" \
        --cal-datasets "acdc_${C}_train" "acdc_${C}_val" \
        --test-datasets "acdc_${C}_train" "acdc_${C}_val" \
        --methods region_crc heuristic argmax
  done
done

step "4. supplementary measurements x6-x9 (21 rebuilds the per-temperature tables)"
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes; do
  run python scripts/21_temperature_leakfree.py \
    --model "$MODEL" --dataset cityscapes_val --scheme cityscapes_val_half --force
done
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes segformer_b2_cityscapes_mcdrop8; do
  run python scripts/22_union_marked_area.py \
    --model "$MODEL" --dataset cityscapes_val --scheme cityscapes_val_half \
    --experiment "e1_indist__${MODEL}" --method region_crc --rho 0.5 \
    --alphas 0.2 --seeds $(seq 0 24)
done
run python scripts/23_tierb_target_pool.py \
  --model segformer_b2_cityscapes --scheme cityscapes_val_half \
  --cal-datasets cityscapes_val --embedding dinov2_vitb14 \
  --conditions fog night rain snow \
  --classes person rider bicycle --max-seeds 25
run python scripts/24_marida_component_stats.py \
  --model marida_unet_official_holdout

step "5. fourth-panel arms x14-x16"
for K in 20 5 2; do
  run python scripts/05_run_experiments.py \
      --name "x14_tierB_noshift__segformer_b2_cityscapes__dinov2_vitb14__clip${K}" \
      --model segformer_b2_cityscapes --scheme cityscapes_val_half \
      --cal-datasets cityscapes_val --test-datasets cityscapes_val \
      --methods weighted_crc region_crc --embedding dinov2_vitb14 \
      --weight-clip-max "$K"
done
run python scripts/05_run_experiments.py \
    --name "x14_tierB_noshift__segformer_b2_cityscapes__clip_vitb16__clip20" \
    --model segformer_b2_cityscapes --scheme cityscapes_val_half \
    --cal-datasets cityscapes_val --test-datasets cityscapes_val \
    --methods weighted_crc region_crc --embedding clip_vitb16 \
    --weight-clip-max 20
for C in fog night; do
  run python scripts/05_run_experiments.py \
      --name "x15_source_reduced__${C}__segformer_b2_cityscapes" \
      --model segformer_b2_cityscapes --scheme cityscapes_val_half \
      --cal-datasets cityscapes_val \
      --test-datasets "acdc_${C}_train" "acdc_${C}_val" \
      --methods region_crc \
      --alphas 0.1801 0.1602 0.1503 0.1009 0.0024 0.0776 0.0553 0.0441
done
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes \
             mask2former_swinb_cityscapes segformer_b2_cityscapes_mcdrop8; do
  run python scripts/05_run_experiments.py \
      --name "x16_dilation__${MODEL}" \
      --model "${MODEL}__dilation" --scheme cityscapes_val_half \
      --cal-datasets cityscapes_val --test-datasets cityscapes_val \
      --methods region_crc argmax
done

step "6. compare against the uniform-grid backup (informational, not strict)"
echo "CSV files: $(ls results/experiments/*.csv 2>/dev/null | wc -l) (243 expected: 225 + 18 new arms)"
run python scripts/compare_experiments.py \
    --old "$EXP_BACKUP" --new results/experiments \
    --out results/loggrid_diff.csv

step "7. derived artifacts"
run python scripts/08_make_tables.py --rho 0.5
run python scripts/08_make_tables.py --rho 0.1 --suffix=-rho0
run python scripts/15_tierb_summary.py --rho 0.5
run python scripts/13_pareto_tradeoff.py
run python scripts/14_uncertainty_tables.py --rho 0.5
run python scripts/16_clip_window.py
run python scripts/09_make_figures.py
run python scripts/11_make_schematic.py
run python scripts/12_make_qualitative.py --benchmark loveda

step "8. audit (FAILs list the article constants that still carry uniform-grid values)"
if [ "$DRY" = 1 ]; then
  echo ">>> python scripts/18_audit.py > audit_loggrid.txt"
else
  python scripts/18_audit.py > audit_loggrid.txt 2>&1 || true
  grep -E "SUMMARY" audit_loggrid.txt || true
  echo "FAIL lines: $(grep -c '^\[FAIL\]' audit_loggrid.txt || true)"
fi

echo
echo "RESULT: LOGGRID FULL DONE"
date
