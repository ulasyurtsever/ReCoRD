#!/usr/bin/env bash
# Fourth-panel batch (2026-09-07): every server experiment the panel asked for,
# in one run, on top of a full stage-5 re-derivation with the unified capture
# rule.
#
# WHAT CHANGED IN THE CODE. The capture rule is now decided at the storage
# precision of the curves everywhere (05_run_experiments.py had three bare
# "< rho" comparisons, 26_marida_confidence.py one; audit 960 scans for them).
# That moves only rho = 0.1 diagnostic columns and the argmax rows at
# rho = 0.1, so the whole stage-5 tree is rewritten and compared against a
# backup, as in run_p1_rerun.sh.
#
# NEW ARMS (all CPU passes over existing caches; no training):
#   x13 n_t = 50, 100  sequence-disjoint tier A at the two larger budgets
#   x14                tier B with NO shift: Cityscapes val calibration half
#                      against its own test half, so the certificate's
#                      no-shift floor kappa/(n+kappa) is measured, at
#                      kappa = 20, 5, 2 (DINOv2) and kappa = 20 (CLIP)
#   x15                "source CRC at the reduced level": region CRC from
#                      source calibration at alpha' = (alpha - p)/(1 - p) with
#                      p the mean conservative test mass of each informative
#                      clip ratio of the p3 sweep (alpha = 0.2: ratios 4, 8,
#                      10, 20, 40 -> 0.1801, 0.1602, 0.1503, 0.1009, 0.0024;
#                      alpha = 0.1: ratios 4, 8, 10 -> 0.0776, 0.0553, 0.0441)
#   x16                dilation CRC: the argmax mask dilated by a calibrated
#                      radius (stage 30 tables, then stage 5 on <model>__dilation)
#   x17                Mask2Former on the log-tail grid (RECORD_GRID=logtail,
#                      separate results root, stage 4 rebuilt there), e1/e2/e3
#
# ORDER. Stage 5 first, then the arms that read stage-5 outputs (x10 reproduces
# e4_tierB; x16/x17 are independent), then compare, then derived artifacts and
# the audit. The comparison is --strict: it stops the run if rho = 0.5 moved or
# lambda-hat rose, which the capture-rule change cannot cause.
#
# Usage (repository root, conda env `record`):
#   nohup bash scripts/run_panel4.sh > panel4.log 2>&1 &
# Dry run: DRY_RUN=1 bash scripts/run_panel4.sh

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
DRY="${DRY_RUN:-0}"
BACKUP=results/experiments_pre_panel4
LOGROOT="$PWD/results_loggrid"

run() {
  echo ">>> $*"
  if [ "$DRY" = 1 ]; then return 0; fi
  "$@"
}
step() { echo; echo "############ $*"; date; }

step "0. preflight"
run python -m pytest -q -p no:cacheprovider          # expected: 131 passed
if [ -d "$BACKUP" ]; then
  echo "backup $BACKUP already exists; kept as is"
else
  run cp -a results/experiments "$BACKUP"
fi
mkdir -p _to_delete/panel4
for f in results/experiments/e1_cs_indist_b2*; do
  [ -e "$f" ] && run mv "$f" _to_delete/panel4/
done
echo "CSV files before: $(ls results/experiments/*.csv 2>/dev/null | wc -l) (225 expected)"

step "1. stage-5 matrices (unified capture rule)"
run bash scripts/run_stage5_full.sh
export STAGE=experiments
run bash scripts/run_stage5_loveda_marida.sh
unset STAGE
run bash scripts/run_stage5_extras.sh
run bash scripts/run_stage5_baselines.sh
run bash scripts/run_stage5_supplementary.sh

step "2. referee-response arms x10-x13 (n_t = 25)"
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
for C in fog night rain snow; do
  run python scripts/05_run_experiments.py \
      --name "x13_tierA25_seqdisjoint__${C}__segformer_b2_cityscapes" \
      --model segformer_b2_cityscapes \
      --scheme "acdc_${C}_targetcal25_seqdisjoint" \
      --cal-datasets "acdc_${C}_train" "acdc_${C}_val" \
      --test-datasets "acdc_${C}_train" "acdc_${C}_val" \
      --methods region_crc heuristic argmax
done

step "3. supplementary measurements x6-x9"
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes; do
  run python scripts/21_temperature_leakfree.py \
    --model "$MODEL" --dataset cityscapes_val --scheme cityscapes_val_half
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

step "4a. x13 at n_t = 50 and 100 (sequence-disjoint tier A)"
run python scripts/28_acdc_sequence_schemes.py --sizes 50 100
for N in 50 100; do
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

step "4b. x14 tier B without shift (Cityscapes val half vs half)"
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

step "4c. x15 source CRC at the reduced level (fog, night)"
for C in fog night; do
  run python scripts/05_run_experiments.py \
      --name "x15_source_reduced__${C}__segformer_b2_cityscapes" \
      --model segformer_b2_cityscapes --scheme cityscapes_val_half \
      --cal-datasets cityscapes_val \
      --test-datasets "acdc_${C}_train" "acdc_${C}_val" \
      --methods region_crc \
      --alphas 0.1801 0.1602 0.1503 0.1009 0.0024 0.0776 0.0553 0.0441
done

step "4d. x16 dilation CRC (in-distribution, four Cityscapes models)"
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes \
             mask2former_swinb_cityscapes segformer_b2_cityscapes_mcdrop8; do
  run python scripts/30_build_dilation_tables.py --model "$MODEL" --datasets cityscapes_val
  run python scripts/05_run_experiments.py \
      --name "x16_dilation__${MODEL}" \
      --model "${MODEL}__dilation" --scheme cityscapes_val_half \
      --cal-datasets cityscapes_val --test-datasets cityscapes_val \
      --methods region_crc argmax
done

step "4e. x17 Mask2Former on the log-tail grid (separate results root)"
# Stage 4 and stage 5 must see the same grid, so both run with RECORD_GRID set
# and write under results_loggrid/; the CSVs are then copied into
# results/experiments under the x17 prefix. The uniform-grid tree is untouched.
mkdir -p "$LOGROOT/raw" "$LOGROOT/experiments"
export RECORD_GRID=logtail RECORD_RESULTS_ROOT="$LOGROOT"
run python scripts/04_build_region_tables.py --model mask2former_swinb_cityscapes \
    --datasets cityscapes_val acdc_fog_train acdc_fog_val acdc_night_train acdc_night_val \
               acdc_rain_train acdc_rain_val acdc_snow_train acdc_snow_val
run python scripts/05_run_experiments.py --name "x17_loggrid__e1_indist__mask2former_swinb_cityscapes" \
    --model mask2former_swinb_cityscapes --scheme cityscapes_val_half \
    --cal-datasets cityscapes_val --test-datasets cityscapes_val \
    --methods region_crc pixel_crc heuristic argmax
for C in fog night rain snow; do
  run python scripts/05_run_experiments.py --name "x17_loggrid__e2_break__${C}__mask2former_swinb_cityscapes" \
      --model mask2former_swinb_cityscapes --scheme cityscapes_val_half \
      --cal-datasets cityscapes_val --test-datasets "acdc_${C}_train" "acdc_${C}_val" \
      --methods region_crc pixel_crc heuristic argmax
  for N in 25 50 100; do
    run python scripts/05_run_experiments.py --name "x17_loggrid__e3_tierA${N}__${C}__mask2former_swinb_cityscapes" \
        --model mask2former_swinb_cityscapes --scheme "acdc_${C}_targetcal${N}" \
        --cal-datasets "acdc_${C}_train" "acdc_${C}_val" \
        --test-datasets "acdc_${C}_train" "acdc_${C}_val" \
        --methods region_crc heuristic argmax
  done
done
unset RECORD_GRID RECORD_RESULTS_ROOT
if [ "$DRY" = 1 ]; then
  echo ">>> cp results_loggrid/experiments/x17_loggrid__*.csv|.meta.json results/experiments/"
else
  cp "$LOGROOT"/experiments/x17_loggrid__*.csv "$LOGROOT"/experiments/x17_loggrid__*.meta.json results/experiments/
fi

step "5. compare against the backup (new x13/x14/x15/x16/x17 files are 'only in new')"
echo "CSV files after: $(ls results/experiments/*.csv 2>/dev/null | wc -l)"
run python scripts/compare_experiments.py \
    --old "$BACKUP" --new results/experiments \
    --out results/panel4_diff.csv --strict

step "6. derived artifacts"
run python scripts/08_make_tables.py --rho 0.5
run python scripts/08_make_tables.py --rho 0.1 --suffix=-rho0
run python scripts/15_tierb_summary.py --rho 0.5
run python scripts/13_pareto_tradeoff.py
run python scripts/14_uncertainty_tables.py --rho 0.5
run python scripts/16_clip_window.py
run python scripts/09_make_figures.py

step "7. audit (FAILs on rho = 0.1 argmax cells are expected until the text is updated)"
if [ "$DRY" = 1 ]; then
  echo ">>> python scripts/18_audit.py > audit_panel4.txt"
else
  python scripts/18_audit.py > audit_panel4.txt 2>&1 || true
  grep -E "^\[FAIL\]|SUMMARY" audit_panel4.txt || true
fi

echo
echo "RESULT: PANEL4 DONE"
date
