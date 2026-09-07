#!/usr/bin/env bash
# P1 re-run: regenerate every stage-5 output with the corrected capture rule.
#
# WHY. src/record/losses.py now decides capture at the storage precision of the
# coverage curves (float16). Before commit 448e186 two drivers widened the
# curves to float32 first, so a component covered by exactly one tenth of its
# pixels was scored "missed" at rho = 0.1. Every CSV under results/experiments
# was written by that code. This driver rewrites all of them from the
# unchanged stage-3/4 artifacts (caches and region tables), measures the move
# against a backup, and rebuilds the derived tables and figures.
#
# WHAT IT DOES NOT DO. No training, no inference, no cache or region-table
# rebuild: results/raw and cache/ are inputs here, never outputs. The FP
# subgrid rebuild (step 1 of run_referee_response.sh) is stage 4 and is
# skipped for the same reason.
#
# ORDER. Stage-5 matrices first (e/l/h/c/x/p blocks), then the referee arms
# that read them (x10 reproduces e4_tierB and aborts if it cannot), then the
# supplementary measurements of phase 7b, then the comparison, then tables and
# figures, then the audit. The comparison runs BEFORE the tables: if rho = 0.5
# moved or lambda-hat rose anywhere, the run stops there, because that would
# be a second defect and tables built on it would be worthless.
#
# RESUMABLE. Experiment CSVs are rewritten deterministically, so after an
# interruption re-running this script is safe; to save time comment out the
# stage-5 drivers whose ">>> ... DONE" line already appears in the log.
#
# Usage (from the repository root, conda env `record`):
#   nohup bash scripts/run_p1_rerun.sh > p1_rerun.log 2>&1 &
#   tail -f p1_rerun.log
# Dry run (prints every command, executes none):
#   DRY_RUN=1 bash scripts/run_p1_rerun.sh

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
DRY="${DRY_RUN:-0}"
BACKUP=results/experiments_pre_p1

run() {
  echo ">>> $*"
  if [ "$DRY" = 1 ]; then return 0; fi
  "$@"
}
step() { echo; echo "############ $*"; date; }

step "0. preflight: the corrected code must be the code that runs"
run python -m pytest -q -p no:cacheprovider          # expected: 122 passed
if [ -d "$BACKUP" ]; then
  echo "backup $BACKUP already exists; kept as is, NOT overwritten"
else
  run cp -a results/experiments "$BACKUP"
fi
# The pilot file e1_cs_indist_b2.* (7 Aug, old region-FNR definition) was
# removed from the repository on 2 Sep because it pooled into the e1 block. If
# a copy survived here, the table generator refuses to run. Park it, do not
# delete it.
mkdir -p _to_delete/p1_rerun
for f in results/experiments/e1_cs_indist_b2*; do
  [ -e "$f" ] && run mv "$f" _to_delete/p1_rerun/
done
n_before=$(ls results/experiments/*.csv 2>/dev/null | wc -l)
echo "CSV files before the re-run: $n_before (expected 225)"

step "1. stage-5 matrices"
run bash scripts/run_stage5_full.sh                   # e1 e2 e3 e4_tierB e4_tierB_knn (104)
export STAGE=experiments                              # MARIDA: reuse checkpoints, caches, tables
run bash scripts/run_stage5_loveda_marida.sh          # l1-l4 (14), h1-h3 (11), c1 (16)
unset STAGE
run bash scripts/run_stage5_extras.sh                 # x1 (6), x2 (2), x3 (6 runs, 12 files)
run bash scripts/run_stage5_baselines.sh              # 04b tables, x4 (2), x5 (2)
run bash scripts/run_stage5_supplementary.sh          # p1 (2), p3 (24), p4 (4), p5 (2), p6/p7 (8)

step "2. referee-response arms (run_referee_response.sh steps 2, 3, 4, 6)"
# step 1 of that driver is stage 4 (unchanged), step 5 duplicates baselines above.
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

step "3. supplementary measurements quoted in the text (run_all.sh phase 7b)"
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes; do
  run python scripts/21_temperature_leakfree.py \
    --model "$MODEL" --dataset cityscapes_val --scheme cityscapes_val_half
done
# Published x7 run: 25 seeds at alpha = 0.2 only. The stage's defaults (one
# seed, three levels) are NOT that run; the first pass of this driver used them
# and produced three-row files (caught by audit 941 and the sidecar diff).
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

step "4. compare against the backup (stops the run on a violated expectation)"
n_after=$(ls results/experiments/*.csv 2>/dev/null | wc -l)
echo "CSV files after the re-run: $n_after (expected 225)"
run python scripts/compare_experiments.py \
    --old "$BACKUP" --new results/experiments \
    --out results/p1_rerun_diff.csv --strict

step "5. derived artifacts"
run python scripts/08_make_tables.py --rho 0.5
run python scripts/08_make_tables.py --rho 0.1 --suffix=-rho0
run python scripts/15_tierb_summary.py --rho 0.5
run python scripts/13_pareto_tradeoff.py
run python scripts/14_uncertainty_tables.py --rho 0.5
run python scripts/16_clip_window.py
run python scripts/09_make_figures.py

step "6. audit"
# FAIL lines are expected here and are part of the deliverable: the audit pins
# the numbers the article currently prints, and every rho = 0.1 quantity that
# moved will show up as a FAIL until the text is corrected to the new value.
# A FAIL on a rho = 0.5 quantity, by contrast, would be a defect. Both go to
# audit_post_p1.txt; the summary lines are echoed below.
if [ "$DRY" = 1 ]; then
  echo ">>> python scripts/18_audit.py > audit_post_p1.txt"
else
  python scripts/18_audit.py > audit_post_p1.txt 2>&1 || true
  grep -E "^\[FAIL\]|SUMMARY" audit_post_p1.txt || true
fi

echo
echo "RESULT: P1 RERUN DONE"
date
