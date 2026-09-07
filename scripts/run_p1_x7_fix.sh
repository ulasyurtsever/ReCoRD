#!/usr/bin/env bash
# Follow-up to run_p1_rerun.sh (2026-09-07): redo the x7 union-area block with
# the published arguments, then rebuild the derived artifacts and re-audit.
#
# The first pass called 22_union_marked_area.py without --seeds/--alphas and
# the stage's defaults (one seed, three levels) replaced the published files
# (25 seeds, alpha = 0.2 only) with three-row files. Audit checks 98, 933, 934
# and 941 failed on them, and the sidecar `seeds` field pinned the cause.
# Nothing else in the re-run is affected; every other sidecar carries the
# published arguments.
#
# Usage (repository root, conda env `record`):
#   nohup bash scripts/run_p1_x7_fix.sh > p1_x7_fix.log 2>&1 &
#   tail -f p1_x7_fix.log
# Dry run: DRY_RUN=1 bash scripts/run_p1_x7_fix.sh

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

step "1. x7 union of the per-class masks, published arguments"
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes segformer_b2_cityscapes_mcdrop8; do
  run python scripts/22_union_marked_area.py \
    --model "$MODEL" --dataset cityscapes_val --scheme cityscapes_val_half \
    --experiment "e1_indist__${MODEL}" --method region_crc --rho 0.5 \
    --alphas 0.2 --seeds $(seq 0 24)
done

step "2. compare again (the x7 files must now align, 25 rows each)"
run python scripts/compare_experiments.py \
    --old "$BACKUP" --new results/experiments \
    --out results/p1_rerun_diff.csv --strict

step "3. derived artifacts"
run python scripts/08_make_tables.py --rho 0.5
run python scripts/08_make_tables.py --rho 0.1 --suffix=-rho0
run python scripts/15_tierb_summary.py --rho 0.5
run python scripts/13_pareto_tradeoff.py
run python scripts/14_uncertainty_tables.py --rho 0.5
run python scripts/16_clip_window.py
run python scripts/09_make_figures.py

step "4. audit"
if [ "$DRY" = 1 ]; then
  echo ">>> python scripts/18_audit.py > audit_post_p1.txt"
else
  python scripts/18_audit.py > audit_post_p1.txt 2>&1 || true
  grep -E "^\[FAIL\]|SUMMARY" audit_post_p1.txt || true
fi

echo
echo "RESULT: P1 X7 FIX DONE"
date
