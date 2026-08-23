#!/usr/bin/env bash
# MARIDA held-out protocol: one model per held-out tile or season, fitted
# without any patch of that axis and without the calibration group, followed by
# caching, region tables, and the calibration experiments.
#
# A held-out tile or season must be absent from every stage of model
# fitting, not only from the calibration set, so each axis needs its own model.
# Stage 2b partitions each axis into four disjoint groups at scene granularity
# (train / monitor / calibration / test); this script fits one model per axis
# from the train group alone, selects its checkpoint on the monitor group, and
# calibrates and evaluates on the remaining two.
#
# Steps, per scheme:
#   1. train the U-Net on the scheme's train group, selecting the checkpoint on
#      its monitor group (07, GPU)
#   2. cache posteriors for the new model over the three MARIDA dataset keys
#      (03, GPU)
#   3. build region tables from that cache (04, CPU)
#   4. run the calibration experiments: calibrate on the scheme's calibration
#      group, test on its test group (05, CPU)
#
# Resumable: training overwrites its checkpoint; caching and table stages skip
# completed work; experiment CSVs are rewritten deterministically.
#
# Usage:
#   nohup bash scripts/run_marida_holdout.sh > marida_holdout.log 2>&1 &
#   tail -f marida_holdout.log

set -euo pipefail
cd "$(dirname "$0")/.."

# STAGE=all (default) trains, caches, tabulates and runs the experiments.
# STAGE=experiments reuses the existing checkpoints, caches and region tables
# and re-runs stage 5 only, for re-deriving the experiment CSVs without
# retraining or re-caching.
STAGE="${STAGE:-all}"

MARIDA_SETS=(marida_train marida_val marida_test)
REGIONS=(16PCC 16PDC 16PEC 18QYF 48PZC)
SEASONS=(winter spring summer autumn)

SCHEMES=(marida_official_holdout)
for R in "${REGIONS[@]}"; do SCHEMES+=("marida_holdout_region_${R}"); done
for S in "${SEASONS[@]}"; do SCHEMES+=("marida_holdout_season_${S}"); done

echo "=== Step 0: regenerate MARIDA held-out schemes ==="
python scripts/02b_generate_marida_splits.py

for SCHEME in "${SCHEMES[@]}"; do
  if [ ! -f "splits/${SCHEME}.json" ]; then
    echo "${SCHEME}: scheme absent, skipped"
    continue
  fi
  MODEL="marida_unet_${SCHEME#marida_}"
  case "$SCHEME" in
    marida_official_holdout) NAME="h1_official__${MODEL}" ;;
    marida_holdout_region_*) NAME="h2_region__${SCHEME#marida_holdout_region_}__${MODEL}" ;;
    marida_holdout_season_*) NAME="h3_season__${SCHEME#marida_holdout_season_}__${MODEL}" ;;
  esac

  echo
  if [ "$STAGE" = all ]; then
    echo "=== ${SCHEME}: train (${MODEL}) ==="
    python scripts/07_train_marida_unet.py --scheme "$SCHEME" --seed 0

    echo "=== ${SCHEME}: cache posteriors ==="
    python scripts/03_precompute_cache.py --model "$MODEL" --datasets "${MARIDA_SETS[@]}"

    echo "=== ${SCHEME}: region tables ==="
    python scripts/04_build_region_tables.py --model "$MODEL" --datasets "${MARIDA_SETS[@]}"
  fi

  echo "=== ${SCHEME}: experiments (${NAME}) ==="
  python scripts/05_run_experiments.py --name "$NAME" \
    --model "$MODEL" --scheme "$SCHEME" \
    --cal-split calibration --test-split test \
    --cal-datasets "${MARIDA_SETS[@]}" --test-datasets "${MARIDA_SETS[@]}" \
    --methods region_crc pixel_crc heuristic argmax \
    --bootstrap-ci 4000 --bootstrap-clusters splits/marida_meta.csv
done

# The ensemble-versus-single comparison lives on the official axis. Its members
# must be fitted under the same discipline as the single model: the official val
# split is the calibration set, so no member may select its checkpoint on it.
ENS=marida_unet_official_holdout_ens5
if [ "$STAGE" = all ]; then
  echo
  echo "=== official axis: deep-ensemble members (seeds 1-4) ==="
  for S in 1 2 3 4; do
    python scripts/07_train_marida_unet.py --scheme marida_official_holdout --seed "$S"
  done
  echo "=== official axis: ensemble cache and tables (${ENS}) ==="
  python scripts/03_precompute_cache.py --model "$ENS" --datasets "${MARIDA_SETS[@]}"
  python scripts/04_build_region_tables.py --model "$ENS" --datasets "${MARIDA_SETS[@]}"
fi
python scripts/05_run_experiments.py --name "h1_official__${ENS}" \
  --model "$ENS" --scheme marida_official_holdout \
  --cal-split calibration --test-split test \
  --cal-datasets "${MARIDA_SETS[@]}" --test-datasets "${MARIDA_SETS[@]}" \
  --methods region_crc pixel_crc heuristic argmax \
  --bootstrap-ci 4000 --bootstrap-clusters splits/marida_meta.csv

echo
echo "=== audit: no test patch may appear in any fitting group ==="
# A nonzero exit means a test patch reached a fitting group, which invalidates
# every number below it, so the run stops here rather than reporting PASS.
python scripts/18_audit.py --section holdout

echo "RESULT: PASS"
