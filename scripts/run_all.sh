#!/usr/bin/env bash
# End-to-end reproduction: verification, splits, training, caching, region
# tables, the full experiment matrix, every reported table and figure, and
# the regression suite that recomputes them independently.
#
# Every stage is idempotent: caching and table stages skip completed work and
# experiment outputs are rewritten deterministically. If a run is interrupted,
# re-running this script resumes from where it stopped (training scripts
# retrain and overwrite their checkpoints; remove or keep checkpoints/
# accordingly).
#
# Prerequisites (see README):
#   - datasets extracted under the configured data root (RECORD_DATA_ROOT)
#   - Python environment installed (pip install -r requirements.txt && pip install -e .)
#   - CUDA GPU for the training and inference-caching phases; runtime is
#     dominated by those phases and scales with the GPU
#
# Usage:
#   nohup bash scripts/run_all.sh > run_all.log 2>&1 &
#   tail -f run_all.log

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Phase 0: environment and data verification ==="
python scripts/00_verify_environment.py
python scripts/01_verify_datasets.py

echo "=== Phase 1: split schemes ==="
python scripts/02_generate_splits.py

echo "=== Phase 2: Cityscapes/ACDC inference caches, embeddings, region tables ==="
bash scripts/run_stage3_full.sh

echo "=== Phase 3: LoveDA training, caches, region tables ==="
# MARIDA is not built here. Each of its models is tied to the axis it is held
# out from, so its whole chain lives in scripts/run_marida_holdout.sh, which
# phase 5 reaches through run_stage5_loveda_marida.sh.
bash scripts/run_train_full.sh

echo "=== Phase 4: experiment matrix (Cityscapes -> ACDC) ==="
bash scripts/run_stage5_full.sh

echo "=== Phase 5: experiment matrix (LoveDA, MARIDA), baselines, ablations ==="
bash scripts/run_stage5_loveda_marida.sh
bash scripts/run_stage5_extras.sh
bash scripts/run_stage5_baselines.sh

echo "=== Phase 6: supplementary blocks (Pareto, clip sweep, tier-B controls) ==="
bash scripts/run_stage5_supplementary.sh

echo "=== Phase 7: detector-level and calibration-level measurements ==="
python scripts/10_marida_test_f1.py
python scripts/20_temperature_seed_spread.py

echo "=== Phase 8: tables ==="
python scripts/08_make_tables.py
python scripts/14_uncertainty_tables.py --rho 0.5
python scripts/15_tierb_summary.py

echo "=== Phase 9: figures ==="
python scripts/09_make_figures.py
python scripts/13_pareto_tradeoff.py
python scripts/16_clip_window.py
python scripts/11_make_schematic.py
python scripts/12_make_qualitative.py --benchmark loveda

echo "=== Phase 10: recompute every published quantity from the result files ==="
python scripts/18_audit.py

echo "=== ALL PHASES DONE ==="
