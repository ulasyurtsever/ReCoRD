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

echo "=== Phase 7b: supplementary measurements quoted in the text ==="
# Stages 21-24 write x6/x7/x8/x9, which carry nineteen quantities the article
# reports: the leakage-free temperatures, the union of the per-class masks, the
# tier-B target-pool diagnostics, and the MARIDA component-size statistics.
# They were reachable only by hand until now, so a clean checkout produced a
# results tree the audit accepted while those quantities were simply absent.
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes; do
  python scripts/21_temperature_leakfree.py \
    --model "$MODEL" --dataset cityscapes_val --scheme cityscapes_val_half
done
# --seeds and --alphas are spelled out because the stage's defaults (one seed,
# three levels) are not the published run (25 seeds, alpha = 0.2 only). The
# 2026-09-07 re-run called it with the defaults and the audit's union checks
# (98, 933, 934, 941) failed on a three-row file; the sidecar's `seeds` field
# is what identified the cause.
for MODEL in segformer_b2_cityscapes segformer_b5_cityscapes \
             segformer_b2_cityscapes_mcdrop8; do
  python scripts/22_union_marked_area.py \
    --model "$MODEL" --dataset cityscapes_val --scheme cityscapes_val_half \
    --experiment "e1_indist__${MODEL}" --method region_crc --rho 0.5 \
    --alphas 0.2 --seeds $(seq 0 24)
done
python scripts/23_tierb_target_pool.py \
  --model segformer_b2_cityscapes --scheme cityscapes_val_half \
  --cal-datasets cityscapes_val --embedding dinov2_vitb14 \
  --conditions fog night rain snow \
  --classes person rider bicycle --max-seeds 25
python scripts/24_marida_component_stats.py \
  --model marida_unet_official_holdout

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
