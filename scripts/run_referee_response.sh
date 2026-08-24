#!/usr/bin/env bash
# Everything the referee response needs from a machine that has the caches.
#
# ORDER MATTERS. Stage 4 must be rebuilt first: the false-positive subgrid was
# densified (41 -> 80 points) after a review found that the old uniform subgrid
# forced almost every calibrated threshold onto lambda_max, where the component
# count degenerates. Stage 5 does NOT stop on a stale cache: it warns and
# records fp_components_per_image as NaN, so a run that skips this step still
# produces results, just without that column. Run it first anyway -- the column
# is the whole point of the rebuild.
#
# None of the steps below re-runs model inference. Every one is a CPU pass over
# the caches that stage 3 already wrote, except step 5, which reads the MARIDA
# rasters directly.
#
# Run from the repository root. Each step is independent once step 1 is done,
# so a failure can be resumed by commenting out what already succeeded.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

echo "############ 1. rebuild the region tables on the densified FP subgrid"
bash scripts/run_fp_subgrid_rebuild.sh

echo "############ 2. tier B: is the vacuity the ceiling's fault?"
# Charges the test point its own estimated weight instead of the ceiling, and
# reports the weight distribution on the TARGET embeddings, which the published
# pipeline never scored. Also refits with the source side unfiltered, so the
# q(x)/p(x|c) conditioning the estimator actually uses is measured rather than
# argued about. The ceiling arm must reproduce the published e4_tierB numbers;
# the script aborts if it does not.
# ACDC. --rhos is passed explicitly: the stage defaults to 0.5 alone, but the
# published tier-B matrix covers both capture levels and the reproduction
# check can only compare the cells it is given.
python scripts/25_tierb_test_charge.py \
    --model segformer_b2_cityscapes --scheme cityscapes_val_half \
    --cal-datasets cityscapes_val --embedding dinov2_vitb14 \
    --conditions fog night rain snow --rhos 0.5 0.1 --max-seeds 25

# LoveDA, both directions. These need --classes, because the LoveDA tables
# carry building and water rather than the ACDC critical classes, and an
# explicit --out-name: the default name is keyed on the model alone, so it is
# the same for every embedding, and a later CLIP run would overwrite the
# DINOv2 one. (The two directions use different models and would not have
# collided with each other.) The name follows the convention in the stage's
# own docstring, direction and embedding both spelled out.
#
# Only one clip was published for LoveDA (kappa=20, run l4_tierB), so the
# sweep's kappa=2 and kappa=5 arms have no published counterpart to reproduce.
# The stage reports them as gaps and still passes, because the condition is
# reproduced at kappa=20; it fails only if a condition matches no clip at all.
python scripts/25_tierb_test_charge.py \
    --model segformer_b2_loveda_urban --scheme loveda_urban_val_half \
    --cal-datasets loveda_Val_Urban --embedding dinov2_vitb14 \
    --conditions urban2rural --dataset-template loveda_Val_Rural \
    --classes building water --rhos 0.5 0.1 --max-seeds 25 \
    --out-name x10_tierb_test_charge__loveda_urban2rural__dinov2_vitb14

python scripts/25_tierb_test_charge.py \
    --model segformer_b2_loveda_rural --scheme loveda_rural_val_half \
    --cal-datasets loveda_Val_Rural --embedding dinov2_vitb14 \
    --conditions rural2urban --dataset-template loveda_Val_Urban \
    --classes building water --rhos 0.5 0.1 --max-seeds 25 \
    --out-name x10_tierb_test_charge__loveda_rural2urban__dinov2_vitb14

echo "############ 3. MARIDA annotation confidence"
# MARIDA flags every annotation High/Moderate/Low and the pipeline ignored it.
# Roughly half the debris pixels are not High, and the uncertain ones are the
# smallest, which is where the residual risk sits. Reports the composition and
# the region miss rate restricted to components carrying a High-confidence
# pixel.
# --out-stem on the second call: the stem does not carry the model name, so
# both runs left at the default would write the same two files and the
# ensemble would overwrite the single model without a word.
python scripts/26_marida_confidence.py
python scripts/26_marida_confidence.py \
    --model marida_unet_official_holdout_ens5 \
    --experiment h1_official__marida_unet_official_holdout_ens5 \
    --out-stem x11_marida_confidence__ens5

echo "############ 4. triage: a permutation band, not just an expectation"
# Five of the six triage settings are single partitions with one realized
# random ordering. The closed-form floor is the right estimand but it is not a
# band; this draws 1000 permutations so the reader can see whether the
# marked-area score beats chance by more than permutation noise.
python scripts/27_triage_permutation_band.py

echo "############ 5. class-conditional LAC and the measured pixel FNR"
# The published LAC row calibrates on an all-class pixel-coverage target and is
# then scored on three rare classes, which a referee called unfair. The
# class-conditional variant is the fair version. --measure-pixel-fnr records
# the quantity pixel CRC is said to control, which was asserted and never
# measured.
for M in segformer_b2_cityscapes segformer_b5_cityscapes; do
  python scripts/04b_build_baseline_tables.py --model "$M" \
      --datasets cityscapes_val --temperature-cal-scheme cityscapes_val_half
  python scripts/05_run_experiments.py --name "x4_lac__$M" \
      --model "$M" --scheme cityscapes_val_half \
      --cal-datasets cityscapes_val --test-datasets cityscapes_val \
      --methods lac_global region_crc pixel_crc \
      --lac-variants marginal class_conditional --measure-pixel-fnr
done

echo "############ 6. sequence-disjoint tier A on ACDC"
# The published tier-A schemes draw target labels from the same driving
# sequences they are tested on: at n_t=25, 91-96% of test frames share a
# sequence with a calibration frame. These schemes hold whole sequences out.
python scripts/28_acdc_sequence_schemes.py
for C in fog night rain snow; do
  python scripts/05_run_experiments.py \
      --name "e3_tierA25_seqdisjoint__${C}__segformer_b2_cityscapes" \
      --model segformer_b2_cityscapes \
      --scheme "acdc_${C}_targetcal25_seqdisjoint" \
      --cal-datasets "acdc_${C}_train" "acdc_${C}_val" \
      --test-datasets "acdc_${C}_train" "acdc_${C}_val" \
      --methods region_crc heuristic argmax
done

echo "############ 7. regenerate the derived artifacts"
python scripts/08_make_tables.py --rho 0.5
python scripts/15_tierb_summary.py --rho 0.5
python scripts/09_make_figures.py

echo "############ 8. audit"
python scripts/18_audit.py

cat <<'NOTE'

STILL OPEN after this run, and deliberately so:
  * The tail-refined lambda grid for Mask2Former (referee item 17). That means
    changing LAMBDA_GRID and rebuilding every table, which would move every
    marked-area number in the article. The text now scopes the Mask2Former
    claim to the family of masks the present grid generates; the experiment is
    a separate, deliberate piece of work.
NOTE
