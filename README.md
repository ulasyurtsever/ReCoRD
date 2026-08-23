# ReCoRD

Region-level conformal risk control for multi-class semantic segmentation under
distribution shift.

Given a pretrained segmentation model, the framework selects class-specific
thresholds on held-out calibration data such that the expected fraction of
missed ground-truth regions (connected components) of designated critical
classes is bounded by a chosen risk level, with finite-sample,
distribution-free validity. Deployment under distribution shift is handled by a
two-tier calibration scheme (exact recalibration on a small labeled target set;
importance-weighted calibration over foundation-model embeddings otherwise)
together with a score-distribution drift monitor.

> Paper in preparation. Citation information will be added upon publication.

## Repository layout

```
configs/          Experiment and dataset configuration (YAML)
src/record/       Core library (installable package)
scripts/          Numbered pipeline stages and orchestrator scripts
splits/           Committed calibration/test split definitions (JSON)
tests/            Unit and integration tests (pytest)
```

The pipeline also writes:

```
cache/                Per-image posterior caches and embeddings
checkpoints/          Locally trained model weights (LoveDA, MARIDA)
results/raw/          Per-region sufficient-statistic tables
results/experiments/  Experiment CSVs with .meta.json sidecars
results/tables/       LaTeX tables
results/figures/      PDF figures
```

The first three are large and are excluded from version control; they are
rebuilt by the pipeline from the datasets. The experiment CSVs and the tables
and figures derived from them are committed, so every reported number can be
regenerated without repeating inference.

## Storage roots

Each storage location resolves relative to the repository root by default and
can be redirected through an environment variable:

| Contents | Default | Override |
|---|---|---|
| Datasets | `../../00_datasets` | `RECORD_DATA_ROOT` |
| Inference caches, embeddings | `./cache` | `RECORD_CACHE_ROOT` |
| Statistic tables, experiment CSVs | `./results` | `RECORD_RESULTS_ROOT` |
| Split definitions | `./splits` | `RECORD_SPLITS_ROOT` |

```bash
export RECORD_DATA_ROOT=/data/datasets
export RECORD_CACHE_ROOT=/data/record/cache
export RECORD_RESULTS_ROOT=/data/record/results
```

The variables must be set before any script runs. Existing files are not
relocated: move the directory first, then point the variable at the new
location. Trained checkpoints are always written to `./checkpoints` inside the
repository.

Approximate storage for the full pipeline: datasets 41 GB, caches 50 GB,
checkpoints 1 GB, results under 1 GB.

## Setup

Requires Python 3.11 or later. The inference and training stages require a
CUDA GPU; the experiment stages run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Dataset acquisition is documented in `configs/datasets.yaml`. All datasets are
publicly available from their original providers and are not redistributed
here.

## Usage

Full reproduction. One command produces every number, table and figure in the
article, then recomputes them independently:

```bash
nohup bash scripts/run_all.sh > run_all.log 2>&1 &
tail -f run_all.log
```

It runs ten phases in dependency order: environment and dataset verification,
split generation, inference caching and region tables, LoveDA and MARIDA
training, the experiment matrix on all three benchmarks, the positive control
on a random patch-level split, the detector-level and temperature measurements,
the tables, the figures, and finally `18_audit.py`, which recomputes every
published quantity from the CSVs on a code path independent of the table and
figure builders. A nonzero exit from that last phase means the pipeline has
stopped reproducing a published value.

All stages are idempotent. Caching and table stages skip completed work, and
experiment outputs are rewritten deterministically, so an interrupted run is
resumed by re-running the same script. The exception is training, which
restarts from scratch and overwrites its checkpoints.

## Pipeline stages

| Stage | Script | Compute | Description |
|---|---|---|---|
| 0 | `00_verify_environment.py` | CPU | Check Python/PyTorch/backend and required packages |
| 1 | `01_verify_datasets.py` | CPU | Validate dataset layout and file counts against the manifest |
| 2 | `02_generate_splits.py` | CPU | Generate all seeded calibration/test splits (committed to `splits/`) |
| 2b | `02b_generate_marida_splits.py` | CPU | MARIDA held-out region and season schemes: four disjoint groups (train/monitor/calibration/test) per axis, partitioned at scene granularity |
| 3 | `03_precompute_cache.py` | GPU | Per-image posterior caches (`--mc-dropout`, `--smoke`) |
| 3b | `03b_precompute_embeddings.py` | GPU | DINOv2/CLIP image embeddings for tier-B weighting |
| 4 | `04_build_region_tables.py` | CPU | Per-region λ-coverage curves and image statistics from the cache |
| 4b | `04b_build_baseline_tables.py` | CPU | Temperature fit, tempered region tables, and pixel-coverage curves for the post-hoc baselines |
| 5 | `05_run_experiments.py` | CPU | Calibration experiments over seeds, methods, and risk levels |
| 6 | `06_train_loveda.py` | GPU | LoveDA SegFormer fine-tuning, one model per domain |
| 7 | `07_train_marida_unet.py` | GPU | MARIDA multispectral U-Net; `--seed` for the official-split ensemble members, `--scheme` for one model per held-out axis |
| 8 | `08_make_tables.py` | CPU | LaTeX tables from experiment CSVs (`results/tables/`) |
| 9 | `09_make_figures.py` | CPU | Figures from experiment CSVs (`results/figures/`) |
| 10 | `10_marida_test_f1.py` | CPU | Pixel-level marine-debris F1 on the official MARIDA test split |
| 11 | `11_make_schematic.py` | CPU | Method schematic (synthetic illustration, no data) |
| 12 | `12_make_qualitative.py` | CPU | Qualitative panel on one test scene: argmax versus region CRC at the calibrated threshold (`--benchmark loveda\|marida`) |
| 13 | `13_pareto_tradeoff.py` | CPU | Area-matched comparison of region and pixel CRC over a level sweep |
| 14 | `14_uncertainty_tables.py` | CPU | Standard errors and Clopper--Pearson intervals for the validity tables |
| 15 | `15_tierb_summary.py` | CPU | Tier-B weight diagnostics: conservative test mass, effective sample size, informative-draw fraction |
| 16 | `16_clip_window.py` | CPU | Tier-B behaviour as a function of the clip ratio |
| 18 | `18_audit.py` | CPU | Regression suite over the released result files: every published numeric and qualitative quantity, recomputed on an independent code path. `--section holdout` additionally checks each experiment's test group against the data its model was fitted on |
| 20 | `20_temperature_seed_spread.py` | CPU | Refits the temperature on each seeded calibration draw and reports the spread, bounding how much the fitted scalar can depend on its own test half |
| 21 | `21_temperature_leakfree.py` | CPU | Leak-free temperature scaling: refits the scalar on each draw's own calibration list and recalibrates that draw against tables built from its own temperature, so the tempered baseline never sees its test half |
| 22 | `22_union_marked_area.py` | CPU | Marked area of the union of the critical-class masks at the selected thresholds, measuring the cost of monitoring several classes at once |
| 23 | `23_tierb_target_pool.py` | CPU | Refits the tier-B density ratio with the target side restricted to class-bearing target images, so the conditioning of the two populations matches the weighted statement, and reports both fits side by side |

Orchestrators: `run_all.sh` (every phase in dependency order; this is the
only script a reproducer needs to run),
`run_stage3_full.sh` (Cityscapes/ACDC caching and tables),
`run_train_full.sh` (LoveDA training, caching, and tables; MARIDA is built by
`run_marida_holdout.sh` instead, since each of its models is tied to the axis it
is held out from),
`run_stage5_full.sh` (Cityscapes/ACDC experiment matrix),
`run_stage5_loveda_marida.sh` (LoveDA matrix and the weight-clip ablation; it
delegates the MARIDA blocks to `run_marida_holdout.sh`),
`run_marida_holdout.sh` (the MARIDA held-out protocol end to end: one model
per region and season, then caching, tables and experiments),
`run_stage5_extras.sh` (shared-threshold, size-weighted, and triage runs),
`run_stage5_baselines.sh` (post-hoc baselines),
`run_stage5_supplementary.sh` (the area-matched sweep, the clip-ratio grid,
the cross-fitted weights, the tier-B holdout controls and the MARIDA
in-distribution control: the `p1`--`p7` blocks under
`results/experiments/`).

GPU inference runs once and is cached; downstream analyses, tables, and
figures are pure arithmetic over the cached statistics.

## Held-out evaluation on MARIDA

MARIDA patches carry an MGRS tile and an acquisition date, and the region and
season axes are defined on that metadata rather than on the official split. A
patch-level partition is not enough there: patches from one acquisition scene
are neither independent of nor exchangeable with patches from another, and the
official train, val and test splits each span several scenes of the same tiles.

Stage 2b therefore partitions each axis into four disjoint groups at *scene*
granularity: `train` (gradient updates), `monitor` (checkpoint selection),
`calibration` (conformal calibration) and `test` (the held-out region or
season). Stage 7 fits one model per axis from `train` and `monitor` alone, and
records the scheme name and digests of both id lists in the checkpoint's
`config.json`. Stage 5 then calibrates on `calibration` and evaluates on
`test`, so the held-out axis is absent from every stage of model fitting.

`18_audit.py --section holdout` verifies this end to end: for every
experiment it reads the checkpoint provenance, reconstructs the fitting sets,
and fails if a single test or calibration patch appears among them. Run it
after any change to the split generation or the training scripts.

Run against a fresh clone, the suite recomputes the numeric quantities in
full and records as notes the checks that need artifacts excluded from version
control: the checkpoint-provenance checks need `checkpoints/`, and the
component-size checks need `results/raw/`. Both become active once the
pipeline has produced them.

The scene is also the resampling unit for the reported intervals. A held-out
tile can contain as few as three acquisition scenes, and the patches of one
scene share illumination, sea state and annotator, so an image-level bootstrap
overstates the precision the design carries. `05_run_experiments.py
--bootstrap-clusters splits/marida_meta.csv` promotes the unit from the image
to the scene; without the flag every image is its own cluster and the two
estimators agree. `run_marida_holdout.sh` passes it for every MARIDA block, and
accepts `STAGE=experiments` to re-run stage 5 alone against existing
checkpoints, caches and tables.

## Reproducibility

All randomness is seeded. Split files are generated deterministically from the
base seed in `configs/datasets.yaml` and committed to version control; every
experiment references splits by file, never by re-sampling. Each result file
is accompanied by a `.meta.json` sidecar recording the producing script,
configuration, and git revision.

## License

Code is released under the MIT License (see `LICENSE`). Datasets are subject
to their own licenses and terms of use; consult each provider before use.
