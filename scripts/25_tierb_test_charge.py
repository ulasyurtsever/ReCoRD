#!/usr/bin/env python
"""Stage 25: who pays for the tier-B test point, and what that charge decides.

Weighted CRC has to charge the unseen test point a weight. The published runs
charge it the clipping ceiling (``test_weight=clip[1]`` in stage 5): the
largest weight the estimator is allowed to return, hence an upper bound on
whatever the true test point would have been given. That is safe, and the
method section says in as many words that it is stricter than necessary. It is
also what makes the tier-B rows vacuous, because the conservative mass it buys,

    p_test = kappa / (sum_i w_i + kappa),

exceeds alpha on its own once the calibration weights have collapsed onto the
floor: with w_i = ell for every i the sum is ell*n, and at
(ell, kappa, n) = (0.05, 20, 200) the mass is 0.66 against an alpha of at most
0.2. A referee is entitled to ask whether the reported vacuity is a property of
importance-weighted CRC or an artifact of that ceiling. Nothing in the
published files answers it, because the pipeline never scores the estimated
ratio anywhere except on the source side. This stage answers it, with four arms
written side by side into one CSV so that no one has to join files:

    published/ceiling   the published fit, the published ceiling charge,
                        recomputed here from the same tables
    published/self      the published fit, the test point charged its own
                        w-hat(X_test) -- the alternative the method section
                        names and never runs
    unfiltered/ceiling  the ratio refitted with the source side unfiltered,
                        ceiling charge
    unfiltered/self     that refit, self charge

and, for each fit, the distribution of w-hat evaluated ON THE TARGET
EMBEDDINGS. That last quantity is the one the reader needs. If target-side
w-hat sits at the ceiling for most target images, the ceiling charge is a
faithful bound on what the test point would have been given and the vacuity is
a property of the shift; if it sits far below, the ceiling is a formality that
decides the result by itself.

The self charge, and why it is not a deployable procedure
---------------------------------------------------------
w-hat(X_test) is defined only once X_test is known, so the arm is evaluated
transductively, exactly as the published tier-B protocol already treats the
target pool: every target image plays the test point in turn, is charged the
weight the fitted ratio gives it, gets its own threshold from that charge, and
is scored at that threshold. Two things follow. The density ratio was fitted on
a target pool containing the image it is then applied to -- the same
transductive dependence ``--weight-holdout`` exists to remove in stage 5 -- and
there is no single calibrated threshold at the end, only a family of them, one
per target image, so the reported risk averages over thresholds rather than
measuring one. The arm is therefore an optimistic bound on what charging the
true weight could buy, not a procedure anyone can ship. That is the right
instrument for the referee's question: if even this reading stays vacuous, the
ceiling was not the reason.

The unfiltered refit
--------------------
The published pipeline restricts the source side of the domain classifier to
calibration images that *contain* the class, because those are the images the
weighted risk sums over, while leaving the target side unconditioned. What
comes back is dQ(x) / dP(x | c present), not the class-conditional ratio the
weighted-CRC statement is written for. Stage 23 attacks that mismatch from the
target side, restricting the target pool to class-bearing images; this stage
attacks it from the source side, refitting with every calibration image of the
seed's split and evaluating the fitted ratio at the class-bearing calibration
points the risk actually sums over. Neither refit is the correct estimator --
that one conditions both sides -- but the two bracket the direction and the
size of the error.

Trust
-----
The ceiling arm is recomputed here rather than joined in from the published
CSVs, so that both charges come out of one code path and one fit. It is then
checked row by row against the published ``e4_tierB__*`` numbers (and, for the
narrower ceilings, against the ``c1_clip*__*`` ablation) on lambda,
feasibility, ESS, p_test, realized risk, marked area and the two sample sizes,
and the stage aborts on the first disagreement, before writing anything, so
that a file which disagrees with the published runs never reaches
``results/``. Without that check the whole
comparison would rest on the assumption that this file reimplements stage 5
faithfully; with it, the assumption is tested on every run. A published run is
accepted as the reference for a cell only if its ``.meta.json`` agrees on
model, embedding, scheme, both dataset pools, estimator and clip interval, and
records neither holdout, so the reference lookup cannot silently match the
wrong file.

Cost
----
Two logistic fits per (condition, class, seed) -- one per source-side pool,
each scoring the calibration points and the whole target pool from a single fit
-- so 600 fits for four ACDC conditions at three classes and 25 seeds, a few
minutes of CPU and no cache beyond the stage-4 tables and the embeddings.
Nothing else is expensive: the weights depend on neither alpha, rho nor the
clip, so one fit serves the whole grid, and the per-target-point threshold
search is one array operation over the few distinct charges the clip leaves
behind.

Nothing here is random. The density ratio is fitted with ``random_state=0``,
the value the published runs used, because the ceiling arm has to reproduce
them; no other step draws, so this stage derives no seeds.

Writes ``results/experiments/x10_tierb_test_charge__<model>.csv``, one row per
(condition, class, seed, clip, rho, alpha, arm), plus a ``.meta.json`` sidecar,
and prints one line per (condition, clip, arm) carrying the informative
fraction and the median target-side weight. The default name does not carry the
embedding key, so a second embedding needs ``--out-name`` or it overwrites the
first.

Examples
--------
    python scripts/25_tierb_test_charge.py \
        --model segformer_b2_cityscapes --scheme cityscapes_val_half \
        --cal-datasets cityscapes_val --embedding dinov2_vitb14 \
        --conditions fog night rain snow --max-seeds 25

LoveDA, one direction per invocation -- the target pool there is a different
dataset rather than a condition of the same one, so it arrives through
``--dataset-template`` and the direction is spelled as the condition:

    python scripts/25_tierb_test_charge.py \
        --model segformer_b2_loveda_urban --scheme loveda_urban_val_half \
        --cal-datasets loveda_Val_Urban --embedding dinov2_vitb14 \
        --conditions urban2rural --dataset-template loveda_Val_Rural \
        --classes building water --max-seeds 25 \
        --out-name x10_tierb_test_charge__loveda_urban2rural__dinov2_vitb14
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from record.evaluation import index_rows
from record.grid import LAMBDA_GRID
from record.paths import results_dir, splits_dir
from record.splits import load_scheme
from record.weights import clip_weights, logistic_density_ratio

SCRIPTS = Path(__file__).resolve().parent
ALPHAS = (0.05, 0.10, 0.20)
RHOS = (0.5,)
CLIPS = ((0.05, 2.0), (0.05, 5.0), (0.05, 20.0))
UNCLIPPED = (1e-12, 1e12)
CRITICAL = ("person", "rider", "bicycle")
LOSS_BOUND = 1.0

# Candidate names of the published run a cell must reproduce. Several are tried
# because the ceiling that stage 5 shipped with (kappa = 20) lives in the e4
# block while the narrower ceilings only exist in the c1 clip ablation, and the
# LoveDA block names itself without the model. Matching the wrong file is
# prevented by the sidecar check in ``load_reference``, not by the ordering.
REFERENCE_TEMPLATES = (
    "e4_tierB__{cond}__{model}__{embedding}",
    "c1_clip{kappa}__{cond}__{model}__{embedding}",
    "l4_tierB__{cond}__{embedding}",
)

# Published column -> column of this stage. The published ``controlled_risk``
# is deliberately absent: stage 5 fills it with the realized test risk, not
# with the risk the threshold search controlled, so it duplicates region_fnr
# for weighted CRC and would test nothing that region_fnr does not.
COMPARE = (
    ("lam", "lam_mean"),
    ("weight_ess", "weight_ess"),
    ("weight_p_test", "p_test_mean"),
    ("region_fnr", "region_fnr"),
    ("marked_area_fraction", "marked_area_fraction"),
    ("n_test_images", "n_test_images"),
    ("n_cal_images", "n_cal"),
)
# Loose enough to absorb a float32 marked-area mean summed in a different
# order, tight enough that a different fit, a different pool or a different
# clip cannot slip through.
REF_RTOL = 1e-6
REF_ATOL = 1e-9


def load_stage(filename: str, alias: str):
    """Import a numbered stage script for reuse.

    Stage module names begin with a digit and cannot be imported by name. The
    alternative is to copy the table loader into this file, which is the code
    most likely to drift away from the tables the reported results are built
    from.
    """
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_clip(text: str) -> tuple[float, float]:
    """Parse one ``lo:hi`` clip interval from the command line."""
    lo, _, hi = text.partition(":")
    if not hi:
        raise argparse.ArgumentTypeError(f"clip '{text}' is not 'lo:hi'")
    clip = (float(lo), float(hi))
    if not 0 < clip[0] <= clip[1]:
        raise argparse.ArgumentTypeError(f"invalid clip interval {clip}")
    return clip


def fitted_ratio(source_emb, target_emb, eval_emb, c_reg=1.0, random_state=0):
    """The unclipped logistic density ratio at chosen evaluation points.

    This is ``record.weights.logistic_density_ratio`` with two things deferred.
    The clip is deferred so that one fit serves every clip interval, and it is
    inverted by construction: ``UNCLIPPED`` lies far outside anything the
    classifier can return, so ``clip_weights`` is the identity on its output.
    The evaluation points are passed in so that one fit serves both sides of
    the ratio -- the calibration points the risk sums over and the target pool
    the test point is drawn from -- which is what halves the cost of the stage.
    Copying the estimator body here instead would let this file drift away from
    the estimator the published numbers came out of, which is the one thing the
    reproduction check could not detect.
    """
    return logistic_density_ratio(source_emb, target_emb, clip=UNCLIPPED,
                                  c_reg=c_reg, random_state=random_state,
                                  eval_emb=eval_emb)


def weight_stats(prefix: str, raw: np.ndarray, clip: tuple[float, float]):
    """Clipped weights and their summary, for one side of the ratio."""
    w = clip_weights(raw, clip)
    stats = {
        f"{prefix}_w_min": float(w.min()),
        f"{prefix}_w_median": float(np.median(w)),
        f"{prefix}_w_mean": float(w.mean()),
        f"{prefix}_w_max": float(w.max()),
        # The clipped fractions are read off the raw ratio rather than off the
        # clipped weights: a weight that happens to land on a bound without
        # having been moved there is not a constraint that bound imposed, and
        # the whole question here is how often the bound binds.
        f"{prefix}_frac_floor": float(np.mean(raw <= clip[0])),
        f"{prefix}_frac_ceiling": float(np.mean(raw >= clip[1])),
        f"{prefix}_raw_median": float(np.median(raw)),
        f"{prefix}_raw_max": float(raw.max()),
    }
    return w, stats


def charge_thresholds(cal_losses, cal_weights, test_weights, alpha):
    """Weighted-CRC threshold index for each of several test-point charges.

    ``record.crc.weighted_crc_threshold`` answers this for one charge. The self
    arm needs one answer per target image, and looping over a thousand images
    across the alpha grid would be the entire runtime of the stage, so the same
    arithmetic is done here in one pass, in the same order and with the same
    ``risk <= alpha`` comparison, so that a constant charge reproduces the
    library exactly. Clipping leaves only a handful of distinct charges --
    every image whose raw ratio falls outside the interval lands on a bound --
    so the risk curve is built once per distinct charge and read back through
    the inverse index.

    Returns the selected grid index, whether the level was attainable at all,
    and the controlled risk at the selection, one entry per charge.
    """
    losses = np.asarray(cal_losses, dtype=np.float64)
    w = np.asarray(cal_weights, dtype=np.float64)
    charges = np.asarray(test_weights, dtype=np.float64)
    if np.isnan(losses).any():
        raise ValueError("calibration losses contain NaN; filter undefined rows first")
    if (w < 0).any() or (charges < 0).any():
        raise ValueError("weights must be nonnegative")

    weighted = (losses * w[:, None]).sum(axis=0)
    uniq, inverse = np.unique(charges, return_inverse=True)
    total = w.sum() + uniq
    risk = weighted[None, :] / total[:, None] + (uniq / total)[:, None] * LOSS_BOUND
    attainable = risk <= alpha
    feasible = attainable.any(axis=1)
    idx = np.where(feasible, attainable.argmax(axis=1), LAMBDA_GRID.size - 1)
    return (idx[inverse], feasible[inverse],
            risk[np.arange(uniq.size), idx][inverse])


def realized(test_losses, test_area, lam_index):
    """Realized risk and marked area when each target image has its own lambda.

    With a constant ``lam_index`` this is ``evaluate_at_threshold`` on the same
    arrays; the reproduction check against the published runs is what keeps the
    two definitions from drifting, which is why the ceiling arm does not call
    the library function separately. Images without ground-truth components
    carry no loss and are excluded from the risk average, exactly as there,
    while every image counts toward the marked area.
    """
    rows = np.arange(test_losses.shape[0])
    loss_col = test_losses[rows, lam_index]
    defined = ~np.isnan(loss_col)
    if not defined.any():
        raise ValueError("no target image contains ground-truth components")
    return (float(loss_col[defined].mean()),
            float(test_area[rows, lam_index].mean()),
            int(defined.sum()))


def load_reference(cond, clip, args, test_datasets):
    """The published stage-5 run whose numbers the ceiling arm must reproduce.

    Returns ``(name, frame)`` or ``(None, None)``. The sidecar decides: a run
    is a reference for this cell only if it was produced from the same model,
    embedding, scheme and dataset pools with the same estimator and the same
    clip interval and neither holdout. That check, not the order of the
    templates, is what makes trying several names safe.
    """
    exp_dir = results_dir("experiments")
    for template in args.reference_templates:
        name = template.format(cond=cond, model=args.model,
                               embedding=args.embedding, kappa=f"{clip[1]:g}")
        csv_path, meta_path = exp_dir / f"{name}.csv", exp_dir / f"{name}.meta.json"
        if not (csv_path.exists() and meta_path.exists()):
            continue
        argv = json.loads(meta_path.read_text()).get("argv", {})
        matches = (
            argv.get("model") == args.model
            and argv.get("embedding") == args.embedding
            and argv.get("scheme") == args.scheme
            and list(argv.get("cal_datasets") or []) == list(args.cal_datasets)
            and list(argv.get("test_datasets") or []) == list(test_datasets)
            and argv.get("weight_estimator") == "logistic"
            and argv.get("loss", "region") == "region"
            and float(argv.get("weight_clip_min", -1.0)) == clip[0]
            and float(argv.get("weight_clip_max", -1.0)) == clip[1]
            and float(argv.get("weight_holdout") or 0.0) == 0.0
            and float(argv.get("weight_source_holdout") or 0.0) == 0.0)
        if matches:
            return name, pd.read_csv(csv_path)
    return None, None


def check_reference(mine: pd.DataFrame, published: pd.DataFrame, name: str) -> tuple[int, float]:
    """Compare the recomputed ceiling arm against a published run, or abort.

    Returns the number of rows compared and the worst relative deviation seen.
    Rows of this stage with no published counterpart are reported by the caller
    and not treated as failures -- the published grid need not cover an alpha
    or a rho this invocation asked for -- but a counterpart that disagrees is
    fatal, because every conclusion drawn from the other arms rests on the
    ceiling arm being the published procedure.
    """
    keys = ["seed", "class_name", "alpha", "rho"]
    pub = published[published["method"] == "weighted_crc"].copy()
    missing = [c for c, _ in COMPARE if c not in pub.columns]
    if missing:
        raise RuntimeError(
            f"published run '{name}' is missing columns {missing}; it was "
            "written by a different version of stage 5 and cannot be used as "
            "a reference")
    pub = pub[keys + ["feasible"] + [c for c, _ in COMPARE]].rename(
        columns={c: f"pub_{c}" for c, _ in COMPARE} | {"feasible": "pub_feasible"})
    for frame in (mine, pub):
        frame["seed"] = frame["seed"].astype(int)
        frame["alpha"] = frame["alpha"].astype(float).round(6)
        frame["rho"] = frame["rho"].astype(float).round(6)

    merged = mine.merge(pub, on=keys, how="inner")
    if merged.empty:
        return 0, 0.0

    worst = 0.0
    failures = []
    for pub_col, my_col in COMPARE:
        a = merged[my_col].to_numpy(dtype=float)
        b = merged[f"pub_{pub_col}"].to_numpy(dtype=float)
        bad = ~np.isclose(a, b, rtol=REF_RTOL, atol=REF_ATOL, equal_nan=True)
        scale = np.maximum(np.abs(b), 1e-12)
        worst = max(worst, float(np.nanmax(np.abs(a - b) / scale)))
        if bad.any():
            row = merged.loc[bad].iloc[0]
            failures.append(
                f"  {pub_col}: {int(bad.sum())}/{len(merged)} rows differ, "
                f"e.g. seed={row['seed']} class={row['class_name']} "
                f"alpha={row['alpha']} rho={row['rho']}: "
                f"this stage {float(row[my_col]):.10g} vs published "
                f"{float(row[f'pub_{pub_col}']):.10g}")
    flip = merged["frac_feasible"].to_numpy(dtype=float) != \
        merged["pub_feasible"].to_numpy(dtype=bool).astype(float)
    if flip.any():
        failures.append(f"  feasible: {int(flip.sum())}/{len(merged)} rows differ")

    if failures:
        raise RuntimeError(
            f"the ceiling arm does not reproduce '{name}':\n"
            + "\n".join(failures)
            + "\n\nThe comparison between charges is only meaningful if the "
              "ceiling arm is the published procedure, so this is fatal. "
              "Check first whether the stage-4 tables, the embeddings or "
              "scikit-learn have changed since that run was written; rerun "
              "stage 5 for this cell if they have.")
    return len(merged), worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--cal-datasets", nargs="+", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--conditions", nargs="+",
                        default=["fog", "night", "rain", "snow"])
    parser.add_argument(
        "--dataset-template", default="acdc_{cond}_train,acdc_{cond}_val",
        help="comma-separated target dataset keys, with {cond} substituted; "
             "the escape hatch for pools that are not ACDC conditions and for "
             "smoke tests on synthetic tables")
    parser.add_argument("--classes", nargs="+", default=list(CRITICAL))
    parser.add_argument("--clips", nargs="+", type=parse_clip, default=list(CLIPS),
                        metavar="LO:HI",
                        help="clip intervals; the ceiling doubles as the "
                             "published test-point charge, so it names the arm "
                             "this stage is arguing with")
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    parser.add_argument("--rhos", nargs="+", type=float, default=list(RHOS))
    parser.add_argument("--cal-split", default="calibration",
                        help="scheme entry key holding the source calibration ids")
    parser.add_argument("--max-seeds", type=int, default=25)
    parser.add_argument("--reference-templates", nargs="*",
                        default=list(REFERENCE_TEMPLATES),
                        help="names of the published runs the ceiling arm is "
                             "checked against; pass the flag with no value to "
                             "skip the check, which is what a smoke test on "
                             "synthetic tables does")
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    stage5 = load_stage("05_run_experiments.py", "stage5")
    scheme = load_scheme(splits_dir() / f"{args.scheme}.json")
    seeds = list(scheme["seeds"].items())[: args.max_seeds]
    cal_store = stage5.TableStore(args.model, args.cal_datasets)

    rows: list[dict] = []
    references: list[dict] = []
    for cond in args.conditions:
        test_datasets = [k.format(cond=cond)
                         for k in args.dataset_template.split(",")]
        test_store = stage5.TableStore(args.model, test_datasets)
        keys = list(dict.fromkeys(args.cal_datasets + test_datasets))
        emb_ids, emb = stage5.load_embeddings(args.embedding, keys)

        # The target pool is every target image, class-bearing or not, which is
        # what stage 5 fits against and what stage 23 questions. Keeping it
        # unchanged here is deliberate: this stage varies the charge and the
        # source side, so the pool has to stay where the published runs put it.
        pool = np.arange(len(test_store.image_ids))
        tgt_emb = emb[index_rows(emb_ids, list(test_store.image_ids))]

        for class_name in args.classes:
            views = {rho: (cal_store.class_view(class_name, rho),
                           test_store.class_view(class_name, rho))
                     for rho in args.rhos}

            for seed_key, entry in seeds:
                cal_rows = index_rows(cal_store.image_ids, entry[args.cal_split])
                cal_def = None
                for rho, (cal_view, _) in views.items():
                    defined = stage5.defined_rows(cal_view["losses"], cal_rows)
                    # An image either contains components of the class or it
                    # does not; rho decides how many of them count as missed,
                    # never whether the image has a loss at all. One fit per
                    # seed serves every rho on that assumption, so it is
                    # checked here rather than assumed.
                    if cal_def is None:
                        cal_def = defined
                    elif not np.array_equal(cal_def, defined):
                        raise RuntimeError(
                            f"the defined calibration rows for {class_name} "
                            f"differ between rho values at seed {seed_key}; "
                            "the shared weight fit would be wrong")
                if cal_def.size == 0:
                    continue

                cal_def_emb = emb[index_rows(
                    emb_ids, [cal_store.image_ids[r] for r in cal_def])]
                cal_all_emb = emb[index_rows(
                    emb_ids, [cal_store.image_ids[r] for r in cal_rows])]
                # One fit per source pool, scoring both sides at once: the
                # evaluation block is the calibration points the risk sums
                # over followed by the whole target pool.
                block = np.vstack([cal_def_emb, tgt_emb])
                split = int(cal_def.size)
                fits = {
                    "published": (fitted_ratio(cal_def_emb, tgt_emb, block), split),
                    "unfiltered": (fitted_ratio(cal_all_emb, tgt_emb, block),
                                   int(cal_rows.size)),
                }

                for fit_name, (raw, n_fit) in fits.items():
                    raw_cal, raw_tgt = raw[:split], raw[split:]
                    for clip in args.clips:
                        w_cal, cal_stats = weight_stats("cal", raw_cal, clip)
                        w_tgt, tgt_stats = weight_stats("tgt", raw_tgt, clip)
                        w_sum = float(w_cal.sum())
                        w_sq = float((w_cal ** 2).sum())
                        base = dict(
                            condition=cond, class_name=class_name,
                            seed=int(seed_key), fit=fit_name,
                            n_cal=int(cal_def.size), n_cal_fit=n_fit,
                            n_target=int(pool.size),
                            c_low=clip[0], kappa=clip[1],
                            ratio=clip[1] / clip[0],
                            weight_sum=w_sum, weight_ess=w_sum ** 2 / w_sq,
                            **cal_stats, **tgt_stats)
                        charges = {
                            "ceiling": np.full(pool.size, clip[1]),
                            "self": w_tgt,
                        }

                        for rho, (cal_view, test_view) in views.items():
                            cal_losses = cal_view["losses"][cal_def]
                            test_losses = test_view["losses"][pool]
                            test_area = test_view["marked_area"][pool]
                            for alpha in args.alphas:
                                for charge_name, charge in charges.items():
                                    idx, feasible, risk = charge_thresholds(
                                        cal_losses, w_cal, charge, alpha)
                                    fnr, area, n_def = realized(
                                        test_losses, test_area, idx)
                                    lam = LAMBDA_GRID[idx]
                                    p_test = charge / (w_sum + charge)
                                    rows.append(dict(
                                        base, rho=rho, alpha=alpha,
                                        charge=charge_name,
                                        arm=f"{fit_name}/{charge_name}",
                                        p_test_mean=float(p_test.mean()),
                                        p_test_median=float(np.median(p_test)),
                                        p_test_min=float(p_test.min()),
                                        p_test_max=float(p_test.max()),
                                        # The effective sample size of the
                                        # (n+1)-point weighting the guarantee
                                        # is stated over, alongside the
                                        # calibration-only ESS stage 5 reports.
                                        ess_with_test=float(np.mean(
                                            (w_sum + charge) ** 2
                                            / (w_sq + charge ** 2))),
                                        lam_mean=float(lam.mean()),
                                        lam_median=float(np.median(lam)),
                                        lam_min=float(lam.min()),
                                        lam_max=float(lam.max()),
                                        n_distinct_lam=int(np.unique(idx).size),
                                        # Informative means the threshold stops
                                        # short of the top of the grid, where
                                        # the mask covers everything and the
                                        # guarantee is bought by marking the
                                        # whole image. Same definition as
                                        # stage 15's lam < 1.
                                        frac_informative=float(np.mean(
                                            idx < LAMBDA_GRID.size - 1)),
                                        frac_feasible=float(feasible.mean()),
                                        selection_risk_mean=float(risk.mean()),
                                        region_fnr=fnr,
                                        marked_area_fraction=area,
                                        n_test_images=n_def,
                                        model=args.model,
                                        embedding=args.embedding,
                                        scheme=args.scheme))
        print(f"{cond}: {len(rows)} rows so far")

    frame = pd.DataFrame(rows)

    # The reproduction check. Only the published fit under the ceiling charge
    # has a counterpart in the shipped files; the other three arms are new.
    for cond in args.conditions:
        test_datasets = [k.format(cond=cond)
                         for k in args.dataset_template.split(",")]
        for clip in args.clips:
            mine = frame[(frame["condition"] == cond)
                         & (frame["arm"] == "published/ceiling")
                         & np.isclose(frame["kappa"], clip[1])
                         & np.isclose(frame["c_low"], clip[0])].copy()
            if mine.empty:
                continue
            name, published = load_reference(cond, clip, args, test_datasets)
            if published is None:
                print(f"WARNING: no published reference for {cond} at clip "
                      f"{clip}; its ceiling arm is unchecked")
                references.append({"condition": cond, "kappa": clip[1],
                                   "reference": None, "n_rows": 0})
                continue
            n_rows, worst = check_reference(mine, published, name)
            unmatched = len(mine) - n_rows
            note = f" ({unmatched} rows had no counterpart)" if unmatched else ""
            print(f"reference {name}: {n_rows} rows agree, worst relative "
                  f"deviation {worst:.2e}{note}")
            references.append({"condition": cond, "kappa": clip[1],
                               "reference": name, "n_rows": n_rows,
                               "worst_rel_dev": worst})

    name = args.out_name or f"x10_tierb_test_charge__{args.model}"
    out_dir = results_dir("experiments")
    path = out_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    (out_dir / f"{name}.meta.json").write_text(json.dumps({
        "name": name, "argv": vars(args), "n_records": len(frame),
        "references": references,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    print(f"\nwrote {path}  ({len(frame)} rows)")

    # The answer, one line per (condition, clip, arm). ``w~target`` is the
    # median of the estimated weight over the target images, the quantity the
    # published pipeline never computes; ``at kappa`` and ``at ell`` are the
    # fractions of target images whose raw ratio leaves the clip interval. If
    # ``at kappa`` is near one the ceiling charge is a faithful bound and the
    # vacuity is a property of the shift; if it is near zero the ceiling
    # decided the result on its own.
    pd.set_option("display.width", 200)
    summary = frame.groupby(["condition", "kappa", "arm"]).agg(
        informative=("frac_informative", "mean"),
        w_target_median=("tgt_w_median", "median"),
        at_kappa=("tgt_frac_ceiling", "mean"),
        at_floor=("tgt_frac_floor", "mean"),
        w_cal_median=("cal_w_median", "median"),
        p_test=("p_test_mean", "mean"),
        ess=("weight_ess", "mean"),
        fnr=("region_fnr", "mean"),
        area=("marked_area_fraction", "mean"),
    ).reset_index()
    print("\ninformative fraction and target-side weights, by arm:")
    print(summary.round(4).to_string(index=False))

    print("\ninformative fraction (rows: condition x kappa, columns: arm):")
    print(frame.pivot_table(index=["condition", "kappa"], columns="arm",
                            values="frac_informative").round(4).to_string())

    print("\nNote: the domain classifier is fitted in sample, so it separates "
          "its own training points; that pushes source-side weights toward the "
          "floor and target-side weights toward the ceiling at the same time. "
          "Read the target-side ceiling fraction as an upper bound on how "
          "often the ceiling truly binds, and against the cross-fitted "
          "estimator of stage 15.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
