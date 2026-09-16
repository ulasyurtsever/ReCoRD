#!/usr/bin/env python
"""Stage 27: the sampling distribution of a uniformly random triage ranking.

Under the oracle review model of ``record.evaluation.triage_curve`` the
residual is nonincreasing in the budget by construction, so an absolute
reduction is not evidence that a priority score works: a uniformly random
order already removes a fraction beta of the risk in expectation. Stage 9
therefore measures the marked-area ranking against the closed form

    E[residual(beta)] = (1 - beta) * residual(0),

rather than against the single sampled permutation the stage-5 triage CSVs
carry. That closed form is the correct centre. It is only a centre. Five of
the six deployment settings are single MARIDA partitions evaluated at one
operating point, and on those the one stored permutation departs from its own
expectation by as much as every advantage the marked-area score claims -- the
figure's own docstring puts the departure at up to 29% of the no-review rate.
A reader cannot tell from an expectation whether an advantage is real.

This stage supplies the missing scale. It redraws the random ranking
``--n-permutations`` times per setting and per class, runs the same
``triage_curve`` on each draw, and reports the 5th, 50th and 95th percentiles
of the residual at every budget in units of the no-review rate, so that the
marked-area curve is read against a band instead of against a line. It then
names, per setting, the budgets at which the marked-area curve falls below the
5th percentile -- the budgets at which the score beats chance by more than
permutation noise. Settings with no such budget are reported as such: that is
the finding, not a gap in the output.

The eleven residuals stored per curve are far too few to redraw anything, so
the per-image losses are reconstructed at the recorded operating point, the
way stage 22 reconstructs the deployed masks. The stage-5 ``.meta.json``
sidecar names the model, the scheme and the split keys; the experiment CSV
names the selected ``lam_index`` for each (seed, class); the stage-4 region
tables supply the loss curve that index selects. Nothing is recalibrated and
no threshold is re-derived. The cost is one region-table read per setting plus
``n_permutations`` argsorts per (seed, class) cell -- no model, image or
probability cache is touched, and the full sweep is minutes of CPU.

Two conventions are inherited rather than re-chosen, because changing either
would silently move the published curve:

* the residual's DENOMINATOR is the full test-set size, including images that
  carry no component of the class (``triage_curve`` maps their undefined loss
  to zero), and not the class-bearing count reported as ``n_test_images``;
* per-class curves are aggregated exactly as ``fig_triage_baselines``
  aggregates them: the mean over (seed, class) cells is taken FIRST and the
  ratio is formed between that mean and the mean no-review rate -- it is not
  the mean of per-cell ratios.

Both are verified against the stored triage CSVs before anything is written:
the reconstructed marked-area curve must reproduce the recorded one to
``--max-drift``, and the reconstructed per-image losses must reproduce the
recorded ``region_fnr`` and class-bearing image count. A reconstruction that
does not match the deployed numbers is refused rather than published.

Writes ``results/experiments/x12_triage_permutation_band.csv`` with one row per
(setting, class, budget). The pseudo-class ``all_classes`` carries the
setting-level aggregate, which is the curve the figure draws and shades; the
named classes carry the per-class bands behind it. Sums over the class column
would double-count. A ``.meta.json`` sidecar records the sweep.

Example
-------
    python scripts/27_triage_permutation_band.py \
        --alpha 0.2 --rho 0.5 --n-permutations 1000
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from record.evaluation import index_rows, triage_curve
from record.paths import results_dir, splits_dir
from record.splits import load_scheme
from record.provenance import git_revision

SCRIPTS = Path(__file__).resolve().parent
OUT_STEM = "x12_triage_permutation_band"
AGGREGATE_CLASS = "all_classes"
QUANTILES = (5.0, 50.0, 95.0)


def load_stage(filename: str, alias: str):
    """Import a numbered stage script for reuse.

    Stage module names begin with a digit and cannot be imported by name. The
    alternative is to copy the table loader and the setting list into this
    file, which is the code most likely to drift away from the tables and the
    figure the reported results are built from.
    """
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def truthy(value) -> bool:
    """Normalize a ``feasible`` cell.

    The column reads back as bool from a clean file and as a string where the
    file also carries undefined rows, so it is normalized rather than trusted
    -- the same convention stage 22 follows.
    """
    return str(value).strip().lower() == "true"


def permutation_rng(setting: str, class_name: str, seed_key: str,
                    alpha: float, rho: float) -> np.random.Generator:
    """A generator keyed by the cell it draws for.

    Python's ``hash()`` is salted per process, so a hash-derived seed would
    not reproduce across runs; the key is spelled out and digested instead,
    exactly as ``collect_triage`` seeds its single stored permutation.
    """
    key = (f"triage-permutation-band|{setting}|{class_name}|{seed_key}"
           f"|alpha={alpha:g}|rho={rho:g}").encode()
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def discover_settings(wanted) -> dict:
    """Locate the stage-5 sidecar of every requested triage setting.

    Returns ``{setting: (experiment_name, argv)}``. Two experiments claiming
    one setting is fatal for the same reason it is fatal in stage 9: a curve
    would silently average two distinct runs.
    """
    exp_dir = results_dir("experiments")
    found: dict[str, tuple[str, dict]] = {}
    for meta_path in sorted(exp_dir.glob("x3_triage__*.meta.json")):
        name = meta_path.name[: -len(".meta.json")]
        match = re.match(r"x3_triage__(\w+?)__", name)
        if match is None or match.group(1) not in wanted:
            continue
        setting = match.group(1)
        if setting in found:
            raise ValueError(
                f"triage setting '{setting}' is claimed by both "
                f"'{found[setting][0]}' and '{name}'; a band would average "
                "two distinct runs")
        found[setting] = (name, json.loads(meta_path.read_text())["argv"])
    return found


def test_row_indices(store, scheme: dict, argv: dict, seed_key: str) -> np.ndarray:
    """The test rows of one draw, reproducing stage 5's own resolution.

    Stage 5 indexes the scheme's test ids into the table when calibration and
    test tables coincide, and takes the whole target table otherwise. A
    nonzero ``--weight-holdout`` would then re-partition those rows with an
    RNG of its own; none of the triage settings uses one, and a setting that
    did could not be reconstructed from the CSV alone, so it is refused.
    """
    if float(argv.get("weight_holdout", 0.0) or 0.0) > 0:
        raise ValueError(
            "the experiment used --weight-holdout, which re-partitions the "
            "test set at run time; its per-image losses cannot be "
            "reconstructed from the recorded operating point")
    if argv["cal_datasets"] == argv["test_datasets"]:
        return index_rows(store.image_ids,
                          scheme["seeds"][seed_key][argv.get("test_split", "test")])
    return np.arange(len(store.image_ids))


def collect_cells(stage5, setting: str, name: str, argv: dict,
                  alpha: float, rho: float, max_drift: float) -> list[dict]:
    """Per-image losses and marked areas at the recorded operating point.

    One entry per (seed, class) cell that stage 5 actually emitted a triage
    curve for -- that is, per feasible ``region_crc`` record with a selected
    grid index, which is the exact condition ``collect_triage`` applies.
    """
    exp_path = results_dir("experiments") / f"{name}.csv"
    frame = pd.read_csv(exp_path)
    frame = frame[(frame["method"] == "region_crc")
                  & np.isclose(frame["alpha"].astype(float), alpha)
                  & np.isclose(frame["rho"].astype(float), rho)]
    frame = frame[frame["feasible"].map(truthy) & (frame["lam_index"] >= 0)]
    if frame.empty:
        raise ValueError(f"{name}: no feasible region_crc record at "
                         f"alpha={alpha:g}, rho={rho:g}")

    scheme = load_scheme(splits_dir() / f"{argv['scheme']}.json")
    store = stage5.TableStore(argv["model"], argv["test_datasets"])

    cells: list[dict] = []
    for _, rec in frame.iterrows():
        seed_key = str(rec["seed"])
        rows = test_row_indices(store, scheme, argv, seed_key)
        view = store.class_view(str(rec["class_name"]), rho)
        col = int(rec["lam_index"])
        losses = view["losses"][rows][:, col]
        area = view["marked_area"][rows][:, col]

        # The reconstruction has to land on the deployed operating point, not
        # near it. Both recorded quantities are recomputed from the same
        # column and compared before the column is used for anything else.
        defined = ~np.isnan(losses)
        fnr = float(losses[defined].mean()) if defined.any() else float("nan")
        drift = abs(fnr - float(rec["region_fnr"]))
        if not (drift <= max_drift):
            raise ValueError(
                f"{name} seed {seed_key} class {rec['class_name']}: "
                f"reconstructed region FNR {fnr:.9f} differs from the "
                f"recorded {float(rec['region_fnr']):.9f} by {drift:.2e}")
        if int(defined.sum()) != int(rec["n_test_images"]):
            raise ValueError(
                f"{name} seed {seed_key} class {rec['class_name']}: "
                f"{int(defined.sum())} class-bearing test images "
                f"reconstructed against {int(rec['n_test_images'])} recorded; "
                "the split or the table has moved under the CSV")
        cells.append({
            "setting": setting, "seed": seed_key,
            "class_name": str(rec["class_name"]),
            "losses": losses, "area": area, "n_test": int(losses.size),
        })
    return cells


def stored_curves(name: str, alpha: float, rho: float,
                  budgets: np.ndarray) -> dict:
    """The recorded triage curves of one setting, averaged over cells.

    Returns ``{(ranking, class_name): array over budgets}``, where the
    pseudo-class ``all_classes`` carries the setting-level aggregate. Every
    mean is the unweighted mean over the stored rows that fall in the group,
    which is the aggregation ``fig_triage_baselines`` performs: over seeds for
    a named class, over seeds and classes for the aggregate.
    """
    path = results_dir("experiments") / f"{name}_triage.csv"
    frame = pd.read_csv(path)
    frame = frame[(frame["method"] == "region_crc")
                  & np.isclose(frame["alpha"].astype(float), alpha)
                  & np.isclose(frame["rho"].astype(float), rho)]

    def _mean(sub) -> np.ndarray:
        mean = sub.groupby("budget")["residual_region_fnr"].mean()
        mean = mean.reindex(budgets)
        if mean.isna().any():
            raise ValueError(
                f"{name}: the recorded triage curve does not cover every "
                f"budget in {list(budgets)}")
        return mean.to_numpy(dtype=float)

    out: dict[tuple[str, str], np.ndarray] = {}
    for ranking, sub in frame.groupby("ranking"):
        out[(str(ranking), AGGREGATE_CLASS)] = _mean(sub)
        for class_name, cell in sub.groupby("class_name"):
            out[(str(ranking), str(class_name))] = _mean(cell)
    return out


def band_for_group(cells: list[dict], budgets: np.ndarray, setting: str,
                   alpha: float, rho: float, n_perm: int) -> tuple:
    """Realized marked-area curve and permutation draws for a set of cells.

    Both are means over the cells, taken before any division, so the ratio to
    the no-review rate is formed from two means and not as a mean of ratios.
    Each cell draws its own permutations, so the aggregate of draw ``r`` is an
    aggregate over independent random rankings -- which is what a per-class
    random triage policy would produce.
    """
    realized = np.zeros(budgets.size)
    draws = np.zeros((n_perm, budgets.size))
    for cell in cells:
        realized += triage_curve(cell["area"], cell["losses"], budgets)
        rng = permutation_rng(setting, cell["class_name"], cell["seed"],
                              alpha, rho)
        n = cell["losses"].size
        for r in range(n_perm):
            draws[r] += triage_curve(rng.permutation(n).astype(float),
                                     cell["losses"], budgets)
    return realized / len(cells), draws / len(cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", type=float, default=0.20,
                        help="risk level of the reported operating point")
    parser.add_argument("--rho", type=float, default=0.5,
                        help="capture level of the reported operating point")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="random orders drawn per (seed, class) cell; the "
                             "cost is one argsort of the test set per draw")
    parser.add_argument("--settings", nargs="+", default=None,
                        help="subset of the stage-9 triage settings (default: "
                             "all of them, in the figure's order)")
    parser.add_argument("--max-drift", type=float, default=1e-9,
                        help="tolerance on reproducing the recorded region "
                             "FNR and the recorded marked-area triage curve; "
                             "raise it only to diagnose a mismatch, never to "
                             "publish through one")
    parser.add_argument("--out-stem", default=OUT_STEM)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage5 = load_stage("05_run_experiments.py", "stage5")
    stage9 = load_stage("09_make_figures.py", "stage9")
    labels = dict(stage9.TRIAGE_SETTINGS)
    wanted = list(args.settings) if args.settings else list(labels)
    unknown = [s for s in wanted if s not in labels]
    if unknown:
        print(f"ERROR: unknown triage setting(s) {unknown}; "
              f"stage 9 knows {list(labels)}")
        return 1

    found = discover_settings(set(wanted))
    missing = [s for s in wanted if s not in found]
    if missing:
        print(f"ERROR: no x3_triage__*.meta.json for setting(s) {missing}")
        return 1

    records: list[dict] = []
    summary: list[tuple] = []
    for setting in wanted:
        name, argv = found[setting]
        cells = collect_cells(stage5, setting, name, argv,
                              args.alpha, args.rho, args.max_drift)
        classes = sorted({c["class_name"] for c in cells})
        print(f"{setting}: {len(cells)} (seed, class) cells over "
              f"{len(classes)} class(es), {cells[0]['n_test']} test images, "
              f"{args.n_permutations} permutations each")

        # The budget grid is taken from the stored curves rather than
        # re-chosen, so the band lands on the figure's own x positions.
        stored_frame = pd.read_csv(
            results_dir("experiments") / f"{name}_triage.csv")
        budgets = np.sort(stored_frame["budget"].astype(float).unique())
        stored = stored_curves(name, args.alpha, args.rho, budgets)

        groups = [(AGGREGATE_CLASS, cells)]
        groups += [(c, [x for x in cells if x["class_name"] == c])
                   for c in classes]
        for class_name, group in groups:
            realized, draws = band_for_group(
                group, budgets, setting, args.alpha, args.rho,
                args.n_permutations)
            base = float(realized[0])
            # Every ranking leaves the whole test set unreviewed at beta = 0,
            # so the no-review rate cannot depend on the order. If it did, the
            # normalization below would not be shared by the curve and band.
            spread = float(np.max(np.abs(draws[:, 0] - base)))
            if spread > 1e-12:
                print(f"ERROR: {setting}/{class_name}: the no-review rate "
                      f"varies across permutations by {spread:.2e}")
                return 1
            if base <= 0:
                print(f"  {class_name}: no-review rate is zero, no band "
                      "(the figure skips this curve for the same reason)")
                continue
            q = np.percentile(draws, QUANTILES, axis=0) / base
            realized_rel = realized / base
            closed = 1.0 - budgets
            single = stored.get(("random", class_name))
            single_rel = None if single is None else single / base

            # The reconstruction is only usable if it reproduces the curve
            # stage 5 recorded for this very group, on the same aggregation.
            recorded = stored.get(("area", class_name))
            if recorded is None:
                print(f"ERROR: {setting}: the stored triage CSV carries no "
                      f"marked-area curve for '{class_name}'")
                return 1
            drift = float(np.max(np.abs(recorded - realized)))
            if not (drift <= args.max_drift):
                print(f"ERROR: {setting}/{class_name}: the reconstructed "
                      f"marked-area curve differs from the recorded one by "
                      f"{drift:.2e} (> {args.max_drift:g}); the operating "
                      "point or the aggregation does not match")
                return 1

            below = budgets[realized_rel < q[0]]
            if class_name == AGGREGATE_CLASS:
                summary.append((setting, labels[setting], base, len(group),
                                below, budgets, q, single_rel))
            for i, beta in enumerate(budgets):
                records.append({
                    "setting": setting, "label": labels[setting],
                    "experiment": name, "model": argv["model"],
                    "scheme": argv["scheme"],
                    "class_name": class_name, "n_cells": len(group),
                    "n_test_images_full": int(group[0]["n_test"]),
                    "alpha": args.alpha, "rho": args.rho,
                    "n_permutations": args.n_permutations,
                    "budget": float(beta),
                    "no_review_rate": base,
                    "residual_area": float(realized[i]),
                    "area_rel": float(realized_rel[i]),
                    "q05_rel": float(q[0, i]),
                    "q50_rel": float(q[1, i]),
                    "q95_rel": float(q[2, i]),
                    "closed_form_rel": float(closed[i]),
                    "single_random_rel": (float(single_rel[i])
                                          if single_rel is not None else np.nan),
                    "below_band": bool(realized_rel[i] < q[0, i]),
                })

    if not records:
        print("no setting produced a band")
        return 1

    out = pd.DataFrame(records)
    exp_dir = results_dir("experiments")
    out_csv = exp_dir / f"{args.out_stem}.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(out)} rows)")
    (exp_dir / f"{args.out_stem}.meta.json").write_text(json.dumps({
        "git_revision": git_revision(),
        "alpha": args.alpha, "rho": args.rho,
        "n_permutations": args.n_permutations,
        "settings": {s: found[s][0] for s in wanted},
        "aggregate_class": AGGREGATE_CLASS,
        "quantiles": list(QUANTILES),
        "denominator": "full test-set size (images without components "
                       "contribute zero loss, as in triage_curve)",
        "aggregation": "mean over (seed, class) cells before dividing by the "
                       "no-review rate, matching fig_triage_baselines",
        "max_drift": args.max_drift,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=1))

    print(f"\nSetting-level bands at alpha={args.alpha:g}, rho={args.rho:g} "
          f"({args.n_permutations} permutations, residuals in units of the "
          "no-review rate)\n")
    print(f"{'setting':<34}{'cells':>6}{'no-review':>11}"
          f"{'width@0.25':>12}{'single perm':>13}{'beats band':>12}")
    for setting, label, base, n_cells, below, budgets, q, single in summary:
        mid = int(np.argmin(np.abs(budgets - 0.25)))
        width = q[2, mid] - q[0, mid]
        dev = (float(np.max(np.abs(single - (1.0 - budgets))))
               if single is not None else float("nan"))
        print(f"{label:<34}{n_cells:>6}{base:>11.4f}{width:>12.4f}"
              f"{dev:>13.4f}{len(below):>12}")
    print("\n  'width@0.25' is the 5th-to-95th percentile spread at a 25% "
          "budget; 'single perm' is how far the one permutation stored in the "
          "stage-5 CSV sits from the closed-form line.\n")

    for setting, label, base, n_cells, below, budgets, q, single in summary:
        if below.size:
            budget_list = ", ".join(f"{b:g}" for b in below)
            print(f"{label}: the marked-area ranking falls below the 5th "
                  f"percentile of the random band at budgets {budget_list}.")
        else:
            # A negative result is the finding here, not a missing row.
            print(f"{label}: the marked-area ranking never falls below the "
                  "5th percentile of the random band, at any budget. Its "
                  "advantage over a random ordering is within permutation "
                  "noise for this setting.")

    n_beat = sum(1 for row in summary if row[4].size)
    print(f"\n{n_beat} of {len(summary)} settings beat the permutation band "
          "at one budget or more.")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
