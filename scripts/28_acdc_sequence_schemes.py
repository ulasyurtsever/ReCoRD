#!/usr/bin/env python
"""Stage 28: sequence-disjoint tier-A target-calibration schemes for ACDC.

WHY. The published tier-A schemes (``acdc_<cond>_targetcal<n>``, stage 2) draw
the n_t labeled target frames uniformly from the same pool they are then tested
against. ACDC frames come in driving sequences of tens to a hundred-odd
consecutive frames, so a uniform draw puts a calibration frame in nearly every
sequence: at n_t = 25 between 91% and 96% of test frames sit in a sequence that
also supplied a calibration frame. Those test frames are not exchangeable with
the calibration frames in any interesting sense -- they are seconds of driving
away, under the same fog bank, with the same vehicles in view -- so the tier-A
number partly measures memorization of the sequence rather than transfer to the
condition. This script emits the pessimistic counterpart: whole sequences are
assigned to the calibration side, and no test frame may come from a sequence
that contributed a calibration frame.

The existing schemes are never touched. Outputs are written alongside them as
``acdc_<cond>_targetcal<N>_seqdisjoint.json`` in exactly the stage-2 scheme
format (``target_calibration`` / ``test`` per seed), so stage 5 consumes them
with no change:

    python scripts/05_run_experiments.py --name e2_fog_tierA25_seqdisjoint \
        --model segformer_b2_cityscapes \
        --scheme acdc_fog_targetcal25_seqdisjoint \
        --cal-datasets acdc_fog_train acdc_fog_val \
        --test-datasets acdc_fog_train acdc_fog_val \
        --methods region_crc heuristic argmax

The identifier of an ACDC frame is ``<condition>/<split>/<sequence>/<frame>``,
so the sequence is the third path segment.

Cost of the design. Sequences are large and unequal, so removing every frame of
the consumed sequences from the test side costs test frames -- more than n_t of
them, sometimes many more. That cost is reported per condition and per seed
rather than hidden, and a condition whose sequence count is too small for the
requested n_t is flagged loudly: night has only 6 sequences, so a
sequence-disjoint calibration set there rests on one or two of them and the
draw-to-draw spread is the honest headline, not the mean.

Example
-------
    python scripts/28_acdc_sequence_schemes.py
    python scripts/28_acdc_sequence_schemes.py --conditions fog --sizes 25 \
        --whole-sequences
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from record.datasets import acdc_ids, load_config
from record.paths import splits_dir
from record.splits import SCHEMA_VERSION, write_scheme

# A condition with fewer sequences than this cannot support a
# sequence-disjoint tier-A draw that is both diverse on the calibration side
# and representative on the test side; the schemes are still written, with a
# warning, because the spread they produce is the finding.
MIN_SEQUENCES_FOR_COMFORT = 8

# Below this many sequences left on the test side the scheme is degenerate
# rather than merely uncomfortable, and it is not written at all.
MIN_TEST_SEQUENCES = 2


def sequence_of(image_id: str) -> str:
    """Return the driving-sequence token of an ACDC identifier.

    Identifiers look like ``fog/train/GOPR0475/GOPR0475_frame_000041``; the
    sequence is the third path segment.

    The ``train``/``val`` segment is deliberately NOT part of the key. ACDC's
    official split cuts several recordings across both sides -- in fog,
    GOPR0476, GP010476 and GP020475 appear in train and in val, and GP020475's
    two halves interleave (train frames 1-302, val frames 54-229) -- so keying
    on ``<split>/<sequence>`` would treat one drive as two and put frames of
    the same drive on the calibration and the test side, which is the leak
    this script exists to remove.
    """
    parts = image_id.split("/")
    if len(parts) != 4:
        raise ValueError(
            f"unexpected ACDC identifier '{image_id}': expected "
            "<condition>/<split>/<sequence>/<frame>")
    return parts[2]


def _generator(base_seed: int, scheme: str, seed_index: int) -> np.random.Generator:
    """RNG keyed by base seed, scheme name, and seed index.

    Same construction as :func:`record.splits._generator`: a SHA-256 digest of
    the scheme name, never Python's ``hash()``, which is salted per process and
    would not reproduce across runs. Because the scheme name carries the
    ``_seqdisjoint`` suffix, these draws are independent of the published
    scheme's draws for the same condition and size.
    """
    scheme_key = int.from_bytes(hashlib.sha256(scheme.encode()).digest()[:4], "big")
    return np.random.default_rng([base_seed, scheme_key, seed_index])


def sequence_disjoint_split(pool, n_cal, base_seed, scheme, seed_index,
                            whole_sequences=False):
    """Split ``pool`` into a calibration and a sequence-disjoint test list.

    Sequences are shuffled with the keyed generator and consumed whole until
    they hold at least ``n_cal`` frames. Every frame of every consumed
    sequence is removed from the test side, which is what makes the split
    sequence-disjoint.

    By default the calibration list is then truncated to exactly ``n_cal``
    frames (in shuffled order within the consumed sequences), so that its size
    matches the published tier-A scheme and the two rows differ only in where
    the frames come from -- the comparison the referee is asking for. With
    ``whole_sequences`` the calibration side keeps every frame of the consumed
    sequences instead, which answers a different question (how well does tier A
    do with a sequence-sized labeling budget) and is reported separately.

    Returns ``(cal_ids, test_ids, diagnostics)``.
    """
    by_sequence: dict[str, list[str]] = {}
    for image_id in sorted(pool):
        by_sequence.setdefault(sequence_of(image_id), []).append(image_id)
    sequences = sorted(by_sequence)
    rng = _generator(base_seed, scheme, seed_index)
    order = [sequences[i] for i in rng.permutation(len(sequences))]

    consumed: list[str] = []
    cal_pool: list[str] = []
    for sequence in order:
        if len(cal_pool) >= n_cal:
            break
        consumed.append(sequence)
        cal_pool.extend(by_sequence[sequence])
    if len(cal_pool) < n_cal:
        raise ValueError(
            f"{scheme}: the whole pool ({len(cal_pool)} frames in "
            f"{len(sequences)} sequences) cannot supply {n_cal} calibration "
            "frames sequence-disjointly")

    if whole_sequences:
        cal_ids = sorted(cal_pool)
    else:
        # Truncate inside the consumed sequences, not across the pool: the
        # frames left over stay out of the test side regardless, because their
        # sequence is a calibration sequence.
        keep = rng.permutation(len(cal_pool))[:n_cal]
        cal_ids = sorted(cal_pool[i] for i in keep)

    consumed_set = set(consumed)
    test_ids = sorted(i for i in pool if sequence_of(i) not in consumed_set)
    diagnostics = {
        "n_sequences": len(sequences),
        "n_calibration_sequences": len(consumed),
        "n_test_sequences": len(sequences) - len(consumed),
        "n_calibration_frames": len(cal_ids),
        "n_frames_in_calibration_sequences": len(cal_pool),
        "n_test_frames": len(test_ids),
    }
    return cal_ids, test_ids, diagnostics


def build_scheme(scheme_name, pool, n_cal, base_seed, n_seeds,
                 whole_sequences=False, rng_scheme=None):
    """Build a full scheme dict plus its per-seed diagnostics.

    ``rng_scheme`` keys the draw. It defaults to ``scheme_name`` but is passed
    separately for the ``--whole-sequences`` arm, whose file name carries a
    ``_wholeseq`` suffix while its sequence draw must stay identical to the
    size-matched arm's: the two rows are meant to differ only in whether the
    consumed sequences are truncated to n_t, so they have to consume the same
    sequences in the same order for every seed.
    """
    seeds, diagnostics = {}, []
    for k in range(n_seeds):
        cal, test, diag = sequence_disjoint_split(
            pool, n_cal, base_seed, rng_scheme or scheme_name, k,
            whole_sequences=whole_sequences)
        seeds[str(k)] = {"target_calibration": cal, "test": test}
        diagnostics.append(dict(diag, seed=str(k)))
    realized = [len(seeds[str(k)]["target_calibration"]) for k in range(n_seeds)]
    scheme = {
        "schema_version": SCHEMA_VERSION,
        "scheme": scheme_name,
        "base_seed": base_seed,
        "n_items": len(pool),
        # The realized calibration size, not the requested one. Under
        # --whole-sequences the consumed sequences are kept entire, so the
        # size varies from seed to seed and is larger than n_cal; writing
        # n_cal here would misdescribe every seed. When the sizes agree the
        # field is that common value, as before.
        "n_calibration": realized[0] if len(set(realized)) == 1 else None,
        "n_calibration_requested": n_cal,
        "n_calibration_per_seed": {str(k): realized[k] for k in range(n_seeds)},
        "whole_sequences": bool(whole_sequences),
        "seeds": seeds,
    }
    return scheme, diagnostics


def summarize(values: list[int]) -> dict:
    """Min / median / max of a per-seed quantity, as plain Python numbers."""
    arr = np.asarray(values, dtype=float)
    return {"min": int(arr.min()), "median": float(np.median(arr)),
            "max": int(arr.max())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="ACDC conditions (default: all in datasets.yaml)")
    parser.add_argument("--sizes", nargs="+", type=int, default=None,
                        help="target-calibration sizes (default: the "
                             "splits.target_cal_sizes of datasets.yaml)")
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="seed replicates (default: splits.n_seeds)")
    parser.add_argument("--whole-sequences", action="store_true",
                        help="keep every frame of the consumed sequences as "
                             "calibration instead of truncating to n_t; the "
                             "calibration set is then larger than the "
                             "published tier-A one and the two are no longer "
                             "size-matched")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    sp = cfg["splits"]
    base_seed: int = sp["base_seed"]
    n_seeds: int = args.n_seeds or sp["n_seeds"]
    sizes: list[int] = args.sizes or sp["target_cal_sizes"]
    conditions = args.conditions or cfg["datasets"]["acdc"]["conditions"]
    out_dir = splits_dir()
    warnings: list[str] = []
    written: list[str] = []
    skipped: list[tuple[str, int]] = []

    for cond in conditions:
        pool = acdc_ids(cond, "train", cfg) + acdc_ids(cond, "val", cfg)
        if not pool:
            print(f"ERROR: no ACDC images found for condition '{cond}'")
            return 1
        sequences = sorted({sequence_of(i) for i in pool})
        sizes_per_sequence = sorted(
            (sum(1 for i in pool if sequence_of(i) == s) for s in sequences),
            reverse=True)
        print(f"\n=== {cond}: {len(pool)} frames in {len(sequences)} sequences "
              f"(sizes {sizes_per_sequence}) ===")
        if len(sequences) < MIN_SEQUENCES_FOR_COMFORT:
            msg = (f"{cond}: only {len(sequences)} driving sequences. A "
                   "sequence-disjoint calibration set rests on one or two of "
                   "them, so the seed-to-seed spread, not the mean, is the "
                   "reportable quantity here.")
            warnings.append(msg)
            print(f"WARNING: {msg}")

        for n in sizes:
            scheme_name = rng_scheme = f"acdc_{cond}_targetcal{n}_seqdisjoint"
            if args.whole_sequences:
                # A different scheme, not a variant of the same one: the
                # calibration side is larger and seed-dependent. Without the
                # suffix this run would overwrite the size-matched scheme
                # file and its meta file under the same name.
                scheme_name += "_wholeseq"
            try:
                scheme, diagnostics = build_scheme(
                    scheme_name, pool, n, base_seed, n_seeds,
                    whole_sequences=args.whole_sequences,
                    rng_scheme=rng_scheme)
            except ValueError as exc:
                msg = f"{scheme_name}: not written ({exc})"
                warnings.append(msg)
                print(f"WARNING: {msg}")
                skipped.append((cond, n))
                continue

            test_seqs = [d["n_test_sequences"] for d in diagnostics]
            if min(test_seqs) < MIN_TEST_SEQUENCES:
                msg = (f"{scheme_name}: at least one seed leaves only "
                       f"{min(test_seqs)} test sequence(s); scheme not written")
                warnings.append(msg)
                print(f"WARNING: {msg}")
                skipped.append((cond, n))
                continue

            path = write_scheme(scheme, out_dir)
            written.append(scheme_name)
            cal_seqs = summarize([d["n_calibration_sequences"] for d in diagnostics])
            test_frames = summarize([d["n_test_frames"] for d in diagnostics])
            spent = summarize([d["n_frames_in_calibration_sequences"] for d in diagnostics])
            published_test = len(pool) - n
            meta = {
                "scheme": scheme_name,
                "file": path.name,
                "condition": cond,
                "n_target_calibration": n,
                "n_seeds": n_seeds,
                "base_seed": base_seed,
                "whole_sequences": bool(args.whole_sequences),
                "rng_scheme_key": rng_scheme,
                "n_pool_frames": len(pool),
                "n_sequences": len(sequences),
                "sequence_sizes": sizes_per_sequence,
                "calibration_sequences": cal_seqs,
                "frames_in_calibration_sequences": spent,
                "test_frames": test_frames,
                "test_frames_in_published_scheme": published_test,
                "definition": "calibration frames are drawn from whole "
                              "driving sequences; every frame of a "
                              "calibration sequence is excluded from the test "
                              "side, so no test frame shares a sequence with "
                              "any calibration frame",
                "warnings": [w for w in warnings if w.startswith(f"{cond}:")],
                "generated_utc": datetime.now(timezone.utc).isoformat(),
            }
            meta_path = out_dir / f"{scheme_name}.meta.json"
            meta_path.write_text(json.dumps(meta, indent=1))
            lost = published_test - test_frames["median"]
            print(f"wrote {path.name}: {len(pool)} items, {n_seeds} seeds, "
                  f"calibration consumes {cal_seqs['min']}-{cal_seqs['max']} "
                  f"of {len(sequences)} sequences "
                  f"({spent['min']}-{spent['max']} frames), "
                  f"test keeps {test_frames['min']}-{test_frames['max']} "
                  f"frames (median {test_frames['median']:.0f} vs "
                  f"{published_test} in the published scheme, "
                  f"{lost:.0f} fewer)")

    if warnings:
        print("\n=== WARNINGS ===")
        for msg in warnings:
            print(f"WARNING: {msg}")

    print(f"\nwrote {len(written)} scheme(s); "
          f"{len(skipped)} (condition, size) pair(s) skipped")
    if skipped:
        # A skipped pair is a missing experimental arm, not a tolerable
        # warning: the downstream stage-5 run would silently have one fewer
        # row. Exit non-zero so a driver script stops here.
        for cond, n in skipped:
            print(f"NOT WRITTEN: {cond} at n_t={n}")
        print("\nRESULT: FAIL")
        return 1
    if not written:
        print("\nRESULT: FAIL (no scheme was written)")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
