#!/usr/bin/env python
"""Stage 7: train a U-Net on MARIDA (GPU stage; deep-ensemble members by seed).

Without ``--scheme`` the model trains on the official MARIDA train split and
selects its checkpoint on the official val split; the test split is never
touched. This model is the subject of the official-split case study only.

With ``--scheme NAME`` the training and checkpoint-selection sets are read from
a held-out scheme written by stage 2b (``train`` and ``monitor`` groups). The
held-out tile or season, and the calibration group, are then absent from
every stage of model fitting, which is what makes the corresponding evaluation
a held-out one. One model is trained per held-out axis; the checkpoint lands in
``checkpoints/marida_unet_<scheme suffix>``. Class ids are shifted
by -1 (unlabeled 0 maps to the ignore index); inverse-sqrt-frequency class
weighting counters the strong imbalance. Per-band normalization statistics
are computed from the training split and stored in the checkpoint, making
checkpoints self-contained.

The five-member deep ensemble is obtained by running this script with seeds
0-4; the stage-3 cache averages member posteriors.

Examples
--------
Smoke run:
    python scripts/07_train_marida_unet.py --seed 0 --smoke

Ensemble members (official split):
    for S in 0 1 2 3 4; do python scripts/07_train_marida_unet.py --seed $S; done

One model per held-out tile:
    for R in 16PCC 16PDC 16PEC 18QYF 48PZC; do
      python scripts/07_train_marida_unet.py --scheme marida_holdout_region_$R
    done
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import torch
from tqdm import tqdm

from record.datasets import marida_official_split
from record.marida import (class_mapping, compute_band_stats, load_bands,
                           load_class_mask, marine_debris_class_id,
                           verify_class_presence)
from record.models import pick_device
from record.paths import repo_root, splits_dir
from record.splits import load_scheme
from record.unet import UNet

IGNORE_INDEX = 255


class MaridaTrainSet(torch.utils.data.Dataset):
    def __init__(self, ids: list[str], mean: np.ndarray, std: np.ndarray, seed: int):
        self.ids = ids
        self.mean, self.std = mean[:, None, None], std[:, None, None]
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        bands = (load_bands(self.ids[idx]) - self.mean) / self.std
        target = load_class_mask(self.ids[idx]).astype(np.int64) - 1
        target[target < 0] = IGNORE_INDEX
        k = int(self.rng.integers(0, 4))
        bands = np.rot90(bands, k, axes=(1, 2))
        target = np.rot90(target, k)
        if self.rng.random() < 0.5:
            bands, target = bands[:, :, ::-1], target[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(bands, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(target)))


def class_weights(ids: list[str], n_classes: int) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=np.float64)
    for patch_id in ids:
        mask = load_class_mask(patch_id)
        values = np.bincount(mask.ravel(), minlength=n_classes + 1)
        counts += values[1:n_classes + 1]
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return (weights / weights.mean()).astype(np.float32)


@torch.inference_mode()
def debris_f1(model, ids, mean, std, debris_channel: int, device: str) -> float:
    tp = fp = fn = 0
    for patch_id in ids:
        bands = (load_bands(patch_id) - mean[:, None, None]) / std[:, None, None]
        x = torch.from_numpy(bands.astype(np.float32))[None].to(device)
        pred = model(x).argmax(1)[0].cpu().numpy()
        target = load_class_mask(patch_id).astype(np.int64) - 1
        valid = target >= 0
        p, t = pred[valid] == debris_channel, target[valid] == debris_channel
        tp += int((p & t).sum()); fp += int((p & ~t).sum()); fn += int((~p & t).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def _ids_digest(ids) -> str:
    """Stable digest of an identifier list, order-independent."""
    joined = "\n".join(sorted(ids)).encode()
    return hashlib.sha256(joined).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="ensemble member seed")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--scheme", default=None,
                        help="held-out scheme name (without .json) providing "
                             "the 'train' and 'monitor' id groups; default "
                             "uses the official train/val split")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.epochs, args.eval_every = 2, 1
    device = args.device or pick_device()
    torch.manual_seed(args.seed)

    if args.scheme:
        entry = load_scheme(splits_dir() / f"{args.scheme}.json")["seeds"]["0"]
        missing = [k for k in ("train", "monitor") if k not in entry]
        if missing:
            raise KeyError(f"{args.scheme}: scheme has no {missing} group(s); "
                           "regenerate with scripts/02b_generate_marida_splits.py")
        train_ids, val_ids = list(entry["train"]), list(entry["monitor"])
        held_out = sorted(set(entry.get("test", [])) | set(entry.get("calibration", [])))
        overlap = sorted(set(train_ids + val_ids) & set(held_out))
        if overlap:
            raise AssertionError(
                f"{args.scheme}: {len(overlap)} patch(es) appear in both a fitting "
                f"group and the calibration/test groups, e.g. {overlap[:3]}")
        # Seed 0 is the canonical model of a scheme; further seeds are deep
        # ensemble members and need their own checkpoint directories.
        suffix = args.scheme.replace("marida_", "", 1)
        run_name = suffix if args.seed == 0 else f"{suffix}_s{args.seed}"
    else:
        train_ids = marida_official_split("train")
        val_ids = marida_official_split("val")
        run_name = f"s{args.seed}"
    if args.smoke:
        train_ids, val_ids = train_ids[:16], val_ids[:8]

    n_classes = max(class_mapping().keys())
    debris_channel = marine_debris_class_id() - 1
    for patch_id in train_ids[:8]:
        verify_class_presence(patch_id)
    n_bands = load_bands(train_ids[0]).shape[0]
    print(f"seed={args.seed} train={len(train_ids)} val={len(val_ids)} "
          f"classes={n_classes} bands={n_bands} debris_channel={debris_channel} device={device}")

    mean, std = compute_band_stats(train_ids)
    mean_arr, std_arr = np.array(mean, np.float32), np.array(std, np.float32)
    weights = class_weights(train_ids, n_classes)

    model = UNet(n_bands, n_classes, base_width=args.base_width).to(device)
    dataset = MaridaTrainSet(train_ids, mean_arr, std_arr, args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True,
                                         num_workers=4, drop_last=False)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).to(device), ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = repo_root() / "checkpoints" / f"marida_unet_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    for epoch in tqdm(range(1, args.epochs + 1), unit="epoch"):
        model.train()
        for x, y in loader:
            loss = criterion(model(x.to(device)), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            f1 = debris_f1(model, val_ids, mean_arr, std_arr, debris_channel, device)
            print(f"\nepoch {epoch}: val debris F1 = {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), out_dir / "model.pt")
                with open(out_dir / "config.json", "w") as f:
                    json.dump({
                        "framework": "unet", "in_channels": n_bands,
                        "n_classes": n_classes, "base_width": args.base_width,
                        "band_mean": mean, "band_std": std,
                        "class_id_offset": -1, "debris_channel": debris_channel,
                        "seed": args.seed, "best_val_debris_f1": best_f1,
                        "smoke": args.smoke,
                        # Provenance of the fitting sets, so that the stage-18
                        # audit can check a scheme's test group against the
                        # data this checkpoint actually saw.
                        "scheme": args.scheme,
                        "n_train": len(train_ids), "n_monitor": len(val_ids),
                        "train_ids_sha256": _ids_digest(train_ids),
                        "monitor_ids_sha256": _ids_digest(val_ids),
                        "completed_utc": datetime.now(timezone.utc).isoformat(),
                    }, f, indent=1)

    print(f"checkpoint: {out_dir} (best val debris F1 {best_f1:.4f})")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
