#!/usr/bin/env python
"""Stage 6: fine-tune a SegFormer model on one LoveDA domain (GPU stage).

Trains on the domain's official Train split only; Val splits are reserved for
calibration and testing and are never touched here. A small monitoring subset
is carved out of the training images for checkpoint selection.

LoveDA mask values 1-7 are shifted to labels 0-6; the no-data value 0 maps to
the ignore index.

Outputs ``checkpoints/loveda_<arch>_<domain>/`` (best monitoring mIoU) in
Hugging Face format, directly loadable by the stage-3 caching pipeline.

Examples
--------
Smoke run:
    python scripts/06_train_loveda.py --domain urban --smoke

Full runs (one per transfer direction):
    python scripts/06_train_loveda.py --domain urban
    python scripts/06_train_loveda.py --domain rural
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from record.datasets import loveda_ids
from record.gt import image_path, label_path
from record.labelmaps import LOVEDA_IGNORE_VALUE
from record.models import pick_device
from record.paths import repo_root

IGNORE_INDEX = 255
N_CLASSES = 7
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def loveda_target(mask: np.ndarray) -> np.ndarray:
    """Map raw LoveDA mask values to training labels (0-6, 255=ignore)."""
    target = mask.astype(np.int64) - 1
    target[mask == LOVEDA_IGNORE_VALUE] = IGNORE_INDEX
    return target


class LovedaTrainSet(torch.utils.data.Dataset):
    def __init__(self, ids: list[str], dataset_key_domain: str, crop: int, seed: int):
        self.ids = ids
        self.domain = dataset_key_domain
        self.crop = crop
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        img = np.array(Image.open(image_path(f"loveda_Train_{self.domain}", image_id)).convert("RGB"))
        mask = np.array(Image.open(label_path(f"loveda_Train_{self.domain}", image_id)))
        h, w = mask.shape
        c = self.crop
        top = int(self.rng.integers(0, max(h - c, 0) + 1))
        left = int(self.rng.integers(0, max(w - c, 0) + 1))
        img, mask = img[top:top + c, left:left + c], mask[top:top + c, left:left + c]
        if self.rng.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        x = ((img.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(loveda_target(mask))))


@torch.inference_mode()
def monitor_miou(model, ids: list[str], domain: str, device: str) -> float:
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for image_id in ids:
        img = np.array(Image.open(image_path(f"loveda_Train_{domain}", image_id)).convert("RGB"))
        mask = np.array(Image.open(label_path(f"loveda_Train_{domain}", image_id)))
        x = ((img.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
        logits = model(pixel_values=x).logits
        logits = torch.nn.functional.interpolate(logits, size=mask.shape, mode="bilinear",
                                                 align_corners=False)
        pred = logits.argmax(1)[0].cpu().numpy()
        target = loveda_target(mask)
        valid = target != IGNORE_INDEX
        confusion += np.bincount(target[valid] * N_CLASSES + pred[valid],
                                 minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    inter = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - inter
    iou = inter / np.maximum(union, 1)
    return float(iou[union > 0].mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=["urban", "rural"])
    parser.add_argument("--arch", default="segformer_b2", choices=["segformer_b2"],
                        help="encoder architecture (nvidia/mit-b2)")
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--monitor-frac", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run to validate the pipeline (no usable checkpoint)")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    args = parse_args()
    if args.smoke:
        args.iters, args.eval_every, args.crop, args.batch = 8, 4, 96, 2
    device = args.device or pick_device()
    torch.manual_seed(args.seed)

    domain_title = args.domain.capitalize()
    all_ids = loveda_ids("Train", domain_title)
    rng = np.random.default_rng(args.seed)
    monitor_n = max(int(len(all_ids) * args.monitor_frac), 4)
    perm = rng.permutation(len(all_ids))
    monitor_ids = [all_ids[i] for i in perm[:monitor_n]]
    train_ids = [all_ids[i] for i in perm[monitor_n:]]
    if args.smoke:
        train_ids, monitor_ids = train_ids[:8], monitor_ids[:2]
    print(f"domain={domain_title} train={len(train_ids)} monitor={len(monitor_ids)} device={device}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=N_CLASSES, semantic_loss_ignore_index=IGNORE_INDEX)
    model.train().to(device)

    dataset = LovedaTrainSet(train_ids, domain_title, args.crop, args.seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4,
        drop_last=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    out_dir = repo_root() / "checkpoints" / f"loveda_{args.arch}_{args.domain}"
    best_miou, step = -1.0, 0
    progress = tqdm(total=args.iters, unit="it")
    while step < args.iters:
        for x, y in loader:
            if step >= args.iters:
                break
            lr = args.lr * (1 - step / args.iters)
            for group in optimizer.param_groups:
                group["lr"] = lr
            outputs = model(pixel_values=x.to(device), labels=y.to(device))
            optimizer.zero_grad(set_to_none=True)
            outputs.loss.backward()
            optimizer.step()
            step += 1
            progress.update(1)
            progress.set_postfix(loss=float(outputs.loss))

            if step % args.eval_every == 0 or step == args.iters:
                model.eval()
                miou = monitor_miou(model, monitor_ids, domain_title, device)
                model.train()
                print(f"\nstep {step}: monitor mIoU = {miou:.4f}")
                if miou > best_miou:
                    best_miou = miou
                    model.save_pretrained(out_dir)
                    SegformerImageProcessor(do_resize=False).save_pretrained(out_dir)
    progress.close()

    with open(out_dir / "training_meta.json", "w") as f:
        json.dump({
            "domain": args.domain, "arch": args.arch, "iters": args.iters,
            "batch": args.batch, "crop": args.crop, "lr": args.lr, "seed": args.seed,
            "best_monitor_miou": best_miou, "smoke": args.smoke,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }, f, indent=1)
    print(f"checkpoint: {out_dir} (best monitor mIoU {best_miou:.4f})")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
