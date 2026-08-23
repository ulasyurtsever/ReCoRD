"""Ground-truth image and label loading for supported datasets.

Dataset keys used across stages:

- ``cityscapes_val``, ``cityscapes_train``
- ``acdc_<condition>_<split>`` (e.g. ``acdc_fog_train``)
- ``loveda_<split>_<domain>`` (e.g. ``loveda_Val_Urban``)

Image identifiers follow :mod:`record.datasets`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from record.datasets import acdc_ids, cityscapes_ids, load_config, loveda_ids, marida_official_split
from record.paths import data_root


def parse_dataset_key(dataset_key: str) -> tuple[str, dict]:
    """Split a dataset key into (family, parameters)."""
    parts = dataset_key.split("_")
    family = parts[0]
    if family == "cityscapes" and len(parts) == 2:
        return family, {"split": parts[1]}
    if family == "acdc" and len(parts) == 3:
        return family, {"condition": parts[1], "split": parts[2]}
    if family == "loveda" and len(parts) == 3:
        return family, {"split": parts[1], "domain": parts[2]}
    if family == "marida" and len(parts) == 2:
        return family, {"split": parts[1]}
    raise KeyError(f"unrecognized dataset key: {dataset_key}")


def list_ids(dataset_key: str, cfg: dict | None = None) -> list[str]:
    """Enumerate image identifiers for a dataset key."""
    cfg = cfg or load_config()
    family, p = parse_dataset_key(dataset_key)
    if family == "cityscapes":
        return cityscapes_ids(p["split"], cfg)
    if family == "acdc":
        return acdc_ids(p["condition"], p["split"], cfg)
    if family == "loveda":
        return loveda_ids(p["split"], p["domain"], cfg)
    if family == "marida":
        return marida_official_split(p["split"], cfg)
    raise KeyError(dataset_key)


def image_path(dataset_key: str, image_id: str, cfg: dict | None = None) -> Path:
    """Return the RGB image path for an identifier."""
    cfg = cfg or load_config()
    family, p = parse_dataset_key(dataset_key)
    root = data_root()
    if family == "cityscapes":
        ds = cfg["datasets"]["cityscapes"]
        return root / ds["relpath"] / ds["images_dir"] / p["split"] / f"{image_id}_leftImg8bit.png"
    if family == "acdc":
        ds = cfg["datasets"]["acdc"]
        return root / ds["relpath"] / ds["images_dir"] / f"{image_id}_rgb_anon.png"
    if family == "loveda":
        ds = cfg["datasets"]["loveda"]
        split, domain, frame = image_id.split("/")
        return root / ds["relpath"] / split / domain / "images_png" / f"{frame}.png"
    raise KeyError(dataset_key)


def label_path(dataset_key: str, image_id: str, cfg: dict | None = None) -> Path:
    """Return the ground-truth label path for an identifier."""
    cfg = cfg or load_config()
    family, p = parse_dataset_key(dataset_key)
    root = data_root()
    if family == "cityscapes":
        ds = cfg["datasets"]["cityscapes"]
        return root / ds["relpath"] / ds["labels_dir"] / p["split"] / f"{image_id}{ds['label_suffix']}"
    if family == "acdc":
        ds = cfg["datasets"]["acdc"]
        return root / ds["relpath"] / ds["labels_dir"] / f"{image_id}{ds['label_suffix']}"
    if family == "loveda":
        ds = cfg["datasets"]["loveda"]
        split, domain, frame = image_id.split("/")
        return root / ds["relpath"] / split / domain / "masks_png" / f"{frame}.png"
    raise KeyError(dataset_key)


def load_label_array(dataset_key: str, image_id: str, cfg: dict | None = None) -> np.ndarray:
    """Load the raw integer label mask for an identifier."""
    family, _ = parse_dataset_key(dataset_key)
    if family == "marida":
        from record.marida import load_class_mask

        return load_class_mask(image_id)
    return np.array(Image.open(label_path(dataset_key, image_id, cfg)))


def load_image_array(dataset_key: str, image_id: str, cfg: dict | None = None) -> np.ndarray:
    """Load the RGB image as an ``(H, W, 3)`` uint8 array."""
    return np.array(Image.open(image_path(dataset_key, image_id, cfg)).convert("RGB"))


def load_model_input(dataset_key: str, image_id: str, cfg: dict | None = None) -> np.ndarray:
    """Load the array a segmentation model consumes for this dataset.

    RGB datasets yield ``(H, W, 3)`` uint8; MARIDA yields raw multispectral
    bands ``(C, H, W)`` float32 (normalization is applied by the model).
    """
    family, _ = parse_dataset_key(dataset_key)
    if family == "marida":
        from record.marida import load_bands

        return load_bands(image_id)
    return load_image_array(dataset_key, image_id, cfg)
