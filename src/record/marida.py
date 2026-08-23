"""MARIDA data access: multispectral patches, class mapping, patch metadata.

MARIDA distributes Sentinel-2 patches as multi-band GeoTIFFs accompanied by
``*_cl.tif`` class masks and ``*_conf.tif`` confidence masks, with official
split lists under ``splits/``. The released ``labels_mapping.txt`` is a JSON
dictionary mapping each patch file to a 15-element class-presence vector
(index ``i`` corresponds to class id ``i + 1``); it does not contain class
names. Class names follow the MARIDA reference implementation
(``cat_mapping`` in marine-debris/marine-debris.github.io, ``utils/assets.py``;
Kikaki et al., 2022) and are cross-checked against the released presence
vectors and class masks by :func:`verify_class_presence`.

Patch identifiers follow ``S2_<d-m-yy>_<MGRS tile>_<index>``; the tile token
serves as the region proxy for leave-region-out splits and the date token
provides the season axis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from record.datasets import load_config
from record.paths import data_root

_ID_PATTERN = re.compile(r"^S2_(?P<day>\d{1,2})-(?P<month>\d{1,2})-(?P<year>\d{2})_(?P<tile>[0-9A-Z]+)_(?P<index>\d+)$")


@dataclass(frozen=True)
class PatchMeta:
    """Metadata parsed from a MARIDA patch identifier."""

    patch_id: str
    tile: str
    day: int
    month: int
    year: int


def marida_root(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return data_root() / cfg["datasets"]["marida"]["relpath"]


def canonical_patch_id(raw: str) -> str:
    """Normalize a patch reference to the canonical ``S2_...`` identifier.

    The released split lists (``splits/*_X.txt``) omit the ``S2_`` prefix and
    ``labels_mapping.txt`` keys carry a ``.tif`` suffix; patch filenames use
    the full prefixed stem. This helper accepts any of those forms.
    """
    patch_id = raw.strip().removesuffix(".tif")
    if not patch_id.startswith("S2_"):
        patch_id = f"S2_{patch_id}"
    return patch_id


def parse_patch_id(patch_id: str) -> PatchMeta:
    """Parse tile and date metadata from a patch identifier."""
    match = _ID_PATTERN.match(patch_id)
    if not match:
        raise ValueError(f"unrecognized MARIDA patch id: {patch_id}")
    g = match.groupdict()
    return PatchMeta(patch_id=patch_id, tile=g["tile"],
                     day=int(g["day"]), month=int(g["month"]), year=int(g["year"]))


@lru_cache(maxsize=1)
def _patch_index() -> dict[str, Path]:
    """Map patch id -> image tif path (excludes ``_cl``/``_conf`` files)."""
    patches_dir = marida_root() / load_config()["datasets"]["marida"]["patches_dir"]
    index: dict[str, Path] = {}
    for path in patches_dir.rglob("*.tif"):
        stem = path.stem
        if stem.endswith("_cl") or stem.endswith("_conf"):
            continue
        index[stem] = path
    if not index:
        raise FileNotFoundError(f"no MARIDA patches found under {patches_dir}")
    return index


def patch_paths(patch_id: str) -> tuple[Path, Path]:
    """Return (image tif, class-mask tif) paths for a patch id."""
    image = _patch_index().get(patch_id)
    if image is None:
        raise KeyError(f"MARIDA patch not found: {patch_id}")
    mask = image.with_name(image.stem + "_cl" + image.suffix)
    if not mask.exists():
        raise FileNotFoundError(f"class mask missing for {patch_id}: {mask}")
    return image, mask


def load_bands(patch_id: str) -> np.ndarray:
    """Load the raw multispectral bands of a patch as float32 ``(C, H, W)``.

    Values are returned unnormalized; per-band normalization is a model
    property and is applied by the model wrapper using statistics stored in
    its checkpoint.
    """
    import rasterio

    image_path, _ = patch_paths(patch_id)
    with rasterio.open(image_path) as src:
        bands = src.read().astype(np.float32)
    return np.nan_to_num(bands, nan=0.0)


def load_class_mask(patch_id: str) -> np.ndarray:
    """Load the class mask of a patch as a 2-D integer array."""
    import rasterio

    _, mask_path = patch_paths(patch_id)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return mask.astype(np.int32)


# Canonical MARIDA class ids (mask values; 0 = unlabeled). Source:
# ``cat_mapping`` in the MARIDA reference implementation
# (marine-debris/marine-debris.github.io, utils/assets.py; Kikaki et al., 2022).
# The released dataset carries no id-to-name file, so the mapping is fixed here
# and validated against the data by ``verify_class_presence``.
CLASS_NAMES: dict[int, str] = {
    1: "Marine Debris",
    2: "Dense Sargassum",
    3: "Sparse Sargassum",
    4: "Natural Organic Material",
    5: "Ship",
    6: "Clouds",
    7: "Marine Water",
    8: "Sediment-Laden Water",
    9: "Foam",
    10: "Turbid Water",
    11: "Shallow Water",
    12: "Waves",
    13: "Cloud Shadows",
    14: "Wakes",
    15: "Mixed Water",
}


def class_mapping() -> dict[int, str]:
    """Return the canonical {class id: class name} mapping."""
    return dict(CLASS_NAMES)


def marine_debris_class_id() -> int:
    """Return the marine-debris class id."""
    for class_id, name in CLASS_NAMES.items():
        if "debris" in name.lower():
            return class_id
    raise KeyError("no class containing 'debris' in the MARIDA class mapping")


@lru_cache(maxsize=1)
def patch_class_presence() -> dict[str, list[int]]:
    """Parse ``labels_mapping.txt`` into {patch id: class-presence vector}.

    The released file is a JSON dictionary mapping ``<patch id>.tif`` to a
    binary vector of length ``len(CLASS_NAMES)`` where index ``i`` marks the
    presence of class id ``i + 1`` in the patch.
    """
    import json

    path = marida_root() / "labels_mapping.txt"
    raw = json.loads(path.read_text())
    presence: dict[str, list[int]] = {}
    for key, vector in raw.items():
        if len(vector) != len(CLASS_NAMES):
            raise ValueError(
                f"unexpected presence vector length {len(vector)} for {key} in {path}"
            )
        presence[canonical_patch_id(key)] = [int(v) for v in vector]
    if not presence:
        raise ValueError(f"could not parse class presence from {path}")
    return presence


def verify_class_presence(patch_id: str) -> None:
    """Check the released presence vector of a patch against its class mask.

    Confirms the index convention (vector index ``i`` == class id ``i + 1``)
    that ties ``CLASS_NAMES`` to the released data. Raises ``ValueError`` on
    mismatch.
    """
    vector = patch_class_presence().get(patch_id)
    if vector is None:
        raise KeyError(f"patch not present in labels_mapping.txt: {patch_id}")
    mask = load_class_mask(patch_id)
    from_mask = sorted(int(v) for v in np.unique(mask) if v > 0)
    from_vector = sorted(i + 1 for i, flag in enumerate(vector) if flag)
    if from_mask != from_vector:
        raise ValueError(
            f"class presence mismatch for {patch_id}: mask {from_mask} "
            f"vs labels_mapping {from_vector}"
        )


def compute_band_stats(patch_ids: list[str]) -> tuple[list[float], list[float]]:
    """Per-band mean and standard deviation over the given patches."""
    total = count = None
    sq_total = None
    for patch_id in patch_ids:
        bands = load_bands(patch_id)
        flat = bands.reshape(bands.shape[0], -1).astype(np.float64)
        if total is None:
            total = flat.sum(axis=1)
            sq_total = (flat ** 2).sum(axis=1)
            count = flat.shape[1]
        else:
            total += flat.sum(axis=1)
            sq_total += (flat ** 2).sum(axis=1)
            count += flat.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(sq_total / count - mean ** 2, 1e-12))
    return mean.tolist(), std.tolist()
