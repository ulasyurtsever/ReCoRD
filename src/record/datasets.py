"""Dataset inventories: image identifier enumeration and layout verification.

Identifiers are stable, path-derived strings (relative stems) so that split
files remain valid regardless of the absolute data location. Enumeration is
sorted to guarantee deterministic ordering across filesystems.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from record.paths import config_path, data_root


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single inventory check."""

    dataset: str
    item: str
    expected: int
    found: int

    @property
    def passed(self) -> bool:
        return self.expected == self.found


def load_config() -> dict:
    """Load the dataset configuration file."""
    with open(config_path("datasets.yaml")) as f:
        return yaml.safe_load(f)


def _stems(directory: Path, pattern: str) -> list[str]:
    """Return sorted relative path stems for files matching ``pattern``."""
    if not directory.is_dir():
        return []
    return sorted(str(p.relative_to(directory).with_suffix("")) for p in directory.rglob(pattern))


# ---------------------------------------------------------------------------
# Cityscapes
# ---------------------------------------------------------------------------

def cityscapes_ids(split: str, cfg: dict | None = None) -> list[str]:
    """Return image identifiers for a Cityscapes split (``train`` or ``val``).

    An identifier is ``<city>/<frame>`` without the ``_leftImg8bit`` suffix,
    e.g. ``frankfurt/frankfurt_000000_000294``.
    """
    cfg = cfg or load_config()
    ds = cfg["datasets"]["cityscapes"]
    images = data_root() / ds["relpath"] / ds["images_dir"] / split
    return [s.removesuffix("_leftImg8bit") for s in _stems(images, "*_leftImg8bit.png")]


# ---------------------------------------------------------------------------
# ACDC
# ---------------------------------------------------------------------------

def acdc_ids(condition: str, split: str, cfg: dict | None = None) -> list[str]:
    """Return image identifiers for an ACDC condition and split.

    An identifier is ``<condition>/<split>/<sequence>/<frame>`` without the
    ``_rgb_anon`` suffix. Reference (clear-weather) frames are excluded.
    """
    cfg = cfg or load_config()
    ds = cfg["datasets"]["acdc"]
    images = data_root() / ds["relpath"] / ds["images_dir"] / condition / split
    stems = _stems(images, "*_rgb_anon.png")
    return [f"{condition}/{split}/{s.removesuffix('_rgb_anon')}" for s in stems]


# ---------------------------------------------------------------------------
# LoveDA
# ---------------------------------------------------------------------------

def loveda_ids(split: str, domain: str, cfg: dict | None = None) -> list[str]:
    """Return image identifiers for a LoveDA split and domain.

    An identifier is ``<split>/<domain>/<frame>``, e.g. ``Val/Urban/3514``.
    """
    cfg = cfg or load_config()
    ds = cfg["datasets"]["loveda"]
    images = data_root() / ds["relpath"] / split / domain / "images_png"
    return [f"{split}/{domain}/{s}" for s in _stems(images, "*.png")]


# ---------------------------------------------------------------------------
# MARIDA
# ---------------------------------------------------------------------------

def marida_official_split(name: str, cfg: dict | None = None) -> list[str]:
    """Return canonical patch identifiers of an official MARIDA split.

    The released ``splits/*_X.txt`` lists omit the ``S2_`` sensor prefix that
    patch filenames carry; identifiers are canonicalized on load.
    """
    from record.marida import canonical_patch_id

    cfg = cfg or load_config()
    ds = cfg["datasets"]["marida"]
    path = data_root() / ds["relpath"] / ds["splits_dir"] / f"{name}_X.txt"
    return sorted(canonical_patch_id(line) for line in path.read_text().splitlines()
                  if line.strip())


def marida_patch_count(cfg: dict | None = None) -> int:
    """Return the number of ``.tif`` files under the MARIDA patches directory."""
    cfg = cfg or load_config()
    ds = cfg["datasets"]["marida"]
    patches = data_root() / ds["relpath"] / ds["patches_dir"]
    return sum(1 for _ in patches.rglob("*.tif"))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_all(cfg: dict | None = None) -> list[CheckResult]:
    """Run every inventory check and return the results."""
    cfg = cfg or load_config()
    checks: list[CheckResult] = []

    cs = cfg["datasets"]["cityscapes"]["expected"]
    checks.append(CheckResult("cityscapes", "train_images", cs["train_images"],
                              len(cityscapes_ids("train", cfg))))
    checks.append(CheckResult("cityscapes", "val_images", cs["val_images"],
                              len(cityscapes_ids("val", cfg))))

    acdc = cfg["datasets"]["acdc"]
    for cond in acdc["conditions"]:
        checks.append(CheckResult(
            "acdc", f"{cond}/train", acdc["expected"]["train_images_per_condition"],
            len(acdc_ids(cond, "train", cfg))))
        checks.append(CheckResult(
            "acdc", f"{cond}/val", acdc["expected"]["val_images"][cond],
            len(acdc_ids(cond, "val", cfg))))

    loveda = cfg["datasets"]["loveda"]
    for split, domains in loveda["expected"].items():
        for domain, expected in domains.items():
            checks.append(CheckResult(
                "loveda", f"{split}/{domain}", expected,
                len(loveda_ids(split, domain, cfg))))

    marida = cfg["datasets"]["marida"]
    checks.append(CheckResult("marida", "patch_tifs", marida["expected"]["patch_tifs"],
                              marida_patch_count(cfg)))
    official_expected = marida["expected"].get("official_splits", {})
    for name in ("train", "val", "test"):
        found = len(marida_official_split(name, cfg))
        checks.append(CheckResult(
            "marida", f"official_{name}", official_expected.get(name, found), found))

    return checks
