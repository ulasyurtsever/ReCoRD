"""Class-id conventions for ground-truth masks and model outputs.

Cityscapes and ACDC ground truth uses ``labelIds`` encoding; pretrained
Cityscapes models output the 19-class ``trainId`` convention. LoveDA masks
use integer values 0-7 with 0 as no-data. MARIDA class ids follow the
canonical mapping in :mod:`record.marida` (validated against the released
per-patch class-presence file).
"""

from __future__ import annotations

# Critical classes: (name, gtFine labelId, trainId).
CITYSCAPES_CRITICAL: dict[str, tuple[int, int]] = {
    "person": (24, 11),
    "rider": (25, 12),
    "bicycle": (33, 18),
}

# LoveDA mask values (per the official datasheet): 0 no-data, 1 background,
# 2 building, 3 road, 4 water, 5 barren, 6 forest, 7 agriculture.
LOVEDA_CLASS_VALUES: dict[str, int] = {
    "background": 1,
    "building": 2,
    "road": 3,
    "water": 4,
    "barren": 5,
    "forest": 6,
    "agriculture": 7,
}
LOVEDA_IGNORE_VALUE = 0
LOVEDA_CRITICAL: dict[str, int] = {
    "building": LOVEDA_CLASS_VALUES["building"],
    "water": LOVEDA_CLASS_VALUES["water"],
}
# LoveDA models trained with the common 7-class convention output
# class index = mask value - 1.
LOVEDA_VALUE_TO_TRAINID_OFFSET = -1

MARIDA_DEBRIS_CLASS_NAME = "Marine Debris"

# Full Cityscapes labelId -> trainId mapping for the 19 evaluation classes
# (source: cityscapesScripts labels.py). Values absent from this table are
# ignore/void under the trainId convention.
CITYSCAPES_LABELID_TO_TRAINID: dict[int, int] = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}


def trainid_map_for(dataset: str) -> dict[int, int]:
    """Return {ground-truth value: model output channel} over ALL classes."""
    if dataset in ("cityscapes", "acdc"):
        return dict(CITYSCAPES_LABELID_TO_TRAINID)
    if dataset == "loveda":
        return {value: value + LOVEDA_VALUE_TO_TRAINID_OFFSET
                for value in LOVEDA_CLASS_VALUES.values()}
    if dataset == "marida":
        from record.marida import class_mapping

        return {cid: cid - 1 for cid in class_mapping()}
    raise KeyError(f"unknown dataset family: {dataset}")


# Dataset families a model family may be evaluated on. Cityscapes-trained
# models are intentionally applied to ACDC (same label convention): the
# adverse-condition datasets constitute the shifted test distribution.
MODEL_TO_DATASET_FAMILIES: dict[str, frozenset[str]] = {
    "cityscapes": frozenset({"cityscapes", "acdc"}),
    "loveda": frozenset({"loveda"}),
    "marida": frozenset({"marida"}),
}


def compatible_dataset_families(model_family: str) -> frozenset[str]:
    """Return the dataset families a model family may be evaluated on."""
    if model_family not in MODEL_TO_DATASET_FAMILIES:
        raise KeyError(f"unknown model family: {model_family}")
    return MODEL_TO_DATASET_FAMILIES[model_family]


def critical_classes_for(dataset: str) -> dict[str, tuple[int, int]]:
    """Return {class_name: (gt_value, model_channel)} for a dataset family."""
    if dataset in ("cityscapes", "acdc"):
        return dict(CITYSCAPES_CRITICAL)
    if dataset == "loveda":
        return {
            name: (value, value + LOVEDA_VALUE_TO_TRAINID_OFFSET)
            for name, value in LOVEDA_CRITICAL.items()
        }
    if dataset == "marida":
        # Canonical class id from record.marida; models are trained with the
        # same -1 offset (unlabeled 0 -> ignore).
        from record.marida import marine_debris_class_id

        debris_id = marine_debris_class_id()
        return {"marine_debris": (debris_id, debris_id - 1)}
    raise KeyError(f"unknown dataset family: {dataset}")
