"""Tests for class-id conventions and model/dataset family compatibility."""

import pytest

from record.labelmaps import (
    CITYSCAPES_CRITICAL,
    LOVEDA_CRITICAL,
    compatible_dataset_families,
    critical_classes_for,
)


def test_cityscapes_models_cover_acdc():
    families = compatible_dataset_families("cityscapes")
    assert "cityscapes" in families
    assert "acdc" in families  # shifted evaluation of Cityscapes-trained models


def test_other_families_are_closed():
    assert compatible_dataset_families("loveda") == frozenset({"loveda"})
    assert compatible_dataset_families("marida") == frozenset({"marida"})


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        compatible_dataset_families("imagenet")


def test_acdc_shares_cityscapes_critical_classes():
    assert critical_classes_for("acdc") == dict(CITYSCAPES_CRITICAL)
    assert critical_classes_for("cityscapes")["person"] == (24, 11)
    assert critical_classes_for("cityscapes")["rider"] == (25, 12)
    assert critical_classes_for("cityscapes")["bicycle"] == (33, 18)


def test_loveda_channel_offset():
    classes = critical_classes_for("loveda")
    assert classes["building"] == (LOVEDA_CRITICAL["building"], LOVEDA_CRITICAL["building"] - 1)
    assert classes["water"] == (LOVEDA_CRITICAL["water"], LOVEDA_CRITICAL["water"] - 1)


def test_full_cityscapes_trainid_map():
    from record.labelmaps import CITYSCAPES_CRITICAL, trainid_map_for

    m = trainid_map_for("cityscapes")
    assert len(m) == 19 and sorted(m.values()) == list(range(19))
    # Consistent with the critical-class table.
    for _, (gt_value, channel) in CITYSCAPES_CRITICAL.items():
        assert m[gt_value] == channel
    assert trainid_map_for("loveda") == {v: v - 1 for v in range(1, 8)}
