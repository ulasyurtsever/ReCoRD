"""Tests for MARIDA metadata: patch ids, class mapping, presence vectors."""

import json

import numpy as np
import pytest

from record import marida
from record.marida import (CLASS_NAMES, canonical_patch_id, class_mapping,
                           marine_debris_class_id, parse_patch_id,
                           patch_class_presence, verify_class_presence)


def test_parse_standard_id():
    meta = parse_patch_id("S2_14-9-18_16PCC_43")
    assert meta.tile == "16PCC"
    assert (meta.day, meta.month, meta.year) == (14, 9, 18)
    assert meta.patch_id == "S2_14-9-18_16PCC_43"


def test_parse_single_digit_date():
    meta = parse_patch_id("S2_1-1-20_30VWH_0")
    assert (meta.day, meta.month, meta.year) == (1, 1, 20)
    assert meta.tile == "30VWH"


def test_reject_malformed_ids():
    for bad in ("S2_16PCC_43", "L8_14-9-18_16PCC_43", "S2_14-9-18_16PCC", "random"):
        with pytest.raises(ValueError):
            parse_patch_id(bad)


def test_canonical_patch_id_accepts_released_forms():
    # splits/*_X.txt lines omit the S2_ prefix; labels_mapping.txt keys add .tif.
    assert canonical_patch_id("1-12-19_48MYU_0") == "S2_1-12-19_48MYU_0"
    assert canonical_patch_id("S2_1-12-19_48MYU_0.tif") == "S2_1-12-19_48MYU_0"
    assert canonical_patch_id(" S2_1-12-19_48MYU_0\n") == "S2_1-12-19_48MYU_0"
    assert parse_patch_id(canonical_patch_id("1-12-19_48MYU_0")).tile == "48MYU"


def test_class_mapping_is_canonical():
    mapping = class_mapping()
    assert mapping == CLASS_NAMES
    assert len(mapping) == 15
    assert sorted(mapping.keys()) == list(range(1, 16))
    assert marine_debris_class_id() == 1


@pytest.fixture()
def presence_file(tmp_path, monkeypatch):
    """Point marida_root at a temp dir holding a JSON labels_mapping.txt."""
    vec_a = [0] * 15
    vec_a[0] = vec_a[6] = 1  # Marine Debris + Marine Water
    vec_b = [0] * 15
    vec_b[6] = 1
    (tmp_path / "labels_mapping.txt").write_text(json.dumps({
        "S2_1-12-19_48MYU_0.tif": vec_a,
        "S2_1-12-19_48MYU_1.tif": vec_b,
    }))
    monkeypatch.setattr(marida, "marida_root", lambda cfg=None: tmp_path)
    patch_class_presence.cache_clear()
    yield tmp_path
    patch_class_presence.cache_clear()


def test_patch_class_presence_parses_released_json(presence_file):
    presence = patch_class_presence()
    assert set(presence) == {"S2_1-12-19_48MYU_0", "S2_1-12-19_48MYU_1"}
    assert presence["S2_1-12-19_48MYU_0"][0] == 1
    assert presence["S2_1-12-19_48MYU_1"][0] == 0


def test_verify_class_presence_accepts_consistent_mask(presence_file, monkeypatch):
    monkeypatch.setattr(marida, "load_class_mask",
                        lambda pid: np.array([[0, 1], [7, 7]], dtype=np.int32))
    verify_class_presence("S2_1-12-19_48MYU_0")


def test_verify_class_presence_rejects_mismatch(presence_file, monkeypatch):
    monkeypatch.setattr(marida, "load_class_mask",
                        lambda pid: np.array([[0, 2], [7, 7]], dtype=np.int32))
    with pytest.raises(ValueError, match="mismatch"):
        verify_class_presence("S2_1-12-19_48MYU_0")


def test_patch_class_presence_rejects_bad_vector_length(tmp_path, monkeypatch):
    (tmp_path / "labels_mapping.txt").write_text(json.dumps({"S2_1-1-20_30VWH_0.tif": [1, 0]}))
    monkeypatch.setattr(marida, "marida_root", lambda cfg=None: tmp_path)
    patch_class_presence.cache_clear()
    with pytest.raises(ValueError, match="length"):
        patch_class_presence()
    patch_class_presence.cache_clear()
