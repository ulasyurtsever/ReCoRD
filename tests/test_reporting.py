"""Tests for stage-6 reporting helpers (pure, no data required)."""

import numpy as np
import pandas as pd

from record.reporting import fmt, latex_table, parse_axis_fields


def test_fmt_handles_nan():
    assert fmt(0.12345) == "0.123"
    assert fmt(float("nan")) == "--"
    assert fmt(0.5, 2) == "0.50"


def test_parse_axis_fields():
    frame = pd.DataFrame({"experiment": [
        "e1_indist__segformer_b2_cityscapes",
        "e2_break__night__segformer_b5_cityscapes",
        "e3_tierA50__fog__segformer_b2_cityscapes",
        "c1_clip5__rain__segformer_b2_cityscapes__dinov2_vitb14",
        "l2_break__urban2rural",
        "m2_region__16PDC__marida_unet_ens5",
        "m3_season__spring__marida_unet_s0",
    ]})
    out = parse_axis_fields(frame)
    assert list(out["block"]) == ["e1", "e2", "e3", "c1", "l2", "m2", "m3"]
    assert out.loc[1, "condition"] == "night"
    assert out.loc[2, "n_target"] == 50
    assert out.loc[3, "clip_max"] == 5
    assert out.loc[3, "embedding"] == "dinov2_vitb14"
    assert out.loc[4, "condition"] == "urban2rural"
    assert out.loc[5, "region"] == "16PDC"
    assert out.loc[6, "season"] == "spring"


def test_latex_table_structure():
    tex = latex_table(r"a & b \\", "Caption.", "tab:x", "cc", "A & B")
    assert r"\begin{table}" in tex and r"\bottomrule" in tex
    assert r"\label{tab:x}" in tex
