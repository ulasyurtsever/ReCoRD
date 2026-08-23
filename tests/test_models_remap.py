"""Tests for legacy Mask2Former checkpoint key remapping.

Runs offline against a tiny randomly initialized model: the current state
dict is renamed to the legacy layout and must round-trip exactly through
``remap_legacy_mask2former_keys``. Skipped when torch/transformers are not
installed.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from record.models import remap_legacy_mask2former_keys  # noqa: E402


def _tiny_model():
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig

    backbone = SwinConfig(
        image_size=64, embed_dim=8, depths=[1, 1, 1, 1], num_heads=[1, 1, 1, 1],
        out_features=["stage1", "stage2", "stage3", "stage4"])
    # feature sizes must be divisible by the pixel decoder's GroupNorm groups (32)
    config = Mask2FormerConfig(
        backbone_config=backbone, feature_size=32, mask_feature_size=32,
        hidden_dim=32, num_queries=4, encoder_layers=1, decoder_layers=1,
        dim_feedforward=64, num_attention_heads=2)
    return Mask2FormerForUniversalSegmentation(config)


def _to_legacy_layout(state: dict) -> dict:
    """Inverse of the remapping: produce the historical checkpoint layout."""
    legacy = {}
    for key, value in state.items():
        old = key
        for i in range(4):
            old = old.replace(
                f"encoder.hidden_states_norms.stage{i + 1}.", f"encoder.hidden_states_norms.{i}.")
        old = old.replace("pixel_level_module.encoder.embeddings.",
                          "pixel_level_module.encoder.model.embeddings.")
        old = old.replace("pixel_level_module.encoder.encoder.",
                          "pixel_level_module.encoder.model.encoder.")
        legacy[old] = value
    return legacy


def test_legacy_roundtrip_matches_every_key():
    model = _tiny_model()
    current = model.state_dict()
    legacy = _to_legacy_layout(current)
    assert legacy.keys() != current.keys()  # layouts genuinely differ

    remapped, dropped = remap_legacy_mask2former_keys(legacy, set(current.keys()))
    assert dropped == []
    assert set(remapped.keys()) == set(current.keys())
    for key in current:
        torch.testing.assert_close(remapped[key], current[key])


def test_unused_legacy_keys_are_dropped_not_loaded():
    model = _tiny_model()
    current = model.state_dict()
    legacy = _to_legacy_layout(current)
    legacy["model.pixel_level_module.encoder.model.layernorm.weight"] = torch.zeros(3)
    legacy["model.pixel_level_module.encoder.model.layernorm.bias"] = torch.zeros(3)

    remapped, dropped = remap_legacy_mask2former_keys(legacy, set(current.keys()))
    assert len(dropped) == 2
    assert all("layernorm" in k for k in dropped)
    assert set(remapped.keys()) == set(current.keys())


def test_loaded_state_is_usable():
    model = _tiny_model()
    reference = model.state_dict()
    legacy = _to_legacy_layout(reference)

    fresh = _tiny_model()
    remapped, _ = remap_legacy_mask2former_keys(legacy, set(fresh.state_dict().keys()))
    fresh.load_state_dict(remapped, strict=False)
    for key, value in fresh.state_dict().items():
        torch.testing.assert_close(value, reference[key])


def test_registry_specs_have_checkpoint_or_members():
    """Every registered model must resolve a checkpoint description.

    Regression: stage-3 logging assumed a ``checkpoint`` key, which ensemble
    specs (``members`` list) do not carry.
    """
    from record.models import load_model_registry

    registry = load_model_registry()
    for key, spec in registry["segmentation_models"].items():
        desc = spec.get("checkpoint") or f"ensemble[{len(spec['members'])} members]"
        assert desc, key
        assert ("checkpoint" in spec) != ("members" in spec), (
            f"{key}: exactly one of checkpoint/members expected"
        )
