"""Segmentation model loading and per-pixel posterior computation.

Torch and transformers are imported lazily so that analysis stages remain
importable in torch-free environments. All models return a full-resolution
class posterior of shape ``(n_classes, H, W)``.

For Mask2Former, the semantic class scores are computed as the standard
query-to-semantic projection (class probabilities times mask sigmoids) and
normalized over classes to yield a per-pixel posterior; this is the same
aggregation used by the reference post-processing, made explicit here because
thresholds operate on per-class probabilities rather than argmax output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from record.paths import config_path, repo_root


def _resolve_local_checkpoint(checkpoint: str) -> Path | None:
    """Return the path of a local checkpoint directory, or None if not local."""
    path = Path(checkpoint)
    if not path.is_absolute():
        path = repo_root() / path
    return path if path.is_dir() else None


def load_model_registry() -> dict:
    with open(config_path("models.yaml")) as f:
        return yaml.safe_load(f)


def remap_legacy_mask2former_keys(state: dict, expected_keys: set[str]) -> tuple[dict, list[str]]:
    """Translate a legacy Mask2Former state dict to the current key layout.

    Checkpoints exported with early transformers releases nest the Swin
    backbone under ``pixel_level_module.encoder.model`` and index the stage
    norms numerically; current releases flatten the nesting and name the
    norms ``stage1..stage4``. Weights are identical, only names differ.

    Returns the remapped state dict restricted to ``expected_keys`` and the
    list of legacy keys with no counterpart in the current architecture
    (e.g. the unused final backbone layernorm).
    """
    remapped: dict = {}
    for key, value in state.items():
        new_key = key.replace(
            "pixel_level_module.encoder.model.", "pixel_level_module.encoder.")
        for i in range(4):
            new_key = new_key.replace(
                f"encoder.hidden_states_norms.{i}.", f"encoder.hidden_states_norms.stage{i + 1}.")
        remapped[new_key] = value
    dropped = sorted(k for k in remapped if k not in expected_keys)
    return {k: v for k, v in remapped.items() if k in expected_keys}, dropped


def _load_mask2former_legacy(checkpoint: str):
    """Load a legacy-format Mask2Former checkpoint by remapping its keys."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    config = Mask2FormerConfig.from_pretrained(checkpoint)
    model = Mask2FormerForUniversalSegmentation(config)

    try:
        from safetensors.torch import load_file

        state = load_file(hf_hub_download(checkpoint, "model.safetensors"))
    except Exception:
        state = torch.load(
            hf_hub_download(checkpoint, "pytorch_model.bin"),
            map_location="cpu", weights_only=True)

    expected = set(model.state_dict().keys())
    remapped, dropped = remap_legacy_mask2former_keys(state, expected)
    missing = sorted(expected - set(remapped.keys()))
    if missing:
        raise RuntimeError(
            f"{checkpoint}: legacy key remapping left {len(missing)} parameter(s) "
            f"unmatched (e.g. {missing[:3]}); refusing to load a partial model.")
    model.load_state_dict(remapped, strict=False)
    print(f"loaded {checkpoint} via legacy key remapping "
          f"({len(remapped)} tensors, {len(dropped)} unused legacy keys dropped)")
    return model


def _assert_clean_load(loading_info: dict, checkpoint: str) -> None:
    """Fail hard when checkpoint weights did not map onto the architecture.

    Randomly initialized submodules (missing keys) would produce silently
    meaningless posteriors; a library/architecture version mismatch must
    abort the run instead.
    """
    missing = loading_info.get("missing_keys") or []
    if missing:
        raise RuntimeError(
            f"{checkpoint}: {len(missing)} parameter(s) missing from the checkpoint "
            f"(e.g. {missing[:3]}); the architecture does not match the stored weights. "
            "This typically indicates a transformers version incompatibility - "
            "install the version pinned in requirements.txt and retry."
        )


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SegmentationModel:
    """A pretrained semantic segmentation model with a posterior interface."""

    def __init__(self, model_key: str, device: str | None = None):
        import torch  # noqa: F401  (ensures a clear error before HF imports)

        registry = load_model_registry()["segmentation_models"]
        if model_key not in registry:
            raise KeyError(f"unknown model key: {model_key}; known: {sorted(registry)}")
        spec = registry[model_key]
        self.spec = spec
        self.model_key = model_key
        self.framework = spec["framework"]
        self.checkpoint = spec.get("checkpoint")
        self.n_classes = spec["n_classes"]
        self.device = device or pick_device()
        self._load()

    def _load(self) -> None:
        import torch

        if self.framework == "segformer":
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

            self.processor = AutoImageProcessor.from_pretrained(self.checkpoint)
            self.model, info = SegformerForSemanticSegmentation.from_pretrained(
                self.checkpoint, output_loading_info=True)
        elif self.framework == "mask2former":
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

            self.processor = AutoImageProcessor.from_pretrained(self.checkpoint)
            self.model, info = Mask2FormerForUniversalSegmentation.from_pretrained(
                self.checkpoint, output_loading_info=True)
            if info.get("missing_keys"):
                # Official Cityscapes checkpoints predate the Swin backbone
                # key-layout change; recover them via deterministic remapping.
                self.model = _load_mask2former_legacy(self.checkpoint)
                info = {"missing_keys": []}
        elif self.framework == "unet":
            self._load_unet()
            return
        else:
            raise ValueError(f"unsupported framework: {self.framework}")
        _assert_clean_load(info, self.checkpoint)
        self.model.eval().to(torch.device(self.device))

    def _load_unet(self) -> None:
        """Load one or more locally trained U-Net members (deep ensemble)."""
        import json

        import torch

        from record.unet import UNet

        member_paths = self.spec.get("members") or [self.checkpoint]
        self.members: list[tuple] = []
        for entry in member_paths:
            directory = _resolve_local_checkpoint(entry)
            if directory is None:
                raise FileNotFoundError(
                    f"{self.model_key}: checkpoint directory not found: {entry} "
                    "(train it first; see scripts/07_train_marida_unet.py)")
            with open(directory / "config.json") as f:
                cfg = json.load(f)
            net = UNet(cfg["in_channels"], cfg["n_classes"], cfg.get("base_width", 32))
            state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=True)
            net.eval().to(torch.device(self.device))
            mean = torch.tensor(cfg["band_mean"], dtype=torch.float32).view(-1, 1, 1).to(self.device)
            std = torch.tensor(cfg["band_std"], dtype=torch.float32).view(-1, 1, 1).to(self.device)
            self.members.append((net, mean, std))
        self.n_classes = cfg["n_classes"]
        print(f"loaded {len(self.members)} U-Net member(s) for {self.model_key}")

    def enable_mc_sampling(self) -> dict[str, int]:
        """Switch stochastic submodules to sampling mode for MC inference.

        Enables ``nn.Dropout`` layers and stochastic-depth (DropPath) modules
        that have a nonzero rate, leaving normalization layers in eval mode.
        Returns the counts of enabled modules; raises when the architecture
        provides no stochasticity (MC sampling would be a silent no-op).
        """
        import torch.nn as nn

        counts = {"dropout": 0, "drop_path": 0}
        for module in self.model.modules():
            if isinstance(module, nn.Dropout) and module.p > 0:
                module.train()
                counts["dropout"] += 1
            elif "DropPath" in type(module).__name__ and getattr(module, "drop_prob", 0) > 0:
                module.train()
                counts["drop_path"] += 1
        if sum(counts.values()) == 0:
            raise RuntimeError(
                f"{self.checkpoint}: no dropout or drop-path modules with nonzero "
                "rate; MC sampling is not applicable to this checkpoint.")
        return counts

    def posterior_mc_mean(self, image: np.ndarray, n_passes: int) -> np.ndarray:
        """Return the mean posterior over ``n_passes`` stochastic forward passes.

        ``enable_mc_sampling`` must have been called; otherwise all passes are
        identical and the result silently equals a single deterministic pass.
        """
        if n_passes < 2:
            raise ValueError("n_passes must be at least 2")
        acc: np.ndarray | None = None
        for _ in range(n_passes):
            p = self.posterior(image)
            acc = p if acc is None else acc + p
        assert acc is not None
        return acc / n_passes

    def posterior(self, image: np.ndarray) -> np.ndarray:
        """Return the class posterior ``(n_classes, H, W)``.

        RGB frameworks expect an ``(H, W, 3)`` uint8 array; the U-Net
        framework expects raw multispectral bands ``(C, H, W)`` float32 and
        averages softmax posteriors over ensemble members.
        """
        import torch
        import torch.nn.functional as functional

        if self.framework == "unet":
            x = torch.from_numpy(np.asarray(image, dtype=np.float32)).to(self.device)
            acc = None
            with torch.inference_mode():
                for net, mean, std in self.members:
                    p = net(((x - mean) / std)[None]).softmax(dim=1)[0]
                    acc = p if acc is None else acc + p
            return (acc / len(self.members)).float().cpu().numpy()

        h, w = image.shape[:2]
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            if self.framework == "segformer":
                logits = outputs.logits  # (1, C, h/4, w/4)
                logits = functional.interpolate(
                    logits, size=(h, w), mode="bilinear", align_corners=False)
                probs = logits.softmax(dim=1)[0]
            else:  # mask2former
                class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]  # drop no-object
                mask_probs = outputs.masks_queries_logits.sigmoid()
                scores = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
                scores = functional.interpolate(
                    scores, size=(h, w), mode="bilinear", align_corners=False)
                probs = (scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12))[0]
        return probs.float().cpu().numpy()


class EmbeddingModel:
    """A frozen image-embedding model (DINOv2 via timm, or CLIP via open_clip)."""

    def __init__(self, embedding_key: str, device: str | None = None):
        registry = load_model_registry()["embedding_models"]
        if embedding_key not in registry:
            raise KeyError(f"unknown embedding key: {embedding_key}; known: {sorted(registry)}")
        self.spec = registry[embedding_key]
        self.embedding_key = embedding_key
        self.device = device or pick_device()
        self._load()

    def _load(self) -> None:
        import torch

        if self.spec["framework"] == "timm":
            import timm

            self.model = timm.create_model(self.spec["checkpoint"], pretrained=True, num_classes=0)
            cfg = timm.data.resolve_model_data_config(self.model)
            self.transform = timm.data.create_transform(**cfg, is_training=False)
        elif self.spec["framework"] == "open_clip":
            import open_clip

            self.model, _, self.transform = open_clip.create_model_and_transforms(
                self.spec["checkpoint"], pretrained=self.spec["pretrained"])
        else:
            raise ValueError(f"unsupported embedding framework: {self.spec['framework']}")
        self.model.eval().to(torch.device(self.device))

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Return a 1-D embedding for an RGB image array."""
        import torch
        from PIL import Image as PILImage

        pil = PILImage.fromarray(image)
        x = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            if self.spec["framework"] == "open_clip":
                feat = self.model.encode_image(x)
            else:
                feat = self.model(x)
        return feat[0].float().cpu().numpy()


def cache_arrays_for_posterior(
    probs: np.ndarray,
    critical_channels: list[int],
    stride: int,
) -> dict[str, np.ndarray]:
    """Assemble the arrays stored per image by the stage-3 cache."""
    return {
        "critical_probs": probs[critical_channels].astype(np.float16),
        "argmax": probs.argmax(axis=0).astype(np.uint8),
        "strided_probs": probs[:, ::stride, ::stride].astype(np.float16),
        "critical_channels": np.asarray(critical_channels, dtype=np.int16),
    }


def resolve_checkpoint(spec: dict) -> bool:
    """Return True if a model spec's checkpoint(s) are available.

    Local directories (trained checkpoints, ensemble members) are checked on
    disk; anything else is resolved against the Hugging Face Hub.
    """
    entries = spec.get("members") or [spec["checkpoint"]]
    for entry in entries:
        if _resolve_local_checkpoint(entry) is not None:
            continue
        try:
            from huggingface_hub import model_info

            model_info(entry)
        except Exception:
            return False
    return True
