"""Versioned, strict model artifacts. Load only bounded, approved local artifacts.

Weights-only deserialization prevents arbitrary pickle object loading; it is not a
resource quota or a provenance/approval check. Exact optimizer/RNG resume is out of scope.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from nlm.model.config import NLMConfig

ARTIFACT_VERSION = 1
FRAME_SCHEMA_VERSION = "rooted-frame-v1"
PREPROCESSOR_VERSION = "deterministic-v1"


def config_from_dict(raw: Any) -> NLMConfig:
    """Require every field; never silently fill defaults from a partial checkpoint."""
    defaults = asdict(NLMConfig())
    if not isinstance(raw, dict) or raw.keys() != defaults.keys():
        raise ValueError("Artifact must contain exactly the complete NLMConfig fields")
    for key, default in defaults.items():
        value = raw[key]
        if isinstance(default, bool):
            valid = type(value) is bool
        elif isinstance(default, int):
            valid = type(value) is int and value >= (0 if key == "quant_seed" else 1)
        elif isinstance(default, float):
            valid = type(value) in (int, float) and math.isfinite(value) and 0 <= value < 1
        elif isinstance(default, list):
            valid = isinstance(value, list) and all(isinstance(v, str) for v in value)
        else:
            valid = isinstance(value, str) and (value == "auto" or (value.isdigit() and int(value) > 0))
        if not valid:
            raise ValueError(f"Invalid config field: {key}")
    cfg = NLMConfig(**raw)
    for width, heads in [(cfg.hidden_dim, cfg.num_heads), (cfg.hidden_dim, cfg.fusion_num_heads),
                         (cfg.graph_hidden_dim, cfg.graph_num_heads)]:
        if width % heads:
            raise ValueError("Config dimensions must be divisible by head counts")
    if cfg.quant_enabled:
        raise ValueError("Quantized artifacts require a separately versioned loader")
    return cfg


def _validate_state(state: Any, model: nn.Module) -> None:
    expected = model.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("Artifact weight keys do not match model")
    # Validate the entire state before load_state_dict can partially mutate model.
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(f"Invalid weight tensor: {key}")
        if tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype:
            raise ValueError(f"Weight shape/dtype mismatch: {key}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Nonfinite weight: {key}")


def _header(config: NLMConfig) -> Dict[str, Any]:
    raw = asdict(config)
    config_from_dict(raw)
    return {"artifact_version": ARTIFACT_VERSION, "frame_schema_version": FRAME_SCHEMA_VERSION,
            "preprocessor_version": PREPROCESSOR_VERSION, "config": raw}


def _read(path: Path, device: str) -> Dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        try:
            artifact = json.loads(metadata["nlm_artifact"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("Missing/invalid safetensors artifact header") from exc
        if not isinstance(artifact, dict):
            raise ValueError("Artifact header must be a mapping")
        artifact["model_state_dict"] = load_file(str(path), device=device)
        return artifact
    if path.suffix not in (".pt", ".pth"):
        raise ValueError("Use .pt/.pth or .safetensors; formats never silently fall back")
    return torch.load(path, map_location=device, weights_only=True)


def _validate_artifact(artifact: Any) -> NLMConfig:
    if not isinstance(artifact, dict):
        raise ValueError("Artifact must be a mapping")
    if type(artifact.get("artifact_version")) is not int or artifact["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError("Unsupported artifact version; legacy artifacts need explicit offline migration")
    if (artifact.get("frame_schema_version") != FRAME_SCHEMA_VERSION or
            artifact.get("preprocessor_version") != PREPROCESSOR_VERSION):
        raise ValueError("Artifact schema/preprocessor version mismatch")
    return config_from_dict(artifact.get("config"))


def save_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], epoch: int,
                    loss: float, config: NLMConfig, path: str | Path,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
    path = Path(path)
    if path.suffix not in (".pt", ".pth"):
        raise ValueError("Training checkpoints require .pt or .pth")
    if type(epoch) is not int or epoch < 0 or not math.isfinite(loss):
        raise ValueError("Invalid checkpoint epoch/loss")
    if getattr(model, "config", None) != config:
        raise ValueError("Model/config mismatch")
    json.dumps(metadata or {}, allow_nan=False)
    artifact = {**_header(config), "model_state_dict": model.state_dict(),
                "epoch": epoch, "loss": loss, "metadata": metadata or {}}
    _validate_state(artifact["model_state_dict"], model)
    if optimizer is not None:
        artifact["optimizer_state_dict"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def load_checkpoint(path: str | Path, model: Optional[nn.Module] = None,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    device: str = "cpu") -> Dict[str, Any]:
    artifact = _read(Path(path), device)
    config = _validate_artifact(artifact)
    if model is None:
        from nlm.model.nlm_model import NatureLearningModel
        model = NatureLearningModel(config).to(device)
    elif getattr(model, "config", None) != config:
        raise ValueError("Supplied model config differs from artifact")
    _validate_state(artifact.get("model_state_dict"), model)
    if optimizer is not None and "optimizer_state_dict" not in artifact:
        raise ValueError("Artifact has no optimizer state")
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(artifact["optimizer_state_dict"])
    return artifact


def load_model(path: str | Path, device: str = "cpu") -> nn.Module:
    """Reconstruct a model from complete artifact config; strict load, eval mode."""
    from nlm.model.nlm_model import NatureLearningModel
    artifact = _read(Path(path), device)
    config = _validate_artifact(artifact)
    model = NatureLearningModel(config).to(device)
    _validate_state(artifact.get("model_state_dict"), model)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    return model.eval()


def export_weights(model: nn.Module, path: str | Path, config: NLMConfig) -> None:
    path = Path(path)
    if getattr(model, "config", None) != config:
        raise ValueError("Model/config mismatch")
    header = _header(config)
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    _validate_state(tensors, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file
        save_file(tensors, str(path), metadata={"nlm_artifact": json.dumps(header)})
    elif path.suffix in (".pt", ".pth"):
        torch.save({**header, "model_state_dict": tensors}, path)
    else:
        raise ValueError("Use .pt/.pth or .safetensors; formats never silently fall back")


def load_pretrained(model: nn.Module, path: str | Path, device: str = "cpu",
                    strict: bool = True) -> nn.Module:
    if strict is not True:
        raise ValueError("Permissive loading is disabled; migrate artifacts explicitly")
    load_checkpoint(path, model=model, device=device)
    return model
