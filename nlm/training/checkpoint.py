"""
Model Checkpoint Management
============================

Save/load model weights, optimizer state, training metadata.
Uses safetensors when available, falls back to torch.save.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from nlm.model.config import NLMConfig


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    config: NLMConfig,
    path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint.

    Saves model weights, optimizer state, epoch, loss, config, and metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "config": {
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "ff_dim": config.ff_dim,
            "spatial_dim": config.spatial_dim,
            "temporal_dim": config.temporal_dim,
            "spectral_sensory_dim": config.spectral_sensory_dim,
            "world_state_dim": config.world_state_dim,
            "self_state_dim": config.self_state_dim,
            "action_intent_dim": config.action_intent_dim,
            "num_ssm_layers": config.num_ssm_layers,
            "ssm_state_dim": config.ssm_state_dim,
            "graph_hidden_dim": config.graph_hidden_dim,
            "graph_num_layers": config.graph_num_layers,
        },
        "metadata": metadata or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Returns checkpoint dict. Optionally loads into model and optimizer.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def export_weights(
    model: nn.Module,
    path: str | Path,
    config: NLMConfig,
) -> None:
    """Export just model weights for inference (no optimizer state)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file
        tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
        save_file(tensors, str(path))
    except ImportError:
        torch.save(model.state_dict(), path)


def load_pretrained(
    model: nn.Module,
    path: str | Path,
    device: str = "cpu",
    strict: bool = False,
) -> nn.Module:
    """Load pretrained weights, handling architecture changes gracefully."""
    path = Path(path)

    try:
        from safetensors.torch import load_file
        state_dict = load_file(str(path), device=device)
    except (ImportError, Exception):
        state_dict = torch.load(path, map_location=device, weights_only=True)

    # Handle checkpoint format (may have 'model_state_dict' key)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    return model
