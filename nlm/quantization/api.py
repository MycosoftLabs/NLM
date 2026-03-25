"""
Model-Level Quantization API
==============================

Top-level functions for applying TurboQuant to an entire NLM model:
- quantize_model(): post-training quantization of weights, attention, SSM
- estimate_memory_savings(): calculate compression ratios
"""

from __future__ import annotations

import fnmatch
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from nlm.quantization.attention_wrappers import (
    QuantizedCyclicalAttention,
    QuantizedSparseAttentionFusion,
    QuantizedSpatialLocalityAttention,
)
from nlm.quantization.config import QuantConfig
from nlm.quantization.linear import QuantizedLinear
from nlm.quantization.ssm_compress import QuantizedSelectiveSSMBlock


def quantize_model(
    model: nn.Module,
    config: QuantConfig,
) -> nn.Module:
    """Apply TurboQuant to an NLM model (post-training, in-place).

    Steps:
    1. Replace nn.Linear layers with QuantizedLinear (if quantize_weights)
    2. Replace attention modules with quantized wrappers (if quantize_kv_cache)
    3. Replace SSM blocks with compressed versions (if quantize_ssm_state)

    Args:
        model: Trained NatureLearningModel (or any nn.Module).
        config: QuantConfig specifying what/how to quantize.

    Returns:
        The same model with quantized components (modified in-place).
    """
    if not config.enabled:
        return model

    if config.quantize_kv_cache:
        _quantize_attention_modules(model, config)

    if config.quantize_weights:
        _quantize_linear_layers(
            model,
            bit_width=config.weight_bit_width,
            skip_patterns=config.skip_modules,
            seed=config.seed,
        )

    if config.quantize_ssm_state:
        _quantize_ssm_blocks(model, config)

    return model


def _quantize_linear_layers(
    module: nn.Module,
    bit_width: int = 4,
    skip_patterns: Optional[List[str]] = None,
    seed: int = 42,
    _prefix: str = "",
) -> None:
    """Recursively replace nn.Linear with QuantizedLinear.

    Skips modules whose full name matches any pattern in skip_patterns.
    Also skips Linear layers inside QuantizedLinear (already quantized)
    and inside quantization infrastructure modules.
    """
    skip_patterns = skip_patterns or []

    for name, child in list(module.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name

        # Skip if name matches a skip pattern
        if any(fnmatch.fnmatch(full_name, pat) for pat in skip_patterns):
            continue

        # Skip modules that are already quantized
        if isinstance(child, (QuantizedLinear, QuantizedCyclicalAttention,
                              QuantizedSpatialLocalityAttention,
                              QuantizedSparseAttentionFusion,
                              QuantizedSelectiveSSMBlock)):
            continue

        if isinstance(child, nn.Linear):
            q_linear = QuantizedLinear.from_linear(
                child, bit_width=bit_width, seed=seed
            )
            setattr(module, name, q_linear)
        else:
            _quantize_linear_layers(
                child,
                bit_width=bit_width,
                skip_patterns=skip_patterns,
                seed=seed,
                _prefix=full_name,
            )


def _quantize_attention_modules(
    model: nn.Module, config: QuantConfig
) -> None:
    """Replace attention modules with quantized wrappers."""
    # Import here to avoid circular imports at module level
    from nlm.model.attention import CyclicalAttention, SpatialLocalityAttention
    from nlm.model.fusion import SparseAttentionFusion

    for name, child in list(model.named_children()):
        if isinstance(child, CyclicalAttention):
            setattr(model, name, QuantizedCyclicalAttention(child, config))
        elif isinstance(child, SpatialLocalityAttention):
            setattr(model, name, QuantizedSpatialLocalityAttention(child, config))
        elif isinstance(child, SparseAttentionFusion):
            setattr(model, name, QuantizedSparseAttentionFusion(child, config))
        else:
            _quantize_attention_modules(child, config)


def _quantize_ssm_blocks(model: nn.Module, config: QuantConfig) -> None:
    """Replace SelectiveSSMBlock with QuantizedSelectiveSSMBlock."""
    from nlm.model.ssm_blocks import SelectiveSSMBlock

    for name, child in list(model.named_children()):
        if isinstance(child, SelectiveSSMBlock):
            setattr(
                model,
                name,
                QuantizedSelectiveSSMBlock(
                    child,
                    bit_width=config.ssm_state_bit_width,
                    seed=config.seed,
                ),
            )
        else:
            _quantize_ssm_blocks(child, config)


def estimate_memory_savings(
    model: nn.Module,
    config: QuantConfig,
) -> Dict[str, Any]:
    """Calculate memory savings from quantization.

    Returns:
        Dict with:
            original_bytes: total float32 parameter memory
            quantized_bytes: estimated memory after quantization
            compression_ratio: original / quantized
            per_module: dict of module name → {original, quantized, ratio}
    """
    original_total = 0
    quantized_total = 0
    per_module = {}

    for name, param in model.named_parameters():
        param_bytes = param.numel() * param.element_size()
        original_total += param_bytes

        # Estimate quantized size
        if config.quantize_weights and "weight" in name:
            # Approximate: bit_width bits per element + overhead
            q_bytes = (param.numel() * config.weight_bit_width + 7) // 8
            # Add per-row scale (float16) if it's a 2D weight
            if param.ndim == 2:
                q_bytes += param.shape[0] * 2
            quantized_total += q_bytes
            per_module[name] = {
                "original_bytes": param_bytes,
                "quantized_bytes": q_bytes,
                "ratio": param_bytes / max(q_bytes, 1),
            }
        else:
            quantized_total += param_bytes
            per_module[name] = {
                "original_bytes": param_bytes,
                "quantized_bytes": param_bytes,
                "ratio": 1.0,
            }

    # Add rotation matrix overhead (amortized)
    if config.quantize_weights:
        # One rotation matrix per unique input dimension
        dims_seen = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                dims_seen.add(module.in_features)
        for dim in dims_seen:
            rotation_bytes = dim * dim * 4  # float32
            codebook_bytes = (2**config.weight_bit_width) * 4
            quantized_total += rotation_bytes + codebook_bytes

    return {
        "original_bytes": original_total,
        "quantized_bytes": quantized_total,
        "compression_ratio": original_total / max(quantized_total, 1),
        "original_mb": original_total / (1024 * 1024),
        "quantized_mb": quantized_total / (1024 * 1024),
        "savings_mb": (original_total - quantized_total) / (1024 * 1024),
        "per_module": per_module,
    }
