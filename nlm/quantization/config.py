"""
TurboQuant Configuration
========================

Dataclass configuration for TurboQuant quantization parameters.
Follows the flat-field convention of NLMConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QuantConfig:
    """Configuration for TurboQuant quantization.

    Controls weight quantization (TurboQuant_mse), KV cache quantization
    (TurboQuant_prod for keys, scalar for values), and SSM state compression.
    """

    enabled: bool = False

    # --- Weight quantization (TurboQuant_mse on nn.Linear) ---
    quantize_weights: bool = True
    weight_bit_width: int = 4  # 2, 3, or 4 bits

    # --- KV cache quantization ---
    quantize_kv_cache: bool = True
    key_bit_width: int = 3  # TurboQuant_prod (inner-product preserving)
    value_bit_width: int = 4  # per-token scalar quantization

    # --- SSM state compression (TurboQuant_mse) ---
    quantize_ssm_state: bool = False
    ssm_state_bit_width: int = 4

    # --- Shared settings ---
    seed: int = 42
    skip_modules: List[str] = field(default_factory=list)
