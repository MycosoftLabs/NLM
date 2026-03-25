"""
Quantized Attention Wrappers
==============================

Drop-in replacements for CyclicalAttention, SpatialLocalityAttention,
and SparseAttentionFusion with TurboQuant KV cache compression.

Each wrapper:
1. Copies all parameters from the original module
2. Intercepts K/V after projection to quantize into cache
3. Computes attention scores via TurboQuant asymmetric estimation
4. Preserves all original biases (cyclical, spatial)
5. Uses the exact same forward signature as the original
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlm.quantization.config import QuantConfig
from nlm.quantization.kv_cache import QuantizedKVCache


class QuantizedCyclicalAttention(nn.Module):
    """CyclicalAttention with TurboQuant KV cache compression.

    Wraps the original CyclicalAttention, replacing the K/V storage
    with quantized cache while preserving the cyclical temporal bias.
    """

    def __init__(self, original: nn.Module, quant_config: QuantConfig):
        super().__init__()
        self.d_model = original.d_model
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim

        # Copy projection layers
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj

        # Copy cyclical bias components
        self.cycle_proj = original.cycle_proj
        self.cycle_scale = original.cycle_scale
        self.dropout = original.dropout

        # Quantized KV cache
        self.kv_cache = QuantizedKVCache(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            key_bit_width=quant_config.key_bit_width,
            value_bit_width=quant_config.value_bit_width,
            seed=quant_config.seed,
        )

    def forward(
        self,
        x: torch.Tensor,
        temporal_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Quantize K and V into cache
        self.kv_cache.reset()
        self.kv_cache.append(K, V)

        # Compute attention scores via TurboQuant asymmetric estimation
        quantized_keys = self.kv_cache.get_quantized_keys()
        attn = self.kv_cache.key_quantizer.attention_scores(Q, quantized_keys)
        attn = attn / math.sqrt(self.head_dim)

        # Cyclical bias (unchanged from original)
        cycle_emb = self.cycle_proj(temporal_features)
        cycle_emb = cycle_emb.permute(0, 2, 1)
        cycle_norm = F.normalize(cycle_emb, dim=-1)
        cycle_bias = torch.bmm(
            cycle_norm.reshape(batch * self.num_heads, seq_len, 1),
            cycle_norm.reshape(batch * self.num_heads, 1, seq_len),
        ).view(batch, self.num_heads, seq_len, seq_len)
        attn = attn + cycle_bias * self.cycle_scale.view(1, -1, 1, 1)

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Multiply by dequantized values
        V_deq = self.kv_cache.get_dequantized_values()
        out = torch.matmul(attn, V_deq)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.o_proj(out)


class QuantizedSpatialLocalityAttention(nn.Module):
    """SpatialLocalityAttention with TurboQuant KV cache compression.

    Preserves geographic distance decay bias while compressing K/V.
    """

    def __init__(self, original: nn.Module, quant_config: QuantConfig):
        super().__init__()
        self.d_model = original.d_model
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim

        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj

        self.log_bandwidth = original.log_bandwidth
        self.dropout = original.dropout

        self.kv_cache = QuantizedKVCache(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            key_bit_width=quant_config.key_bit_width,
            value_bit_width=quant_config.value_bit_width,
            seed=quant_config.seed,
        )

    def forward(
        self,
        x: torch.Tensor,
        spatial_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Quantize K and V
        self.kv_cache.reset()
        self.kv_cache.append(K, V)

        # Attention scores via TurboQuant
        quantized_keys = self.kv_cache.get_quantized_keys()
        attn = self.kv_cache.key_quantizer.attention_scores(Q, quantized_keys)
        attn = attn / math.sqrt(self.head_dim)

        # Spatial distance decay bias (unchanged)
        coords_diff = spatial_coords.unsqueeze(2) - spatial_coords.unsqueeze(1)
        dist_sq = (coords_diff**2).sum(dim=-1)
        bandwidth = torch.exp(self.log_bandwidth)
        spatial_bias = -dist_sq.unsqueeze(1) / (2.0 * bandwidth.view(1, -1, 1, 1) ** 2)
        attn = attn + spatial_bias

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        V_deq = self.kv_cache.get_dequantized_values()
        out = torch.matmul(attn, V_deq)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.o_proj(out)


class QuantizedSparseAttentionFusion(nn.Module):
    """SparseAttentionFusion with TurboQuant on the cross-stream attention.

    The cross-stream gating is preserved in full precision;
    only the self-attention K/V across stream tokens is quantized.
    """

    def __init__(self, original: nn.Module, quant_config: QuantConfig):
        super().__init__()
        self.hidden_dim = original.hidden_dim
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.num_streams = original.num_streams

        # Copy all sub-modules
        self.stream_projs = original.stream_projs
        self.gating = original.gating
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.ff = original.ff
        self.norm1 = original.norm1
        self.norm2 = original.norm2
        self.dropout = original.dropout

        self.kv_cache = QuantizedKVCache(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            key_bit_width=quant_config.key_bit_width,
            value_bit_width=quant_config.value_bit_width,
            seed=quant_config.seed,
        )

    def forward(
        self,
        spatial: torch.Tensor,
        temporal: torch.Tensor,
        spectral_sensory: torch.Tensor,
        world_state: torch.Tensor,
        self_state: torch.Tensor,
        action_intent: torch.Tensor,
    ) -> torch.Tensor:
        raw_streams = [spatial, temporal, spectral_sensory, world_state, self_state, action_intent]

        # Project each stream to hidden_dim (full precision)
        projected = [proj(s) for proj, s in zip(self.stream_projs, raw_streams)]

        # Cross-stream gating (full precision)
        gated = self.gating(projected)

        # Stack as sequence for self-attention
        x = torch.stack(gated, dim=1)
        batch, num_tokens, _ = x.shape

        # Self-attention with quantized K/V
        residual = x
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm).view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Quantize K/V
        self.kv_cache.reset()
        self.kv_cache.append(K, V)

        # Compute attention via TurboQuant
        quantized_keys = self.kv_cache.get_quantized_keys()
        attn = self.kv_cache.key_quantizer.attention_scores(Q, quantized_keys)
        attn = attn / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        V_deq = self.kv_cache.get_dequantized_values()
        out = torch.matmul(attn, V_deq)
        out = out.transpose(1, 2).reshape(batch, num_tokens, self.hidden_dim)
        out = self.o_proj(out)
        x = residual + self.dropout(out)

        # FF (unchanged)
        x = x + self.dropout(self.ff(self.norm2(x)))

        return x.mean(dim=1)
