"""
Quantized KV Cache
===================

Memory-efficient KV cache using TurboQuant compression:
- Keys: TurboQuant_prod (preserves inner products for Q @ K^T)
- Values: Per-token scalar quantization (values are linearly combined)

This follows the architecture shown in the QJL paper (arxiv 2406.03482):
    K → QJL(·)          → compressed keys in cache
    V → Per-token Quant  → compressed values in cache
    Attention: softmax(q^T K^T) V computed via asymmetric estimation
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlm.quantization.turbo_prod import QuantizedProdResult, TurboQuantProd


class ScalarValueQuantizer(nn.Module):
    """Per-token uniform scalar quantization for attention values.

    For each token's value vector, stores (min, max, quantized_ints).
    Uses uniform quantization within [min, max] range.
    """

    def __init__(self, bit_width: int = 4):
        super().__init__()
        self.bit_width = bit_width
        self.n_levels = 2**bit_width

    def quantize(
        self, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize values with per-token min/max scaling.

        Args:
            v: Value tensor of shape (..., dim).

        Returns:
            Tuple of (indices, v_min, v_max):
                indices: (..., dim) int16
                v_min: (..., 1) float
                v_max: (..., 1) float
        """
        v_min = v.min(dim=-1, keepdim=True).values
        v_max = v.max(dim=-1, keepdim=True).values

        # Avoid division by zero for constant vectors
        v_range = (v_max - v_min).clamp(min=1e-8)

        # Normalize to [0, 1] then scale to [0, n_levels - 1]
        normalized = (v - v_min) / v_range
        indices = (normalized * (self.n_levels - 1)).round().clamp(0, self.n_levels - 1)

        return indices.to(torch.int16), v_min, v_max

    def dequantize(
        self,
        indices: torch.Tensor,
        v_min: torch.Tensor,
        v_max: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct values from indices and min/max.

        Args:
            indices: (..., dim) int16.
            v_min: (..., 1) float.
            v_max: (..., 1) float.

        Returns:
            Reconstructed values of shape (..., dim).
        """
        v_range = v_max - v_min
        return v_min + (indices.float() / (self.n_levels - 1)) * v_range


class QuantizedKVCache(nn.Module):
    """Quantized Key-Value cache for attention.

    Keys use TurboQuant_prod (inner-product preserving) and values
    use per-token scalar quantization. Supports incremental appending
    for autoregressive generation.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int = 1,
        key_bit_width: int = 3,
        value_bit_width: int = 4,
        max_seq_len: int = 4096,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Key quantizer: TurboQuant_prod
        self.key_quantizer = TurboQuantProd(
            head_dim, bit_width=key_bit_width, seed=seed, device=device
        )
        # Value quantizer: per-token scalar
        self.value_quantizer = ScalarValueQuantizer(bit_width=value_bit_width)

        # Cache storage (pre-allocated for efficiency)
        self._cache_len = 0
        self._key_mse_indices: Optional[torch.Tensor] = None
        self._key_qjl_signs: Optional[torch.Tensor] = None
        self._key_residual_norm: Optional[torch.Tensor] = None
        self._value_indices: Optional[torch.Tensor] = None
        self._value_min: Optional[torch.Tensor] = None
        self._value_max: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Clear the cache."""
        self._cache_len = 0
        self._key_mse_indices = None
        self._key_qjl_signs = None
        self._key_residual_norm = None
        self._value_indices = None
        self._value_min = None
        self._value_max = None

    @property
    def seq_len(self) -> int:
        return self._cache_len

    def append(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Quantize and append new key/value pairs to cache.

        Args:
            keys: (batch, num_heads, new_len, head_dim)
            values: (batch, num_heads, new_len, head_dim)
        """
        batch, heads, new_len, dim = keys.shape

        # Quantize keys with TurboQuant_prod
        key_result = self.key_quantizer.quantize(keys)

        # Quantize values with scalar quantization
        val_indices, val_min, val_max = self.value_quantizer.quantize(values)

        if self._key_mse_indices is None:
            # First append — initialize cache
            self._key_mse_indices = key_result.mse_indices
            self._key_qjl_signs = key_result.qjl_signs
            self._key_residual_norm = key_result.residual_norm
            self._value_indices = val_indices
            self._value_min = val_min
            self._value_max = val_max
        else:
            # Concatenate along sequence dimension (dim=2)
            self._key_mse_indices = torch.cat(
                [self._key_mse_indices, key_result.mse_indices], dim=2
            )
            self._key_qjl_signs = torch.cat(
                [self._key_qjl_signs, key_result.qjl_signs], dim=2
            )
            self._key_residual_norm = torch.cat(
                [self._key_residual_norm, key_result.residual_norm], dim=2
            )
            self._value_indices = torch.cat(
                [self._value_indices, val_indices], dim=2
            )
            self._value_min = torch.cat([self._value_min, val_min], dim=2)
            self._value_max = torch.cat([self._value_max, val_max], dim=2)

        self._cache_len += new_len

    def get_quantized_keys(self) -> QuantizedProdResult:
        """Return the quantized key cache."""
        return QuantizedProdResult(
            mse_indices=self._key_mse_indices,
            qjl_signs=self._key_qjl_signs,
            residual_norm=self._key_residual_norm,
        )

    def get_dequantized_values(self) -> torch.Tensor:
        """Dequantize and return cached values."""
        return self.value_quantizer.dequantize(
            self._value_indices, self._value_min, self._value_max
        )

    def compute_attention(
        self,
        query: torch.Tensor,
        scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention output using quantized cache.

        Uses TurboQuant_prod asymmetric estimator for Q @ K^T,
        then applies softmax, then multiplies by dequantized values.

        Args:
            query: (batch, num_heads, seq_q, head_dim)
            scale: Attention scale factor. Defaults to 1/√head_dim.
            mask: Optional attention mask.

        Returns:
            Attention output of shape (batch, num_heads, seq_q, head_dim).
        """
        if self._cache_len == 0:
            raise RuntimeError("Cache is empty. Call append() first.")

        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        # Compute attention scores via TurboQuant asymmetric estimation
        quantized_keys = self.get_quantized_keys()
        attn_scores = self.key_quantizer.attention_scores(query, quantized_keys)
        attn_scores = attn_scores * scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Get dequantized values and compute weighted sum
        values = self.get_dequantized_values()
        output = torch.matmul(attn_weights, values)

        return output

    def memory_bytes(self) -> int:
        """Estimate current cache memory usage in bytes."""
        if self._cache_len == 0:
            return 0

        batch, heads, seq, dim = self._key_mse_indices.shape
        key_bits_per_token = self.key_quantizer.bit_width
        val_bits_per_token = self.value_quantizer.bit_width

        # Keys: MSE indices + QJL signs + residual norm
        key_bytes = batch * heads * seq * dim * key_bits_per_token // 8
        key_bytes += batch * heads * seq * 4  # residual norms (float32)

        # Values: indices + min/max per token
        val_bytes = batch * heads * seq * dim * val_bits_per_token // 8
        val_bytes += batch * heads * seq * 8  # min + max (float32 each)

        return key_bytes + val_bytes
