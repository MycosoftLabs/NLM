"""
Quantized Linear Layer
=======================

Drop-in replacement for nn.Linear with TurboQuant_mse weight quantization.

Stores quantized weight indices + per-row scale factors instead of
full-precision weights. Dequantizes on-the-fly during forward pass.

Weight rows are normalized to unit norm before quantization (the TurboQuant
assumption), with the original norm stored as a per-row scale factor
in float16 for efficiency.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlm.quantization.turbo_mse import TurboQuantMSE


class QuantizedLinear(nn.Module):
    """nn.Linear with TurboQuant_mse weight quantization.

    Weights are quantized post-training. During forward pass,
    weights are dequantized on-the-fly for computation.

    Memory: bit_width bits per weight element + 2 bytes per output row (scale).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bit_width: int = 4,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width

        # TurboQuant MSE quantizer for the in_features dimension
        self.quantizer = TurboQuantMSE(in_features, bit_width=bit_width, seed=seed, device=device)

        # Quantized storage (will be populated by from_linear or manual init)
        self.register_buffer(
            "weight_indices",
            torch.zeros(out_features, in_features, dtype=torch.int16, device=device),
        )
        # Per-row scale factor (original row norms, stored in float16)
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, dtype=torch.float16, device=device),
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bit_width: int = 4,
        seed: int = 42,
    ) -> "QuantizedLinear":
        """Convert a trained nn.Linear to quantized form (post-training).

        Steps:
        1. Extract weight matrix W ∈ ℝ^{out × in}
        2. Per-row normalize: scale_i = ‖W_i‖, W_norm_i = W_i / scale_i
        3. Quantize W_norm using TurboQuantMSE
        4. Store indices + scale + bias

        Args:
            linear: Trained nn.Linear module.
            bit_width: Quantization bit width (2, 3, or 4).
            seed: Random seed for rotation matrix.

        Returns:
            QuantizedLinear with quantized weights.
        """
        device = linear.weight.device
        q_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bit_width=bit_width,
            seed=seed,
            device=device,
        )

        with torch.no_grad():
            weight = linear.weight.data  # (out_features, in_features)

            # Per-row normalization
            row_norms = torch.linalg.norm(weight, dim=-1)  # (out_features,)
            row_norms = row_norms.clamp(min=1e-8)
            weight_normalized = weight / row_norms.unsqueeze(-1)

            # Quantize normalized weights
            indices = q_linear.quantizer.quantize(weight_normalized)
            q_linear.weight_indices.copy_(indices)
            q_linear.weight_scale.copy_(row_norms.to(torch.float16))

            # Copy bias if present
            if linear.bias is not None and q_linear.bias is not None:
                q_linear.bias.data.copy_(linear.bias.data)

        return q_linear

    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize weight matrix on-the-fly.

        Returns:
            Weight tensor of shape (out_features, in_features).
        """
        w_hat = self.quantizer.dequantize(self.weight_indices)
        # Re-apply scale
        w_hat = w_hat * self.weight_scale.float().unsqueeze(-1)
        return w_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantized weights.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        weight = self._dequantize_weight()
        return F.linear(x, weight, self.bias)

    def memory_bytes(self) -> int:
        """Actual memory used by quantized weights (excluding rotation/codebook)."""
        # Indices: out * in * bit_width bits
        index_bytes = (self.out_features * self.in_features * self.bit_width + 7) // 8
        # Scales: out * 2 bytes (float16)
        scale_bytes = self.out_features * 2
        # Bias: out * 4 bytes (float32) if present
        bias_bytes = self.out_features * 4 if self.bias is not None else 0
        return index_bytes + scale_bytes + bias_bytes

    def original_memory_bytes(self) -> int:
        """Memory that would be used by the original float32 weights."""
        weight_bytes = self.out_features * self.in_features * 4
        bias_bytes = self.out_features * 4 if self.bias is not None else 0
        return weight_bytes + bias_bytes

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bit_width={self.bit_width}"
        )
