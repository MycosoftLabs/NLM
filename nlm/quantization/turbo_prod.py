"""
TurboQuant Inner Product Quantizer
====================================

Algorithm 2 from TurboQuant (arxiv 2504.19874):

    Input: dimension d, bit-width b
    Instantiate TurboQuant_mse with bit-width b-1
    Generate random projection matrix S ∈ ℝ^{d×d}, S_{i,j} ~ N(0,1)

    Quant_prod(x):
        idx ← Quant_mse(x)
        r ← x - DeQuant_mse(idx)     # residual
        qjl ← sign(S · r)            # 1-bit QJL of residual
        γ ← ‖r‖₂
        output: (idx, qjl, γ)

    DeQuant_prod(idx, qjl, γ):
        x̃_mse ← DeQuant_mse(idx)
        x̃_qjl ← (√(π/2)/d) · γ · S^T · qjl
        output: x̃_mse + x̃_qjl

Properties:
    - E[⟨y, x̃⟩] = ⟨y, x⟩  (UNBIASED inner product estimator)
    - D_prod ≤ (√3·π²·‖y‖²/d) · (1/4^b)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from nlm.quantization.qjl import QJLTransform
from nlm.quantization.turbo_mse import TurboQuantMSE


@dataclass
class QuantizedProdResult:
    """Result of TurboQuant_prod quantization.

    Stores MSE indices (b-1 bits per coordinate) + QJL sign bits
    (1 bit per coordinate) + residual norm (1 float per vector).
    """

    mse_indices: torch.Tensor  # (..., dim) int16
    qjl_signs: torch.Tensor  # (..., dim) int8, values in {-1, +1}
    residual_norm: torch.Tensor  # (...,) float

    def to(self, device: torch.device | str) -> "QuantizedProdResult":
        return QuantizedProdResult(
            mse_indices=self.mse_indices.to(device),
            qjl_signs=self.qjl_signs.to(device),
            residual_norm=self.residual_norm.to(device),
        )


class TurboQuantProd(nn.Module):
    """TurboQuant for inner product preservation.

    Two-stage quantizer: TurboQuant_mse (b-1 bits) + QJL residual (1 bit).
    The combined estimator is UNBIASED for inner products.
    """

    def __init__(
        self,
        dim: int,
        bit_width: int = 3,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ):
        """
        Args:
            dim: Vector dimension.
            bit_width: Total bit budget per coordinate.
                Stage 1 (MSE) uses bit_width-1 bits.
                Stage 2 (QJL) uses 1 bit for residual.
            seed: Random seed.
            device: Target device.
        """
        super().__init__()
        self.dim = dim
        self.bit_width = bit_width

        if bit_width < 2:
            raise ValueError("TurboQuantProd requires bit_width >= 2 (1 for MSE + 1 for QJL)")

        # Stage 1: MSE quantizer with b-1 bits
        self.mse_quantizer = TurboQuantMSE(
            dim, bit_width=bit_width - 1, seed=seed, device=device
        )

        # Stage 2: QJL on residual (1 bit per coordinate)
        # Use different seed to ensure independence
        self.qjl = QJLTransform(dim, seed=seed + 1000, device=device)

    def quantize(self, x: torch.Tensor) -> QuantizedProdResult:
        """Quantize vector with inner-product-preserving two-stage method.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            QuantizedProdResult with MSE indices, QJL signs, and residual norm.
        """
        # Stage 1: MSE quantization
        mse_indices = self.mse_quantizer.quantize(x)

        # Compute residual
        x_mse = self.mse_quantizer.dequantize(mse_indices)
        residual = x - x_mse

        # Stage 2: QJL on residual
        qjl_signs, residual_norm = self.qjl.quantize(residual)

        return QuantizedProdResult(
            mse_indices=mse_indices,
            qjl_signs=qjl_signs,
            residual_norm=residual_norm,
        )

    def dequantize(self, result: QuantizedProdResult) -> torch.Tensor:
        """Full reconstruction: MSE dequantization + QJL residual correction.

        Args:
            result: QuantizedProdResult from quantize().

        Returns:
            Reconstructed tensor of shape (..., dim).
        """
        x_mse = self.mse_quantizer.dequantize(result.mse_indices)
        x_qjl = self.qjl.dequantize(result.qjl_signs, result.residual_norm)
        return x_mse + x_qjl

    def attention_scores(
        self,
        Q: torch.Tensor,
        quantized_K: QuantizedProdResult,
    ) -> torch.Tensor:
        """Compute attention scores Q @ K^T using asymmetric estimation.

        Decomposes into two efficient operations:
            score = Q @ DeQuant_mse(K_idx)^T
                  + (√(π/2)/d) · (Q @ S^T) @ diag(γ) @ qjl_signs^T

        Args:
            Q: Query tensor of shape (batch, heads, seq_q, head_dim).
            quantized_K: Quantized keys with:
                mse_indices: (batch, heads, seq_k, head_dim)
                qjl_signs: (batch, heads, seq_k, head_dim)
                residual_norm: (batch, heads, seq_k)

        Returns:
            Attention scores of shape (batch, heads, seq_q, seq_k).
        """
        # Stage 1: Q @ K_mse^T
        K_mse = self.mse_quantizer.dequantize(quantized_K.mse_indices)
        scores_mse = torch.matmul(Q, K_mse.transpose(-2, -1))

        # Stage 2: QJL correction
        # Project queries: S·q for each query
        Sq = Q @ self.qjl.projection.t()  # (batch, heads, seq_q, dim)

        # Dot with sign bits: (batch, heads, seq_q, dim) @ (batch, heads, dim, seq_k)
        qjl_float = quantized_K.qjl_signs.float()
        scores_qjl = torch.matmul(Sq, qjl_float.transpose(-2, -1))

        # Scale by (√(π/2)/d) · γ_k
        scores_qjl = scores_qjl * self.qjl.scale_factor
        scores_qjl = scores_qjl * quantized_K.residual_norm.unsqueeze(-2)

        return scores_mse + scores_qjl

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, QuantizedProdResult]:
        """Quantize and dequantize (for evaluation).

        Returns:
            Tuple of (reconstructed x̃, QuantizedProdResult).
        """
        result = self.quantize(x)
        x_hat = self.dequantize(result)
        return x_hat, result
