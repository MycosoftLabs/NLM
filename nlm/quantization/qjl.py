"""
Quantized Johnson-Lindenstrauss (QJL) Transform
=================================================

From QJL paper (arxiv 2406.03482):

    Q_qjl(x) = sign(S · x)           → {-1, +1}^m
    Q_qjl^{-1}(z) = (√(π/2)/m) · S^T · z    → ℝ^d

Key property (asymmetric inner product estimation):
    ⟨S·q, QJL(S,k)⟩ · (√(π/2)/m) · ‖k‖₂ ≈_ε ⟨q, k⟩

The query q is projected (but NOT quantized) as S·q ∈ ℝ^m.
The key k is projected AND sign-quantized as sign(S·k) ∈ {±1}^m.
The inner product is estimated asymmetrically.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from nlm.quantization.rotation import generate_projection_matrix


class QJLTransform(nn.Module):
    """QJL: 1-bit sign quantization with Gaussian projection.

    Projects vectors via random Gaussian matrix S, then takes the sign.
    Provides unbiased inner product estimation via asymmetric decoding.
    """

    def __init__(
        self,
        dim: int,
        projection_dim: Optional[int] = None,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ):
        """
        Args:
            dim: Input vector dimension d.
            projection_dim: Output dimension m. Defaults to dim.
            seed: Random seed for reproducibility.
            device: Target device.
        """
        super().__init__()
        self.dim = dim
        self.projection_dim = projection_dim or dim

        # S ∈ ℝ^{m×d}, S_{ij} ~ N(0,1)
        S = generate_projection_matrix(
            dim, m=self.projection_dim, seed=seed, device=device
        )
        self.register_buffer("projection", S)

        # Precompute scaling factor: √(π/2) / m
        self.register_buffer(
            "scale_factor",
            torch.tensor(math.sqrt(math.pi / 2.0) / self.projection_dim),
        )

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize vector to sign bits + norm.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Tuple of:
                sign_bits: {-1, +1} tensor of shape (..., projection_dim), dtype=int8
                norm: ‖x‖₂ tensor of shape (...,)
        """
        norm = torch.linalg.norm(x, dim=-1)

        # Project: S · x (batch-compatible via right-multiply by S^T)
        projected = x @ self.projection.t()  # (..., projection_dim)

        # Sign quantization
        sign_bits = torch.sign(projected)
        # Replace zeros with +1 (rare but possible)
        sign_bits[sign_bits == 0] = 1.0
        sign_bits = sign_bits.to(torch.int8)

        return sign_bits, norm

    def dequantize(
        self, sign_bits: torch.Tensor, norm: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct vector from sign bits and norm.

        x̃ = (√(π/2)/m) · ‖x‖₂ · S^T · z

        Args:
            sign_bits: {-1, +1} tensor of shape (..., projection_dim).
            norm: ‖x‖₂ tensor of shape (...,).

        Returns:
            Reconstructed tensor of shape (..., dim).
        """
        z = sign_bits.float()
        # S^T · z: (d, m) @ (..., m) → via (..., m) @ (m, d)
        reconstructed = z @ self.projection  # (..., dim)
        # Scale by (√(π/2)/m) · norm
        reconstructed = reconstructed * self.scale_factor * norm.unsqueeze(-1)
        return reconstructed

    def project_query(self, q: torch.Tensor) -> torch.Tensor:
        """Project query vector (without quantization) for asymmetric estimation.

        Args:
            q: Query tensor of shape (..., dim).

        Returns:
            Projected query S·q of shape (..., projection_dim).
        """
        return q @ self.projection.t()

    def estimate_inner_product(
        self,
        q: torch.Tensor,
        sign_bits_k: torch.Tensor,
        norm_k: torch.Tensor,
    ) -> torch.Tensor:
        """Asymmetric inner product estimation: ⟨q, k⟩ ≈ ⟨Sq, sign(Sk)⟩ · scale · ‖k‖.

        This is the key operation for attention: project q once, then
        dot-product with stored sign bits for each cached key.

        Args:
            q: Query tensor of shape (..., dim).
            sign_bits_k: Quantized key signs of shape (..., projection_dim).
            norm_k: Key norms of shape (...,).

        Returns:
            Estimated inner product, shape depends on input broadcasting.
        """
        # Project query
        Sq = self.project_query(q)  # (..., projection_dim)

        # Dot product with sign bits
        dots = (Sq * sign_bits_k.float()).sum(dim=-1)  # (...,)

        # Scale: (√(π/2)/m) · ‖k‖₂
        estimated = dots * self.scale_factor * norm_k

        return estimated
