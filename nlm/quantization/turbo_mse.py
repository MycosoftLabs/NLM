"""
TurboQuant MSE Quantizer
=========================

Algorithm 1 from TurboQuant (arxiv 2504.19874):

    Input: dimension d, bit-width b
    Generate random rotation matrix Π ∈ ℝ^{d×d}
    Construct codebook: centroids c_1,...,c_{2^b} ∈ [-1,1]

    Quant_mse(x):
        y ← x @ Π^T               (rotate)
        idx_j ← nearest(y_j, codebook)  for each j
        output: idx

    DeQuant_mse(idx):
        ỹ_j ← codebook[idx_j]     for each j
        x̃ ← ỹ @ Π                (inverse rotation)
        output: x̃

After rotation, each coordinate follows a Beta distribution:
    f_X(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^{(d-3)/2}

MSE distortion bound: D_mse ≤ (√3·π/2) · (1/4^b) ≈ 2.7/4^b
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from nlm.quantization.codebook import compute_codebook
from nlm.quantization.rotation import generate_rotation_matrix


class TurboQuantMSE(nn.Module):
    """TurboQuant MSE quantizer: random rotation + scalar quantization.

    Stores rotation matrix Π and codebook as non-trainable buffers.
    Supports arbitrary leading batch dimensions.
    """

    def __init__(
        self,
        dim: int,
        bit_width: int = 4,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.dim = dim
        self.bit_width = bit_width
        self.n_codes = 2**bit_width

        # Generate and register rotation matrix
        rotation = generate_rotation_matrix(dim, seed=seed, device=device)
        self.register_buffer("rotation", rotation)

        # Compute and register codebook
        codebook = compute_codebook(dim, bit_width).to(device)
        self.register_buffer("codebook", codebook)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize vectors to codebook indices.

        Args:
            x: Input tensor of shape (..., dim). Must be normalized
               (each vector should have unit norm for optimal performance).

        Returns:
            Indices tensor of shape (..., dim), dtype=torch.int16.
        """
        # Rotate: y = x @ Π^T
        y = x @ self.rotation.t()

        # Nearest centroid lookup via searchsorted on sorted codebook
        # codebook is sorted, so we can use efficient binary search
        # Compute boundaries between centroids
        boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2.0
        indices = torch.searchsorted(boundaries, y).clamp(0, self.n_codes - 1)

        return indices.to(torch.int16)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct vectors from codebook indices.

        Args:
            indices: Index tensor of shape (..., dim).

        Returns:
            Reconstructed tensor of shape (..., dim).
        """
        # Look up centroids
        y_hat = self.codebook[indices.long()]

        # Inverse rotation: x̃ = ỹ @ Π
        x_hat = y_hat @ self.rotation

        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize and immediately dequantize (for evaluation).

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Tuple of (reconstructed x̃, indices).
        """
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices
