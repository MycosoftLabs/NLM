"""
Random Rotation and Projection Matrices
=========================================

Generates orthogonal rotation matrices (via QR decomposition of Gaussian)
and i.i.d. Gaussian projection matrices for QJL.

The rotation matrix Π ensures that after rotation, each coordinate of a
unit-norm vector follows a Beta distribution concentrated around zero,
enabling efficient scalar quantization.
"""

from __future__ import annotations

from typing import Optional

import torch


def generate_rotation_matrix(
    d: int,
    seed: Optional[int] = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a random orthogonal rotation matrix via QR decomposition.

    Produces a Haar-distributed orthogonal matrix Π ∈ ℝ^{d×d}
    satisfying Π^T Π = I.

    Args:
        d: Matrix dimension.
        seed: Random seed for reproducibility.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Orthogonal matrix of shape (d, d).
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    # Generate random Gaussian matrix and compute QR decomposition
    gaussian = torch.randn(d, d, generator=gen, dtype=dtype)
    Q, R = torch.linalg.qr(gaussian)

    # Ensure uniqueness: multiply by sign of diagonal of R
    # This gives a proper Haar-distributed orthogonal matrix
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)

    return Q.to(device=device)


def generate_projection_matrix(
    d: int,
    m: Optional[int] = None,
    seed: Optional[int] = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate random Gaussian projection matrix for QJL.

    Produces S ∈ ℝ^{m×d} with S_{i,j} ~ N(0, 1) i.i.d.

    Args:
        d: Input dimension.
        m: Projection dimension. Defaults to d if None.
        seed: Random seed for reproducibility.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Projection matrix of shape (m, d).
    """
    if m is None:
        m = d

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    S = torch.randn(m, d, generator=gen, dtype=dtype)
    return S.to(device=device)
