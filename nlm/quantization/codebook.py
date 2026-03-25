"""
Beta Distribution Codebook Computation
========================================

Computes optimal Lloyd-Max codebooks for the Beta distribution induced
by random rotation of unit-norm vectors.

After rotation by orthogonal Π, each coordinate of a d-dimensional
unit-norm vector follows:

    f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}

defined on [-1, 1]. For large d, this converges to N(0, 1/d).

The Lloyd-Max algorithm finds centroids c_1,...,c_{2^b} minimizing
the MSE distortion under this distribution.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import torch


def beta_pdf(x: torch.Tensor, d: int) -> torch.Tensor:
    """Evaluate the Beta distribution PDF for d-dimensional rotation.

    f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}

    Args:
        x: Points in [-1, 1] at which to evaluate.
        d: Vector dimension (affects distribution shape).

    Returns:
        PDF values at x, same shape as x.
    """
    if d < 3:
        # For d=2, uniform on [-1, 1]: f(x) = 0.5
        return torch.full_like(x, 0.5)

    log_norm = (
        math.lgamma(d / 2.0)
        - 0.5 * math.log(math.pi)
        - math.lgamma((d - 1) / 2.0)
    )
    exponent = (d - 3) / 2.0

    # Clamp to avoid numerical issues at boundaries
    x_clamped = x.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    log_density = exponent * torch.log(1.0 - x_clamped**2)

    return torch.exp(log_norm + log_density)


def _lloyd_max_iteration(
    centroids: torch.Tensor,
    grid: torch.Tensor,
    pdf_vals: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """One iteration of Lloyd-Max algorithm.

    1. Compute boundaries as midpoints between adjacent centroids.
    2. Update each centroid as the conditional expectation E[x | x ∈ region_i].
    """
    n_codes = centroids.shape[0]

    # Boundaries: midpoints between adjacent centroids
    boundaries = torch.zeros(n_codes + 1, device=centroids.device, dtype=centroids.dtype)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(1, n_codes):
        boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

    # Update centroids as conditional expectations
    new_centroids = torch.zeros_like(centroids)
    for i in range(n_codes):
        mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
        if i == n_codes - 1:
            mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])

        weighted = grid * pdf_vals * mask.float()
        weight_sum = (pdf_vals * mask.float()).sum()

        if weight_sum > 1e-12:
            new_centroids[i] = (weighted.sum() * dx) / (weight_sum * dx)
        else:
            new_centroids[i] = centroids[i]

    return new_centroids


def compute_codebook(
    d: int,
    b: int,
    num_iterations: int = 100,
    grid_size: int = 10000,
) -> torch.Tensor:
    """Compute optimal Lloyd-Max codebook for the Beta distribution.

    Uses iterative refinement to find 2^b centroids that minimize
    MSE distortion under the Beta distribution induced by dimension d.

    Args:
        d: Vector dimension (affects Beta distribution shape).
        b: Bit width (codebook has 2^b entries).
        num_iterations: Number of Lloyd-Max iterations.
        grid_size: Number of grid points for numerical integration.

    Returns:
        Sorted centroids tensor of shape (2^b,).
    """
    # Check for known closed-form codebooks first
    known = get_known_codebook(d, b)
    if known is not None:
        return known

    n_codes = 2**b

    # Create fine grid over [-1, 1] for numerical integration
    grid = torch.linspace(-1.0, 1.0, grid_size)
    dx = 2.0 / grid_size
    pdf_vals = beta_pdf(grid, d)

    # Initialize centroids: use quantiles of the distribution
    cdf = torch.cumsum(pdf_vals * dx, dim=0)
    cdf = cdf / cdf[-1]  # normalize

    centroids = torch.zeros(n_codes)
    for i in range(n_codes):
        target = (2 * i + 1) / (2 * n_codes)
        idx = torch.searchsorted(cdf, target).clamp(0, grid_size - 1)
        centroids[i] = grid[idx]

    # Lloyd-Max iterations
    for _ in range(num_iterations):
        centroids = _lloyd_max_iteration(centroids, grid, pdf_vals, dx)

    return centroids.sort()[0]


def get_known_codebook(d: int, b: int) -> Optional[torch.Tensor]:
    """Return analytically known codebooks for small bit widths.

    From TurboQuant paper:
    - b=1: centroids = {±√(2/(πd))}
    - b=2: centroids = {±0.453/√d, ±1.51/√d}

    Args:
        d: Vector dimension.
        b: Bit width.

    Returns:
        Sorted centroids tensor, or None if no closed form is known.
    """
    if b == 1:
        c = math.sqrt(2.0 / (math.pi * d))
        return torch.tensor([-c, c])

    if b == 2:
        sqrt_d = math.sqrt(d)
        return torch.tensor([-1.51 / sqrt_d, -0.453 / sqrt_d,
                             0.453 / sqrt_d, 1.51 / sqrt_d])

    return None


@lru_cache(maxsize=32)
def cached_codebook(d: int, b: int) -> torch.Tensor:
    """Cached version of compute_codebook for repeated access."""
    return compute_codebook(d, b)
