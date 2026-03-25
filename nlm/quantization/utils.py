"""
Quantization Utilities
=======================

Bit packing/unpacking, distortion metrics, theoretical bounds,
and memory estimation helpers.
"""

from __future__ import annotations

import math
from typing import Dict

import torch


# --- Bit Packing ---


def pack_indices(indices: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Pack integer indices into bytes for storage efficiency.

    Packing ratios:
        bit_width=1: 8 indices per byte
        bit_width=2: 4 indices per byte
        bit_width=4: 2 indices per byte

    Args:
        indices: Integer tensor with values in [0, 2^bit_width - 1].
        bit_width: Bits per index (1, 2, or 4).

    Returns:
        Packed uint8 tensor.
    """
    if bit_width not in (1, 2, 4):
        raise ValueError(f"pack_indices supports bit_width in (1, 2, 4), got {bit_width}")

    flat = indices.flatten().to(torch.uint8)
    vals_per_byte = 8 // bit_width

    # Pad to multiple of vals_per_byte
    pad_len = (vals_per_byte - len(flat) % vals_per_byte) % vals_per_byte
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8, device=flat.device)])

    flat = flat.view(-1, vals_per_byte)
    packed = torch.zeros(flat.shape[0], dtype=torch.uint8, device=flat.device)
    for i in range(vals_per_byte):
        packed |= flat[:, i] << (i * bit_width)

    return packed


def unpack_indices(
    packed: torch.Tensor, bit_width: int, num_elements: int
) -> torch.Tensor:
    """Unpack bytes back to integer indices.

    Args:
        packed: Packed uint8 tensor from pack_indices.
        bit_width: Bits per index (1, 2, or 4).
        num_elements: Number of original indices to recover.

    Returns:
        Integer tensor of shape (num_elements,).
    """
    if bit_width not in (1, 2, 4):
        raise ValueError(f"unpack_indices supports bit_width in (1, 2, 4), got {bit_width}")

    vals_per_byte = 8 // bit_width
    mask = (1 << bit_width) - 1

    result = []
    for i in range(vals_per_byte):
        result.append((packed >> (i * bit_width)) & mask)

    unpacked = torch.stack(result, dim=-1).flatten()
    return unpacked[:num_elements].to(torch.int16)


def pack_sign_bits(signs: torch.Tensor) -> torch.Tensor:
    """Pack {-1, +1} sign bits into boolean bytes.

    Args:
        signs: Int8 tensor with values in {-1, +1}.

    Returns:
        Boolean tensor (1 bit per element, packed by PyTorch).
    """
    return (signs > 0).to(torch.bool)


def unpack_sign_bits(packed: torch.Tensor) -> torch.Tensor:
    """Unpack boolean bytes back to {-1, +1} int8.

    Args:
        packed: Boolean tensor from pack_sign_bits.

    Returns:
        Int8 tensor with values in {-1, +1}.
    """
    return packed.to(torch.int8) * 2 - 1


# --- Distortion Metrics ---


def compute_mse_distortion(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> float:
    """Compute normalized MSE between original and reconstructed tensors.

    Returns MSE / ‖original‖² for comparison with theoretical bounds.
    """
    mse = ((original - reconstructed) ** 2).mean().item()
    orig_norm_sq = (original**2).mean().item()
    if orig_norm_sq < 1e-12:
        return 0.0
    return mse / orig_norm_sq


def compute_inner_product_error(
    Q: torch.Tensor,
    K: torch.Tensor,
    estimated_scores: torch.Tensor,
) -> Dict[str, float]:
    """Compare exact Q@K^T with estimated attention scores.

    Args:
        Q: Query tensor (..., dim).
        K: Key tensor (..., dim).
        estimated_scores: Estimated Q@K^T from quantized method.

    Returns:
        Dict with max_error, mean_error, relative_error.
    """
    exact_scores = torch.matmul(Q, K.transpose(-2, -1))
    diff = (exact_scores - estimated_scores).abs()
    exact_norm = exact_scores.abs().mean().item()

    return {
        "max_error": diff.max().item(),
        "mean_error": diff.mean().item(),
        "relative_error": diff.mean().item() / max(exact_norm, 1e-12),
    }


# --- Theoretical Bounds ---


def theoretical_mse_bound(b: int) -> float:
    """Theoretical MSE distortion upper bound for TurboQuant_mse.

    D_mse ≤ (√3·π/2) · (1/4^b)

    Args:
        b: Bit width.

    Returns:
        Upper bound on normalized MSE distortion.
    """
    return math.sqrt(3) * math.pi / 2.0 * (1.0 / 4**b)


def theoretical_prod_bound(d: int, b: int, y_norm_sq: float = 1.0) -> float:
    """Theoretical inner product distortion upper bound for TurboQuant_prod.

    D_prod ≤ (√3·π²·‖y‖²/d) · (1/4^b)

    Args:
        d: Vector dimension.
        b: Bit width.
        y_norm_sq: ‖y‖² (query norm squared).

    Returns:
        Upper bound on inner product distortion.
    """
    return math.sqrt(3) * math.pi**2 * y_norm_sq / d * (1.0 / 4**b)


def information_theoretic_lower_bound(b: int) -> float:
    """Information-theoretic lower bound on MSE distortion.

    D(B) ≥ 1/4^b (sphere limiting bound).
    """
    return 1.0 / 4**b


# --- Memory Estimation ---


def estimate_tensor_memory(
    shape: tuple,
    bit_width: int,
    include_overhead: bool = True,
) -> int:
    """Estimate memory in bytes for a quantized tensor.

    Args:
        shape: Tensor shape.
        bit_width: Bits per element.
        include_overhead: Whether to include rotation matrix / codebook overhead.

    Returns:
        Estimated bytes.
    """
    num_elements = 1
    for s in shape:
        num_elements *= s

    # Quantized data
    data_bits = num_elements * bit_width
    data_bytes = math.ceil(data_bits / 8)

    if include_overhead:
        dim = shape[-1] if shape else 0
        # Rotation matrix: dim × dim × 4 bytes (float32)
        rotation_bytes = dim * dim * 4
        # Codebook: 2^b × 4 bytes
        codebook_bytes = (2**bit_width) * 4
        return data_bytes + rotation_bytes + codebook_bytes

    return data_bytes


def compression_ratio(
    original_bytes: int, quantized_bytes: int
) -> float:
    """Compute compression ratio."""
    if quantized_bytes == 0:
        return float("inf")
    return original_bytes / quantized_bytes
