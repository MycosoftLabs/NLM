"""
SSM State Compression
======================

Compresses Mamba-style SSM hidden states between timesteps using
TurboQuant_mse. The hidden state h ∈ ℝ^{batch × d_inner × d_state}
is flattened and quantized to reduce memory during long-sequence
recurrence.

This is the most experimental component — SSM states accumulate
quantization error across timesteps, so higher bit widths (4-bit)
are recommended.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlm.quantization.turbo_mse import TurboQuantMSE


class SSMStateCompressor(nn.Module):
    """Compress SSM hidden states using TurboQuant_mse.

    Flattens (d_inner, d_state) → (d_inner * d_state), applies
    TurboQuant_mse, then reshapes back. Scale factor preserves norms.
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int,
        bit_width: int = 4,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.flat_dim = d_inner * d_state

        self.quantizer = TurboQuantMSE(
            self.flat_dim, bit_width=bit_width, seed=seed, device=device
        )

    def compress(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize SSM state.

        Args:
            h: State tensor of shape (batch, d_inner, d_state).

        Returns:
            Tuple of (indices, scale):
                indices: (batch, flat_dim) int16
                scale: (batch,) float — original state norm
        """
        batch = h.shape[0]
        flat = h.reshape(batch, self.flat_dim)

        # Store norm for rescaling after dequantization
        scale = torch.linalg.norm(flat, dim=-1)
        # Normalize to unit norm (TurboQuant assumption)
        scale_safe = scale.clamp(min=1e-8)
        flat_norm = flat / scale_safe.unsqueeze(-1)

        indices = self.quantizer.quantize(flat_norm)
        return indices, scale

    def decompress(
        self, indices: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize SSM state.

        Args:
            indices: (batch, flat_dim) int16.
            scale: (batch,) float.

        Returns:
            Reconstructed state of shape (batch, d_inner, d_state).
        """
        flat_hat = self.quantizer.dequantize(indices)
        flat_hat = flat_hat * scale.unsqueeze(-1)
        return flat_hat.reshape(-1, self.d_inner, self.d_state)


class QuantizedSelectiveSSMBlock(nn.Module):
    """SelectiveSSMBlock with compressed hidden state between timesteps.

    Wraps the original SSM block, adding state compression in the
    _selective_scan recurrence. The hidden state is quantized after
    each state update to simulate reduced-precision storage.
    """

    def __init__(
        self,
        original: nn.Module,
        bit_width: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.d_model = original.d_model
        self.d_state = original.d_state
        self.d_inner = original.d_inner
        self.d_conv = original.d_conv

        # Copy all original parameters
        self.in_proj = original.in_proj
        self.conv1d = original.conv1d
        self.x_proj = original.x_proj
        self.A_log = original.A_log
        self.D = original.D
        self.dt_proj = original.dt_proj
        self.out_proj = original.out_proj
        self.dropout = original.dropout
        self.norm = original.norm

        # State compressor
        self.state_compressor = SSMStateCompressor(
            d_inner=self.d_inner,
            d_state=self.d_state,
            bit_width=bit_width,
            seed=seed,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)

        x_conv = x_main.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_ssm = self.x_proj(x_conv)
        B = x_ssm[:, :, : self.d_state]
        C = x_ssm[:, :, self.d_state : 2 * self.d_state]
        dt = x_ssm[:, :, -1:]

        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)

        # Selective scan with state compression
        y = self._compressed_selective_scan(x_conv, dt, A, B, C)

        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return y + residual

    def _compressed_selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Selective scan with quantized state compression between steps."""
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[0]

        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt[:, t, :].unsqueeze(-1))
            dB = dt[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1)

            h = dA * h + dB * x[:, t, :].unsqueeze(-1)

            # Compress state between timesteps (every step)
            indices, scale = self.state_compressor.compress(h)
            h = self.state_compressor.decompress(indices, scale)

            y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)
