"""
NLM SSM/Mamba Blocks

State Space Model blocks for temporal encoding and long-horizon
state evolution. Primary backbone — linear complexity in sequence
length, natural fit for continuous state evolution.

This is a simplified reference implementation. Production deployment
should use optimized Mamba kernels (mamba-ssm package).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMBlock(nn.Module):
    """
    Selective State Space Model block (Mamba-style).

    Implements the S6 selective scan mechanism:
    - Input-dependent state transition matrices (selectivity)
    - Linear recurrence for O(L) complexity
    - Gated output with residual connection
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection (to 2x for gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters
        # B, C are input-dependent (selective)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # A is fixed (discretized log-space)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.unsqueeze(0).expand(self.d_inner, -1)))

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # Project and split for gating
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 1D conv
        x_conv = x_inner.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :x_inner.shape[1]]  # trim to original length
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameters (input-dependent B, C, and delta)
        x_proj = self.x_proj(x_conv)  # (B, L, 2*d_state + 1)
        B = x_proj[:, :, :self.d_state]          # (B, L, d_state)
        C = x_proj[:, :, self.d_state:2*self.d_state]  # (B, L, d_state)
        delta = F.softplus(x_proj[:, :, -1:])     # (B, L, 1)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative for stability

        # Selective scan (simplified sequential for clarity)
        y = self._selective_scan(x_conv, A, B, C, delta)

        # Skip connection with D
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # Gate
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        return y + residual

    def _selective_scan(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified selective scan.

        In production, use the optimized CUDA kernel from mamba-ssm.
        """
        batch, seq_len, d_inner = x.shape

        # Initialize state
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            # Discretize: A_bar = exp(delta * A)
            dt = delta[:, t, :]  # (B, 1)
            A_discrete = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)

            # B_bar = delta * B
            B_t = dt.unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # (B, 1, d_state) -> (B, d_inner, d_state)
            B_t = B_t.expand_as(h)

            # State update: h = A_bar * h + B_bar * x
            x_t = x[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            h = A_discrete * h + B_t * x_t

            # Output: y = C * h
            C_t = C[:, t, :]  # (B, d_state)
            y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1)  # (B, d_inner)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, d_inner)


class SSMStack(nn.Module):
    """Stack of SSM blocks for deep temporal processing."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            SSMBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process through all SSM layers."""
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
