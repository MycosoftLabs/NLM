"""
Selective State Space Model Blocks (Mamba-style)
================================================

SSM/Mamba trunk for temporal sensory processing:
- Linear-time scaling with sequence length
- Selective gating for content-aware state evolution
- Hardware-friendly recurrent execution
- Efficient for continuous sensor streams and long-horizon state

This is the temporal backbone of the NLM, not a transformer.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlm.model.config import NLMConfig


class SelectiveSSMBlock(nn.Module):
    """A single Selective State Space Model block.

    Inspired by Mamba: input-dependent state transitions with
    selective gating. Linear complexity in sequence length.

    Architecture:
    1. Linear projection to expanded inner dim
    2. 1D depthwise convolution for local context
    3. Selective SSM core (discretized state-space)
    4. Gated output with SiLU activation
    5. Linear projection back to model dim
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        # Input projection (2x for gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D depthwise convolution for local temporal context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt

        # Learnable SSM parameters
        # A is structured as diagonal negative (ensures stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))  # log for numerical stability
        self.D = nn.Parameter(torch.ones(self.d_inner))  # skip connection

        # dt (discretization step) projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        batch, seq_len, _ = x.shape

        # Project and split into main path and gate
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_main, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 1D convolution for local context
        x_conv = x_main.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # trim padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameter projection (input-dependent B, C, dt)
        x_ssm = self.x_proj(x_conv)  # (B, L, d_state*2 + 1)
        B = x_ssm[:, :, :self.d_state]  # (B, L, d_state)
        C = x_ssm[:, :, self.d_state:2*self.d_state]  # (B, L, d_state)
        dt = x_ssm[:, :, -1:]  # (B, L, 1)

        # Discretization step
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # Diagonal A matrix (negative for stability)
        A = -torch.exp(self.A_log)  # (d_state,)

        # Selective scan (sequential for correctness; parallel impl would use associative scan)
        y = self._selective_scan(x_conv, dt, A, B, C)

        # Skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        # Gate with SiLU
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        return y + residual

    def _selective_scan(
        self,
        x: torch.Tensor,  # (B, L, d_inner)
        dt: torch.Tensor,  # (B, L, d_inner)
        A: torch.Tensor,   # (d_state,)
        B: torch.Tensor,   # (B, L, d_state)
        C: torch.Tensor,   # (B, L, d_state)
    ) -> torch.Tensor:
        """Selective scan implementation.

        For each timestep:
            h[t] = exp(A * dt[t]) * h[t-1] + dt[t] * B[t] * x[t]
            y[t] = C[t] @ h[t]
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[0]

        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            # Discretize A for this timestep
            dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt[:, t, :].unsqueeze(-1))  # (B, d_inner, d_state)
            dB = dt[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # (B, d_inner, d_state)

            # State update
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)  # (B, d_inner, d_state)

            # Output
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, d_inner)


class TemporalSSMStack(nn.Module):
    """Stack of SSM blocks for temporal processing.

    The temporal backbone of the NLM. Processes sequences of
    nature observations with linear-time complexity.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            SelectiveSSMBlock(
                d_model=config.hidden_dim,
                d_state=config.ssm_state_dim,
                d_conv=config.ssm_conv_width,
                expand=config.ssm_expand_factor,
                dropout=config.dropout,
            )
            for _ in range(config.num_ssm_layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
