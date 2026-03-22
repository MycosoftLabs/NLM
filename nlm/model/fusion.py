"""
Sparse Attention Fusion
========================

Cross-stream integration via sparse attention — used only where
global cross-modal binding is necessary. Not the default compute path.

The bulk of temporal processing is handled by SSM blocks.
The bulk of structural reasoning is handled by graph encoders.
Fusion happens selectively between streams.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlm.model.config import NLMConfig


class CrossStreamGating(nn.Module):
    """Learned gating to control information flow between streams.

    Each stream can selectively attend to or ignore other streams
    based on content relevance.
    """

    def __init__(self, d_model: int, num_streams: int = 6, dropout: float = 0.1):
        super().__init__()
        self.num_streams = num_streams
        self.gate_proj = nn.Linear(d_model * num_streams, num_streams * num_streams)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, streams: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            streams: list of 6 tensors, each (batch, d_model)
        Returns:
            list of 6 tensors, each (batch, d_model) — gated
        """
        batch = streams[0].size(0)
        concat = torch.cat(streams, dim=-1)  # (B, d_model * 6)

        # Compute gating matrix
        gates = self.gate_proj(concat)  # (B, 36)
        gates = gates.view(batch, self.num_streams, self.num_streams)
        gates = torch.sigmoid(gates)  # (B, 6, 6)

        # Project values
        values = [self.value_proj(s) for s in streams]  # list of (B, d_model)
        value_stack = torch.stack(values, dim=1)  # (B, 6, d_model)

        # Gated combination: each stream gets weighted sum of all streams
        gated = torch.bmm(gates, value_stack)  # (B, 6, d_model)

        # Output with residual
        outputs = []
        for i in range(self.num_streams):
            out = self.output_proj(gated[:, i])
            out = self.dropout(out)
            out = self.norm(out + streams[i])
            outputs.append(out)

        return outputs


class SparseAttentionFusion(nn.Module):
    """Sparse cross-modal attention for fusing all 6 streams.

    Only a fraction of attention weights are non-zero (controlled by sparsity).
    This keeps the fusion layer efficient while allowing global binding.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.fusion_num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.num_streams = 6
        self.top_k = max(1, int(self.num_streams * (1.0 - config.fusion_sparsity)))

        # Stream projection to unified dim
        self.stream_projs = nn.ModuleList([
            nn.Linear(dim, config.hidden_dim)
            for dim in [
                config.spatial_dim, config.temporal_dim,
                config.spectral_sensory_dim, config.world_state_dim,
                config.self_state_dim, config.action_intent_dim,
            ]
        ])

        # Cross-stream gating
        self.gating = CrossStreamGating(config.hidden_dim, self.num_streams, config.dropout)

        # Multi-head attention across concatenated stream tokens
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.hidden_dim),
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        spatial: torch.Tensor,           # (batch, spatial_dim)
        temporal: torch.Tensor,           # (batch, temporal_dim)
        spectral_sensory: torch.Tensor,   # (batch, spectral_sensory_dim)
        world_state: torch.Tensor,        # (batch, world_state_dim)
        self_state: torch.Tensor,         # (batch, self_state_dim)
        action_intent: torch.Tensor,      # (batch, action_intent_dim)
    ) -> torch.Tensor:
        """Fuse all 6 streams into a single hidden representation.

        Returns: (batch, hidden_dim)
        """
        raw_streams = [spatial, temporal, spectral_sensory, world_state, self_state, action_intent]

        # Project each stream to hidden_dim
        projected = [proj(s) for proj, s in zip(self.stream_projs, raw_streams)]

        # Cross-stream gating
        gated = self.gating(projected)

        # Stack as sequence for self-attention: (batch, 6, hidden_dim)
        x = torch.stack(gated, dim=1)
        batch, num_tokens, _ = x.shape

        # Self-attention across streams
        residual = x
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm).view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(batch, num_tokens, self.hidden_dim)
        out = self.o_proj(out)
        x = residual + self.dropout(out)

        # FF
        x = x + self.dropout(self.ff(self.norm2(x)))

        # Pool across stream tokens → single vector
        return x.mean(dim=1)  # (batch, hidden_dim)
