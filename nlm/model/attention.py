"""
Physics-Informed Attention Mechanisms
=====================================

Attention that respects natural structure:
- CyclicalAttention: events at similar cycle phases attend more strongly
- SpatialLocalityAttention: geographic distance decay on attention logits
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlm.model.config import NLMConfig


class CyclicalAttention(nn.Module):
    """Attention modulated by natural cycles.

    Events at similar temporal cycle phases attend more strongly
    to each other (dawn↔dawn, full-moon↔full-moon).

    Adds a learned cyclical bias to standard scaled dot-product attention,
    computed from the temporal encoding of each position.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Cyclical bias: projects temporal features to per-head bias
        self.cycle_proj = nn.Linear(12, num_heads)  # 12 = temporal feature dim
        self.cycle_scale = nn.Parameter(torch.ones(num_heads) * 0.1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                     # (batch, seq_len, d_model)
        temporal_features: torch.Tensor,      # (batch, seq_len, 12)
        mask: Optional[torch.Tensor] = None,  # (batch, seq_len) bool
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Cyclical bias: similarity of temporal phases
        cycle_emb = self.cycle_proj(temporal_features)  # (B, L, num_heads)
        cycle_emb = cycle_emb.permute(0, 2, 1)  # (B, H, L)
        # Cosine similarity between temporal embeddings → bias
        cycle_norm = F.normalize(cycle_emb, dim=-1)
        cycle_bias = torch.bmm(
            cycle_norm.reshape(batch * self.num_heads, seq_len, 1),
            cycle_norm.reshape(batch * self.num_heads, 1, seq_len),
        ).view(batch, self.num_heads, seq_len, seq_len)

        # Scale and add bias
        attn = attn + cycle_bias * self.cycle_scale.view(1, -1, 1, 1)

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.o_proj(out)


class SpatialLocalityAttention(nn.Module):
    """Attention with geographic distance decay.

    Closer spatial points attend more strongly. Uses geodesic distance
    as a soft mask on attention logits.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Spatial decay: learned per-head bandwidth (in log-km)
        self.log_bandwidth = nn.Parameter(torch.ones(num_heads) * math.log(50.0))  # ~50km default

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                   # (batch, seq_len, d_model)
        spatial_coords: torch.Tensor,       # (batch, seq_len, 3) — lat, lon, alt (normalized)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Spatial distance decay
        # Compute pairwise L2 distance between normalized coordinates
        coords_diff = spatial_coords.unsqueeze(2) - spatial_coords.unsqueeze(1)  # (B, L, L, 3)
        dist_sq = (coords_diff ** 2).sum(dim=-1)  # (B, L, L)

        # Gaussian decay per head
        bandwidth = torch.exp(self.log_bandwidth)  # (H,)
        spatial_bias = -dist_sq.unsqueeze(1) / (2.0 * bandwidth.view(1, -1, 1, 1) ** 2)

        attn = attn + spatial_bias

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.o_proj(out)
