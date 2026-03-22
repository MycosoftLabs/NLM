"""
NLM Sparse Attention Fusion

Cross-stream integration layer. Sparse attention (not dense self-attention)
across the 6 stream embeddings. This is where the streams talk to each other.

Architecture:
  6 stream embeddings -> sparse cross-attention -> fused representation

NOT transformer-first. The fusion uses sparse attention only for
cross-stream integration, not as the primary backbone.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseStreamAttention(nn.Module):
    """
    Sparse attention across stream embeddings.

    Each stream attends to a subset of other streams based on
    learned routing weights. More efficient than full O(n²) attention
    for small n (6 streams) this is mostly for architectural clarity.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, n_streams: int = 6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_streams = n_streams
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Learned sparsity mask: which streams attend to which
        # Initialized to allow all connections, learned during training
        self.routing_logits = nn.Parameter(torch.zeros(n_streams, n_streams))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, stream_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Cross-stream attention.

        Args:
            stream_embeddings: (batch, n_streams, d_model)

        Returns:
            (batch, n_streams, d_model)
        """
        residual = stream_embeddings
        x = self.norm(stream_embeddings)

        batch, n_streams, d = x.shape

        # Multi-head QKV
        Q = self.q_proj(x).view(batch, n_streams, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch, n_streams, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch, n_streams, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply learned sparsity routing
        routing_mask = torch.sigmoid(self.routing_logits)  # (n_streams, n_streams), soft [0,1]
        # Expand for batch and heads
        routing_mask = routing_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, n_streams, n_streams)
        scores = scores * routing_mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (batch, n_heads, n_streams, d_head)

        out = out.transpose(1, 2).contiguous().view(batch, n_streams, d)
        out = self.o_proj(out)

        return out + residual


class HybridFusionCore(nn.Module):
    """
    The hybrid core that fuses 6 stream embeddings.

    Architecture:
    1. Sparse cross-stream attention (inter-stream communication)
    2. Per-stream feed-forward (intra-stream processing)
    3. Global readout (produce single fused representation)

    Multiple layers for deeper fusion.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_streams: int = 6,
        n_layers: int = 3,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers

        self.attention_layers = nn.ModuleList([
            SparseStreamAttention(d_model, n_heads, n_streams)
            for _ in range(n_layers)
        ])

        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

        # Global readout: attention-weighted pooling across streams
        self.readout_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )
        self.readout_norm = nn.LayerNorm(d_model)

    def forward(self, stream_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Fuse 6 stream embeddings into a single representation.

        Args:
            stream_embeddings: (batch, 6, d_model)

        Returns:
            (batch, d_model) — the fused world state representation
        """
        h = stream_embeddings

        for attn_layer, ff_layer in zip(self.attention_layers, self.ff_layers):
            # Cross-stream attention
            h = attn_layer(h)
            # Per-stream feed-forward with residual
            h = h + ff_layer(h)

        # Global readout
        attn_weights = F.softmax(self.readout_attn(h), dim=1)  # (batch, 6, 1)
        fused = (h * attn_weights).sum(dim=1)  # (batch, d_model)
        fused = self.readout_norm(fused)

        return fused
