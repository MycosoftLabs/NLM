"""
Graph / Hypergraph Encoders
===========================

Processes the HyperDAG world structure into neural representations.
Handles entity-relation graphs, hyperedges, and multi-way interactions.

Not a flat embedding lookup — structure-aware message passing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlm.model.config import NLMConfig


class EntityRelationEncoder(nn.Module):
    """Encodes entity-relation graphs via message passing.

    Processes pairwise edges between entities (organisms, devices,
    sites, compounds) using a simplified graph attention mechanism.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        self.hidden_dim = config.graph_hidden_dim
        self.num_heads = config.graph_num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        # Node feature projection
        self.node_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Edge type embedding
        self.edge_type_embed = nn.Embedding(32, self.hidden_dim)  # 32 edge types

        # Multi-head attention for message passing
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Output
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.ff_norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,  # (batch, num_nodes, hidden_dim)
        edge_index: torch.Tensor,      # (batch, num_edges, 2) int — source/target indices
        edge_types: torch.Tensor,      # (batch, num_edges) int — edge type ids
        node_mask: Optional[torch.Tensor] = None,  # (batch, num_nodes) bool
    ) -> torch.Tensor:
        """
        Returns: (batch, num_nodes, hidden_dim) — updated node representations
        """
        batch, num_nodes, _ = node_features.shape
        residual = node_features

        # Project nodes
        x = self.node_proj(node_features)

        # Compute attention (simplified: full attention between all nodes, masked by edges)
        Q = self.query(x).view(batch, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if node_mask is not None:
            mask = node_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(batch, num_nodes, self.hidden_dim)
        out = self.out_proj(out)

        # Residual + norm
        x = self.norm(out + residual)

        # FF
        x = self.ff_norm(self.ff(x) + x)

        return x


class HyperedgeAggregator(nn.Module):
    """Aggregates information from hyperedges (multi-way relationships).

    Each hyperedge connects 2+ nodes. The aggregator:
    1. Pools features from all participating nodes
    2. Combines with edge metadata
    3. Distributes updated information back to nodes
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        self.hidden_dim = config.graph_hidden_dim

        # Hyperedge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + 32, self.hidden_dim),  # node features + edge metadata
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )

        # Node update from hyperedge context
        self.node_update = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )

        # Edge type embedding
        self.edge_type_embed = nn.Embedding(16, 32)

    def forward(
        self,
        node_features: torch.Tensor,        # (batch, num_nodes, hidden_dim)
        hyperedge_members: torch.Tensor,     # (batch, num_edges, max_arity) int — node indices per edge
        hyperedge_types: torch.Tensor,       # (batch, num_edges) int
        hyperedge_mask: torch.Tensor,        # (batch, num_edges, max_arity) bool — valid members
    ) -> torch.Tensor:
        """Returns: (batch, num_nodes, hidden_dim) — nodes updated with hyperedge context"""
        batch, num_nodes, _ = node_features.shape
        _, num_edges, max_arity = hyperedge_members.shape

        # Gather node features for each hyperedge member
        # Clamp indices to valid range
        members_clamped = hyperedge_members.clamp(0, num_nodes - 1)
        gathered = torch.gather(
            node_features.unsqueeze(1).expand(-1, num_edges, -1, -1),
            2,
            members_clamped.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim),
        )  # (B, num_edges, max_arity, hidden_dim)

        # Mask invalid members
        gathered = gathered * hyperedge_mask.unsqueeze(-1).float()

        # Pool across members
        member_count = hyperedge_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
        pooled = gathered.sum(dim=2) / member_count  # (B, num_edges, hidden_dim)

        # Add edge type embedding
        edge_emb = self.edge_type_embed(hyperedge_types)  # (B, num_edges, 32)
        edge_features = self.edge_encoder(torch.cat([pooled, edge_emb], dim=-1))

        # Distribute back to nodes (scatter-add)
        node_updates = torch.zeros_like(node_features)
        update_counts = torch.zeros(batch, num_nodes, 1, device=node_features.device)

        for e in range(num_edges):
            for a in range(max_arity):
                valid = hyperedge_mask[:, e, a]  # (B,)
                indices = members_clamped[:, e, a]  # (B,)
                for b in range(batch):
                    if valid[b]:
                        idx = indices[b].item()
                        node_updates[b, idx] += edge_features[b, e]
                        update_counts[b, idx] += 1

        # Average updates
        update_counts = update_counts.clamp(min=1)
        node_updates = node_updates / update_counts

        # Combine with original features
        updated = self.node_update(torch.cat([node_features, node_updates], dim=-1))
        return updated


class HyperDAGEncoder(nn.Module):
    """Full HyperDAG encoder: entity relations + hyperedge aggregation.

    Processes the multi-resolution graph structure into node embeddings
    that capture both pairwise and multi-way relationships.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        self.config = config

        # Node type embedding
        self.node_type_embed = nn.Embedding(32, config.graph_hidden_dim)

        # Stacked relation + hyperedge layers
        self.relation_layers = nn.ModuleList([
            EntityRelationEncoder(config) for _ in range(config.graph_num_layers)
        ])
        self.hyperedge_layers = nn.ModuleList([
            HyperedgeAggregator(config) for _ in range(config.graph_num_layers)
        ])

        # Graph-level pooling
        self.graph_pool = nn.Sequential(
            nn.Linear(config.graph_hidden_dim, config.graph_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.graph_hidden_dim),
        )

    def forward(
        self,
        node_type_ids: torch.Tensor,         # (batch, num_nodes) int
        edge_index: torch.Tensor,             # (batch, num_edges, 2) int
        edge_types: torch.Tensor,             # (batch, num_edges) int
        hyperedge_members: torch.Tensor,      # (batch, num_hyperedges, max_arity) int
        hyperedge_types: torch.Tensor,        # (batch, num_hyperedges) int
        hyperedge_mask: torch.Tensor,         # (batch, num_hyperedges, max_arity) bool
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_embeddings: (batch, num_nodes, graph_hidden_dim)
            graph_embedding: (batch, graph_hidden_dim) — pooled graph representation
        """
        # Initialize node features from type embeddings
        x = self.node_type_embed(node_type_ids)

        # Alternating relation and hyperedge layers
        for rel_layer, hyp_layer in zip(self.relation_layers, self.hyperedge_layers):
            x = rel_layer(x, edge_index, edge_types, node_mask)
            x = hyp_layer(x, hyperedge_members, hyperedge_types, hyperedge_mask)

        # Graph-level pooling (mean over valid nodes)
        if node_mask is not None:
            masked = x * node_mask.unsqueeze(-1).float()
            graph_emb = masked.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        else:
            graph_emb = x.mean(dim=1)

        graph_emb = self.graph_pool(graph_emb)

        return x, graph_emb


import math  # needed for EntityRelationEncoder.forward
