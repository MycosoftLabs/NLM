"""
NLM Graph Encoders

Graph-aware, stateful encoders for world-state and self-state streams.
Uses message-passing (GNN-style) over entity-relation subgraphs.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphMessagePassingLayer(nn.Module):
    """
    Single message-passing layer for graph encoding.

    Implements: h_i' = Update(h_i, Aggregate({Message(h_i, h_j, e_ij) : j in N(i)}))
    """

    def __init__(self, d_node: int = 128, d_edge: int = 64):
        super().__init__()
        self.d_node = d_node
        self.d_edge = d_edge

        # Message function: MLP(concat(h_i, h_j, e_ij))
        self.message_mlp = nn.Sequential(
            nn.Linear(d_node * 2 + d_edge, d_node),
            nn.SiLU(),
            nn.Linear(d_node, d_node),
        )

        # Update function: GRU-style
        self.update_gate = nn.Linear(d_node * 2, d_node)
        self.update_transform = nn.Linear(d_node * 2, d_node)

        self.norm = nn.LayerNorm(d_node)

    def forward(
        self,
        node_features: torch.Tensor,       # (num_nodes, d_node)
        edge_index: torch.Tensor,           # (2, num_edges)
        edge_features: Optional[torch.Tensor] = None,  # (num_edges, d_edge)
    ) -> torch.Tensor:
        """Message passing step."""
        num_nodes = node_features.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Compute messages
        h_src = node_features[src]  # (num_edges, d_node)
        h_dst = node_features[dst]  # (num_edges, d_node)

        if edge_features is not None:
            msg_input = torch.cat([h_src, h_dst, edge_features], dim=-1)
        else:
            # Zero edge features
            zero_edge = torch.zeros(src.shape[0], self.d_edge, device=node_features.device)
            msg_input = torch.cat([h_src, h_dst, zero_edge], dim=-1)

        messages = self.message_mlp(msg_input)  # (num_edges, d_node)

        # Aggregate messages (mean)
        aggregated = torch.zeros(num_nodes, self.d_node, device=node_features.device)
        count = torch.zeros(num_nodes, 1, device=node_features.device)
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones_like(dst.unsqueeze(-1).float()))
        count = count.clamp(min=1)
        aggregated = aggregated / count

        # Update
        gate_input = torch.cat([node_features, aggregated], dim=-1)
        gate = torch.sigmoid(self.update_gate(gate_input))
        transform = torch.tanh(self.update_transform(gate_input))
        updated = gate * node_features + (1 - gate) * transform

        return self.norm(updated)


class WorldStateGraphEncoder(nn.Module):
    """
    Graph-aware encoder for world-state stream (Stream 4).

    Encodes the entity-relation subgraph from MINDEX (species,
    compounds, devices, sites, weather, etc.) into a fixed-dimension
    embedding using multi-layer message passing.

    Stateful: maintains a hidden state across frames for temporal
    consistency.
    """

    def __init__(
        self,
        d_node: int = 128,
        d_edge: int = 64,
        d_output: int = 256,
        n_layers: int = 3,
        max_nodes: int = 512,
    ):
        super().__init__()
        self.d_node = d_node
        self.d_output = d_output
        self.max_nodes = max_nodes

        # Node feature projection (from raw properties)
        self.node_proj = nn.Linear(d_node, d_node)

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            GraphMessagePassingLayer(d_node, d_edge) for _ in range(n_layers)
        ])

        # Readout: graph-level embedding from node embeddings
        self.readout_attn = nn.Linear(d_node, 1)
        self.readout_proj = nn.Linear(d_node, d_output)

        # Stateful: GRU for temporal state
        self.state_gru = nn.GRUCell(d_output, d_output)

        # Hidden state (managed externally per sequence)
        self._hidden: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int = 1, device: str = "cpu"):
        """Reset the temporal hidden state."""
        self._hidden = torch.zeros(batch_size, self.d_output, device=device)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a world-state graph into a fixed embedding.

        Args:
            node_features: (num_nodes, d_node)
            edge_index: (2, num_edges)
            edge_features: optional (num_edges, d_edge)

        Returns:
            (1, d_output) or (batch, d_output) if batched
        """
        h = self.node_proj(node_features)

        # Message passing
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_features)

        # Attention-weighted readout
        attn_weights = F.softmax(self.readout_attn(h), dim=0)  # (num_nodes, 1)
        graph_emb = (h * attn_weights).sum(dim=0, keepdim=True)  # (1, d_node)
        graph_emb = self.readout_proj(graph_emb)  # (1, d_output)

        # Update temporal state
        if self._hidden is not None:
            self._hidden = self.state_gru(graph_emb, self._hidden)
            return self._hidden
        return graph_emb


class SelfStateGraphEncoder(nn.Module):
    """
    Graph-aware encoder for self-state stream (Stream 5).

    Encodes the agent/service/device graph into a fixed embedding.
    Lighter than WorldStateGraphEncoder (fewer entities typically).
    """

    def __init__(
        self,
        d_node: int = 64,
        d_edge: int = 32,
        d_output: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        self.d_node = d_node
        self.d_output = d_output

        self.node_proj = nn.Linear(d_node, d_node)

        self.mp_layers = nn.ModuleList([
            GraphMessagePassingLayer(d_node, d_edge) for _ in range(n_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(d_node, d_output),
            nn.SiLU(),
            nn.Linear(d_output, d_output),
        )

        self.state_gru = nn.GRUCell(d_output, d_output)
        self._hidden: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int = 1, device: str = "cpu"):
        self._hidden = torch.zeros(batch_size, self.d_output, device=device)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.node_proj(node_features)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_features)

        # Mean readout (simpler for self-state)
        graph_emb = h.mean(dim=0, keepdim=True)
        graph_emb = self.readout(graph_emb)

        if self._hidden is not None:
            self._hidden = self.state_gru(graph_emb, self._hidden)
            return self._hidden
        return graph_emb
