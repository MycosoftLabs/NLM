"""
Multi-Resolution Merkle HyperDAG
=================================

The world model backbone. Not a flat vector store — a structured graph
with cryptographic integrity across multiple resolution layers.

Layers:
  L0: Raw sensory events, waveform windows, spectral windows, device packets
  L1: Fused observations, normalized state estimates, anomalies, summaries
  L2: Entities and pairwise relations (organisms, species, devices, sites, compounds)
  L3: Hyperedges — multi-way events (fungus + VOC spike + humidity + intervention)
  L4: Causal lineage DAG (derivations, predictions, interventions, outcomes)

Merkle wrapping provides integrity and replay across all levels.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


class DAGLayer(IntEnum):
    """Resolution layers in the HyperDAG."""
    RAW_EVENTS = 0
    FUSED_OBSERVATIONS = 1
    ENTITIES = 2
    HYPEREDGES = 3
    CAUSAL_LINEAGE = 4


@dataclass
class HyperNode:
    """A node in the HyperDAG.

    Nodes exist at a specific resolution layer and carry typed data.
    """

    node_id: str
    layer: DAGLayer
    node_type: str  # "event", "observation", "entity", "organism", "device", "site", "compound", etc.
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    geolocation: Optional[Tuple[float, float, float]] = None
    merkle_hash: str = ""
    parent_ids: List[str] = field(default_factory=list)  # nodes this was derived from
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if not self.merkle_hash:
            self.merkle_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "node_id": self.node_id,
            "layer": int(self.layer),
            "node_type": self.node_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "parent_ids": sorted(self.parent_ids),
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "layer": int(self.layer),
            "node_type": self.node_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "geolocation": list(self.geolocation) if self.geolocation else None,
            "merkle_hash": self.merkle_hash,
            "parent_ids": self.parent_ids,
            "tags": list(self.tags),
        }


@dataclass
class HyperEdge:
    """A hyperedge connecting multiple nodes.

    Represents multi-way relationships:
    e.g., fungus + VOC spike + humidity jump + intervention + location
    """

    edge_id: str
    node_ids: List[str]  # 2+ nodes participating
    edge_type: str  # "co-occurrence", "causal", "correlation", "intervention", "spatial_cluster"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    merkle_hash: str = ""

    def __post_init__(self):
        if not self.merkle_hash:
            content = json.dumps({
                "edge_id": self.edge_id,
                "node_ids": sorted(self.node_ids),
                "edge_type": self.edge_type,
                "weight": self.weight,
            }, sort_keys=True, default=str)
            self.merkle_hash = hashlib.sha256(content.encode()).hexdigest()

    @property
    def arity(self) -> int:
        return len(self.node_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "node_ids": self.node_ids,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "merkle_hash": self.merkle_hash,
            "arity": self.arity,
        }


@dataclass
class CausalEdge:
    """A directed causal link in the lineage DAG (Layer 4).

    Records what produced what, in what order, with what confidence.
    """

    source_id: str
    target_id: str
    causal_type: str  # "derived_from", "predicted", "intervened", "observed_outcome"
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "causal_type": self.causal_type,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class HyperDAG:
    """Multi-Resolution Merkle HyperDAG.

    The world model backbone combining:
    - Graph traversal
    - Hypergraph reasoning
    - Causal lineage
    - Cryptographic trust via Merkle hashing
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, HyperNode] = {}
        self.hyperedges: Dict[str, HyperEdge] = {}
        self.causal_edges: List[CausalEdge] = []

        # Indices for fast lookup
        self._layer_index: Dict[DAGLayer, Set[str]] = defaultdict(set)
        self._type_index: Dict[str, Set[str]] = defaultdict(set)
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)  # node_id → connected node_ids
        self._node_to_hyperedges: Dict[str, Set[str]] = defaultdict(set)
        self._children: Dict[str, Set[str]] = defaultdict(set)  # parent → children (derivation)
        self._parents: Dict[str, Set[str]] = defaultdict(set)  # child → parents

    # --- Node Operations ---

    def add_node(self, node: HyperNode) -> None:
        self.nodes[node.node_id] = node
        self._layer_index[node.layer].add(node.node_id)
        self._type_index[node.node_type].add(node.node_id)
        for parent_id in node.parent_ids:
            self._children[parent_id].add(node.node_id)
            self._parents[node.node_id].add(parent_id)

    def get_node(self, node_id: str) -> Optional[HyperNode]:
        return self.nodes.get(node_id)

    def get_nodes_by_layer(self, layer: DAGLayer) -> List[HyperNode]:
        return [self.nodes[nid] for nid in self._layer_index.get(layer, set()) if nid in self.nodes]

    def get_nodes_by_type(self, node_type: str) -> List[HyperNode]:
        return [self.nodes[nid] for nid in self._type_index.get(node_type, set()) if nid in self.nodes]

    # --- Hyperedge Operations ---

    def add_hyperedge(self, edge: HyperEdge) -> None:
        self.hyperedges[edge.edge_id] = edge
        for nid in edge.node_ids:
            self._node_to_hyperedges[nid].add(edge.edge_id)
            for other_nid in edge.node_ids:
                if other_nid != nid:
                    self._adjacency[nid].add(other_nid)

    def get_hyperedges_for_node(self, node_id: str) -> List[HyperEdge]:
        return [
            self.hyperedges[eid]
            for eid in self._node_to_hyperedges.get(node_id, set())
            if eid in self.hyperedges
        ]

    def get_neighbors(self, node_id: str) -> List[HyperNode]:
        return [
            self.nodes[nid]
            for nid in self._adjacency.get(node_id, set())
            if nid in self.nodes
        ]

    # --- Causal Operations ---

    def add_causal_edge(self, edge: CausalEdge) -> None:
        self.causal_edges.append(edge)
        self._children[edge.source_id].add(edge.target_id)
        self._parents[edge.target_id].add(edge.source_id)

    def get_descendants(self, node_id: str, max_depth: int = 10) -> List[str]:
        """BFS to find all descendants in the causal DAG."""
        visited = set()
        queue = [node_id]
        depth = 0
        while queue and depth < max_depth:
            next_queue = []
            for nid in queue:
                for child in self._children.get(nid, set()):
                    if child not in visited:
                        visited.add(child)
                        next_queue.append(child)
            queue = next_queue
            depth += 1
        return list(visited)

    def get_ancestors(self, node_id: str, max_depth: int = 10) -> List[str]:
        """BFS to find all ancestors in the causal DAG."""
        visited = set()
        queue = [node_id]
        depth = 0
        while queue and depth < max_depth:
            next_queue = []
            for nid in queue:
                for parent in self._parents.get(nid, set()):
                    if parent not in visited:
                        visited.add(parent)
                        next_queue.append(parent)
            queue = next_queue
            depth += 1
        return list(visited)

    # --- Spatial Queries ---

    def get_nodes_in_radius(
        self, lat: float, lon: float, radius_km: float, layer: Optional[DAGLayer] = None,
    ) -> List[HyperNode]:
        """Find nodes within a geographic radius."""
        import math
        results = []
        candidates = self._layer_index.get(layer, set()) if layer is not None else set(self.nodes.keys())
        for nid in candidates:
            node = self.nodes.get(nid)
            if node and node.geolocation:
                dist = self._haversine(lat, lon, node.geolocation[0], node.geolocation[1])
                if dist <= radius_km:
                    results.append(node)
        return results

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in km."""
        import math
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # --- Merkle Integrity ---

    def compute_layer_root(self, layer: DAGLayer) -> str:
        """Compute a Merkle root for an entire layer."""
        node_ids = sorted(self._layer_index.get(layer, set()))
        hashes = [self.nodes[nid].merkle_hash for nid in node_ids if nid in self.nodes]
        if not hashes:
            return hashlib.sha256(b"empty_layer").hexdigest()
        combined = "||".join(hashes)
        return hashlib.sha256(combined.encode()).hexdigest()

    def compute_dag_root(self) -> str:
        """Compute the overall DAG Merkle root across all layers."""
        layer_roots = [self.compute_layer_root(layer) for layer in DAGLayer]
        combined = "||".join(layer_roots)
        return hashlib.sha256(combined.encode()).hexdigest()

    # --- Stats ---

    def stats(self) -> Dict[str, Any]:
        return {
            "total_nodes": len(self.nodes),
            "total_hyperedges": len(self.hyperedges),
            "total_causal_edges": len(self.causal_edges),
            "nodes_per_layer": {
                layer.name: len(self._layer_index.get(layer, set()))
                for layer in DAGLayer
            },
            "node_types": {t: len(ids) for t, ids in self._type_index.items()},
            "dag_root": self.compute_dag_root(),
        }
