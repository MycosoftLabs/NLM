"""
NLM Multi-Resolution Merkle HyperDAG

Five-layer graph structure for representing knowledge at multiple
resolutions, from raw sensory events up to causal lineage.

Layer 0: Raw sensory events (waveforms, spectral windows, packets)
Layer 1: Fused observations (normalized states, anomalies, summaries)
Layer 2: Entities and pairwise relations (organisms, devices, sites)
Layer 3: Hyperedges / multi-way events (species X + compound Y + site Z)
Layer 4: Causal lineage DAG (directed cause-effect chains)

Every node at every layer is Merkle-hashed for tamper-evident integrity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set

from nlm.core.merkle import MerkleTree, merkle_root, sha256


class DAGLayer(IntEnum):
    """The five layers of the HyperDAG."""

    RAW_EVENTS = 0
    FUSED_OBSERVATIONS = 1
    ENTITIES_RELATIONS = 2
    HYPEREDGES = 3
    CAUSAL_LINEAGE = 4


@dataclass
class DAGNode:
    """A node in the HyperDAG."""

    node_id: str = ""
    layer: DAGLayer = DAGLayer.RAW_EVENTS
    node_type: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    content_hash: bytes = b""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # References to child nodes in lower layers
    child_ids: List[str] = field(default_factory=list)
    # References to parent nodes in higher layers
    parent_ids: List[str] = field(default_factory=list)

    def compute_hash(self) -> bytes:
        """Compute content hash from node properties."""
        import json
        payload = json.dumps({
            "node_id": self.node_id,
            "layer": int(self.layer),
            "node_type": self.node_type,
            "properties": self.properties,
            "child_ids": sorted(self.child_ids),
        }, sort_keys=True, default=str).encode("utf-8")
        self.content_hash = sha256(payload)
        return self.content_hash


@dataclass
class DAGEdge:
    """A pairwise edge in the HyperDAG (Layer 2)."""

    source_id: str = ""
    target_id: str = ""
    relation_type: str = ""
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    content_hash: bytes = b""

    def compute_hash(self) -> bytes:
        import json
        payload = json.dumps({
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "properties": self.properties,
        }, sort_keys=True, default=str).encode("utf-8")
        self.content_hash = sha256(payload)
        return self.content_hash


@dataclass
class HyperEdge:
    """
    A multi-way event connecting 3+ entities (Layer 3).

    Example: "species X responds to compound Y at site Z during event W"
    """

    hyperedge_id: str = ""
    participant_ids: List[str] = field(default_factory=list)
    event_type: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    content_hash: bytes = b""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def compute_hash(self) -> bytes:
        import json
        payload = json.dumps({
            "hyperedge_id": self.hyperedge_id,
            "participant_ids": sorted(self.participant_ids),
            "event_type": self.event_type,
            "properties": self.properties,
        }, sort_keys=True, default=str).encode("utf-8")
        self.content_hash = sha256(payload)
        return self.content_hash


@dataclass
class CausalLink:
    """A directed causal link in the Layer 4 DAG."""

    cause_id: str = ""
    effect_id: str = ""
    mechanism: str = ""
    confidence: float = 0.5
    lag_seconds: float = 0.0
    content_hash: bytes = b""

    def compute_hash(self) -> bytes:
        import json
        payload = json.dumps({
            "cause_id": self.cause_id,
            "effect_id": self.effect_id,
            "mechanism": self.mechanism,
            "confidence": self.confidence,
        }, sort_keys=True, default=str).encode("utf-8")
        self.content_hash = sha256(payload)
        return self.content_hash


class MerkleHyperDAG:
    """
    Multi-Resolution Merkle HyperDAG.

    In-memory graph with Merkle-hashed nodes at all five layers.
    Designed to be synced to/from MINDEX for persistence.
    """

    def __init__(self):
        self.nodes: Dict[str, DAGNode] = {}
        self.edges: Dict[str, DAGEdge] = {}  # keyed by "source->target"
        self.hyperedges: Dict[str, HyperEdge] = {}
        self.causal_links: Dict[str, CausalLink] = {}
        self._layer_index: Dict[DAGLayer, Set[str]] = {
            layer: set() for layer in DAGLayer
        }

    # ── Node Operations ─────────────────────────────────────────

    def add_node(self, node: DAGNode) -> bytes:
        """Add a node and compute its hash. Returns content_hash."""
        node.compute_hash()
        self.nodes[node.node_id] = node
        self._layer_index[node.layer].add(node.node_id)
        return node.content_hash

    def get_node(self, node_id: str) -> Optional[DAGNode]:
        return self.nodes.get(node_id)

    def get_layer_nodes(self, layer: DAGLayer) -> List[DAGNode]:
        """Get all nodes at a given layer."""
        return [self.nodes[nid] for nid in self._layer_index[layer] if nid in self.nodes]

    # ── Edge Operations ─────────────────────────────────────────

    def add_edge(self, edge: DAGEdge) -> bytes:
        """Add a pairwise edge. Returns content_hash."""
        edge.compute_hash()
        key = f"{edge.source_id}->{edge.target_id}"
        self.edges[key] = edge
        return edge.content_hash

    def get_edges_from(self, node_id: str) -> List[DAGEdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges.values() if e.source_id == node_id]

    def get_edges_to(self, node_id: str) -> List[DAGEdge]:
        """Get all edges targeting a node."""
        return [e for e in self.edges.values() if e.target_id == node_id]

    # ── HyperEdge Operations ────────────────────────────────────

    def add_hyperedge(self, he: HyperEdge) -> bytes:
        """Add a multi-way hyperedge. Returns content_hash."""
        he.compute_hash()
        self.hyperedges[he.hyperedge_id] = he
        return he.content_hash

    def get_hyperedges_for(self, entity_id: str) -> List[HyperEdge]:
        """Get all hyperedges involving a given entity."""
        return [he for he in self.hyperedges.values() if entity_id in he.participant_ids]

    # ── Causal Links ────────────────────────────────────────────

    def add_causal_link(self, link: CausalLink) -> bytes:
        """Add a causal link. Returns content_hash."""
        link.compute_hash()
        key = f"{link.cause_id}=>{link.effect_id}"
        self.causal_links[key] = link
        return link.content_hash

    def get_causal_chain(self, start_id: str, max_depth: int = 10) -> List[CausalLink]:
        """Trace causal chain forward from a node."""
        chain = []
        visited = set()
        frontier = [start_id]

        for _ in range(max_depth):
            next_frontier = []
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                for link in self.causal_links.values():
                    if link.cause_id == nid and link.effect_id not in visited:
                        chain.append(link)
                        next_frontier.append(link.effect_id)
            frontier = next_frontier
            if not frontier:
                break

        return chain

    # ── Merkle Integrity ────────────────────────────────────────

    def layer_merkle_root(self, layer: DAGLayer) -> bytes:
        """Compute Merkle root for all nodes at a layer."""
        nodes = self.get_layer_nodes(layer)
        if not nodes:
            from nlm.core.merkle import GENESIS_ROOT
            return GENESIS_ROOT
        hashes = sorted([n.content_hash for n in nodes if n.content_hash])
        return merkle_root(hashes)

    def full_merkle_root(self) -> bytes:
        """Compute Merkle root over all five layer roots."""
        layer_roots = [self.layer_merkle_root(layer) for layer in DAGLayer]
        return merkle_root(layer_roots)

    # ── Subgraph Extraction ─────────────────────────────────────

    def extract_subgraph(
        self,
        root_id: str,
        max_depth: int = 2,
        layers: Optional[List[DAGLayer]] = None,
    ) -> Dict[str, Any]:
        """
        Extract a subgraph centered on a node.

        Returns dict with nodes, edges, and hyperedges within max_depth.
        """
        allowed_layers = set(layers) if layers else set(DAGLayer)
        visited_nodes: Set[str] = set()
        result_edges: List[DAGEdge] = []
        result_hyperedges: List[HyperEdge] = []

        frontier = {root_id}
        for _ in range(max_depth):
            next_frontier: Set[str] = set()
            for nid in frontier:
                if nid in visited_nodes:
                    continue
                visited_nodes.add(nid)
                node = self.nodes.get(nid)
                if node and node.layer in allowed_layers:
                    # Follow edges
                    for edge in self.get_edges_from(nid):
                        result_edges.append(edge)
                        next_frontier.add(edge.target_id)
                    for edge in self.get_edges_to(nid):
                        result_edges.append(edge)
                        next_frontier.add(edge.source_id)
                    # Follow hyperedges
                    for he in self.get_hyperedges_for(nid):
                        result_hyperedges.append(he)
                        next_frontier.update(he.participant_ids)
                    # Follow parent/child links
                    if node:
                        next_frontier.update(node.child_ids)
                        next_frontier.update(node.parent_ids)
            frontier = next_frontier - visited_nodes
            if not frontier:
                break

        result_nodes = [self.nodes[nid] for nid in visited_nodes if nid in self.nodes]

        return {
            "nodes": result_nodes,
            "edges": result_edges,
            "hyperedges": result_hyperedges,
            "root_id": root_id,
            "node_count": len(result_nodes),
        }

    # ── Stats ───────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """Return counts per layer and totals."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_hyperedges": len(self.hyperedges),
            "total_causal_links": len(self.causal_links),
            **{f"layer_{int(layer)}_nodes": len(ids) for layer, ids in self._layer_index.items()},
        }
