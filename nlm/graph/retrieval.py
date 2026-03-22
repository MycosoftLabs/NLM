"""
GraphRAG Retrieval
==================

Graph-native retrieval combining graph traversal with vector similarity.
Not vector-only — uses the HyperDAG structure for reasoning.

Designed to anticipate sparse-matrix compilation paths for
graph traversal, propagation, and constrained routing.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from nlm.graph.hyperdag import CausalEdge, DAGLayer, HyperDAG, HyperEdge, HyperNode


class GraphRetriever:
    """Graph-native retrieval over HyperDAG.

    Supports:
    - Local neighborhood expansion (k-hop)
    - Type-constrained traversal
    - Causal chain retrieval
    - Hyperedge-based multi-way lookup
    - Spatial proximity queries
    """

    def __init__(self, dag: HyperDAG) -> None:
        self.dag = dag

    def retrieve_neighborhood(
        self, seed_id: str, max_hops: int = 2, max_nodes: int = 50,
        allowed_types: Optional[Set[str]] = None,
    ) -> List[HyperNode]:
        """Expand from a seed node through k hops of the graph.

        Returns nodes reachable within max_hops, optionally filtered by type.
        """
        visited: Set[str] = {seed_id}
        frontier = {seed_id}
        results = []

        seed = self.dag.get_node(seed_id)
        if seed:
            results.append(seed)

        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for nid in frontier:
                for neighbor in self.dag.get_neighbors(nid):
                    if neighbor.node_id not in visited:
                        if allowed_types is None or neighbor.node_type in allowed_types:
                            visited.add(neighbor.node_id)
                            next_frontier.add(neighbor.node_id)
                            results.append(neighbor)
                            if len(results) >= max_nodes:
                                return results
            frontier = next_frontier
            if not frontier:
                break

        return results

    def retrieve_by_type_and_proximity(
        self, node_type: str, lat: float, lon: float,
        radius_km: float = 50.0, limit: int = 20,
    ) -> List[HyperNode]:
        """Retrieve nodes of a specific type near a location."""
        nodes = self.dag.get_nodes_by_type(node_type)
        scored = []
        for node in nodes:
            if node.geolocation:
                dist = self.dag._haversine(lat, lon, node.geolocation[0], node.geolocation[1])
                if dist <= radius_km:
                    scored.append((dist, node))
        scored.sort(key=lambda x: x[0])
        return [node for _, node in scored[:limit]]

    def retrieve_causal_chain(
        self, node_id: str, direction: str = "forward", max_depth: int = 10,
    ) -> List[HyperNode]:
        """Follow causal lineage forward (descendants) or backward (ancestors)."""
        if direction == "forward":
            ids = self.dag.get_descendants(node_id, max_depth)
        else:
            ids = self.dag.get_ancestors(node_id, max_depth)
        return [self.dag.nodes[nid] for nid in ids if nid in self.dag.nodes]

    def retrieve_hyperedge_context(self, node_id: str) -> Dict[str, Any]:
        """Get all hyperedges a node participates in, with co-participating nodes.

        Returns the multi-way context for a given node.
        """
        edges = self.dag.get_hyperedges_for_node(node_id)
        context = {
            "node_id": node_id,
            "hyperedges": [],
        }
        for edge in edges:
            participants = []
            for nid in edge.node_ids:
                if nid != node_id:
                    node = self.dag.get_node(nid)
                    if node:
                        participants.append({
                            "node_id": nid,
                            "node_type": node.node_type,
                            "layer": int(node.layer),
                        })
            context["hyperedges"].append({
                "edge_id": edge.edge_id,
                "edge_type": edge.edge_type,
                "weight": edge.weight,
                "arity": edge.arity,
                "participants": participants,
            })
        return context

    def retrieve_multi_resolution(
        self, node_id: str, include_layers: Optional[List[DAGLayer]] = None,
    ) -> Dict[str, List[HyperNode]]:
        """Retrieve connected nodes across multiple resolution layers.

        Starting from a node, find related nodes at each layer:
        - L0: raw events that contributed
        - L1: fused observations derived from those events
        - L2: entities related to those observations
        - L3: hyperedges connecting those entities
        - L4: causal lineage
        """
        if include_layers is None:
            include_layers = list(DAGLayer)

        result: Dict[str, List[HyperNode]] = {}

        ancestors = set(self.dag.get_ancestors(node_id, max_depth=5))
        descendants = set(self.dag.get_descendants(node_id, max_depth=5))
        neighbors = {n.node_id for n in self.dag.get_neighbors(node_id)}
        all_related = ancestors | descendants | neighbors | {node_id}

        for layer in include_layers:
            layer_nodes = []
            for nid in all_related:
                node = self.dag.get_node(nid)
                if node and node.layer == layer:
                    layer_nodes.append(node)
            result[layer.name] = layer_nodes

        return result


class HybridRetriever:
    """Combines graph traversal with vector similarity.

    GraphRAG-style retrieval: use the graph structure for reasoning
    about relationships, and vectors for semantic similarity.
    """

    def __init__(self, dag: HyperDAG) -> None:
        self.dag = dag
        self.graph_retriever = GraphRetriever(dag)
        self._embeddings: Dict[str, np.ndarray] = {}  # node_id → embedding

    def set_embedding(self, node_id: str, embedding: np.ndarray) -> None:
        self._embeddings[node_id] = embedding

    def retrieve(
        self,
        query_embedding: Optional[np.ndarray] = None,
        seed_node_id: Optional[str] = None,
        top_k: int = 20,
        graph_hops: int = 2,
        vector_weight: float = 0.5,
        graph_weight: float = 0.5,
    ) -> List[Tuple[str, float, HyperNode]]:
        """Hybrid retrieval combining graph distance and vector similarity.

        Returns list of (node_id, combined_score, node) sorted by score.
        """
        candidates: Dict[str, float] = {}

        # Graph-based candidates
        if seed_node_id is not None:
            graph_nodes = self.graph_retriever.retrieve_neighborhood(
                seed_node_id, max_hops=graph_hops, max_nodes=top_k * 3,
            )
            for i, node in enumerate(graph_nodes):
                # Score decays with distance from seed
                graph_score = 1.0 / (1.0 + i * 0.1)
                candidates[node.node_id] = graph_score * graph_weight

        # Vector-based candidates
        if query_embedding is not None and self._embeddings:
            similarities = []
            for nid, emb in self._embeddings.items():
                sim = self._cosine_similarity(query_embedding, emb)
                similarities.append((nid, sim))
            similarities.sort(key=lambda x: -x[1])
            for nid, sim in similarities[:top_k * 3]:
                vector_score = max(0.0, sim) * vector_weight
                candidates[nid] = candidates.get(nid, 0.0) + vector_score

        # Sort and return
        scored = []
        for nid, score in candidates.items():
            node = self.dag.get_node(nid)
            if node:
                scored.append((nid, score, node))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
