"""
NLM Graph Retrieval

GraphRAG-style retrieval from the Multi-Resolution Merkle HyperDAG.
Given a query (frame or natural-language), retrieves relevant subgraphs
for the WorldStateGraphEncoder and SelfStateGraphEncoder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from nlm.core.frames import RootedNatureFrame
from nlm.graph.hyperdag import DAGLayer, DAGNode, MerkleHyperDAG
from nlm.mindex.client import MINDEXClient, GraphQuery, Subgraph

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a graph retrieval query."""

    nodes: List[DAGNode] = field(default_factory=list)
    edges: List[Any] = field(default_factory=list)
    hyperedges: List[Any] = field(default_factory=list)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    source: str = ""  # "local" or "mindex"


class GraphRetriever:
    """
    Retrieves relevant subgraphs for model consumption.

    Two modes:
    1. Local: queries the in-memory MerkleHyperDAG
    2. MINDEX: queries the remote MINDEX graph store
    """

    def __init__(
        self,
        local_dag: Optional[MerkleHyperDAG] = None,
        mindex_client: Optional[MINDEXClient] = None,
    ):
        self.local_dag = local_dag or MerkleHyperDAG()
        self.mindex_client = mindex_client

    def retrieve_for_frame(
        self,
        frame: RootedNatureFrame,
        max_depth: int = 2,
        max_nodes: int = 100,
    ) -> RetrievalResult:
        """
        Retrieve relevant subgraph for a RootedNatureFrame.

        Uses device_ids, sensor_ids, and geolocation to find relevant
        entities in the local DAG.
        """
        # Collect seed entity IDs from the frame
        seed_ids: List[str] = []
        seed_ids.extend(frame.ground_truth.device_ids)
        seed_ids.extend(frame.ground_truth.sensor_ids)

        # Add location-based seed
        geo = frame.ground_truth.geolocation
        if geo.latitude != 0.0 or geo.longitude != 0.0:
            # Look for site nodes near this location
            for node in self.local_dag.get_layer_nodes(DAGLayer.ENTITIES_RELATIONS):
                node_lat = node.properties.get("latitude", 0)
                node_lon = node.properties.get("longitude", 0)
                if node_lat and node_lon:
                    dist = _haversine_km(geo.latitude, geo.longitude, node_lat, node_lon)
                    if dist < 10:  # within 10km
                        seed_ids.append(node.node_id)

        if not seed_ids:
            return RetrievalResult(source="local")

        # Extract subgraphs from each seed and merge
        all_nodes: Dict[str, DAGNode] = {}
        all_edges = []
        all_hyperedges = []

        for seed_id in seed_ids[:10]:  # limit seeds
            subgraph = self.local_dag.extract_subgraph(
                root_id=seed_id, max_depth=max_depth
            )
            for node in subgraph["nodes"]:
                all_nodes[node.node_id] = node
            all_edges.extend(subgraph["edges"])
            all_hyperedges.extend(subgraph["hyperedges"])

        # Trim to max_nodes
        nodes = list(all_nodes.values())[:max_nodes]

        return RetrievalResult(
            nodes=nodes,
            edges=all_edges,
            hyperedges=all_hyperedges,
            source="local",
        )

    async def retrieve_from_mindex(
        self,
        entity_types: Optional[List[str]] = None,
        root_entity_id: Optional[str] = None,
        max_depth: int = 2,
        limit: int = 100,
    ) -> RetrievalResult:
        """Retrieve subgraph from MINDEX graph store."""
        if not self.mindex_client:
            logger.warning("No MINDEX client configured for graph retrieval")
            return RetrievalResult(source="mindex")

        query = GraphQuery(
            entity_types=entity_types or [],
            root_entity_id=root_entity_id,
            max_depth=max_depth,
            limit=limit,
        )

        try:
            subgraph = await self.mindex_client.query_graph(query)
            # Convert MINDEX nodes to DAGNodes
            dag_nodes = [
                DAGNode(
                    node_id=n.node_id,
                    node_type=n.node_type,
                    layer=DAGLayer.ENTITIES_RELATIONS,
                    properties=n.properties,
                )
                for n in subgraph.nodes
            ]
            return RetrievalResult(
                nodes=dag_nodes,
                edges=subgraph.edges,
                source="mindex",
            )
        except Exception as e:
            logger.error(f"MINDEX graph retrieval failed: {e}")
            return RetrievalResult(source="mindex")

    async def retrieve_combined(
        self,
        frame: RootedNatureFrame,
        max_depth: int = 2,
        max_nodes: int = 100,
    ) -> RetrievalResult:
        """Retrieve from both local DAG and MINDEX, merge results."""
        local = self.retrieve_for_frame(frame, max_depth, max_nodes)

        if self.mindex_client:
            device_ids = frame.ground_truth.device_ids
            root_id = device_ids[0] if device_ids else None
            mindex = await self.retrieve_from_mindex(
                root_entity_id=root_id, max_depth=max_depth, limit=max_nodes
            )
            # Merge: local nodes take precedence
            seen_ids = {n.node_id for n in local.nodes}
            for node in mindex.nodes:
                if node.node_id not in seen_ids:
                    local.nodes.append(node)
            local.edges.extend(mindex.edges)
            local.source = "combined"

        return local


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometers."""
    import math
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
