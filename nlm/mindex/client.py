"""
NLM MINDEX Client

Unified client for all MINDEX operations. Replaces ad-hoc httpx calls
scattered throughout the codebase. MINDEX is the central knowledge
substrate — ledger, graph store, vector store, time-series store.

NLM does not own MINDEX persistence; it consumes MINDEX via this client.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nlm.core.frames import RootedNatureFrame

logger = logging.getLogger(__name__)


@dataclass
class VectorHit:
    """A result from MINDEX vector similarity search."""

    mindex_id: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphQuery:
    """Query specification for MINDEX graph store."""

    entity_types: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    root_entity_id: Optional[str] = None
    max_depth: int = 2
    limit: int = 100
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    """A node in a MINDEX graph result."""

    node_id: str = ""
    node_type: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge in a MINDEX graph result."""

    source_id: str = ""
    target_id: str = ""
    relation_type: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subgraph:
    """A subgraph returned from MINDEX graph queries."""

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)


@dataclass
class TrainingRecord:
    """Record of a training batch for provenance tracking."""

    record_id: str = ""
    model_version: str = ""
    frame_roots: List[str] = field(default_factory=list)  # hex-encoded
    loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MINDEXClient:
    """
    Unified client for all MINDEX operations.

    Replaces ad-hoc httpx calls in engine.py, client.py, etc.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0):
        self.base_url = (
            base_url
            or os.getenv("MINDEX_API_URL", "http://localhost:8003")
        ).rstrip("/")
        self.timeout = timeout
        logger.info(f"MINDEXClient initialized: {self.base_url}")

    # ── Frame Storage ───────────────────────────────────────────

    async def store_frame(self, frame: RootedNatureFrame) -> str:
        """
        Store a RootedNatureFrame in MINDEX.

        Returns the frame_root hex as the storage key.
        """
        import httpx

        payload = frame.to_dict()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/frames",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("frame_root", frame.frame_root.hex())

    async def get_frame(self, frame_root_hex: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored frame by its Merkle root."""
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.base_url}/api/frames/{frame_root_hex}",
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()

    # ── Merkle Lineage ──────────────────────────────────────────

    async def store_merkle_root(self, root_hex: str, metadata: Dict[str, Any]) -> None:
        """Store a Merkle root with associated metadata."""
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/merkle/roots",
                json={"root": root_hex, "metadata": metadata},
            )
            resp.raise_for_status()

    async def get_lineage(self, frame_root_hex: str, depth: int = 100) -> List[str]:
        """
        Get the lineage chain for a frame (list of parent frame_root hexes).
        """
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.base_url}/api/merkle/lineage/{frame_root_hex}",
                params={"depth": depth},
            )
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            return resp.json().get("lineage", [])

    # ── Graph Store ─────────────────────────────────────────────

    async def query_graph(self, query: GraphQuery) -> Subgraph:
        """Query the MINDEX entity-relation graph."""
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/graph/query",
                json={
                    "entity_types": query.entity_types,
                    "relation_types": query.relation_types,
                    "root_entity_id": query.root_entity_id,
                    "max_depth": query.max_depth,
                    "limit": query.limit,
                    "filters": query.filters,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            nodes = [GraphNode(**n) for n in data.get("nodes", [])]
            edges = [GraphEdge(**e) for e in data.get("edges", [])]
            return Subgraph(nodes=nodes, edges=edges)

    async def store_graph_nodes(self, nodes: List[GraphNode]) -> int:
        """Store graph nodes in MINDEX. Returns count stored."""
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/graph/nodes",
                json=[{"node_id": n.node_id, "node_type": n.node_type,
                        "properties": n.properties} for n in nodes],
            )
            resp.raise_for_status()
            return resp.json().get("count", 0)

    async def store_graph_edges(self, edges: List[GraphEdge]) -> int:
        """Store graph edges in MINDEX. Returns count stored."""
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/graph/edges",
                json=[{"source_id": e.source_id, "target_id": e.target_id,
                        "relation_type": e.relation_type,
                        "properties": e.properties} for e in edges],
            )
            resp.raise_for_status()
            return resp.json().get("count", 0)

    # ── Vector Store ────────────────────────────────────────────

    async def store_vectors(
        self,
        vectors: List[Tuple[str, np.ndarray, Dict[str, Any]]],
    ) -> int:
        """
        Store embedding vectors in MINDEX.

        Each tuple is (id, vector, metadata).
        Returns count stored.
        """
        import httpx

        payload = [
            {"id": vid, "vector": vec.tolist(), "metadata": meta}
            for vid, vec, meta in vectors
        ]
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/vectors",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json().get("count", 0)

    async def search_vectors(
        self,
        embedding: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        """Nearest-neighbor search in MINDEX vector store."""
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/vectors/search",
                json={
                    "vector": embedding.tolist(),
                    "k": k,
                    "filters": filters or {},
                },
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
            return [VectorHit(**h) for h in hits]

    # ── Training Provenance ─────────────────────────────────────

    async def store_training_record(self, record: TrainingRecord) -> str:
        """Store a training provenance record. Returns record_id."""
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/training/records",
                json={
                    "record_id": record.record_id,
                    "model_version": record.model_version,
                    "frame_roots": record.frame_roots,
                    "loss": record.loss,
                    "metrics": record.metrics,
                    "timestamp": record.timestamp.isoformat(),
                },
            )
            resp.raise_for_status()
            return resp.json().get("record_id", record.record_id)

    # ── Unified Search (legacy compatibility) ───────────────────

    async def search_unified(
        self, query: str, limit: int = 50, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Unified search across all MINDEX domains."""
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.base_url}/api/search/unified",
                params={"q": query, "limit": limit, **kwargs},
            )
            if resp.status_code == 200:
                return resp.json().get("results", [])
            return []
