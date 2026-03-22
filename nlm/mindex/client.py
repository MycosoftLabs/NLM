"""
MINDEX Client
=============

Typed client for MINDEX — the central persistence, provenance,
and knowledge substrate for all Mycosoft systems.

MINDEX is:
- Ledger for provenance and Merkle roots
- Graph store for entities and relations
- Vector store for embeddings and fingerprints
- Time-series store for sensor data

This client provides typed methods for NLM-specific operations:
frame persistence, lineage storage, embedding queries, and
training data retrieval.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from nlm.core.merkle import LineageRecord


class MINDEXClient:
    """Client for MINDEX persistence and knowledge operations.

    All methods are async-ready and handle connection failures gracefully.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.environ.get("MINDEX_API_URL", "http://localhost:8003")
        self._http = None

    async def _get_http(self):
        if self._http is None:
            import httpx
            self._http = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self._http

    # --- Lineage / Provenance ---

    async def store_lineage(self, record: LineageRecord) -> Dict[str, Any]:
        """Store a Merkle lineage record in MINDEX."""
        http = await self._get_http()
        try:
            resp = await http.post("/api/lineage/store", json=record.to_dict())
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {"status": "error", "stored": False, "record": record.to_dict()}

    async def get_lineage(self, frame_root: str) -> Optional[LineageRecord]:
        """Retrieve a lineage record by frame root."""
        http = await self._get_http()
        try:
            resp = await http.get(f"/api/lineage/{frame_root}")
            if resp.status_code == 200:
                return LineageRecord.from_dict(resp.json())
            return None
        except Exception:
            return None

    async def get_lineage_chain(self, frame_root: str, depth: int = 100) -> List[LineageRecord]:
        """Walk back through lineage from a frame root."""
        http = await self._get_http()
        try:
            resp = await http.get(f"/api/lineage/chain/{frame_root}", params={"depth": depth})
            if resp.status_code == 200:
                return [LineageRecord.from_dict(r) for r in resp.json().get("records", [])]
            return []
        except Exception:
            return []

    # --- Frame Persistence ---

    async def store_frame(self, frame_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a RootedNatureFrame (as dict) to MINDEX."""
        http = await self._get_http()
        try:
            resp = await http.post("/api/frames/store", json=frame_dict)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {"status": "error", "stored": False}

    async def get_frame(self, frame_root: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored frame by its root hash."""
        http = await self._get_http()
        try:
            resp = await http.get(f"/api/frames/{frame_root}")
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    # --- Vector / Embedding Store ---

    async def store_embedding(
        self, entity_id: str, vector: List[float], metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Store an embedding vector in MINDEX."""
        http = await self._get_http()
        try:
            payload = {"entity_id": entity_id, "vector": vector, "metadata": metadata or {}}
            resp = await http.post("/api/vectors/store", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {"status": "error", "stored": False}

    async def search_vectors(
        self, query_vector: List[float], top_k: int = 10, filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in MINDEX."""
        http = await self._get_http()
        try:
            payload = {"vector": query_vector, "top_k": top_k, "filters": filters or {}}
            resp = await http.post("/api/vectors/search", json=payload)
            if resp.status_code == 200:
                return resp.json().get("results", [])
            return []
        except Exception:
            return []

    # --- Graph Store ---

    async def store_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Store an entity node in the knowledge graph."""
        http = await self._get_http()
        try:
            resp = await http.post("/api/graph/entities", json=entity)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {"status": "error"}

    async def store_relation(self, relation: Dict[str, Any]) -> Dict[str, Any]:
        """Store a relation edge in the knowledge graph."""
        http = await self._get_http()
        try:
            resp = await http.post("/api/graph/relations", json=relation)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {"status": "error"}

    async def query_graph(self, query: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a graph query against MINDEX."""
        http = await self._get_http()
        try:
            payload = {"query": query, "params": params or {}}
            resp = await http.post("/api/graph/query", json=payload)
            if resp.status_code == 200:
                return resp.json()
            return {"results": []}
        except Exception:
            return {"results": []}

    # --- Time-Series Store ---

    async def store_timeseries(
        self, series_id: str, timestamp: str, values: Dict[str, float],
    ) -> Dict[str, Any]:
        """Store a time-series data point."""
        http = await self._get_http()
        try:
            payload = {"series_id": series_id, "timestamp": timestamp, "values": values}
            resp = await http.post("/api/timeseries/store", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {"status": "error"}

    async def query_timeseries(
        self, series_id: str, start: str, end: str, resolution: str = "raw",
    ) -> List[Dict[str, Any]]:
        """Query time-series data for a given range."""
        http = await self._get_http()
        try:
            params = {"series_id": series_id, "start": start, "end": end, "resolution": resolution}
            resp = await http.get("/api/timeseries/query", params=params)
            if resp.status_code == 200:
                return resp.json().get("points", [])
            return []
        except Exception:
            return []

    # --- Training Data ---

    async def get_training_frames(
        self, limit: int = 1000, offset: int = 0, filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve frames for model training."""
        http = await self._get_http()
        try:
            params = {"limit": limit, "offset": offset}
            if filters:
                params["filters"] = str(filters)
            resp = await http.get("/api/frames/training", params=params)
            if resp.status_code == 200:
                return resp.json().get("frames", [])
            return []
        except Exception:
            return []

    async def close(self):
        if self._http:
            await self._http.aclose()
            self._http = None
