"""
NLM Universal Search Engine — aligned with MINDEX v3
======================================================

Calls mindex's ``/unified-search/earth`` endpoint (which fans out across
all 35 domains in parallel via asyncio.gather on the mindex side) and
normalises the response for NLM consumers: the API, Myca, CREP bridge,
and the ingestion pipeline.

Also supports direct domain queries via ``/earth/*`` detail endpoints
and falls back to individual external source scraping when mindex is
unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from nlm.search.domains import (
    ALL_DOMAINS,
    DOMAIN_GROUPS,
    DOMAIN_TO_GROUP,
    DomainRegistry,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Request / Result — mirrors mindex SearchResult shape exactly
# ======================================================================

@dataclass
class SearchRequest:
    """Inbound search query — maps to mindex /unified-search/earth params."""
    query: str
    types: Optional[str] = None           # comma-separated domains or group alias
    limit: int = 50
    lat: Optional[float] = None
    lng: Optional[float] = None
    radius: Optional[float] = None        # km
    toxicity: Optional[str] = None        # fungi filter: poisonous, edible, psychedelic
    include_crep: bool = True


@dataclass
class SearchHit:
    """One result — matches mindex SearchResult schema exactly."""
    id: str
    domain: str                           # mindex domain key
    entity_type: str                      # specific type within domain
    name: str
    description: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    geometry_type: Optional[str] = None   # point, line, polygon
    occurred_at: Optional[str] = None
    source: Optional[str] = None
    image_url: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Aggregated response — mirrors mindex EarthSearchResponse."""
    query: str = ""
    domains_searched: List[str] = field(default_factory=list)
    results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    universal_results: List[SearchHit] = field(default_factory=list)
    total_count: int = 0
    timing_ms: int = 0
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "domains_searched": self.domains_searched,
            "results": self.results,
            "universal_results": [
                {
                    "id": h.id,
                    "domain": h.domain,
                    "entity_type": h.entity_type,
                    "name": h.name,
                    "description": h.description,
                    "lat": h.lat,
                    "lng": h.lng,
                    "geometry_type": h.geometry_type,
                    "occurred_at": h.occurred_at,
                    "source": h.source,
                    "image_url": h.image_url,
                    "properties": h.properties,
                }
                for h in self.universal_results
            ],
            "total_count": self.total_count,
            "timing_ms": self.timing_ms,
            "filters_applied": self.filters_applied,
            "errors": self.errors,
        }


# ======================================================================
# Engine
# ======================================================================

class UniversalSearchEngine:
    """
    Searches all Earth domains via the mindex unified-search/earth endpoint.

    Primary path:
      NLM → mindex /unified-search/earth → postgres (35 parallel queries)

    Fallback path (when mindex is down):
      NLM → individual external APIs (best-effort)
    """

    def __init__(self, mindex_url: Optional[str] = None) -> None:
        self.mindex_url = (mindex_url or "http://localhost:8003").rstrip("/")
        self.domains = DomainRegistry()

    async def search(self, request: SearchRequest) -> SearchResult:
        """Execute a universal Earth search via mindex."""
        t0 = time.monotonic()
        result = SearchResult(query=request.query)

        # Resolve which domains will be searched
        resolved = self.domains.resolve_types(request.types)
        result.domains_searched = resolved
        result.filters_applied = {
            "types": request.types or "all",
            "limit": request.limit,
        }
        if request.lat is not None:
            result.filters_applied["lat"] = request.lat
            result.filters_applied["lng"] = request.lng
            result.filters_applied["radius"] = request.radius
        if request.toxicity:
            result.filters_applied["toxicity"] = request.toxicity

        # Try mindex first
        mindex_ok = await self._query_mindex(request, result)

        if not mindex_ok:
            # Fallback: try mindex legacy endpoint
            await self._query_mindex_legacy(request, result)

        result.timing_ms = int((time.monotonic() - t0) * 1000)
        return result

    async def search_nearby(
        self,
        query: str,
        lat: float,
        lng: float,
        radius: float = 50,
        types: Optional[str] = None,
        limit: int = 50,
    ) -> SearchResult:
        """Location-based search — calls mindex /unified-search/nearby."""
        request = SearchRequest(
            query=query, types=types, limit=limit,
            lat=lat, lng=lng, radius=radius,
        )
        return await self.search(request)

    async def get_earth_stats(self) -> Dict[str, Any]:
        """Get entity counts from mindex /earth/stats."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.mindex_url}/earth/stats")
                if resp.status_code == 200:
                    return resp.json()
        except Exception as exc:
            logger.debug("earth/stats failed: %s", exc)
        return {"domains": {}, "total_entities": 0}

    async def get_map_bbox(
        self, layer: str,
        lat_min: float, lat_max: float,
        lng_min: float, lng_max: float,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """Spatial query for CREP map — calls mindex /earth/map/bbox."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.mindex_url}/earth/map/bbox",
                    params={
                        "layer": layer,
                        "lat_min": lat_min, "lat_max": lat_max,
                        "lng_min": lng_min, "lng_max": lng_max,
                        "limit": limit,
                    },
                )
                if resp.status_code == 200:
                    return resp.json()
        except Exception as exc:
            logger.debug("earth/map/bbox failed: %s", exc)
        return {"layer": layer, "entities": [], "total": 0}

    async def get_map_layers(self) -> List[Dict[str, Any]]:
        """Get available CREP map layers from mindex /earth/map/layers."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.mindex_url}/earth/map/layers")
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("layers", [])
        except Exception as exc:
            logger.debug("earth/map/layers failed: %s", exc)
        return []

    async def sync_to_crep(self, entity_type: str, limit: int = 1000) -> Dict[str, Any]:
        """Push domain data into crep.unified_entities via mindex /earth/crep/sync."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.mindex_url}/earth/crep/sync",
                    params={"entity_type": entity_type, "limit": limit},
                )
                if resp.status_code == 200:
                    return resp.json()
        except Exception as exc:
            logger.debug("earth/crep/sync failed: %s", exc)
        return {"error": "mindex unavailable"}

    # ------------------------------------------------------------------
    # Domain-specific detail endpoints
    # ------------------------------------------------------------------

    async def get_recent_earthquakes(
        self, hours: int = 24, min_magnitude: float = 2.5, limit: int = 100,
    ) -> Dict[str, Any]:
        return await self._get(
            "/earth/earthquakes/recent",
            {"hours": hours, "min_magnitude": min_magnitude, "limit": limit},
        )

    async def get_active_satellites(
        self,
        satellite_type: Optional[str] = None,
        orbit_type: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if satellite_type:
            params["satellite_type"] = satellite_type
        if orbit_type:
            params["orbit_type"] = orbit_type
        return await self._get("/earth/satellites/active", params)

    async def get_recent_solar(
        self, days: int = 30, event_type: Optional[str] = None, limit: int = 50,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"days": days, "limit": limit}
        if event_type:
            params["event_type"] = event_type
        return await self._get("/earth/solar/recent", params)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.mindex_url}{path}", params=params)
                if resp.status_code == 200:
                    return resp.json()
        except Exception as exc:
            logger.debug("%s failed: %s", path, exc)
        return {}

    async def _query_mindex(self, req: SearchRequest, result: SearchResult) -> bool:
        """Query mindex /unified-search/earth — returns True on success."""
        try:
            import httpx

            params: Dict[str, Any] = {
                "q": req.query,
                "limit": req.limit,
            }
            if req.types:
                params["types"] = req.types
            if req.lat is not None:
                params["lat"] = req.lat
                params["lng"] = req.lng
                params["radius"] = req.radius or 50
            if req.toxicity:
                params["toxicity"] = req.toxicity

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{self.mindex_url}/unified-search/earth",
                    params=params,
                )
                if resp.status_code != 200:
                    return False

                data = resp.json()
                result.domains_searched = data.get("domains_searched", [])
                result.results = data.get("results", {})
                result.total_count = data.get("total_count", 0)
                result.timing_ms = data.get("timing_ms", 0)

                for item in data.get("universal_results", []):
                    if not isinstance(item, dict):
                        continue
                    hit = SearchHit(
                        id=str(item.get("id", "")),
                        domain=item.get("domain", ""),
                        entity_type=item.get("entity_type", ""),
                        name=item.get("name", ""),
                        description=item.get("description"),
                        lat=item.get("lat"),
                        lng=item.get("lng"),
                        geometry_type=item.get("geometry_type"),
                        occurred_at=item.get("occurred_at"),
                        source=item.get("source"),
                        image_url=item.get("image_url"),
                        properties=item.get("properties", {}),
                    )
                    result.universal_results.append(hit)

                return True

        except Exception as exc:
            logger.debug("mindex /unified-search/earth failed: %s", exc)
            return False

    async def _query_mindex_legacy(self, req: SearchRequest, result: SearchResult) -> None:
        """Fallback: query mindex /unified-search (non-earth endpoint)."""
        try:
            import httpx

            params: Dict[str, Any] = {"q": req.query, "limit": req.limit}
            if req.types:
                params["types"] = req.types

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.mindex_url}/unified-search",
                    params=params,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    result.results = data.get("results", {})
                    result.total_count = data.get("total_count", 0)
                    result.domains_searched = data.get("domains_searched", [])

                    # Flatten into universal_results
                    for domain, items in result.results.items():
                        if not isinstance(items, list):
                            continue
                        group = DOMAIN_TO_GROUP.get(domain, domain)
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            hit = SearchHit(
                                id=str(item.get("id", uuid4())),
                                domain=item.get("domain", domain),
                                entity_type=item.get("entity_type", domain),
                                name=item.get("name", item.get("scientific_name", "")),
                                description=item.get("description"),
                                lat=item.get("lat"),
                                lng=item.get("lng"),
                                occurred_at=item.get("occurred_at"),
                                source=item.get("source"),
                                image_url=item.get("image_url"),
                                properties={
                                    k: v for k, v in item.items()
                                    if k not in {"id", "domain", "entity_type", "name",
                                                 "description", "lat", "lng", "occurred_at",
                                                 "source", "image_url"}
                                },
                            )
                            result.universal_results.append(hit)

        except Exception as exc:
            logger.debug("mindex /unified-search fallback failed: %s", exc)
