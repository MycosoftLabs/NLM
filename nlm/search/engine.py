"""
NLM Universal Search Engine
=============================

Parallel fan-out search across every Earth domain.  Accepts a free-text
query, resolves which domains and data sources are relevant, queries them
concurrently, normalises the results, and returns a unified response that
can be:

1. Returned directly via the API (for agents / Myca)
2. Rendered on the CREP map
3. Piped into the ingestion pipeline for mindex local storage
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from nlm.search.domains import DomainRegistry, SearchDomain
from nlm.search.sources import DataSource, DataSourceRegistry

logger = logging.getLogger(__name__)


# ======================================================================
# Request / Result models
# ======================================================================

@dataclass
class SearchRequest:
    """Inbound search query."""
    query: str
    domains: Optional[List[str]] = None       # restrict to these domain keys
    sources: Optional[List[str]] = None       # restrict to these source keys
    location: Optional[Dict[str, float]] = None  # lat, lon, radius_km
    time_range: Optional[Dict[str, str]] = None  # start, end (ISO)
    limit: int = 50
    offset: int = 0
    include_crep: bool = True                 # attach CREP layer hints
    include_mindex: bool = True               # queue for mindex ingestion


@dataclass
class SearchHit:
    """One result item."""
    id: str
    domain_key: str
    source_key: str
    title: str
    description: str = ""
    score: float = 1.0
    location: Optional[Dict[str, float]] = None  # lat, lon
    timestamp: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    crep_layer: Optional[str] = None
    geojson: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Aggregated search response."""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    total_hits: int = 0
    hits: List[SearchHit] = field(default_factory=list)
    domains_searched: List[str] = field(default_factory=list)
    sources_searched: List[str] = field(default_factory=list)
    crep_layers: List[str] = field(default_factory=list)
    timing_ms: float = 0
    errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "total_hits": self.total_hits,
            "hits": [
                {
                    "id": h.id,
                    "domain": h.domain_key,
                    "source": h.source_key,
                    "title": h.title,
                    "description": h.description,
                    "score": h.score,
                    "location": h.location,
                    "timestamp": h.timestamp,
                    "data": h.data,
                    "crep_layer": h.crep_layer,
                    "geojson": h.geojson,
                }
                for h in self.hits
            ],
            "domains_searched": self.domains_searched,
            "sources_searched": self.sources_searched,
            "crep_layers": self.crep_layers,
            "timing_ms": round(self.timing_ms, 2),
            "errors": self.errors,
        }


# ======================================================================
# Engine
# ======================================================================

class UniversalSearchEngine:
    """
    Parallel fan-out search across all Earth domains.

    Flow:
    1. Classify query → resolve matching domains
    2. Determine data sources for those domains
    3. Fan-out concurrent queries (async tasks)
    4. Merge, rank, and normalise results
    5. Attach CREP layer hints
    6. Optionally queue for mindex ingestion
    """

    def __init__(
        self,
        domain_registry: Optional[DomainRegistry] = None,
        source_registry: Optional[DataSourceRegistry] = None,
        mindex_url: Optional[str] = None,
    ) -> None:
        self.domains = domain_registry or DomainRegistry()
        self.sources = source_registry or DataSourceRegistry()
        self.mindex_url = mindex_url or "http://localhost:8003"

    async def search(self, request: SearchRequest) -> SearchResult:
        """Execute a universal search."""
        t0 = time.monotonic()
        result = SearchResult(query=request.query)

        # 1. Resolve domains
        matched_domains = self._resolve_domains(request)
        result.domains_searched = [d.key for d in matched_domains]

        # 2. Resolve sources
        source_keys = self._resolve_sources(matched_domains, request)
        result.sources_searched = list(source_keys)

        # 3. Fan-out search
        tasks = []
        for domain in matched_domains:
            for sk in domain.source_keys:
                if sk not in source_keys:
                    continue
                src = self.sources.get(sk)
                if src:
                    tasks.append(self._query_source(src, domain, request))

        # Always include local mindex
        tasks.append(self._query_mindex(request, matched_domains))

        # Run all concurrently
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. Merge results
        seen_ids: Set[str] = set()
        for tr in task_results:
            if isinstance(tr, Exception):
                result.errors.append({
                    "error": str(tr),
                    "type": type(tr).__name__,
                })
                continue
            if isinstance(tr, list):
                for hit in tr:
                    if hit.id not in seen_ids:
                        seen_ids.add(hit.id)
                        result.hits.append(hit)

        # 5. Sort by score
        result.hits.sort(key=lambda h: h.score, reverse=True)

        # 6. Apply limit/offset
        result.hits = result.hits[request.offset:request.offset + request.limit]
        result.total_hits = len(seen_ids)

        # 7. Collect CREP layers
        if request.include_crep:
            layers: Set[str] = set()
            for d in matched_domains:
                if d.crep_layer:
                    layers.add(d.crep_layer)
            for h in result.hits:
                if h.crep_layer:
                    layers.add(h.crep_layer)
            result.crep_layers = sorted(layers)

        result.timing_ms = (time.monotonic() - t0) * 1000
        return result

    # ------------------------------------------------------------------
    # Domain resolution
    # ------------------------------------------------------------------

    def _resolve_domains(self, req: SearchRequest) -> List[SearchDomain]:
        if req.domains:
            resolved = []
            for dk in req.domains:
                d = self.domains.get(dk)
                if d:
                    resolved.append(d)
            return resolved or self.domains.list_domains()

        # Auto-classify from query text
        matched = self.domains.search_domains(req.query)
        if matched:
            return matched

        # Fallback: all top-level domains
        roots = self.domains.list_roots()
        return [
            d for d in self.domains.list_domains()
            if d.key in roots
        ]

    def _resolve_sources(
        self, domains: List[SearchDomain], req: SearchRequest,
    ) -> Set[str]:
        if req.sources:
            return set(req.sources)
        keys: Set[str] = set()
        for d in domains:
            keys.update(d.source_keys)
        return keys

    # ------------------------------------------------------------------
    # Source querying (stub implementations — real connectors go here)
    # ------------------------------------------------------------------

    async def _query_source(
        self,
        source: DataSource,
        domain: SearchDomain,
        request: SearchRequest,
    ) -> List[SearchHit]:
        """
        Query an external data source.

        In production each source key maps to a dedicated connector class
        that knows the API specifics.  Here we use a generic httpx request
        with best-effort normalisation.
        """
        hits: List[SearchHit] = []
        try:
            import httpx

            params = self._build_params(source, domain, request)
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    source.base_url,
                    params=params,
                )
                if resp.status_code == 200:
                    data = resp.json() if source.response_format.value == "json" else {}
                    hits = self._normalise_response(
                        data, source, domain, request,
                    )
        except Exception as exc:
            logger.debug("Source %s query failed: %s", source.key, exc)
        return hits

    async def _query_mindex(
        self,
        request: SearchRequest,
        domains: List[SearchDomain],
    ) -> List[SearchHit]:
        """Query the local mindex database."""
        hits: List[SearchHit] = []
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self.mindex_url}/api/search/unified",
                    params={
                        "q": request.query,
                        "limit": request.limit,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("results", []):
                        if not isinstance(item, dict):
                            continue
                        hit = SearchHit(
                            id=str(item.get("id", uuid4())),
                            domain_key=item.get("domain", "mindex"),
                            source_key="mindex",
                            title=item.get("name", item.get("title", "")),
                            description=item.get("description", ""),
                            score=float(item.get("score", 0.8)),
                            location=item.get("location"),
                            timestamp=item.get("timestamp"),
                            data=item,
                            crep_layer=item.get("crep_layer"),
                        )
                        hits.append(hit)
        except Exception as exc:
            logger.debug("mindex query failed: %s", exc)
        return hits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_params(
        self, source: DataSource, domain: SearchDomain,
        request: SearchRequest,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"q": request.query, "limit": request.limit}
        if request.location and source.supports_geospatial:
            params["lat"] = request.location.get("lat")
            params["lon"] = request.location.get("lon")
            params["radius"] = request.location.get("radius_km", 50)
        if request.time_range and source.supports_temporal:
            params["start"] = request.time_range.get("start")
            params["end"] = request.time_range.get("end")
        return params

    def _normalise_response(
        self,
        data: Any,
        source: DataSource,
        domain: SearchDomain,
        request: SearchRequest,
    ) -> List[SearchHit]:
        hits: List[SearchHit] = []
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ("results", "data", "items", "records",
                        "features", "observations", "occurrences"):
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break

        for item in items[:request.limit]:
            if not isinstance(item, dict):
                continue

            title = (
                item.get("name")
                or item.get("title")
                or item.get("scientificName")
                or item.get("canonical_name")
                or item.get("label")
                or str(item.get("id", ""))
            )

            loc = None
            if "decimalLatitude" in item:
                loc = {
                    "lat": item["decimalLatitude"],
                    "lon": item.get("decimalLongitude"),
                }
            elif "latitude" in item:
                loc = {
                    "lat": item["latitude"],
                    "lon": item.get("longitude"),
                }
            elif "coordinates" in item:
                coords = item["coordinates"]
                if isinstance(coords, list) and len(coords) >= 2:
                    loc = {"lat": coords[1], "lon": coords[0]}

            geojson = None
            if "geometry" in item:
                geojson = {
                    "type": "Feature",
                    "geometry": item["geometry"],
                    "properties": {
                        "title": title,
                        "source": source.key,
                        "domain": domain.key,
                    },
                }

            hit = SearchHit(
                id=str(item.get("id", item.get("key", uuid4()))),
                domain_key=domain.key,
                source_key=source.key,
                title=title,
                description=item.get("description", item.get("summary", "")),
                score=float(item.get("score", item.get("confidence", 0.5))),
                location=loc,
                timestamp=item.get("timestamp", item.get("eventDate")),
                data=item,
                crep_layer=domain.crep_layer,
                geojson=geojson,
            )
            hits.append(hit)

        return hits
