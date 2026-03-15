"""
NLM Myca Query Interface
==========================

Natural-language interface for Myca to query, explain, and visualise
anything about any Earth domain.  Myca calls into this module to:

1. Understand a user question and classify it to domains
2. Run a universal search
3. Synthesise an answer with context from NLM physics/biology/chemistry
4. Return map-ready data for CREP rendering
5. Optionally trigger ingestion so the data is stored locally

This is the bridge between the NLM intelligence layer and the Myca
conversational agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nlm.search.domains import DomainRegistry
from nlm.search.engine import SearchRequest, SearchResult, UniversalSearchEngine
from nlm.search.crep import CREPMapBridge
from nlm.search.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


@dataclass
class MycaAnswer:
    """Structured answer that Myca returns to the user or agent."""
    query: str
    summary: str
    domains_used: List[str] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    search_result: Optional[Dict[str, Any]] = None
    crep_geojson: Optional[Dict[str, Any]] = None
    nlm_context: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "summary": self.summary,
            "domains_used": self.domains_used,
            "sources_used": self.sources_used,
            "search_result": self.search_result,
            "crep_geojson": self.crep_geojson,
            "nlm_context": self.nlm_context,
            "suggestions": self.suggestions,
        }


class MycaQueryInterface:
    """
    Myca's gateway into the NLM universal search system.

    Usage::

        myca = MycaQueryInterface()
        answer = await myca.ask("Show me all earthquakes near Tokyo in 2024")
        answer = await myca.ask("What birds are in Central Park right now?")
        answer = await myca.ask("Where are the nearest cell towers to me?",
                                location={"lat": 40.7, "lon": -74.0})
    """

    def __init__(
        self,
        engine: Optional[UniversalSearchEngine] = None,
        crep_bridge: Optional[CREPMapBridge] = None,
        pipeline: Optional[IngestionPipeline] = None,
    ) -> None:
        self.engine = engine or UniversalSearchEngine()
        self.crep = crep_bridge or CREPMapBridge()
        self.pipeline = pipeline or IngestionPipeline()

    async def ask(
        self,
        query: str,
        *,
        location: Optional[Dict[str, float]] = None,
        time_range: Optional[Dict[str, str]] = None,
        domains: Optional[List[str]] = None,
        limit: int = 50,
        include_map: bool = True,
        ingest: bool = False,
    ) -> MycaAnswer:
        """
        Answer any Earth-related question.

        Args:
            query: Natural-language question
            location: Optional lat/lon for spatial queries
            time_range: Optional start/end ISO strings
            domains: Restrict to specific domain keys
            limit: Max results
            include_map: Generate CREP GeoJSON
            ingest: Store results in mindex for future use
        """
        # 1. Build search request
        request = SearchRequest(
            query=query,
            domains=domains,
            location=location,
            time_range=time_range,
            limit=limit,
            include_crep=include_map,
        )

        # 2. Run universal search
        result = await self.engine.search(request)

        # 3. Build CREP map data
        crep_geojson = None
        if include_map:
            layers = self.crep.build_layers(result)
            crep_geojson = self.crep.to_geojson_collection(layers)

        # 4. Synthesise summary
        summary = self._synthesise_summary(query, result)

        # 5. Build NLM context (physics/biology enrichment)
        nlm_context = await self._get_nlm_context(query, result, location)

        # 6. Generate follow-up suggestions
        suggestions = self._generate_suggestions(query, result)

        answer = MycaAnswer(
            query=query,
            summary=summary,
            domains_used=result.domains_searched,
            sources_used=result.sources_searched,
            search_result=result.to_dict(),
            crep_geojson=crep_geojson,
            nlm_context=nlm_context,
            suggestions=suggestions,
        )

        # 7. Optionally ingest for local storage/training
        if ingest and result.hits:
            try:
                for domain_key in set(h.domain_key for h in result.hits):
                    await self.pipeline.ingest_domain(domain_key)
            except Exception as exc:
                logger.warning("Myca ingestion failed: %s", exc)

        return answer

    async def list_domains(self) -> List[Dict[str, Any]]:
        """List all searchable domains for Myca to present."""
        registry = self.engine.domains
        return [
            {
                "key": d.key,
                "label": d.label,
                "description": d.description,
                "parent": d.parent_key,
                "source_count": len(d.source_keys),
                "crep_layer": d.crep_layer,
                "tags": sorted(d.tags),
            }
            for d in registry.list_domains()
        ]

    async def list_sources(self) -> List[Dict[str, Any]]:
        """List all data sources for Myca to present."""
        registry = self.engine.sources
        return [
            {
                "key": s.key,
                "name": s.name,
                "description": s.description,
                "base_url": s.base_url,
                "supports_geospatial": s.supports_geospatial,
                "supports_temporal": s.supports_temporal,
                "tags": s.tags,
            }
            for s in registry.list_sources()
        ]

    async def crep_layer_config(self) -> List[Dict[str, Any]]:
        """Return all CREP layer definitions."""
        return self.crep.build_layer_config()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _synthesise_summary(
        self, query: str, result: SearchResult,
    ) -> str:
        if not result.hits:
            return (
                f"No results found for '{query}'. "
                f"Searched {len(result.domains_searched)} domains "
                f"and {len(result.sources_searched)} sources."
            )

        hit_count = result.total_hits
        top_domains = result.domains_searched[:5]
        domain_str = ", ".join(top_domains)

        top_titles = [h.title for h in result.hits[:3] if h.title]
        titles_str = ", ".join(top_titles) if top_titles else "various records"

        return (
            f"Found {hit_count} results for '{query}' across "
            f"{domain_str}. Top matches: {titles_str}. "
            f"Search completed in {result.timing_ms:.0f}ms."
        )

    async def _get_nlm_context(
        self,
        query: str,
        result: SearchResult,
        location: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Enrich with NLM physics/biology context when relevant."""
        context: Dict[str, Any] = {}

        # If location provided, get field physics context
        if location:
            try:
                from nlm.physics.field_physics import FieldPhysicsModel
                import time as _time

                fpm = FieldPhysicsModel()
                loc = (
                    location.get("lat", 0),
                    location.get("lon", 0),
                    location.get("alt", 0),
                )
                ts = _time.time()
                context["atmospheric"] = fpm.get_atmospheric_conditions(loc, ts)
                context["geomagnetic"] = fpm.get_geomagnetic_field(loc, ts)
                context["lunar"] = fpm.get_lunar_gravitational_influence(loc, ts)
            except Exception:
                pass

        # Tag domains present
        domain_roots = set(d.split(".")[0] for d in result.domains_searched)
        context["domain_roots"] = sorted(domain_roots)

        return context

    def _generate_suggestions(
        self, query: str, result: SearchResult,
    ) -> List[str]:
        suggestions: List[str] = []
        domain_roots = set(
            d.split(".")[0] for d in result.domains_searched
        )

        if "life" in domain_roots:
            suggestions.append("Show these species on the CREP map")
            suggestions.append("What compounds does this species produce?")
        if "environment" in domain_roots:
            suggestions.append("Show historical trends for this area")
            suggestions.append("What species are affected by these conditions?")
        if "infrastructure" in domain_roots:
            suggestions.append("What pollution sources are nearby?")
            suggestions.append("Show the power grid in this region")
        if "space" in domain_roots:
            suggestions.append("Show current solar activity")
            suggestions.append("What satellites are overhead right now?")
        if "transportation" in domain_roots:
            suggestions.append("Show all vessels in this area")
            suggestions.append("What flights are overhead?")

        if not suggestions:
            suggestions.append("Show results on the CREP map")
            suggestions.append("Ingest this data for offline access")

        return suggestions[:4]
