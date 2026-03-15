"""
NLM Myca Query Interface — aligned with MINDEX v3
====================================================

Natural-language interface for Myca to query, explain, and visualise
anything about any Earth domain.  Myca calls into this module to:

1. Understand a user question and classify it to domains
2. Run a universal search via mindex /unified-search/earth
3. Synthesise an answer with context from NLM physics/biology/chemistry
4. Return map-ready data for CREP rendering
5. Optionally trigger mindex CREP sync for local storage

This is the bridge between the NLM intelligence layer and the Myca
conversational agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nlm.search.domains import DOMAIN_GROUPS, DOMAIN_TO_GROUP, DomainRegistry
from nlm.search.engine import SearchRequest, SearchResult, UniversalSearchEngine
from nlm.search.crep import CREPMapBridge

logger = logging.getLogger(__name__)


@dataclass
class MycaAnswer:
    """Structured answer that Myca returns to the user or agent."""
    query: str
    summary: str
    domains_used: List[str] = field(default_factory=list)
    search_result: Optional[Dict[str, Any]] = None
    crep_geojson: Optional[Dict[str, Any]] = None
    nlm_context: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "summary": self.summary,
            "domains_used": self.domains_used,
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
                                lat=40.7, lng=-74.0)
    """

    def __init__(
        self,
        engine: Optional[UniversalSearchEngine] = None,
        crep_bridge: Optional[CREPMapBridge] = None,
    ) -> None:
        self.engine = engine or UniversalSearchEngine()
        self.crep = crep_bridge or CREPMapBridge()

    async def ask(
        self,
        query: str,
        *,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        radius: Optional[float] = None,
        types: Optional[str] = None,
        limit: int = 50,
        include_map: bool = True,
    ) -> MycaAnswer:
        """
        Answer any Earth-related question.

        Args:
            query: Natural-language question
            lat/lng/radius: Optional spatial filter
            types: Comma-separated domain keys or group alias
            limit: Max results
            include_map: Generate CREP GeoJSON
        """
        # 1. Build search request (mindex-aligned params)
        request = SearchRequest(
            query=query,
            types=types,
            limit=limit,
            lat=lat,
            lng=lng,
            radius=radius,
        )

        # 2. Run universal search via mindex
        result = await self.engine.search(request)

        # 3. Build CREP map data
        crep_geojson = None
        if include_map:
            layers = self.crep.build_layers(result)
            crep_geojson = self.crep.to_geojson_collection(layers)

        # 4. Synthesise summary
        summary = self._synthesise_summary(query, result)

        # 5. Build NLM context (physics/biology enrichment)
        nlm_context = await self._get_nlm_context(query, result, lat, lng)

        # 6. Generate follow-up suggestions
        suggestions = self._generate_suggestions(query, result)

        return MycaAnswer(
            query=query,
            summary=summary,
            domains_used=result.domains_searched,
            search_result=result.to_dict(),
            crep_geojson=crep_geojson,
            nlm_context=nlm_context,
            suggestions=suggestions,
        )

    async def list_domains(self) -> List[Dict[str, Any]]:
        """List all searchable domains for Myca to present."""
        registry = self.engine.domains
        return [
            {
                "key": d.key,
                "label": d.label,
                "description": d.description,
                "group": d.group,
                "table": d.table,
                "crep_layer": d.crep_layer,
                "tags": sorted(d.tags),
            }
            for d in registry.list_domains()
        ]

    async def list_groups(self) -> List[str]:
        """List all domain group aliases."""
        return self.engine.domains.list_groups()

    async def crep_layer_config(self) -> List[Dict[str, Any]]:
        """Return all CREP layer definitions."""
        return self.crep.build_layer_config()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _synthesise_summary(
        self, query: str, result: SearchResult,
    ) -> str:
        if not result.universal_results:
            return (
                f"No results found for '{query}'. "
                f"Searched {len(result.domains_searched)} domains."
            )

        count = result.total_count
        top_domains = result.domains_searched[:5]
        domain_str = ", ".join(top_domains)

        top_names = [h.name for h in result.universal_results[:3] if h.name]
        names_str = ", ".join(top_names) if top_names else "various records"

        return (
            f"Found {count} results for '{query}' across "
            f"{domain_str}. Top matches: {names_str}. "
            f"Search completed in {result.timing_ms}ms."
        )

    async def _get_nlm_context(
        self,
        query: str,
        result: SearchResult,
        lat: Optional[float],
        lng: Optional[float],
    ) -> Dict[str, Any]:
        """Enrich with NLM physics/biology context when relevant."""
        context: Dict[str, Any] = {}

        # If location provided, get field physics context
        if lat is not None and lng is not None:
            try:
                from nlm.physics.field_physics import FieldPhysicsModel
                import time as _time

                fpm = FieldPhysicsModel()
                loc = (lat, lng, 0.0)
                ts = _time.time()
                context["atmospheric"] = fpm.get_atmospheric_conditions(loc, ts)
                context["geomagnetic"] = fpm.get_geomagnetic_field(loc, ts)
                context["lunar"] = fpm.get_lunar_gravitational_influence(loc, ts)
            except Exception:
                pass

        # Tag domain groups present
        groups = set()
        for d in result.domains_searched:
            g = DOMAIN_TO_GROUP.get(d)
            if g:
                groups.add(g)
        context["domain_groups"] = sorted(groups)

        return context

    def _generate_suggestions(
        self, query: str, result: SearchResult,
    ) -> List[str]:
        suggestions: List[str] = []
        groups = set()
        for d in result.domains_searched:
            g = DOMAIN_TO_GROUP.get(d)
            if g:
                groups.add(g)

        if "biological" in groups:
            suggestions.append("Show these species on the CREP map")
            suggestions.append("What compounds does this species produce?")
        if "earth_events" in groups:
            suggestions.append("Show historical trends for this area")
            suggestions.append("What species are affected by these conditions?")
        if "infrastructure" in groups:
            suggestions.append("What pollution sources are nearby?")
            suggestions.append("Show the power grid in this region")
        if "space" in groups:
            suggestions.append("Show current solar activity")
            suggestions.append("What satellites are overhead right now?")
        if "transport" in groups:
            suggestions.append("Show all vessels in this area")
            suggestions.append("What flights are overhead?")

        if not suggestions:
            suggestions.append("Show results on the CREP map")
            suggestions.append("Search nearby for related data")

        return suggestions[:4]
