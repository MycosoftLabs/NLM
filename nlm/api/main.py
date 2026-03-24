"""
NLM FastAPI Application
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NLM API",
    description="Nature Learning Model API",
    version="0.1.0"
)

# CORS middleware: wildcard allowed only in development.
env = os.getenv("NLM_ENV", os.getenv("ENV", "development")).lower()
cors_origins = os.getenv("NLM_CORS_ORIGINS", "")
allow_origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
if not allow_origins:
    allow_origins = ["*"] if env in {"dev", "development", "local"} else ["https://mycosoft.com", "https://sandbox.mycosoft.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "NLM API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "nlm"
    }


@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    return {
        "status": "ready",
        "service": "nlm"
    }

@app.get("/api/training/status")
async def training_status():
    """
    Exposes live PyTorch training metrics so the dashboard can render real-time charts.
    """
    import json
    from pathlib import Path
    
    metrics_file = Path(__file__).resolve().parent.parent / "telemetry" / "training_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                return data
        except Exception as e:
            return {"status": "degraded", "error": str(e)}
    
    # Provide a fallback starting state before train.py kicks off
    return {
        "latest": {
            "epoch": 0,
            "loss": 4.5,
            "accuracy": 0.0,
            "learning_rate": 0.001,
            "throughput": 0,
            "signal_samples": 3100000,
            "overall_progress": 0,
            "status": "waiting"
        },
        "history": []
    }


class EnvironmentalProcessRequest(BaseModel):
    """Request body for environmental data processing."""
    temperature: float
    humidity: float
    co2: Optional[float] = None
    pressure: Optional[float] = None
    timestamp: Optional[str] = None
    location: Optional[Dict[str, float]] = None


class PredictionRequest(BaseModel):
    """Request body for prediction generation."""
    entity_type: str = "generic"
    entity_id: str = "default"
    time_horizon: str = "30d"
    conditions: Optional[Dict[str, Any]] = None


class RecommendationRequest(BaseModel):
    """Request body for recommendations."""
    scenario: str = "optimal_growth_conditions"
    constraints: Optional[Dict[str, Any]] = None


@app.post("/api/environmental/process")
async def process_environmental(req: EnvironmentalProcessRequest):
    """
    Process environmental data through NLM.
    Returns insights and predictions based on temperature, humidity, and other factors.
    """
    from nlm.client import NLMClient

    client = NLMClient()
    ts = None
    if req.timestamp:
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(req.timestamp.replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)
    result = await client.process_environmental_data(
        temperature=req.temperature,
        humidity=req.humidity,
        co2=req.co2,
        pressure=req.pressure,
        location=req.location,
        timestamp=ts,
    )
    return result


@app.post("/api/predict")
async def predict(req: PredictionRequest):
    """Generate a prediction for entity growth, fruiting, or environmental trend."""
    from nlm.client import NLMClient

    client = NLMClient()
    result = await client.predict(
        entity_type=req.entity_type,
        entity_id=req.entity_id,
        time_horizon=req.time_horizon,
        conditions=req.conditions,
    )
    return result


@app.post("/api/recommend")
async def recommend(req: RecommendationRequest):
    """Get recommendations for cultivation or environmental optimization."""
    from nlm.client import NLMClient

    client = NLMClient()
    result = await client.recommend(
        scenario=req.scenario,
        constraints=req.constraints,
    )
    return result


class VerifiedTelemetrySample(BaseModel):
    device_slug: str = Field(..., min_length=1, max_length=200)
    stream_key: str = Field(..., min_length=1, max_length=200)
    recorded_at: datetime
    value_numeric: Optional[float] = None
    value_json: Optional[Dict[str, Any]] = None
    value_unit: Optional[str] = None
    verified: bool = True
    envelope_seq: Optional[int] = None
    envelope_msg_id: Optional[str] = None


@app.post("/api/telemetry/ingest-verified")
async def ingest_verified(samples: List[VerifiedTelemetrySample]):
    """
    Accept verified telemetry samples for learning pipelines.

    Note: This endpoint currently validates and acknowledges receipt; storage and
    training orchestration is handled by downstream NLM jobs.
    """
    if not samples:
        raise HTTPException(status_code=400, detail="no_samples")
    verified_count = sum(1 for s in samples if s.verified)
    logger.info("Received %d telemetry samples (%d verified)", len(samples), verified_count)
    return {
        "success": True,
        "received": len(samples),
        "verified": verified_count,
        "received_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# NLM Full Endpoints - Translate, NMF, Tokens, Predict/Fruiting, Query/Knowledge
# Extended: February 17, 2026
# =============================================================================


class TranslateRequest(BaseModel):
    """Request body for translation layer (raw -> NMF)."""
    raw: Dict[str, Any] = Field(..., description="Raw sensor/environmental data")
    envelopes: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional telemetry envelopes")
    source_id: str = Field(default="", description="Source identifier")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context")


class NMFCreateRequest(BaseModel):
    """Request body for creating a Nature Message Frame."""
    raw: Dict[str, Any] = Field(..., description="Raw sensor/environmental data")
    envelopes: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional telemetry envelopes")
    source_id: str = Field(default="", description="Source identifier")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context")


class FruitingPredictRequest(BaseModel):
    """Request body for fruiting prediction."""
    entity_id: str = Field(default="generic", description="Species or entity identifier")
    time_horizon: str = Field(default="30d", description="Prediction horizon (e.g. 7d, 30d)")
    conditions: Optional[Dict[str, Any]] = Field(default=None, description="Environmental conditions")
    location: Optional[Dict[str, float]] = Field(default=None, description="Lat/lon/alt")


class KnowledgeQueryRequest(BaseModel):
    """Request body for knowledge graph query."""
    query: str = Field(..., min_length=1, description="Query string")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context")


@app.post("/api/translate")
async def api_translate(req: TranslateRequest):
    """
    Translate raw environmental data through the NLM translation layer.

    Raw -> Normalized -> Bio-Tokens -> Nature Message Frame (NMF).
    Returns the full NMF as JSON.
    """
    from nlm.telemetry.translation_layer import translate as translate_layer

    nmf = translate_layer(
        raw=req.raw,
        envelopes=req.envelopes,
        source_id=req.source_id,
        context=req.context,
    )
    return nmf.to_dict()


@app.post("/api/nmf/create")
async def api_nmf_create(req: NMFCreateRequest):
    """
    Create a Nature Message Frame from raw data.

    Builds NMF with bio-tokens, structured output, and optional envelopes.
    """
    from nlm.telemetry.translation_layer import build_nmf

    nmf = build_nmf(
        raw=req.raw,
        envelopes=req.envelopes,
        source_id=req.source_id,
        context=req.context,
    )
    return nmf.to_dict()


@app.get("/api/tokens/vocabulary")
async def api_tokens_vocabulary():
    """
    Get the Bio-Token vocabulary (micro-speak codes and semantic labels).
    """
    from nlm.telemetry.bio_tokens import BIO_TOKEN_VOCABULARY, all_tokens, all_semantics

    return {
        "vocabulary": BIO_TOKEN_VOCABULARY,
        "token_codes": all_tokens(),
        "semantic_labels": list(all_semantics()),
        "count": len(BIO_TOKEN_VOCABULARY),
    }


@app.post("/api/predict/fruiting")
async def api_predict_fruiting(req: FruitingPredictRequest):
    """
    Generate fruiting probability prediction.

    Uses NLM physics and environmental models for fungal fruiting conditions.
    """
    from nlm.client import NLMClient

    client = NLMClient()
    conditions = req.conditions or {}
    if req.location:
        conditions["location"] = req.location
    result = await client.predict(
        entity_type="fruiting_conditions",
        entity_id=req.entity_id,
        time_horizon=req.time_horizon,
        conditions=conditions,
    )
    return result


@app.post("/api/query/knowledge")
async def api_query_knowledge(req: KnowledgeQueryRequest):
    """
    Query the knowledge graph via MINDEX.

    Returns entities and relations matching the query.
    """
    from nlm.client import NLMClient

    client = NLMClient()
    result = await client.query_knowledge_graph(
        query=req.query,
        context=req.context,
        limit=req.limit,
    )
    return result


# =============================================================================
# Universal Earth Search — aligned with MINDEX v3
# All 35 domains: species, earth events, atmosphere, water, infrastructure,
# signals, transport, space, monitoring, military, telemetry, knowledge
# =============================================================================


class UnifiedSearchRequest(BaseModel):
    """Request body for universal Earth search — mirrors mindex /unified-search/earth."""
    query: str = Field(..., min_length=1, description="Free-text search query")
    types: Optional[str] = Field(default=None, description="Comma-separated domain keys or group alias (e.g. 'earthquakes,volcanoes' or 'earth_events')")
    limit: int = Field(default=50, ge=1, le=500, description="Max results")
    lat: Optional[float] = Field(default=None, description="Latitude for spatial search")
    lng: Optional[float] = Field(default=None, description="Longitude for spatial search")
    radius: Optional[float] = Field(default=None, description="Search radius in km")
    toxicity: Optional[str] = Field(default=None, description="Fungi filter: poisonous, edible, psychedelic")


class MycaAskRequest(BaseModel):
    """Request body for Myca natural-language queries."""
    query: str = Field(..., min_length=1, description="Natural-language question")
    lat: Optional[float] = Field(default=None, description="Latitude for spatial context")
    lng: Optional[float] = Field(default=None, description="Longitude for spatial context")
    radius: Optional[float] = Field(default=None, description="Search radius in km")
    types: Optional[str] = Field(default=None, description="Restrict to domain keys or group alias")
    limit: int = Field(default=50, ge=1, le=500)
    include_map: bool = Field(default=True, description="Generate CREP map data")


class SyncRequest(BaseModel):
    """Request body for triggering mindex ETL sync."""
    domain: Optional[str] = Field(default=None, description="Sync a single domain")
    tier: Optional[str] = Field(default=None, description="Sync all domains in a tier (realtime, hourly, daily, weekly)")
    all_domains: bool = Field(default=False, description="Sync every domain")


@app.get("/api/search/earth")
async def api_search_earth(
    q: str,
    types: Optional[str] = None,
    limit: int = 50,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    radius: Optional[float] = None,
    toxicity: Optional[str] = None,
):
    """
    Universal Earth Search — mirrors mindex /unified-search/earth.

    Searches across ALL 35 domains in parallel via mindex: every species,
    every earth event, every infrastructure element, every signal, every
    satellite, every vessel, every flight — everything on Earth.
    """
    from nlm.search.engine import UniversalSearchEngine, SearchRequest

    engine = UniversalSearchEngine()
    search_req = SearchRequest(
        query=q,
        types=types,
        limit=limit,
        lat=lat,
        lng=lng,
        radius=radius,
        toxicity=toxicity,
    )
    result = await engine.search(search_req)
    return result.to_dict()


@app.post("/api/search/earth")
async def api_search_earth_post(req: UnifiedSearchRequest):
    """Universal Earth Search (POST variant)."""
    from nlm.search.engine import UniversalSearchEngine, SearchRequest

    engine = UniversalSearchEngine()
    search_req = SearchRequest(
        query=req.query,
        types=req.types,
        limit=req.limit,
        lat=req.lat,
        lng=req.lng,
        radius=req.radius,
        toxicity=req.toxicity,
    )
    result = await engine.search(search_req)
    return result.to_dict()


@app.post("/api/myca/ask")
async def api_myca_ask(req: MycaAskRequest):
    """
    Myca AI query endpoint.

    Natural-language interface for Myca to ask anything about Earth.
    Returns a synthesised answer with search results, NLM context
    (physics, biology, chemistry), CREP map data, and follow-up
    suggestions.
    """
    from nlm.search.myca import MycaQueryInterface

    myca = MycaQueryInterface()
    answer = await myca.ask(
        query=req.query,
        lat=req.lat,
        lng=req.lng,
        radius=req.radius,
        types=req.types,
        limit=req.limit,
        include_map=req.include_map,
    )
    return answer.to_dict()


@app.get("/api/search/domains")
async def api_search_domains(group: Optional[str] = None):
    """
    List all 35 searchable Earth domains — mirrors mindex domain taxonomy.

    Optionally filter by group alias (biological, earth_events, atmosphere,
    water, infrastructure, signals, transport, space, monitoring, military,
    telemetry).
    """
    from nlm.search.domains import DomainRegistry, DOMAIN_GROUPS

    registry = DomainRegistry()
    if group and group in DOMAIN_GROUPS:
        keys = DOMAIN_GROUPS[group]
        domains = [registry.get(k) for k in keys if registry.get(k)]
    else:
        domains = registry.list_domains()

    return {
        "domains": [
            {
                "key": d.key,
                "label": d.label,
                "description": d.description,
                "group": d.group,
                "table": d.table,
                "crep_layer": d.crep_layer,
                "tags": sorted(d.tags),
            }
            for d in domains
        ],
        "total": len(domains),
        "groups": registry.list_groups(),
    }


@app.get("/api/search/sources")
async def api_search_sources(query: Optional[str] = None):
    """
    List all external data sources NLM can ingest from.

    Includes biodiversity databases, weather services, satellite feeds,
    seismic networks, air quality monitors, ocean buoys, ship trackers,
    flight trackers, power plant registries, and more.
    """
    from nlm.search.sources import DataSourceRegistry

    registry = DataSourceRegistry()
    sources = registry.search(query) if query else registry.list_sources()
    return {
        "sources": [
            {
                "key": s.key,
                "name": s.name,
                "description": s.description,
                "base_url": s.base_url,
                "auth_method": s.auth_method.value,
                "supports_geospatial": s.supports_geospatial,
                "supports_temporal": s.supports_temporal,
                "tags": s.tags,
            }
            for s in sources
        ],
        "total": len(sources),
    }


@app.get("/api/crep/layers")
async def api_crep_layers():
    """
    List all 16 CREP map layer definitions — mirrors mindex earth.py layers.

    Layers: earthquakes, volcanoes, wildfires, facilities, antennas, aircraft,
    vessels, airports, ports, cameras, military, buoys, weather, air_quality,
    wifi_hotspots, species.
    """
    from nlm.search.crep import CREPMapBridge

    bridge = CREPMapBridge()
    return {"layers": bridge.build_layer_config()}


@app.get("/api/earth/stats")
async def api_earth_stats():
    """Get entity counts per domain from mindex."""
    from nlm.search.engine import UniversalSearchEngine

    engine = UniversalSearchEngine()
    return await engine.get_earth_stats()


@app.post("/api/sync")
async def api_sync(req: SyncRequest):
    """
    Trigger mindex ETL sync.

    Sync a single domain, a tier (realtime/hourly/daily/weekly), or all domains.
    """
    from nlm.search.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()

    if req.all_domains:
        jobs = await pipeline.sync_all()
        return {
            "status": "completed",
            "jobs": pipeline.get_jobs(),
            "total_jobs": len(jobs),
        }
    elif req.tier:
        jobs = await pipeline.sync_tier(req.tier)
        return {
            "status": "completed",
            "jobs": pipeline.get_jobs(),
            "total_jobs": len(jobs),
        }
    elif req.domain:
        job = await pipeline.sync_domain(req.domain)
        return {
            "status": job.status,
            "job": {
                "domain": job.domain,
                "tier": job.tier,
                "records_synced": job.records_synced,
                "errors": job.errors,
            },
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide domain, tier, or set all_domains=true",
        )

