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
    # TODO: Check database connectivity
    return {
        "status": "ready",
        "service": "nlm"
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

