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

