"""
NLM Mindex Ingestion Pipeline — aligned with MINDEX v3
========================================================

Orchestrates data ingestion from external sources into the local mindex
database.  Calls mindex's ETL sync endpoints rather than scraping directly:

- mindex /earth/crep/sync  — push domain data into crep.unified_entities
- mindex /earth/etl/sync   — trigger ETL connectors for a domain

The pipeline runs as a background scheduler.  Each domain has a sync tier:
  real-time (15min): earthquakes, wildfires, aircraft, solar_events
  hourly: weather, air_quality, vessels, storms
  daily: satellites, species, facilities, antennas, cameras, military
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nlm.search.domains import ALL_DOMAINS, DOMAIN_TABLES, DomainRegistry

logger = logging.getLogger(__name__)


# Sync tiers — how often each domain should be refreshed
SYNC_TIERS: Dict[str, List[str]] = {
    "realtime": [
        "earthquakes", "wildfires", "aircraft", "solar_events",
        "lightning", "storms",
    ],
    "hourly": [
        "weather", "air_quality", "vessels", "buoys",
        "stream_gauges", "tornadoes", "floods",
    ],
    "daily": [
        "satellites", "species", "taxa", "facilities", "antennas",
        "cameras", "military_installations", "airports", "ports",
        "spaceports", "launches", "wifi_hotspots", "internet_cables",
        "power_grid", "water_systems", "signal_measurements",
        "greenhouse_gas", "remote_sensing",
    ],
    "weekly": [
        "compounds", "genetics", "observations", "devices",
        "telemetry", "research", "crep_entities",
    ],
}


@dataclass
class SyncJob:
    """Tracks one ingestion/sync run."""
    domain: str
    tier: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    records_synced: int = 0
    errors: List[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | completed | failed


class IngestionPipeline:
    """
    Orchestrates data ingestion via mindex ETL endpoints.

    Usage::

        pipeline = IngestionPipeline(mindex_url="http://localhost:8003")
        await pipeline.sync_domain("earthquakes")
        await pipeline.sync_tier("realtime")
        await pipeline.sync_all()
    """

    def __init__(
        self,
        mindex_url: str = "http://localhost:8003",
        concurrency: int = 10,
    ) -> None:
        self.mindex_url = mindex_url.rstrip("/")
        self.concurrency = concurrency
        self._jobs: List[SyncJob] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def sync_domain(self, domain: str) -> SyncJob:
        """Sync a single domain via mindex ETL."""
        tier = self._get_tier(domain)
        return await self._run_sync(domain, tier)

    async def sync_tier(self, tier: str) -> List[SyncJob]:
        """Sync all domains in a tier (realtime, hourly, daily, weekly)."""
        domains = SYNC_TIERS.get(tier, [])
        return await self._run_parallel(domains, tier)

    async def sync_all(self) -> List[SyncJob]:
        """Sync every domain."""
        return await self._run_parallel(list(ALL_DOMAINS), "all")

    async def sync_crep(self, entity_type: str, limit: int = 1000) -> Dict[str, Any]:
        """Push domain data into crep.unified_entities via mindex."""
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
            logger.debug("CREP sync failed for %s: %s", entity_type, exc)
        return {"error": "mindex unavailable"}

    def get_jobs(self) -> List[Dict[str, Any]]:
        """Return summary of all sync jobs."""
        return [
            {
                "domain": j.domain,
                "tier": j.tier,
                "status": j.status,
                "records_synced": j.records_synced,
                "started_at": j.started_at,
                "finished_at": j.finished_at,
                "errors": j.errors,
            }
            for j in self._jobs
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_tier(self, domain: str) -> str:
        for tier, domains in SYNC_TIERS.items():
            if domain in domains:
                return tier
        return "daily"

    async def _run_parallel(self, domains: List[str], tier: str) -> List[SyncJob]:
        sem = asyncio.Semaphore(self.concurrency)
        jobs: List[SyncJob] = []

        async def _bounded(d: str) -> SyncJob:
            async with sem:
                return await self._run_sync(d, tier)

        results = await asyncio.gather(
            *[_bounded(d) for d in domains],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, SyncJob):
                jobs.append(r)
        return jobs

    async def _run_sync(self, domain: str, tier: str) -> SyncJob:
        job = SyncJob(
            domain=domain,
            tier=tier,
            started_at=datetime.now(timezone.utc).isoformat(),
            status="running",
        )
        self._jobs.append(job)

        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Call mindex ETL sync endpoint
                resp = await client.post(
                    f"{self.mindex_url}/earth/etl/sync",
                    params={"domain": domain},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    job.records_synced = data.get("records_synced", 0)
                    job.status = "completed"
                else:
                    job.status = "failed"
                    job.errors.append(f"mindex returned {resp.status_code}")

        except Exception as exc:
            job.status = "failed"
            job.errors.append(str(exc))
            logger.debug("ETL sync failed for %s: %s", domain, exc)
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()

        return job
