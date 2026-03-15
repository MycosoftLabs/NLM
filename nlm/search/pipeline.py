"""
NLM Mindex Ingestion Pipeline
===============================

Scrapes data from external sources and stores it in the local mindex
database for:

1. Low-latency search (no round-trip to external APIs)
2. NLM training data (offline model improvement)
3. CREP map pre-rendering (instant layer display)

The pipeline runs as a background scheduler.  Each source has an
ingestion adapter that knows how to fetch, normalise, and store records.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from nlm.search.domains import DomainRegistry
from nlm.search.sources import DataSource, DataSourceRegistry

logger = logging.getLogger(__name__)


@dataclass
class IngestionRecord:
    """One normalised record ready for mindex storage."""
    source_key: str
    domain_key: str
    external_id: str
    title: str
    description: str = ""
    location: Optional[Dict[str, float]] = None
    timestamp: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    geojson: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class IngestionJob:
    """Tracks one ingestion run."""
    source_key: str
    domain_keys: List[str]
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    records_fetched: int = 0
    records_stored: int = 0
    errors: List[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | completed | failed


class IngestionPipeline:
    """
    Fetches data from external sources and stores into mindex.

    Usage::

        pipeline = IngestionPipeline(mindex_url="http://localhost:8003")
        await pipeline.ingest_source("usgs_earthquake")
        await pipeline.ingest_domain("environment.earthquakes")
        await pipeline.ingest_all()
    """

    def __init__(
        self,
        domain_registry: Optional[DomainRegistry] = None,
        source_registry: Optional[DataSourceRegistry] = None,
        mindex_url: str = "http://localhost:8003",
        batch_size: int = 100,
    ) -> None:
        self.domains = domain_registry or DomainRegistry()
        self.sources = source_registry or DataSourceRegistry()
        self.mindex_url = mindex_url.rstrip("/")
        self.batch_size = batch_size
        self._jobs: List[IngestionJob] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_source(self, source_key: str) -> IngestionJob:
        """Ingest all data from a single source."""
        src = self.sources.get(source_key)
        if not src:
            return IngestionJob(
                source_key=source_key, domain_keys=[],
                status="failed", errors=[f"Unknown source: {source_key}"],
            )
        domain_keys = [d.key for d in self.domains.domains_for_source(source_key)]
        return await self._run_job(src, domain_keys)

    async def ingest_domain(self, domain_key: str) -> List[IngestionJob]:
        """Ingest all sources for a domain."""
        domain = self.domains.get(domain_key)
        if not domain:
            return []
        jobs = []
        for sk in domain.source_keys:
            src = self.sources.get(sk)
            if src:
                jobs.append(await self._run_job(src, [domain_key]))
        return jobs

    async def ingest_all(self, concurrency: int = 10) -> List[IngestionJob]:
        """Ingest from every registered source (rate-limited concurrency)."""
        sem = asyncio.Semaphore(concurrency)
        jobs: List[IngestionJob] = []

        async def _bounded(src: DataSource, dk: List[str]):
            async with sem:
                return await self._run_job(src, dk)

        tasks = []
        for src in self.sources.list_sources():
            dk = [d.key for d in self.domains.domains_for_source(src.key)]
            if dk:
                tasks.append(_bounded(src, dk))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, IngestionJob):
                jobs.append(r)
        return jobs

    def get_jobs(self) -> List[Dict[str, Any]]:
        """Return summary of all ingestion jobs."""
        return [
            {
                "source": j.source_key,
                "domains": j.domain_keys,
                "status": j.status,
                "records_fetched": j.records_fetched,
                "records_stored": j.records_stored,
                "started_at": j.started_at,
                "finished_at": j.finished_at,
                "errors": j.errors,
            }
            for j in self._jobs
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_job(
        self, source: DataSource, domain_keys: List[str],
    ) -> IngestionJob:
        job = IngestionJob(
            source_key=source.key,
            domain_keys=domain_keys,
            started_at=datetime.now(timezone.utc).isoformat(),
            status="running",
        )
        self._jobs.append(job)

        try:
            records = await self._fetch_records(source, domain_keys)
            job.records_fetched = len(records)

            stored = await self._store_records(records)
            job.records_stored = stored
            job.status = "completed"
        except Exception as exc:
            job.status = "failed"
            job.errors.append(str(exc))
            logger.error("Ingestion failed for %s: %s", source.key, exc)
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()

        return job

    async def _fetch_records(
        self, source: DataSource, domain_keys: List[str],
    ) -> List[IngestionRecord]:
        """Fetch records from an external source."""
        records: List[IngestionRecord] = []
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    source.base_url,
                    params={"limit": self.batch_size},
                )
                if resp.status_code != 200:
                    return records

                data = resp.json()
                items = []
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    for k in ("results", "data", "items", "features",
                              "records", "observations", "occurrences"):
                        if k in data and isinstance(data[k], list):
                            items = data[k]
                            break

                for item in items[:self.batch_size]:
                    if not isinstance(item, dict):
                        continue
                    rec = IngestionRecord(
                        source_key=source.key,
                        domain_key=domain_keys[0] if domain_keys else source.key,
                        external_id=str(item.get("id", item.get("key", ""))),
                        title=item.get("name", item.get("title", "")),
                        description=item.get("description", ""),
                        data=item,
                        tags=source.tags,
                    )

                    # Extract geospatial
                    if "decimalLatitude" in item:
                        rec.location = {
                            "lat": item["decimalLatitude"],
                            "lon": item.get("decimalLongitude"),
                        }
                    elif "latitude" in item:
                        rec.location = {
                            "lat": item["latitude"],
                            "lon": item.get("longitude"),
                        }
                    if "geometry" in item:
                        rec.geojson = {
                            "type": "Feature",
                            "geometry": item["geometry"],
                            "properties": {"source": source.key},
                        }

                    rec.timestamp = item.get("timestamp", item.get("eventDate"))
                    records.append(rec)

        except Exception as exc:
            logger.debug("Fetch from %s failed: %s", source.key, exc)

        return records

    async def _store_records(self, records: List[IngestionRecord]) -> int:
        """Store records into local mindex database."""
        if not records:
            return 0

        stored = 0
        try:
            import httpx

            payload = [
                {
                    "source_key": r.source_key,
                    "domain_key": r.domain_key,
                    "external_id": r.external_id,
                    "title": r.title,
                    "description": r.description,
                    "location": r.location,
                    "timestamp": r.timestamp,
                    "data": r.data,
                    "geojson": r.geojson,
                    "tags": r.tags,
                }
                for r in records
            ]

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Batch store
                for i in range(0, len(payload), self.batch_size):
                    batch = payload[i:i + self.batch_size]
                    resp = await client.post(
                        f"{self.mindex_url}/api/ingest/batch",
                        json=batch,
                    )
                    if resp.status_code in (200, 201):
                        result = resp.json()
                        stored += result.get("stored", len(batch))
                    else:
                        logger.warning(
                            "mindex batch store returned %d", resp.status_code,
                        )
        except Exception as exc:
            logger.debug("mindex store failed: %s", exc)

        return stored
