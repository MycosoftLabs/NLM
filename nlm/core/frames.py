"""
RootedNatureFrame — The Irreducible Unit of NLM Cognition
=========================================================

The RootedNatureFrame is the canonical data object that the Nature
Learning Model operates on. It replaces RawNatureInput and
NatureMessageFrame as the model's internal cognitive format.

Every frame is:
- Geotagged and timestamped
- Merkle-rooted for integrity and replay
- Composed of complete self-state, world-state, and event-state
- Carrying raw sensory fingerprints in their native physical units

NatureMessageFrame (nlm/telemetry/nmf.py) remains the wire format.
RootedNatureFrame is what the model actually thinks in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from nlm.core.fingerprints import (
    AcousticFingerprint,
    BioelectricFingerprint,
    ChemicalFingerprint,
    MechanicalFingerprint,
    SpectralFingerprint,
    ThermalFingerprint,
)


# ---------------------------------------------------------------------------
# Sub-state dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """What was sensed — raw measured reality."""

    raw_sensor_blobs: Dict[str, bytes] = field(default_factory=dict)  # sensor_id → raw bytes
    normalized_physical: Dict[str, float] = field(default_factory=dict)  # calibrated SI values
    spectral: List[SpectralFingerprint] = field(default_factory=list)
    acoustic: List[AcousticFingerprint] = field(default_factory=list)
    bioelectric: List[BioelectricFingerprint] = field(default_factory=list)
    thermal: List[ThermalFingerprint] = field(default_factory=list)
    chemical: List[ChemicalFingerprint] = field(default_factory=list)
    mechanical: List[MechanicalFingerprint] = field(default_factory=list)
    waveform_refs: List[str] = field(default_factory=list)  # URIs to raw data
    bio_tokens: List[str] = field(default_factory=list)  # derived via TranslationLayer

    def fingerprint_count(self) -> int:
        return (
            len(self.spectral) + len(self.acoustic) + len(self.bioelectric)
            + len(self.thermal) + len(self.chemical) + len(self.mechanical)
        )


@dataclass
class SelfState:
    """Complete internal state of MYCA/MAS."""

    active_agents: List[str] = field(default_factory=list)
    service_health: Dict[str, str] = field(default_factory=dict)  # service → "healthy"/"degraded"/"down"
    memory_indices: Dict[str, str] = field(default_factory=dict)  # index_name → status
    safety_mode: str = "normal"  # "normal", "cautious", "lockdown"
    active_plans: List[str] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    embodiment_readiness: Dict[str, float] = field(default_factory=dict)  # device → 0-1 readiness
    resource_levels: Dict[str, float] = field(default_factory=dict)  # resource → level

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_agents": self.active_agents,
            "service_health": self.service_health,
            "memory_indices": self.memory_indices,
            "safety_mode": self.safety_mode,
            "active_plans": self.active_plans,
            "available_tools": self.available_tools,
            "embodiment_readiness": self.embodiment_readiness,
            "resource_levels": self.resource_levels,
        }


@dataclass
class WorldState:
    """Complete external world state."""

    environmental: Dict[str, float] = field(default_factory=dict)  # temp, humidity, CO2, pressure, wind, AQI
    site_device_graph: Dict[str, Any] = field(default_factory=dict)  # entity-relation snapshot
    external_feeds: Dict[str, Any] = field(default_factory=dict)  # latest feed summaries
    derived_fields: Dict[str, Any] = field(default_factory=dict)  # physics-computed (geomagnetic, lunar, atmospheric)
    alerts_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    weather: Dict[str, Any] = field(default_factory=dict)
    ecological_context: Dict[str, Any] = field(default_factory=dict)  # species, symbiosis, growth stage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environmental": self.environmental,
            "site_device_graph": self.site_device_graph,
            "external_feeds": self.external_feeds,
            "derived_fields": self.derived_fields,
            "alerts_anomalies": self.alerts_anomalies,
            "weather": self.weather,
            "ecological_context": self.ecological_context,
        }


@dataclass
class ActionContext:
    """Recent and intended actions — enables active inference."""

    recent_actions: List[Dict[str, Any]] = field(default_factory=list)
    intended_actions: List[Dict[str, Any]] = field(default_factory=list)
    intervention_metadata: Dict[str, Any] = field(default_factory=dict)
    pending_decisions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recent_actions": self.recent_actions,
            "intended_actions": self.intended_actions,
            "intervention_metadata": self.intervention_metadata,
            "pending_decisions": self.pending_decisions,
        }


@dataclass
class Uncertainty:
    """What we don't know — explicit uncertainty tracking."""

    missingness: Dict[str, bool] = field(default_factory=dict)  # field → is_missing
    sensor_confidence: Dict[str, float] = field(default_factory=dict)  # sensor_id → 0-1
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # field → (lo, hi)
    freshness_seconds: Dict[str, float] = field(default_factory=dict)  # field → age in seconds
    overall_confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "missingness": self.missingness,
            "sensor_confidence": self.sensor_confidence,
            "confidence_intervals": {k: list(v) for k, v in self.confidence_intervals.items()},
            "freshness_seconds": self.freshness_seconds,
            "overall_confidence": self.overall_confidence,
        }


@dataclass
class Provenance:
    """Chain of custody and source attribution."""

    chain_of_custody: List[str] = field(default_factory=list)
    producer: str = ""
    content_hash: str = ""
    source_refs: List[str] = field(default_factory=list)
    ingestion_path: str = ""  # e.g. "device→natureos→mindex→nlm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_of_custody": self.chain_of_custody,
            "producer": self.producer,
            "content_hash": self.content_hash,
            "source_refs": self.source_refs,
            "ingestion_path": self.ingestion_path,
        }


# ---------------------------------------------------------------------------
# The Core Object
# ---------------------------------------------------------------------------


@dataclass
class RootedNatureFrame:
    """The irreducible unit of NLM cognition.

    Every frame is Merkle-rooted:
        frame_root = merkle(self_root || world_root || event_root || parent_frame_root)

    The model consumes the decoded structured state.
    MINDEX stores the roots, lineage, and provenance.
    """

    # --- Merkle identity ---
    frame_root: str = ""
    parent_frame_root: str = ""
    event_root: str = ""
    self_root: str = ""
    world_root: str = ""

    # --- Ground truth anchor ---
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    monotonic_ns: int = field(default_factory=time.monotonic_ns)
    geolocation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # lat, lon, alt
    device_ids: List[str] = field(default_factory=list)

    # --- Composed state ---
    observation: Observation = field(default_factory=Observation)
    self_state: SelfState = field(default_factory=SelfState)
    world_state: WorldState = field(default_factory=WorldState)
    action_context: ActionContext = field(default_factory=ActionContext)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_root": self.frame_root,
            "parent_frame_root": self.parent_frame_root,
            "event_root": self.event_root,
            "self_root": self.self_root,
            "world_root": self.world_root,
            "timestamp": self.timestamp.isoformat(),
            "monotonic_ns": self.monotonic_ns,
            "geolocation": list(self.geolocation),
            "device_ids": self.device_ids,
            "observation": {
                "normalized_physical": self.observation.normalized_physical,
                "bio_tokens": self.observation.bio_tokens,
                "fingerprint_count": self.observation.fingerprint_count(),
                "waveform_refs": self.observation.waveform_refs,
            },
            "self_state": self.self_state.to_dict(),
            "world_state": self.world_state.to_dict(),
            "action_context": self.action_context.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "provenance": self.provenance.to_dict(),
        }
