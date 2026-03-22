"""
NLM RootedNatureFrame — The Central Cognitive Object

Every piece of data the NLM processes enters as or becomes a RootedNatureFrame.
This replaces the flat NatureMessageFrame with a Merkle-rooted, multi-section
structure that captures:

- ground_truth: when, where, which devices
- observation: raw blobs, normalized values, fingerprints
- self_state: agent state, service health, safety mode
- world_state: environmental state, entity graph, alerts
- action_context: recent and intended actions
- uncertainty: per-sensor confidence, missingness, staleness
- provenance: chain of custody, content hash, source references

The frame_root is a Merkle hash committing to self_root, world_root,
event_root, and parent_frame_root — forming a tamper-evident cognitive
lineage chain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from nlm.core.fingerprints import SensoryFingerprint
from nlm.core.merkle import (
    GENESIS_ROOT,
    compute_frame_root,
    content_hash,
    sorted_field_root,
    merkle_root,
    sha256,
)


# ── Enums ────────────────────────────────────────────────────────────


class SafetyMode(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    LOCKDOWN = "lockdown"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# ── Supporting dataclasses ───────────────────────────────────────────


@dataclass
class GeoLocation:
    """Geographic position with datum and accuracy."""

    latitude: float = 0.0
    longitude: float = 0.0
    altitude_m: float = 0.0
    accuracy_m: float = 0.0
    datum: str = "WGS84"

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.latitude, self.longitude, self.altitude_m)


@dataclass
class PhysicalValue:
    """A calibrated physical measurement with SI unit."""

    value: float = 0.0
    unit: str = ""
    calibration_ref: str = ""  # reference to calibration record in MINDEX
    quality_flag: str = "ok"  # ok, out_of_range, frozen, drift


@dataclass
class SensorBlob:
    """Raw byte payload from a sensor with content-type metadata."""

    content_type: str = ""  # e.g., "application/octet-stream", "audio/wav"
    data: bytes = b""
    device_id: str = ""
    sensor_id: str = ""
    content_hash: bytes = b""

    def compute_hash(self) -> bytes:
        self.content_hash = sha256(self.data)
        return self.content_hash


@dataclass
class WaveformRef:
    """Reference to a stored waveform in MINDEX."""

    mindex_id: str = ""
    content_hash: bytes = b""
    sample_rate_hz: int = 0
    duration_seconds: float = 0.0
    channel_count: int = 1


@dataclass
class SpectralDecomposition:
    """Result of a spectral decomposition (FFT, wavelet, etc.)."""

    method: str = ""  # "fft", "wavelet", "stft"
    frequency_bins: List[float] = field(default_factory=list)
    power_values: List[float] = field(default_factory=list)
    window_seconds: float = 0.0


@dataclass
class ServiceStatus:
    """Health status of an internal service."""

    name: str = ""
    healthy: bool = True
    last_heartbeat: Optional[datetime] = None
    error: str = ""


@dataclass
class GraphSnapshotRef:
    """Reference to an entity graph snapshot in MINDEX."""

    mindex_id: str = ""
    node_count: int = 0
    edge_count: int = 0
    timestamp: Optional[datetime] = None


@dataclass
class ExternalFeedStatus:
    """Status of an external data feed."""

    feed_name: str = ""
    active: bool = True
    last_update: Optional[datetime] = None
    latency_ms: float = 0.0


@dataclass
class Alert:
    """An alert generated from observations or predictions."""

    alert_id: str = ""
    severity: AlertSeverity = AlertSeverity.INFO
    message: str = ""
    source: str = ""
    timestamp: Optional[datetime] = None


@dataclass
class Anomaly:
    """A detected anomaly in sensor data or predictions."""

    anomaly_id: str = ""
    anomaly_type: str = ""
    score: float = 0.0
    description: str = ""
    source_sensor: str = ""


@dataclass
class ActionRecord:
    """Record of a recently executed action."""

    action_id: str = ""
    action_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    timestamp: Optional[datetime] = None


@dataclass
class IntendedAction:
    """A planned but not-yet-executed action."""

    action_id: str = ""
    action_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.0
    deadline: Optional[datetime] = None


@dataclass
class InterventionMeta:
    """Metadata about an intervention event."""

    intervention_id: str = ""
    intervention_type: str = ""
    target_entity: str = ""
    expected_outcome: str = ""
    authorized_by: str = ""


@dataclass
class CustodyRecord:
    """Record in the chain of custody for data provenance."""

    actor: str = ""
    action: str = ""  # "created", "transformed", "forwarded", "stored"
    timestamp: Optional[datetime] = None
    content_hash: bytes = b""
    notes: str = ""


# ── Frame Sections ──────────────────────────────────────────────────


@dataclass
class GroundTruth:
    """When, where, and which devices produced this frame."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    monotonic_time: float = 0.0
    geolocation: GeoLocation = field(default_factory=GeoLocation)
    device_ids: List[str] = field(default_factory=list)
    sensor_ids: List[str] = field(default_factory=list)


@dataclass
class Observation:
    """All sensory data captured in this frame."""

    raw_blobs: List[SensorBlob] = field(default_factory=list)
    normalized_values: Dict[str, PhysicalValue] = field(default_factory=dict)
    spectral_decompositions: List[SpectralDecomposition] = field(default_factory=list)
    waveform_refs: List[WaveformRef] = field(default_factory=list)
    fingerprints: List[SensoryFingerprint] = field(default_factory=list)


@dataclass
class SelfState:
    """Internal state of the NLM system at frame time."""

    active_agents: List[str] = field(default_factory=list)
    service_health: Dict[str, ServiceStatus] = field(default_factory=dict)
    memory_indices: Dict[str, str] = field(default_factory=dict)  # MINDEX cursors
    safety_mode: SafetyMode = SafetyMode.NORMAL
    active_plans: List[str] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    embodiment_readiness: Dict[str, float] = field(default_factory=dict)


@dataclass
class WorldState:
    """External world state at frame time."""

    environmental_state: Dict[str, PhysicalValue] = field(default_factory=dict)
    entity_graph_snapshot: Optional[GraphSnapshotRef] = None
    external_feeds: List[ExternalFeedStatus] = field(default_factory=list)
    derived_fields: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Alert] = field(default_factory=list)
    anomalies: List[Anomaly] = field(default_factory=list)


@dataclass
class ActionContext:
    """Recent and intended actions."""

    recent_actions: List[ActionRecord] = field(default_factory=list)
    intended_actions: List[IntendedAction] = field(default_factory=list)
    intervention_metadata: Optional[InterventionMeta] = None


@dataclass
class Uncertainty:
    """Per-sensor uncertainty, missingness, and staleness."""

    missingness: Dict[str, bool] = field(default_factory=dict)
    sensor_confidence: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    freshness: Dict[str, float] = field(default_factory=dict)  # seconds since last reading
    staleness_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class Provenance:
    """Chain of custody and content integrity."""

    chain_of_custody: List[CustodyRecord] = field(default_factory=list)
    producer: str = ""
    content_hash: bytes = b""  # SHA-256 of serialized payload
    source_refs: List[str] = field(default_factory=list)  # MINDEX IDs


# ── RootedNatureFrame ───────────────────────────────────────────────


def _serialize_for_hash(obj: Any) -> bytes:
    """Deterministic serialization for hashing."""
    return json.dumps(obj, sort_keys=True, default=str).encode("utf-8")


@dataclass
class RootedNatureFrame:
    """
    The central cognitive object of NLM.

    Replaces NatureMessageFrame. Every observation, prediction, and
    action flows through this structure. The frame_root is a Merkle
    hash committing to all sections.
    """

    # ── Merkle identity ──
    frame_root: bytes = b""
    parent_frame_root: Optional[bytes] = None
    event_root: bytes = b""
    self_root: bytes = b""
    world_root: bytes = b""

    # ── Sections ──
    ground_truth: GroundTruth = field(default_factory=GroundTruth)
    observation: Observation = field(default_factory=Observation)
    self_state: SelfState = field(default_factory=SelfState)
    world_state: WorldState = field(default_factory=WorldState)
    action_context: ActionContext = field(default_factory=ActionContext)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    provenance: Provenance = field(default_factory=Provenance)

    def compute_roots(self) -> bytes:
        """
        Compute all Merkle roots from current state.

        Sets event_root, self_root, world_root, and frame_root.
        Returns the frame_root.
        """
        # Event root: hash of all observations
        obs_leaves = []
        for blob in self.observation.raw_blobs:
            h = blob.content_hash or blob.compute_hash()
            obs_leaves.append(h)
        for fp in self.observation.fingerprints:
            obs_leaves.append(sha256(fp.raw_hash or _serialize_for_hash(fp.fingerprint_id)))
        for key, pv in sorted(self.observation.normalized_values.items()):
            obs_leaves.append(sha256(_serialize_for_hash({key: pv.value, "unit": pv.unit})))
        self.event_root = merkle_root(obs_leaves) if obs_leaves else GENESIS_ROOT

        # Self root: hash of self_state fields
        self_fields = {
            "safety_mode": sha256(self.self_state.safety_mode.value.encode()),
            "active_agents": sha256(_serialize_for_hash(self.self_state.active_agents)),
            "active_plans": sha256(_serialize_for_hash(self.self_state.active_plans)),
            "available_tools": sha256(_serialize_for_hash(self.self_state.available_tools)),
        }
        self.self_root = sorted_field_root(self_fields)

        # World root: hash of world_state fields
        world_fields = {
            "environmental_state": sha256(_serialize_for_hash(
                {k: v.value for k, v in self.world_state.environmental_state.items()}
            )),
            "derived_fields": sha256(_serialize_for_hash(self.world_state.derived_fields)),
            "alerts": sha256(_serialize_for_hash([a.message for a in self.world_state.alerts])),
        }
        self.world_root = sorted_field_root(world_fields)

        # Frame root
        self.frame_root = compute_frame_root(
            self.self_root,
            self.world_root,
            self.event_root,
            self.parent_frame_root,
        )

        # Provenance content hash
        self.provenance.content_hash = content_hash(
            self.event_root + self.self_root + self.world_root
        )

        return self.frame_root

    def link_parent(self, parent: RootedNatureFrame) -> None:
        """Link this frame to a parent, forming the lineage chain."""
        self.parent_frame_root = parent.frame_root

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for JSON / MINDEX storage)."""
        return {
            "frame_root": self.frame_root.hex() if self.frame_root else "",
            "parent_frame_root": self.parent_frame_root.hex() if self.parent_frame_root else None,
            "event_root": self.event_root.hex() if self.event_root else "",
            "self_root": self.self_root.hex() if self.self_root else "",
            "world_root": self.world_root.hex() if self.world_root else "",
            "ground_truth": {
                "timestamp": self.ground_truth.timestamp.isoformat(),
                "monotonic_time": self.ground_truth.monotonic_time,
                "geolocation": {
                    "latitude": self.ground_truth.geolocation.latitude,
                    "longitude": self.ground_truth.geolocation.longitude,
                    "altitude_m": self.ground_truth.geolocation.altitude_m,
                    "accuracy_m": self.ground_truth.geolocation.accuracy_m,
                    "datum": self.ground_truth.geolocation.datum,
                },
                "device_ids": self.ground_truth.device_ids,
                "sensor_ids": self.ground_truth.sensor_ids,
            },
            "self_state": {
                "safety_mode": self.self_state.safety_mode.value,
                "active_agents": self.self_state.active_agents,
                "active_plans": self.self_state.active_plans,
                "available_tools": self.self_state.available_tools,
            },
            "world_state": {
                "environmental_state": {
                    k: {"value": v.value, "unit": v.unit}
                    for k, v in self.world_state.environmental_state.items()
                },
                "derived_fields": self.world_state.derived_fields,
                "alerts": [
                    {"severity": a.severity.value, "message": a.message}
                    for a in self.world_state.alerts
                ],
                "anomalies": [
                    {"type": a.anomaly_type, "score": a.score, "description": a.description}
                    for a in self.world_state.anomalies
                ],
            },
            "uncertainty": {
                "missingness": self.uncertainty.missingness,
                "sensor_confidence": self.uncertainty.sensor_confidence,
                "staleness_flags": self.uncertainty.staleness_flags,
            },
            "provenance": {
                "producer": self.provenance.producer,
                "content_hash": self.provenance.content_hash.hex() if self.provenance.content_hash else "",
                "source_refs": self.provenance.source_refs,
            },
        }
