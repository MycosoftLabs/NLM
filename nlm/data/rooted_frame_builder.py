"""
NLM RootedFrameBuilder

Assembles complete RootedNatureFrames from:
- Raw observations
- Preconditioned derived fields
- Extracted fingerprints
- Self-state snapshot
- World-state snapshot
- Action context

This is the main entry point for the data pipeline:
  raw -> precondition -> extract fingerprints -> assemble frame -> compute roots
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nlm.core.fingerprints import SensoryFingerprint
from nlm.core.frames import (
    ActionContext,
    ActionRecord,
    CustodyRecord,
    GeoLocation,
    GroundTruth,
    IntendedAction,
    Observation,
    PhysicalValue,
    Provenance,
    RootedNatureFrame,
    SafetyMode,
    SelfState,
    SensorBlob,
    Uncertainty,
    WorldState,
)
from nlm.core.merkle import sha256
from nlm.core.protocols import SignalEnvelope, get_adapter
from nlm.data.fingerprint_extraction import FingerprintExtractor
from nlm.data.preconditioner import DeterministicPreconditioningStack

logger = logging.getLogger(__name__)


class RootedFrameBuilder:
    """
    Builds complete RootedNatureFrames from raw inputs.

    Usage:
        builder = RootedFrameBuilder()
        frame = builder.build(
            raw_data={"temperature_c": 22.5, "humidity_pct": 75},
            lat=37.77, lon=-122.42, alt_m=10,
            device_id="fci-001",
        )
    """

    def __init__(self):
        self.preconditioning_stack = DeterministicPreconditioningStack()
        self.fingerprint_extractor = FingerprintExtractor()

    def build(
        self,
        raw_data: Dict[str, Any],
        *,
        lat: float = 0.0,
        lon: float = 0.0,
        alt_m: float = 0.0,
        timestamp: Optional[float] = None,
        device_id: str = "",
        sensor_id: str = "",
        protocol: str = "generic",
        envelopes: Optional[List[Dict[str, Any]]] = None,
        parent_frame: Optional[RootedNatureFrame] = None,
        self_state: Optional[SelfState] = None,
        action_context: Optional[ActionContext] = None,
        producer: str = "nlm",
    ) -> RootedNatureFrame:
        """
        Full pipeline: raw -> precondition -> fingerprint -> assemble -> root.

        Returns a complete RootedNatureFrame with all Merkle roots computed.
        """
        ts_unix = timestamp or time.time()
        ts_dt = datetime.fromtimestamp(ts_unix, tz=timezone.utc)

        # ── 1. Protocol normalization ──
        adapter = get_adapter(protocol)
        envelope = adapter.normalize(raw_data, device_slug=device_id)

        # ── 2. Normalize raw values to SI ──
        normalized_values: Dict[str, PhysicalValue] = {}
        for key, val in envelope.normalized_values.items():
            unit = envelope.units.get(key, "")
            normalized_values[key] = PhysicalValue(value=val, unit=unit)

        # ── 3. Deterministic preconditioning ──
        derived = self.preconditioning_stack.precondition(
            raw_data, lat=lat, lon=lon, alt_m=alt_m, timestamp=ts_unix,
        )

        # ── 4. Fingerprint extraction ──
        # Combine raw + derived for fingerprint extraction
        all_numeric: Dict[str, float] = {}
        for k, v in raw_data.items():
            try:
                all_numeric[k] = float(v)
            except (TypeError, ValueError):
                pass
        for k, pv in derived.items():
            all_numeric[k] = pv.value

        raw_bytes = str(sorted(raw_data.items())).encode("utf-8")
        fingerprints = self.fingerprint_extractor.extract_all(
            normalized=all_numeric,
            raw_bytes=raw_bytes,
            device_id=device_id,
            sensor_id=sensor_id,
            timestamp=ts_dt,
        )

        # ── 5. Build uncertainty from data completeness ──
        expected_sensors = ["temperature", "humidity", "pressure", "co2", "ph", "light"]
        missingness: Dict[str, bool] = {}
        sensor_confidence: Dict[str, float] = {}
        for sensor in expected_sensors:
            present = any(sensor in k.lower() for k in raw_data)
            missingness[sensor] = not present
            sensor_confidence[sensor] = 0.9 if present else 0.0

        uncertainty = Uncertainty(
            missingness=missingness,
            sensor_confidence=sensor_confidence,
            freshness={sensor: 0.0 for sensor in expected_sensors if not missingness[sensor]},
            staleness_flags={sensor: False for sensor in expected_sensors},
        )

        # ── 6. Build world_state from derived fields ──
        world_state = WorldState(
            environmental_state=normalized_values,
            derived_fields={k: v.value for k, v in derived.items()},
        )

        # ── 7. Assemble observation ──
        raw_blob = SensorBlob(
            content_type="application/json",
            data=raw_bytes,
            device_id=device_id,
            sensor_id=sensor_id,
        )
        raw_blob.compute_hash()

        observation = Observation(
            raw_blobs=[raw_blob],
            normalized_values=normalized_values,
            fingerprints=fingerprints,
        )

        # ── 8. Provenance ──
        provenance = Provenance(
            producer=producer,
            chain_of_custody=[
                CustodyRecord(
                    actor=producer,
                    action="created",
                    timestamp=ts_dt,
                    content_hash=raw_blob.content_hash,
                ),
            ],
        )

        # ── 9. Ground truth ──
        ground_truth = GroundTruth(
            timestamp=ts_dt,
            monotonic_time=ts_unix,
            geolocation=GeoLocation(
                latitude=lat, longitude=lon, altitude_m=alt_m,
            ),
            device_ids=[device_id] if device_id else [],
            sensor_ids=[sensor_id] if sensor_id else [],
        )

        # ── 10. Assemble frame ──
        frame = RootedNatureFrame(
            ground_truth=ground_truth,
            observation=observation,
            self_state=self_state or SelfState(),
            world_state=world_state,
            action_context=action_context or ActionContext(),
            uncertainty=uncertainty,
            provenance=provenance,
        )

        # ── 11. Link parent ──
        if parent_frame is not None:
            frame.link_parent(parent_frame)

        # ── 12. Compute all Merkle roots ──
        frame.compute_roots()

        return frame

    def build_from_envelope(
        self,
        envelope: SignalEnvelope,
        *,
        lat: float = 0.0,
        lon: float = 0.0,
        alt_m: float = 0.0,
        parent_frame: Optional[RootedNatureFrame] = None,
        self_state: Optional[SelfState] = None,
    ) -> RootedNatureFrame:
        """Build a frame from a pre-normalized SignalEnvelope."""
        return self.build(
            raw_data=dict(envelope.normalized_values),
            lat=lat,
            lon=lon,
            alt_m=alt_m,
            timestamp=envelope.timestamp.timestamp(),
            device_id=envelope.device_slug,
            protocol=envelope.protocol_type.value,
            parent_frame=parent_frame,
            self_state=self_state,
            producer=f"nlm.{envelope.protocol_type.value}",
        )
