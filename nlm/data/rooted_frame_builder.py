"""
RootedNatureFrame Builder
=========================

Assembles RootedNatureFrames from device envelopes, preconditioned state,
and Merkle roots. This is the final assembly step before model consumption.

Pipeline: DeviceEnvelope → fingerprints + preprocessing → RootedNatureFrame
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from nlm.core.frames import (
    ActionContext,
    Observation,
    Provenance,
    RootedNatureFrame,
    SelfState,
    Uncertainty,
    WorldState,
)
from nlm.core.merkle import (
    compute_event_root,
    compute_frame_root,
    compute_self_root,
    compute_world_root,
    hash_bytes,
    hash_dict,
)
from nlm.core.protocols import DeviceEnvelope
from nlm.data.fingerprint_extraction import FingerprintExtractor
from nlm.data.preprocessor import NaturePreprocessor


class RootedFrameBuilder:
    """Builds RootedNatureFrames from device data and system state.

    Handles:
    1. Fingerprint extraction from device envelopes
    2. Bio-token generation via TranslationLayer
    3. Physics preconditioning via FieldPhysicsModel
    4. Merkle root computation
    5. Full frame assembly
    """

    def __init__(self) -> None:
        self.preprocessor = NaturePreprocessor()
        self.extractor = FingerprintExtractor()
        self._parent_frame_root: str = ""

    def build(
        self,
        envelope: DeviceEnvelope,
        self_state: Optional[SelfState] = None,
        world_state_override: Optional[WorldState] = None,
        action_context: Optional[ActionContext] = None,
        parent_frame_root: Optional[str] = None,
        producer: str = "nlm",
    ) -> RootedNatureFrame:
        """Build a complete RootedNatureFrame from a device envelope.

        Args:
            envelope: Normalized device data envelope
            self_state: Current MYCA/MAS internal state (defaults to empty)
            world_state_override: Override world state (otherwise derived from envelope)
            action_context: Recent/intended actions
            parent_frame_root: Previous frame root for lineage (auto-tracks if None)
            producer: Service/agent that created this frame
        """
        # Use tracked parent if not provided
        if parent_frame_root is None:
            parent_frame_root = self._parent_frame_root

        # --- Extract fingerprints ---
        fingerprints = self.extractor.extract_all(envelope)

        # --- Preprocess for bio-tokens and physics ---
        preprocessed = self.preprocessor.process_envelope(envelope)

        # --- Build observation ---
        observation = Observation(
            raw_sensor_blobs=envelope.binary_blobs,
            normalized_physical=preprocessed["environmental_normalized"],
            spectral=fingerprints["spectral"],
            acoustic=fingerprints["acoustic"],
            bioelectric=fingerprints["bioelectric"],
            thermal=fingerprints["thermal"],
            chemical=fingerprints["chemical"],
            mechanical=fingerprints["mechanical"],
            waveform_refs=[],
            bio_tokens=preprocessed["bio_tokens"],
        )

        # --- Build world state ---
        if world_state_override is not None:
            world_state = world_state_override
        else:
            world_state = WorldState(
                environmental=preprocessed["environmental_normalized"],
                derived_fields=preprocessed["physics_context"],
            )

        # --- Defaults ---
        if self_state is None:
            self_state = SelfState()
        if action_context is None:
            action_context = ActionContext()

        # --- Compute sensor hashes for event root ---
        sensor_hashes = []
        for sensor_id, blob in envelope.binary_blobs.items():
            sensor_hashes.append(hash_bytes(blob))
        for sensor_id, reading in envelope.readings.items():
            sensor_hashes.append(hash_dict({"sensor_id": sensor_id, "reading": reading}))

        # --- Merkle roots ---
        event_root = compute_event_root(
            timestamp=envelope.recorded_at.isoformat(),
            geolocation=f"{envelope.geolocation()}",
            sensor_hashes=sensor_hashes,
            bio_tokens=preprocessed["bio_tokens"],
        )
        self_root = compute_self_root(self_state.to_dict())
        world_root = compute_world_root(world_state.to_dict())
        frame_root = compute_frame_root(self_root, world_root, event_root, parent_frame_root)

        # --- Uncertainty ---
        sensor_confidence = {}
        missingness = {}
        freshness = {}
        for sensor in envelope.sensors:
            sensor_confidence[sensor.sensor_id] = 1.0 if sensor.operational else 0.0
            missingness[sensor.sensor_id] = sensor.sensor_id not in envelope.readings
        if envelope.age_seconds() > 0:
            freshness["envelope"] = envelope.age_seconds()

        uncertainty = Uncertainty(
            missingness=missingness,
            sensor_confidence=sensor_confidence,
            freshness_seconds=freshness,
            overall_confidence=1.0 if envelope.verified else 0.8,
        )

        # --- Provenance ---
        provenance = Provenance(
            chain_of_custody=[envelope.device_id, producer],
            producer=producer,
            content_hash=hash_dict(observation.normalized_physical),
            source_refs=[f"device:{envelope.device_id}"],
            ingestion_path=f"device→{envelope.header.transport}→nlm",
        )

        # --- Assemble frame ---
        frame = RootedNatureFrame(
            frame_root=frame_root,
            parent_frame_root=parent_frame_root,
            event_root=event_root,
            self_root=self_root,
            world_root=world_root,
            timestamp=envelope.recorded_at,
            geolocation=envelope.geolocation(),
            device_ids=[envelope.device_id],
            observation=observation,
            self_state=self_state,
            world_state=world_state,
            action_context=action_context,
            uncertainty=uncertainty,
            provenance=provenance,
        )

        # Track for next frame
        self._parent_frame_root = frame_root

        return frame

    def build_from_dict(
        self,
        raw_data: Dict[str, Any],
        device_id: str = "synthetic",
        self_state: Optional[SelfState] = None,
        parent_frame_root: str = "",
    ) -> RootedNatureFrame:
        """Build a frame from a plain dictionary (for testing/synthetic data).

        Wraps the dict in a DeviceEnvelope and calls build().
        """
        from nlm.core.protocols import ProtocolHeader

        envelope = DeviceEnvelope(
            device_id=device_id,
            header=ProtocolHeader(protocol_name="synthetic"),
            readings=raw_data,
            recorded_at=datetime.now(timezone.utc),
            received_at=datetime.now(timezone.utc),
            latitude=raw_data.get("latitude", 0.0),
            longitude=raw_data.get("longitude", 0.0),
            altitude=raw_data.get("altitude", 0.0),
        )
        return self.build(
            envelope=envelope,
            self_state=self_state,
            parent_frame_root=parent_frame_root,
            producer="synthetic",
        )
