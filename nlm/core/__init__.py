"""
NLM Core — Ontology primitives for the Nature Learning Model.

Defines the irreducible data types: RootedNatureFrame, sensory fingerprints,
Merkle lineage, and device protocol envelopes.
"""

from nlm.core.frames import (
    RootedNatureFrame,
    SelfState,
    WorldState,
    ActionContext,
    Uncertainty,
    Provenance,
    Observation,
)
from nlm.core.fingerprints import (
    SpectralFingerprint,
    AcousticFingerprint,
    BioelectricFingerprint,
    ThermalFingerprint,
    ChemicalFingerprint,
    MechanicalFingerprint,
)
from nlm.core.merkle import compute_frame_root, merkle_hash, verify_lineage, LineageRecord
from nlm.core.protocols import DeviceEnvelope, SensorMetadata, ProtocolHeader

__all__ = [
    "RootedNatureFrame",
    "SelfState",
    "WorldState",
    "ActionContext",
    "Uncertainty",
    "Provenance",
    "Observation",
    "SpectralFingerprint",
    "AcousticFingerprint",
    "BioelectricFingerprint",
    "ThermalFingerprint",
    "ChemicalFingerprint",
    "MechanicalFingerprint",
    "compute_frame_root",
    "merkle_hash",
    "verify_lineage",
    "LineageRecord",
    "DeviceEnvelope",
    "SensorMetadata",
    "ProtocolHeader",
]
