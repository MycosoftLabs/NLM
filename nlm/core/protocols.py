"""
NLM Device Protocol Definitions

Defines the supported device protocols (FCI, Mushroom1, MycoNode,
SporeBase, Petraeus) and the Mycorrhizae normalized signal envelope
format.

Each protocol adapter normalizes device-specific telemetry into
standard envelopes that feed into the RootedFrameBuilder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ProtocolType(str, Enum):
    """Supported device protocol families."""

    FCI = "fci"                # Fungal Computing Interface
    MUSHROOM1 = "mushroom1"    # Mushroom1 sensor boards
    MYCONODE = "myconode"      # MycoNode distributed sensors
    SPOREBASE = "sporebase"    # SporeBase data loggers
    PETRAEUS = "petraeus"      # Petraeus environmental monitors
    GENERIC = "generic"        # Fallback for unknown protocols


@dataclass
class CalibrationRef:
    """Reference to a sensor calibration record."""

    calibration_id: str = ""
    calibration_date: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    method: str = ""  # e.g., "factory", "field", "cross_calibration"
    mindex_ref: str = ""  # MINDEX ID for full calibration data


@dataclass
class NormalizationMeta:
    """Metadata about how raw data was normalized."""

    method: str = ""  # e.g., "linear_scale", "polynomial", "lookup_table"
    input_range: tuple = (0.0, 1.0)
    output_unit: str = ""  # SI unit
    applied_corrections: List[str] = field(default_factory=list)


@dataclass
class SignalEnvelope:
    """
    Mycorrhizae normalized signal envelope.

    The universal container for device-to-NLM data transfer.
    Protocol-specific raw data is normalized into this format
    before entering the cognitive pipeline.
    """

    # Identity
    envelope_id: str = ""
    sequence_number: int = 0
    message_id: str = ""

    # Source
    protocol_type: ProtocolType = ProtocolType.GENERIC
    device_slug: str = ""
    firmware_version: str = ""
    stream_key: str = ""

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    device_timestamp: Optional[datetime] = None  # device-local clock

    # Payload
    raw_payload: bytes = b""
    normalized_values: Dict[str, float] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)

    # Calibration and normalization metadata
    calibration: Optional[CalibrationRef] = None
    normalization: Optional[NormalizationMeta] = None

    # Quality
    signal_quality: float = 1.0  # [0, 1]
    error_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "sequence_number": self.sequence_number,
            "message_id": self.message_id,
            "protocol_type": self.protocol_type.value,
            "device_slug": self.device_slug,
            "firmware_version": self.firmware_version,
            "stream_key": self.stream_key,
            "timestamp": self.timestamp.isoformat(),
            "normalized_values": self.normalized_values,
            "units": self.units,
            "signal_quality": self.signal_quality,
            "error_flags": self.error_flags,
        }


# ── Protocol Adapters ───────────────────────────────────────────────


class ProtocolAdapter:
    """
    Base class for protocol-specific normalization.

    Each adapter converts raw device payloads into SignalEnvelopes.
    """

    protocol_type: ProtocolType = ProtocolType.GENERIC

    def normalize(self, raw: Dict[str, Any], device_slug: str = "") -> SignalEnvelope:
        """Convert raw device data to a normalized SignalEnvelope."""
        envelope = SignalEnvelope(
            protocol_type=self.protocol_type,
            device_slug=device_slug,
        )
        # Copy numeric values and infer SI units
        for key, value in raw.items():
            try:
                envelope.normalized_values[key] = float(value)
                envelope.units[key] = self._infer_unit(key)
            except (TypeError, ValueError):
                pass
        return envelope

    def _infer_unit(self, key: str) -> str:
        """Infer SI unit from field name. Override for protocol-specific units."""
        unit_map = {
            "temperature": "celsius",
            "temp": "celsius",
            "humidity": "percent",
            "pressure": "hPa",
            "co2": "ppm",
            "ph": "pH",
            "moisture": "percent",
            "light": "lux",
            "voltage": "mV",
            "current": "uA",
            "resistance": "kOhm",
        }
        key_lower = key.lower()
        for pattern, unit in unit_map.items():
            if pattern in key_lower:
                return unit
        return ""


class FCIAdapter(ProtocolAdapter):
    """Fungal Computing Interface protocol adapter."""

    protocol_type = ProtocolType.FCI

    def normalize(self, raw: Dict[str, Any], device_slug: str = "") -> SignalEnvelope:
        envelope = super().normalize(raw, device_slug)
        # FCI-specific: bioelectric channels
        for key in ["voltage_mv", "current_ua", "impedance_ohm"]:
            if key in raw:
                try:
                    envelope.normalized_values[key] = float(raw[key])
                except (TypeError, ValueError):
                    pass
        return envelope


class Mushroom1Adapter(ProtocolAdapter):
    """Mushroom1 sensor board protocol adapter."""

    protocol_type = ProtocolType.MUSHROOM1


class MycoNodeAdapter(ProtocolAdapter):
    """MycoNode distributed sensor protocol adapter."""

    protocol_type = ProtocolType.MYCONODE


class SporeBaseAdapter(ProtocolAdapter):
    """SporeBase data logger protocol adapter."""

    protocol_type = ProtocolType.SPOREBASE


class PetraeusAdapter(ProtocolAdapter):
    """Petraeus environmental monitor protocol adapter."""

    protocol_type = ProtocolType.PETRAEUS


# ── Adapter Registry ────────────────────────────────────────────────


PROTOCOL_ADAPTERS: Dict[ProtocolType, ProtocolAdapter] = {
    ProtocolType.FCI: FCIAdapter(),
    ProtocolType.MUSHROOM1: Mushroom1Adapter(),
    ProtocolType.MYCONODE: MycoNodeAdapter(),
    ProtocolType.SPOREBASE: SporeBaseAdapter(),
    ProtocolType.PETRAEUS: PetraeusAdapter(),
    ProtocolType.GENERIC: ProtocolAdapter(),
}


def get_adapter(protocol: str) -> ProtocolAdapter:
    """Get the adapter for a protocol string."""
    try:
        pt = ProtocolType(protocol.lower())
    except ValueError:
        pt = ProtocolType.GENERIC
    return PROTOCOL_ADAPTERS[pt]
