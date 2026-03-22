"""
Device Protocol Layer
=====================

Normalized envelopes for Mycorrhizae Protocol, FCI, Mushroom1,
MycoNode, SporeBase, Petraeus, and other device families.

Raw device data arrives through normalized envelopes with protocol
metadata attached. The data path supports biological, chemical,
and environmental telemetry together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ProtocolHeader:
    """Wire-level protocol metadata from device communication."""

    protocol_name: str  # "mycorrhizae", "fci", "mushroom1", "myconode", "sporebase", "petraeus"
    protocol_version: str = "1.0"
    device_family: str = ""  # hardware family identifier
    firmware_version: str = ""
    transport: str = "mqtt"  # "mqtt", "http", "ble", "lora", "serial"
    encoding: str = "json"  # "json", "msgpack", "protobuf", "cbor"
    sequence_number: int = 0
    hop_count: int = 0  # how many relays this traversed
    ttl_seconds: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_name": self.protocol_name,
            "protocol_version": self.protocol_version,
            "device_family": self.device_family,
            "firmware_version": self.firmware_version,
            "transport": self.transport,
            "encoding": self.encoding,
            "sequence_number": self.sequence_number,
            "hop_count": self.hop_count,
            "ttl_seconds": self.ttl_seconds,
        }


@dataclass
class SensorMetadata:
    """Metadata about a specific sensor on a device.

    Captures calibration, accuracy, and operational bounds so the
    model knows how much to trust each reading.
    """

    sensor_id: str
    sensor_type: str  # "temperature", "humidity", "bioelectric", "spectral", "chemical", "acoustic", "mechanical", "camera"
    unit: str  # SI unit string: "°C", "Pa", "mV", "Hz", "ppb", "lux"
    accuracy: float = 0.0  # ± in sensor units
    resolution: float = 0.0  # smallest detectable change
    range_min: float = 0.0
    range_max: float = 0.0
    calibration_date: Optional[str] = None
    calibration_ref: str = ""
    sampling_rate_hz: float = 1.0
    operational: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "unit": self.unit,
            "accuracy": self.accuracy,
            "resolution": self.resolution,
            "range": [self.range_min, self.range_max],
            "calibration_date": self.calibration_date,
            "sampling_rate_hz": self.sampling_rate_hz,
            "operational": self.operational,
        }


@dataclass
class DeviceEnvelope:
    """Normalized device data envelope.

    The universal ingestion format for all device families.
    Every observation enters the NLM through this envelope,
    preserving protocol metadata and sensor context.
    """

    # Identity
    device_id: str
    device_slug: str = ""
    site_id: str = ""

    # Protocol context
    header: ProtocolHeader = field(default_factory=lambda: ProtocolHeader(protocol_name="unknown"))
    sensors: List[SensorMetadata] = field(default_factory=list)

    # Timing
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = 0.0

    # Location
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    location_accuracy_m: float = 0.0

    # Payload — raw sensor readings keyed by sensor_id
    readings: Dict[str, Any] = field(default_factory=dict)
    # Raw bytes for binary sensor data (spectra, waveforms)
    binary_blobs: Dict[str, bytes] = field(default_factory=dict)

    # Verification
    verified: bool = False
    signature: str = ""
    envelope_hash: str = ""

    def geolocation(self) -> tuple:
        return (
            self.latitude or 0.0,
            self.longitude or 0.0,
            self.altitude or 0.0,
        )

    def age_seconds(self) -> float:
        delta = self.received_at - self.recorded_at
        return max(0.0, delta.total_seconds())

    def get_sensor_meta(self, sensor_id: str) -> Optional[SensorMetadata]:
        for s in self.sensors:
            if s.sensor_id == sensor_id:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_slug": self.device_slug,
            "site_id": self.site_id,
            "header": self.header.to_dict(),
            "sensors": [s.to_dict() for s in self.sensors],
            "recorded_at": self.recorded_at.isoformat(),
            "received_at": self.received_at.isoformat(),
            "latency_ms": self.latency_ms,
            "geolocation": self.geolocation(),
            "readings": self.readings,
            "binary_blob_keys": list(self.binary_blobs.keys()),
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeviceEnvelope:
        header_data = data.get("header", {})
        header = ProtocolHeader(
            protocol_name=header_data.get("protocol_name", "unknown"),
            protocol_version=header_data.get("protocol_version", "1.0"),
            device_family=header_data.get("device_family", ""),
            firmware_version=header_data.get("firmware_version", ""),
            transport=header_data.get("transport", "mqtt"),
            encoding=header_data.get("encoding", "json"),
            sequence_number=header_data.get("sequence_number", 0),
            hop_count=header_data.get("hop_count", 0),
            ttl_seconds=header_data.get("ttl_seconds", 300),
        )

        sensors = []
        for s in data.get("sensors", []):
            rng = s.get("range", [0.0, 0.0])
            sensors.append(SensorMetadata(
                sensor_id=s["sensor_id"],
                sensor_type=s.get("sensor_type", ""),
                unit=s.get("unit", ""),
                accuracy=s.get("accuracy", 0.0),
                resolution=s.get("resolution", 0.0),
                range_min=rng[0] if len(rng) > 0 else 0.0,
                range_max=rng[1] if len(rng) > 1 else 0.0,
                calibration_date=s.get("calibration_date"),
                sampling_rate_hz=s.get("sampling_rate_hz", 1.0),
                operational=s.get("operational", True),
            ))

        rec_at = data.get("recorded_at")
        if isinstance(rec_at, str):
            rec_at = datetime.fromisoformat(rec_at.replace("Z", "+00:00"))
        else:
            rec_at = datetime.now(timezone.utc)

        recv_at = data.get("received_at")
        if isinstance(recv_at, str):
            recv_at = datetime.fromisoformat(recv_at.replace("Z", "+00:00"))
        else:
            recv_at = datetime.now(timezone.utc)

        geo = data.get("geolocation", (None, None, None))

        return cls(
            device_id=data["device_id"],
            device_slug=data.get("device_slug", ""),
            site_id=data.get("site_id", ""),
            header=header,
            sensors=sensors,
            recorded_at=rec_at,
            received_at=recv_at,
            latency_ms=data.get("latency_ms", 0.0),
            latitude=geo[0] if isinstance(geo, (list, tuple)) and len(geo) > 0 else data.get("latitude"),
            longitude=geo[1] if isinstance(geo, (list, tuple)) and len(geo) > 1 else data.get("longitude"),
            altitude=geo[2] if isinstance(geo, (list, tuple)) and len(geo) > 2 else data.get("altitude"),
            readings=data.get("readings", {}),
            verified=data.get("verified", False),
            signature=data.get("signature", ""),
        )
