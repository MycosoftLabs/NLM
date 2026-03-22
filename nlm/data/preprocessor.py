"""
Nature Preprocessor
===================

Bridges existing telemetry/physics modules to model-ready tensors.
Uses TranslationLayer for bio-token extraction and FieldPhysicsModel
for physics feature computation.

Pipeline: DeviceEnvelope → normalized values → bio-tokens → tensor dict
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nlm.core.protocols import DeviceEnvelope, SensorMetadata


class NaturePreprocessor:
    """Converts raw device data into normalized tensor-ready features.

    Reuses existing modules:
    - nlm.telemetry.translation_layer for raw→normalized→bio-tokens
    - nlm.physics.field_physics.FieldPhysicsModel for physics context
    """

    # Standard normalization ranges for environmental values
    NORMALIZATION_RANGES: Dict[str, Tuple[float, float]] = {
        "temperature_c": (-40.0, 60.0),
        "humidity_pct": (0.0, 100.0),
        "co2_ppm": (0.0, 5000.0),
        "pressure_hpa": (900.0, 1100.0),
        "light_lux": (0.0, 100000.0),
        "ph": (0.0, 14.0),
        "moisture_pct": (0.0, 100.0),
        "wind_speed_mps": (0.0, 100.0),
        "dissolved_oxygen_mg_l": (0.0, 20.0),
        "conductivity_us_cm": (0.0, 10000.0),
    }

    def __init__(self) -> None:
        self._translation_layer = None
        self._physics_model = None

    def _get_translation_layer(self):
        if self._translation_layer is None:
            from nlm.telemetry.translation_layer import normalize_raw, raw_to_bio_tokens
            self._translation_layer = (normalize_raw, raw_to_bio_tokens)
        return self._translation_layer

    def _get_physics_model(self):
        if self._physics_model is None:
            from nlm.physics.field_physics import FieldPhysicsModel
            self._physics_model = FieldPhysicsModel()
        return self._physics_model

    def normalize_value(self, key: str, value: float) -> float:
        """Normalize a single value to [0, 1] range using known bounds."""
        if key in self.NORMALIZATION_RANGES:
            lo, hi = self.NORMALIZATION_RANGES[key]
            if hi == lo:
                return 0.5
            return max(0.0, min(1.0, (value - lo) / (hi - lo)))
        return value

    def normalize_readings(self, readings: Dict[str, Any]) -> Dict[str, float]:
        """Normalize raw sensor readings to [0, 1] range."""
        normalized = {}
        for key, value in readings.items():
            if isinstance(value, (int, float)):
                normalized[key] = self.normalize_value(key, float(value))
        return normalized

    def extract_bio_tokens(self, readings: Dict[str, Any]) -> List[str]:
        """Extract bio-tokens from raw readings using TranslationLayer."""
        normalize_raw, raw_to_bio_tokens = self._get_translation_layer()
        return raw_to_bio_tokens(readings)

    def compute_physics_context(
        self, location: Tuple[float, float, float], timestamp=None
    ) -> Dict[str, float]:
        """Compute deterministic physics features for a location/time.

        Returns flat dict of physics-derived values ready for tensor creation.
        """
        model = self._get_physics_model()

        # FieldPhysicsModel expects unix timestamp, not datetime
        from datetime import datetime
        if isinstance(timestamp, datetime):
            timestamp = timestamp.timestamp()

        geo = model.get_geomagnetic_field(location, timestamp)
        lunar = model.get_lunar_gravitational_influence(location, timestamp)
        atmo = model.get_atmospheric_conditions(location, timestamp)

        return {
            "geo_bx": geo.get("Bx", 0.0),
            "geo_by": geo.get("By", 0.0),
            "geo_bz": geo.get("Bz", 0.0),
            "geo_inclination": geo.get("inclination", 0.0),
            "geo_declination": geo.get("declination", 0.0),
            "geo_field_strength": geo.get("field_strength", 0.0),
            "lunar_phase_angle": lunar.get("phase_angle", 0.0),
            "lunar_phase_fraction": lunar.get("illumination_fraction", 0.0),
            "lunar_gravity": lunar.get("gravitational_acceleration", 0.0),
            "lunar_tidal_potential": lunar.get("tidal_potential", 0.0),
            "atmo_temperature": atmo.get("temperature", 0.0),
            "atmo_pressure": atmo.get("pressure", 0.0),
            "atmo_humidity": atmo.get("humidity", 0.0),
            "atmo_wind_speed": atmo.get("wind_speed", 0.0),
        }

    def encode_spatial(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """Encode geographic coordinates using sinusoidal positional encoding.

        Returns a fixed-size vector encoding the position on Earth's surface.
        """
        features = []
        # Multi-frequency sinusoidal encoding for lat/lon
        for freq in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
            features.extend([
                math.sin(math.radians(lat) * freq),
                math.cos(math.radians(lat) * freq),
                math.sin(math.radians(lon) * freq),
                math.cos(math.radians(lon) * freq),
            ])
        # Altitude (log-scaled)
        features.append(math.log1p(max(0.0, alt)))
        # Raw normalized lat/lon
        features.extend([lat / 90.0, lon / 180.0])
        return np.array(features, dtype=np.float32)

    def encode_temporal(self, timestamp, physics_context: Dict[str, float]) -> np.ndarray:
        """Encode time as multi-scale cyclical features.

        Not learned positions — physics-derived cycles:
        - Time of day (sin/cos)
        - Day of year (sin/cos)
        - Lunar phase (sin/cos) from physics context
        - Solar declination
        """
        from datetime import datetime, timezone

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        elif isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        # Time of day cycle
        hour_frac = (timestamp.hour + timestamp.minute / 60.0) / 24.0
        tod_sin = math.sin(2.0 * math.pi * hour_frac)
        tod_cos = math.cos(2.0 * math.pi * hour_frac)

        # Day of year cycle
        doy = timestamp.timetuple().tm_yday
        doy_frac = doy / 365.25
        doy_sin = math.sin(2.0 * math.pi * doy_frac)
        doy_cos = math.cos(2.0 * math.pi * doy_frac)

        # Lunar phase from physics
        lunar_angle = physics_context.get("lunar_phase_angle", 0.0)
        lunar_sin = math.sin(math.radians(lunar_angle))
        lunar_cos = math.cos(math.radians(lunar_angle))
        lunar_frac = physics_context.get("lunar_phase_fraction", 0.0)

        # Solar declination (approximate)
        solar_dec = 23.44 * math.sin(math.radians((360.0 / 365.0) * (doy - 81)))
        solar_sin = math.sin(math.radians(solar_dec))
        solar_cos = math.cos(math.radians(solar_dec))

        # Week cycle
        dow_frac = timestamp.weekday() / 7.0
        dow_sin = math.sin(2.0 * math.pi * dow_frac)
        dow_cos = math.cos(2.0 * math.pi * dow_frac)

        # Year fraction
        year_frac = (timestamp.month - 1 + timestamp.day / 30.0) / 12.0

        return np.array([
            tod_sin, tod_cos,
            doy_sin, doy_cos,
            lunar_sin, lunar_cos, lunar_frac,
            solar_sin, solar_cos,
            dow_sin, dow_cos,
            year_frac,
        ], dtype=np.float32)

    def process_envelope(self, envelope: DeviceEnvelope) -> Dict[str, Any]:
        """Full preprocessing pipeline for a single device envelope.

        Returns dict of tensor-ready arrays and metadata.
        """
        location = envelope.geolocation()
        physics = self.compute_physics_context(location, envelope.recorded_at)

        normalized = self.normalize_readings(envelope.readings)
        bio_tokens = self.extract_bio_tokens(envelope.readings)
        spatial = self.encode_spatial(*location)
        temporal = self.encode_temporal(envelope.recorded_at, physics)

        # World state: environmental readings + physics-derived fields
        env_values = list(normalized.values())
        physics_values = list(physics.values())

        return {
            "spatial": spatial,
            "temporal": temporal,
            "environmental_normalized": normalized,
            "environmental_vector": np.array(env_values, dtype=np.float32) if env_values else np.zeros(1, dtype=np.float32),
            "physics_context": physics,
            "physics_vector": np.array(physics_values, dtype=np.float32),
            "bio_tokens": bio_tokens,
            "device_id": envelope.device_id,
            "timestamp": envelope.recorded_at,
            "geolocation": location,
        }
