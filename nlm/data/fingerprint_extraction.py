"""
Fingerprint Extraction
======================

Extracts sensory fingerprints from raw device data and binary blobs.
Converts raw measurements into structured fingerprint dataclasses
that preserve physical units and measurement context.
"""

from __future__ import annotations

import struct
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nlm.core.fingerprints import (
    AcousticFingerprint,
    BioelectricFingerprint,
    ChemicalFingerprint,
    MechanicalFingerprint,
    SpectralFingerprint,
    ThermalFingerprint,
)
from nlm.core.protocols import DeviceEnvelope, SensorMetadata


class FingerprintExtractor:
    """Extracts typed sensory fingerprints from device envelopes.

    Each sensor type maps to a specific fingerprint:
    - spectral sensors → SpectralFingerprint
    - acoustic/microphone → AcousticFingerprint
    - bioelectric/FCI probes → BioelectricFingerprint
    - thermal/IR → ThermalFingerprint
    - chemical/VOC/pH → ChemicalFingerprint
    - pressure/accelerometer → MechanicalFingerprint
    """

    SENSOR_TO_FINGERPRINT = {
        "spectral": "spectral",
        "optical": "spectral",
        "infrared": "spectral",
        "uv": "spectral",
        "acoustic": "acoustic",
        "microphone": "acoustic",
        "bioelectric": "bioelectric",
        "fci": "bioelectric",
        "impedance": "bioelectric",
        "thermal": "thermal",
        "ir_camera": "thermal",
        "temperature_array": "thermal",
        "chemical": "chemical",
        "voc": "chemical",
        "ph": "chemical",
        "ion": "chemical",
        "gas": "chemical",
        "pressure": "mechanical",
        "accelerometer": "mechanical",
        "vibration": "mechanical",
        "strain": "mechanical",
    }

    def extract_all(self, envelope: DeviceEnvelope) -> Dict[str, list]:
        """Extract all fingerprints from a device envelope.

        Returns dict keyed by fingerprint type with lists of fingerprint objects.
        """
        result: Dict[str, list] = {
            "spectral": [],
            "acoustic": [],
            "bioelectric": [],
            "thermal": [],
            "chemical": [],
            "mechanical": [],
        }

        for sensor in envelope.sensors:
            fp_type = self.SENSOR_TO_FINGERPRINT.get(sensor.sensor_type)
            if fp_type is None:
                continue

            reading = envelope.readings.get(sensor.sensor_id, {})
            blob = envelope.binary_blobs.get(sensor.sensor_id)
            ts = envelope.recorded_at
            dev_id = envelope.device_id

            if fp_type == "spectral":
                fp = self._extract_spectral(reading, blob, sensor, ts, dev_id)
                if fp:
                    result["spectral"].append(fp)
            elif fp_type == "acoustic":
                fp = self._extract_acoustic(reading, blob, sensor, ts, dev_id)
                if fp:
                    result["acoustic"].append(fp)
            elif fp_type == "bioelectric":
                fp = self._extract_bioelectric(reading, blob, sensor, ts, dev_id)
                if fp:
                    result["bioelectric"].append(fp)
            elif fp_type == "thermal":
                fp = self._extract_thermal(reading, blob, sensor, ts, dev_id)
                if fp:
                    result["thermal"].append(fp)
            elif fp_type == "chemical":
                fp = self._extract_chemical(reading, envelope.readings, sensor, ts, dev_id)
                if fp:
                    result["chemical"].append(fp)
            elif fp_type == "mechanical":
                fp = self._extract_mechanical(reading, blob, sensor, ts, dev_id)
                if fp:
                    result["mechanical"].append(fp)

        # Also extract chemical fingerprint from general environmental readings
        # even without a dedicated chemical sensor
        if not result["chemical"] and envelope.readings:
            fp = self._extract_chemical_from_env(envelope.readings, envelope.recorded_at, envelope.device_id)
            if fp:
                result["chemical"].append(fp)

        return result

    def _extract_spectral(
        self, reading: Any, blob: Optional[bytes], sensor: SensorMetadata,
        timestamp: datetime, device_id: str,
    ) -> Optional[SpectralFingerprint]:
        if isinstance(reading, dict):
            bins = reading.get("wavelength_bins", reading.get("bins", []))
            values = reading.get("energy_values", reading.get("values", []))
            if bins and values:
                return SpectralFingerprint(
                    wavelength_bins=[float(b) for b in bins],
                    energy_values=[float(v) for v in values],
                    source_type=sensor.sensor_type,
                    timestamp=timestamp,
                    device_id=device_id,
                )
        if blob and len(blob) >= 8:
            # Interpret as packed float32 pairs: [bin, value, bin, value, ...]
            n_floats = len(blob) // 4
            if n_floats >= 4 and n_floats % 2 == 0:
                floats = struct.unpack(f"<{n_floats}f", blob[:n_floats * 4])
                bins = [floats[i] for i in range(0, n_floats, 2)]
                values = [floats[i] for i in range(1, n_floats, 2)]
                return SpectralFingerprint(
                    wavelength_bins=bins, energy_values=values,
                    source_type=sensor.sensor_type,
                    timestamp=timestamp, device_id=device_id,
                )
        return None

    def _extract_acoustic(
        self, reading: Any, blob: Optional[bytes], sensor: SensorMetadata,
        timestamp: datetime, device_id: str,
    ) -> Optional[AcousticFingerprint]:
        if isinstance(reading, dict):
            freq = reading.get("frequency_bins", reading.get("frequencies", []))
            mag = reading.get("magnitude", reading.get("amplitudes", []))
            if freq and mag:
                return AcousticFingerprint(
                    frequency_bins=[float(f) for f in freq],
                    magnitude=[float(m) for m in mag],
                    duration_ms=reading.get("duration_ms", 0.0),
                    sample_rate_hz=int(sensor.sampling_rate_hz),
                    timestamp=timestamp,
                    device_id=device_id,
                )
        return None

    def _extract_bioelectric(
        self, reading: Any, blob: Optional[bytes], sensor: SensorMetadata,
        timestamp: datetime, device_id: str,
    ) -> Optional[BioelectricFingerprint]:
        if isinstance(reading, dict):
            voltage = reading.get("voltage_series", reading.get("voltage", []))
            current = reading.get("current_series", reading.get("current", []))
            if isinstance(voltage, (int, float)):
                voltage = [voltage]
            if isinstance(current, (int, float)):
                current = [current]
            return BioelectricFingerprint(
                voltage_series=[float(v) for v in voltage] if voltage else [],
                current_series=[float(c) for c in current] if current else [],
                impedance=float(reading.get("impedance", 0.0)),
                sample_rate_hz=int(sensor.sampling_rate_hz),
                electrode_config=reading.get("electrode_config", "bipolar"),
                timestamp=timestamp,
                device_id=device_id,
            )
        return None

    def _extract_thermal(
        self, reading: Any, blob: Optional[bytes], sensor: SensorMetadata,
        timestamp: datetime, device_id: str,
    ) -> Optional[ThermalFingerprint]:
        if isinstance(reading, dict):
            field_data = reading.get("temperature_field", reading.get("thermal_grid", []))
            if field_data and isinstance(field_data, list):
                if field_data and not isinstance(field_data[0], list):
                    # 1D array → wrap as single row
                    field_data = [field_data]
                return ThermalFingerprint(
                    temperature_field=[[float(v) for v in row] for row in field_data],
                    gradient_magnitude=float(reading.get("gradient", 0.0)),
                    heat_flux=float(reading.get("heat_flux", 0.0)),
                    emissivity=float(reading.get("emissivity", 0.95)),
                    ambient_temperature=float(reading.get("ambient", 20.0)),
                    timestamp=timestamp,
                    device_id=device_id,
                )
        return None

    def _extract_chemical(
        self, reading: Any, all_readings: Dict, sensor: SensorMetadata,
        timestamp: datetime, device_id: str,
    ) -> Optional[ChemicalFingerprint]:
        if isinstance(reading, dict):
            return ChemicalFingerprint(
                voc_concentrations={k: float(v) for k, v in reading.get("voc", {}).items()},
                vsc_concentrations={k: float(v) for k, v in reading.get("vsc", {}).items()},
                ph=float(reading.get("ph", all_readings.get("ph", 7.0))),
                conductivity=float(reading.get("conductivity", 0.0)),
                dissolved_oxygen=float(reading.get("dissolved_oxygen", 0.0)),
                ion_concentrations={k: float(v) for k, v in reading.get("ions", {}).items()},
                timestamp=timestamp,
                device_id=device_id,
            )
        return None

    def _extract_chemical_from_env(
        self, readings: Dict[str, Any], timestamp: datetime, device_id: str,
    ) -> Optional[ChemicalFingerprint]:
        """Extract chemical fingerprint from general environmental readings."""
        has_chem = any(k in readings for k in ("ph", "co2_ppm", "conductivity", "voc", "dissolved_oxygen"))
        if not has_chem:
            return None
        return ChemicalFingerprint(
            ph=float(readings.get("ph", 7.0)),
            conductivity=float(readings.get("conductivity", 0.0)),
            dissolved_oxygen=float(readings.get("dissolved_oxygen", 0.0)),
            timestamp=timestamp,
            device_id=device_id,
        )

    def _extract_mechanical(
        self, reading: Any, blob: Optional[bytes], sensor: SensorMetadata,
        timestamp: datetime, device_id: str,
    ) -> Optional[MechanicalFingerprint]:
        if isinstance(reading, dict):
            force = reading.get("force_vector", reading.get("force", (0.0, 0.0, 0.0)))
            if isinstance(force, list):
                force = tuple(force[:3]) if len(force) >= 3 else (0.0, 0.0, 0.0)
            return MechanicalFingerprint(
                pressure_pa=float(reading.get("pressure_pa", reading.get("pressure", 101325.0))),
                vibration_spectrum=[float(v) for v in reading.get("vibration_spectrum", [])],
                vibration_freq_bins=[float(f) for f in reading.get("vibration_freq_bins", [])],
                force_vector=force,
                strain=float(reading.get("strain", 0.0)),
                moisture_content=float(reading.get("moisture", 0.0)),
                substrate_density=float(reading.get("density", 0.0)),
                timestamp=timestamp,
                device_id=device_id,
            )
        return None
