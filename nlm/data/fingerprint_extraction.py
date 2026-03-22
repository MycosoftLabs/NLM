"""
NLM Fingerprint Extraction

Takes calibrated/normalized sensor data and produces SensoryFingerprint
instances for each modality present in the observation.

This is the bridge between raw physical measurements and the learned
encoder inputs. Fingerprints are continuous vector representations —
not symbolic tokens.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from nlm.core.fingerprints import (
    AcousticFingerprint,
    BioelectricFingerprint,
    ChemicalFingerprint,
    MechanicalFingerprint,
    SensoryFingerprint,
    SpectralFingerprint,
    ThermalFingerprint,
)
from nlm.core.merkle import sha256

logger = logging.getLogger(__name__)


def _serialize_for_hash(data: Any) -> bytes:
    """Quick deterministic serialization for hashing."""
    import json
    return json.dumps(data, sort_keys=True, default=str).encode("utf-8")


class FingerprintExtractor:
    """
    Extracts sensory fingerprints from calibrated sensor data.

    Call extract_all() with a dict of normalized sensor values to get
    all applicable fingerprints for the data present.
    """

    def extract_all(
        self,
        normalized: Dict[str, float],
        raw_bytes: Optional[bytes] = None,
        device_id: str = "",
        sensor_id: str = "",
        timestamp: Optional[datetime] = None,
    ) -> List[SensoryFingerprint]:
        """
        Extract all applicable fingerprints from normalized data.

        Returns a list of SensoryFingerprint subclass instances.
        """
        ts = timestamp or datetime.now(timezone.utc)
        raw_hash = sha256(raw_bytes) if raw_bytes else sha256(_serialize_for_hash(normalized))
        fingerprints: List[SensoryFingerprint] = []

        spectral = self._extract_spectral(normalized, device_id, sensor_id, ts, raw_hash)
        if spectral:
            fingerprints.append(spectral)

        acoustic = self._extract_acoustic(normalized, device_id, sensor_id, ts, raw_hash)
        if acoustic:
            fingerprints.append(acoustic)

        bioelectric = self._extract_bioelectric(normalized, device_id, sensor_id, ts, raw_hash)
        if bioelectric:
            fingerprints.append(bioelectric)

        thermal = self._extract_thermal(normalized, device_id, sensor_id, ts, raw_hash)
        if thermal:
            fingerprints.append(thermal)

        chemical = self._extract_chemical(normalized, device_id, sensor_id, ts, raw_hash)
        if chemical:
            fingerprints.append(chemical)

        mechanical = self._extract_mechanical(normalized, device_id, sensor_id, ts, raw_hash)
        if mechanical:
            fingerprints.append(mechanical)

        return fingerprints

    def _extract_spectral(
        self, data: Dict[str, float], device_id: str, sensor_id: str,
        ts: datetime, raw_hash: bytes,
    ) -> Optional[SpectralFingerprint]:
        """Extract spectral fingerprint from light/wavelength data."""
        light_keys = [k for k in data if any(
            p in k.lower() for p in ["light", "lux", "uv", "ir", "spectral", "wavelength", "ndvi"]
        )]
        if not light_keys:
            return None

        values = np.array([data[k] for k in sorted(light_keys)], dtype=np.float32)
        # Create synthetic wavelength bins if not provided
        n = len(values)
        wavelength_bins = np.linspace(380, 780, n)  # visible spectrum default

        peak_indices = []
        if n > 2:
            for i in range(1, n - 1):
                if values[i] > values[i - 1] and values[i] > values[i + 1]:
                    peak_indices.append(i)
        peak_wavelengths = [float(wavelength_bins[i]) for i in peak_indices]

        bandwidth = float(wavelength_bins[-1] - wavelength_bins[0]) if n > 1 else 0.0

        return SpectralFingerprint(
            device_id=device_id,
            sensor_id=sensor_id,
            timestamp=ts,
            raw_hash=raw_hash,
            wavelength_bins_nm=wavelength_bins,
            spectral_power=values,
            peak_wavelengths_nm=peak_wavelengths,
            bandwidth_nm=bandwidth,
            confidence=0.8 if n > 1 else 0.5,
        )

    def _extract_acoustic(
        self, data: Dict[str, float], device_id: str, sensor_id: str,
        ts: datetime, raw_hash: bytes,
    ) -> Optional[AcousticFingerprint]:
        """Extract acoustic fingerprint from sound/frequency data."""
        acoustic_keys = [k for k in data if any(
            p in k.lower() for p in ["sound", "audio", "freq", "acoustic", "db", "decibel"]
        )]
        if not acoustic_keys:
            return None

        values = np.array([data[k] for k in sorted(acoustic_keys)], dtype=np.float32)
        n = len(values)
        freq_bins = np.logspace(1, 4.3, n)  # 10 Hz to ~20 kHz

        # Spectral centroid
        total_energy = np.sum(values) if np.sum(values) > 0 else 1.0
        centroid = float(np.sum(freq_bins * values) / total_energy)

        return AcousticFingerprint(
            device_id=device_id,
            sensor_id=sensor_id,
            timestamp=ts,
            raw_hash=raw_hash,
            frequency_bins_hz=freq_bins,
            energy_distribution=values,
            spectral_centroid_hz=centroid,
            confidence=0.7,
        )

    def _extract_bioelectric(
        self, data: Dict[str, float], device_id: str, sensor_id: str,
        ts: datetime, raw_hash: bytes,
    ) -> Optional[BioelectricFingerprint]:
        """Extract bioelectric fingerprint from voltage/current/impedance data."""
        bio_keys = [k for k in data if any(
            p in k.lower() for p in ["voltage", "current", "impedance", "resistance",
                                       "bioelectric", "electrode"]
        )]
        if not bio_keys:
            return None

        voltages = []
        currents = []
        resistances = []
        labels = []

        for k in sorted(bio_keys):
            labels.append(k)
            v = data[k]
            if "voltage" in k.lower() or "mv" in k.lower():
                voltages.append(v)
            elif "current" in k.lower() or "ua" in k.lower():
                currents.append(v)
            else:
                resistances.append(v)

        return BioelectricFingerprint(
            device_id=device_id,
            sensor_id=sensor_id,
            timestamp=ts,
            raw_hash=raw_hash,
            channel_labels=labels,
            voltages_mv=np.array(voltages, dtype=np.float32) if voltages else np.array([]),
            currents_ua=np.array(currents, dtype=np.float32) if currents else np.array([]),
            resistances_kohm=np.array(resistances, dtype=np.float32) if resistances else np.array([]),
            confidence=0.8,
        )

    def _extract_thermal(
        self, data: Dict[str, float], device_id: str, sensor_id: str,
        ts: datetime, raw_hash: bytes,
    ) -> Optional[ThermalFingerprint]:
        """Extract thermal fingerprint from temperature data."""
        temp_keys = [k for k in data if any(
            p in k.lower() for p in ["temp", "thermal", "heat", "celsius", "fahrenheit"]
        )]
        if not temp_keys:
            return None

        temps = np.array([data[k] for k in sorted(temp_keys)], dtype=np.float32)

        # Compute gradient if multiple temperature readings
        gradient_mag = 0.0
        if len(temps) > 1:
            gradient_mag = float(np.max(temps) - np.min(temps))

        return ThermalFingerprint(
            device_id=device_id,
            sensor_id=sensor_id,
            timestamp=ts,
            raw_hash=raw_hash,
            temperatures_celsius=temps,
            gradient_magnitude=gradient_mag,
            confidence=0.9,
        )

    def _extract_chemical(
        self, data: Dict[str, float], device_id: str, sensor_id: str,
        ts: datetime, raw_hash: bytes,
    ) -> Optional[ChemicalFingerprint]:
        """Extract chemical fingerprint from gas/pH/conductivity data."""
        chem_keys = [k for k in data if any(
            p in k.lower() for p in ["co2", "ch4", "voc", "vsc", "ph", "conductivity",
                                       "humidity", "moisture", "oxygen", "gas", "ppm", "ppb"]
        )]
        if not chem_keys:
            return None

        gas_conc: Dict[str, float] = {}
        ph_val = None
        ec_val = None
        co2_val = None
        humidity_val = None
        do_val = None

        for k in chem_keys:
            v = data[k]
            kl = k.lower()
            if "ph" == kl or kl.endswith("_ph"):
                ph_val = v
            elif "conductivity" in kl:
                ec_val = v
            elif "co2" in kl:
                co2_val = v
                gas_conc["CO2"] = v
            elif "ch4" in kl or "methane" in kl:
                gas_conc["CH4"] = v
            elif "humidity" in kl:
                humidity_val = v
            elif "oxygen" in kl:
                do_val = v
            elif "voc" in kl:
                gas_conc["VOC_total"] = v
            elif "ppm" in kl or "ppb" in kl or "gas" in kl:
                gas_conc[k] = v

        return ChemicalFingerprint(
            device_id=device_id,
            sensor_id=sensor_id,
            timestamp=ts,
            raw_hash=raw_hash,
            gas_concentrations_ppm=gas_conc,
            ph=ph_val,
            electrical_conductivity_us_cm=ec_val,
            co2_ppm=co2_val,
            humidity_percent=humidity_val,
            dissolved_oxygen_mg_l=do_val,
            confidence=0.8,
        )

    def _extract_mechanical(
        self, data: Dict[str, float], device_id: str, sensor_id: str,
        ts: datetime, raw_hash: bytes,
    ) -> Optional[MechanicalFingerprint]:
        """Extract mechanical fingerprint from pressure/vibration/seismic data."""
        mech_keys = [k for k in data if any(
            p in k.lower() for p in ["pressure", "vibration", "seismic", "strain",
                                       "magnitude", "acceleration", "force"]
        )]
        if not mech_keys:
            return None

        pressure_val = None
        seismic_mag = None
        vibration_amps = []

        for k in sorted(mech_keys):
            v = data[k]
            kl = k.lower()
            if "pressure" in kl:
                pressure_val = v
            elif "magnitude" in kl or "seismic" in kl:
                seismic_mag = v
            elif "vibration" in kl or "acceleration" in kl:
                vibration_amps.append(v)

        return MechanicalFingerprint(
            device_id=device_id,
            sensor_id=sensor_id,
            timestamp=ts,
            raw_hash=raw_hash,
            pressure_pa=pressure_val,
            seismic_magnitude=seismic_mag,
            vibration_amplitudes=np.array(vibration_amps, dtype=np.float32) if vibration_amps else np.array([]),
            confidence=0.8,
        )
