"""
NLM Sensory Fingerprint Types

Six modality-specific fingerprint types representing how NLM perceives
raw physical reality. These are continuous vector representations —
not symbolic tokens. Bio-tokens become a downstream discretization.

The six senses of NLM:
1. Spectral   — wavelengths / spectral bins (sight)
2. Acoustic   — frequency-energy / waveforms (hearing)
3. Bioelectric — voltage / current / impedance (electroception)
4. Thermal    — temperature gradients / heat flux (thermoception)
5. Chemical   — gas vectors / pH / conductivity (smell/taste)
6. Mechanical — pressure / vibration / strain (touch)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np


class FingerprintType(str, Enum):
    SPECTRAL = "spectral"
    ACOUSTIC = "acoustic"
    BIOELECTRIC = "bioelectric"
    THERMAL = "thermal"
    CHEMICAL = "chemical"
    MECHANICAL = "mechanical"


@dataclass
class SensoryFingerprint:
    """
    Base class for all sensory fingerprints.

    Every fingerprint records: what modality, when, from which device,
    with what confidence, and the raw content hash for provenance.
    """

    fingerprint_id: UUID = field(default_factory=uuid4)
    fingerprint_type: FingerprintType = FingerprintType.SPECTRAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    device_id: str = ""
    sensor_id: str = ""
    confidence: float = 1.0
    raw_hash: bytes = b""  # SHA-256 of the raw data this was extracted from

    def vector(self) -> np.ndarray:
        """Return the fingerprint as a flat numeric vector. Override in subclasses."""
        raise NotImplementedError


# ── 1. Spectral Fingerprint ─────────────────────────────────────────


@dataclass
class SpectralFingerprint(SensoryFingerprint):
    """
    NLM sees in wavelengths and spectral power distributions.

    Captures: spectral bins across UV/VIS/NIR/SWIR/thermal-IR,
    peak wavelengths, bandwidth, and full spectral power distribution.
    """

    fingerprint_type: FingerprintType = FingerprintType.SPECTRAL

    # Spectral power distribution: wavelength_nm -> power
    wavelength_bins_nm: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_power: np.ndarray = field(default_factory=lambda: np.array([]))

    # Derived peaks
    peak_wavelengths_nm: List[float] = field(default_factory=list)
    bandwidth_nm: float = 0.0

    # Band indices (e.g., NDVI, EVI)
    band_indices: Dict[str, float] = field(default_factory=dict)

    def vector(self) -> np.ndarray:
        if len(self.spectral_power) > 0:
            return self.spectral_power.astype(np.float32)
        return np.array([], dtype=np.float32)


# ── 2. Acoustic Fingerprint ─────────────────────────────────────────


@dataclass
class AcousticFingerprint(SensoryFingerprint):
    """
    NLM hears in frequency-energy distributions and waveforms.

    Captures: frequency bins, energy per bin, spectral centroid,
    harmonic structure, and a compact waveform digest.
    """

    fingerprint_type: FingerprintType = FingerprintType.ACOUSTIC

    # Frequency-energy distribution
    frequency_bins_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_distribution: np.ndarray = field(default_factory=lambda: np.array([]))

    # Derived features
    peak_frequencies_hz: List[float] = field(default_factory=list)
    spectral_centroid_hz: float = 0.0
    spectral_bandwidth_hz: float = 0.0
    harmonic_ratios: List[float] = field(default_factory=list)

    # Compact waveform digest (e.g., MFCC or learned embedding)
    waveform_digest: np.ndarray = field(default_factory=lambda: np.array([]))

    # Duration of the captured audio window
    duration_seconds: float = 0.0
    sample_rate_hz: int = 0

    def vector(self) -> np.ndarray:
        parts = []
        if len(self.energy_distribution) > 0:
            parts.append(self.energy_distribution)
        if len(self.waveform_digest) > 0:
            parts.append(self.waveform_digest)
        if parts:
            return np.concatenate(parts).astype(np.float32)
        return np.array([], dtype=np.float32)


# ── 3. Bioelectric Fingerprint ──────────────────────────────────────


@dataclass
class BioelectricFingerprint(SensoryFingerprint):
    """
    NLM senses voltage, current, and impedance.

    Captures: multi-channel voltage/current readings, impedance
    spectral profile, electrode configuration.
    """

    fingerprint_type: FingerprintType = FingerprintType.BIOELECTRIC

    # Raw electrical measurements (per channel)
    channel_labels: List[str] = field(default_factory=list)
    voltages_mv: np.ndarray = field(default_factory=lambda: np.array([]))
    currents_ua: np.ndarray = field(default_factory=lambda: np.array([]))
    resistances_kohm: np.ndarray = field(default_factory=lambda: np.array([]))

    # Impedance spectral profile (frequency -> complex impedance)
    impedance_frequencies_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    impedance_magnitude: np.ndarray = field(default_factory=lambda: np.array([]))
    impedance_phase_deg: np.ndarray = field(default_factory=lambda: np.array([]))

    # Electrode configuration
    electrode_config: str = ""  # e.g., "bipolar_2ch", "tetrapolar_4ch"

    def vector(self) -> np.ndarray:
        parts = []
        for arr in [self.voltages_mv, self.currents_ua, self.resistances_kohm,
                     self.impedance_magnitude]:
            if len(arr) > 0:
                parts.append(arr)
        if parts:
            return np.concatenate(parts).astype(np.float32)
        return np.array([], dtype=np.float32)


# ── 4. Thermal Fingerprint ──────────────────────────────────────────


@dataclass
class ThermalFingerprint(SensoryFingerprint):
    """
    NLM senses temperature gradients and heat flux.

    Captures: spatial temperature field, gradient vectors,
    heat flux magnitude and direction.
    """

    fingerprint_type: FingerprintType = FingerprintType.THERMAL

    # Spatial temperature field (flattened grid or point measurements)
    temperatures_celsius: np.ndarray = field(default_factory=lambda: np.array([]))
    measurement_positions: np.ndarray = field(default_factory=lambda: np.array([]))  # Nx3

    # Gradient (computed from spatial field)
    gradient_magnitude: float = 0.0
    gradient_direction: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Heat flux
    heat_flux_w_per_m2: float = 0.0
    heat_flux_direction: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Thermal map digest (compact representation of full thermal image)
    thermal_map_digest: np.ndarray = field(default_factory=lambda: np.array([]))

    def vector(self) -> np.ndarray:
        parts = [np.array([self.gradient_magnitude, self.heat_flux_w_per_m2])]
        if len(self.temperatures_celsius) > 0:
            parts.append(self.temperatures_celsius)
        if len(self.thermal_map_digest) > 0:
            parts.append(self.thermal_map_digest)
        return np.concatenate(parts).astype(np.float32)


# ── 5. Chemical Fingerprint ─────────────────────────────────────────


@dataclass
class ChemicalFingerprint(SensoryFingerprint):
    """
    NLM smells via gas vectors and chemical concentrations.

    Captures: VOC/VSC gas concentration vectors, pH, electrical
    conductivity, humidity/moisture, CO2, and other dissolved or
    airborne chemical species.
    """

    fingerprint_type: FingerprintType = FingerprintType.CHEMICAL

    # Gas vectors (named species -> concentration in ppm or ppb)
    gas_concentrations_ppm: Dict[str, float] = field(default_factory=dict)

    # VOC/VSC summary vector (sensor array response)
    voc_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    vsc_vector: np.ndarray = field(default_factory=lambda: np.array([]))

    # Solution chemistry
    ph: Optional[float] = None
    electrical_conductivity_us_cm: Optional[float] = None
    dissolved_oxygen_mg_l: Optional[float] = None

    # Atmospheric
    co2_ppm: Optional[float] = None
    humidity_percent: Optional[float] = None

    def vector(self) -> np.ndarray:
        parts = []
        if self.gas_concentrations_ppm:
            parts.append(np.array(sorted(self.gas_concentrations_ppm.values())))
        if len(self.voc_vector) > 0:
            parts.append(self.voc_vector)
        if len(self.vsc_vector) > 0:
            parts.append(self.vsc_vector)
        scalars = []
        for v in [self.ph, self.electrical_conductivity_us_cm,
                   self.dissolved_oxygen_mg_l, self.co2_ppm, self.humidity_percent]:
            if v is not None:
                scalars.append(v)
        if scalars:
            parts.append(np.array(scalars))
        if parts:
            return np.concatenate(parts).astype(np.float32)
        return np.array([], dtype=np.float32)


# ── 6. Mechanical Fingerprint ───────────────────────────────────────


@dataclass
class MechanicalFingerprint(SensoryFingerprint):
    """
    NLM senses pressure, touch, and vibration.

    Captures: static pressure, vibration frequency spectrum,
    strain measurements, seismic waveform digest.
    """

    fingerprint_type: FingerprintType = FingerprintType.MECHANICAL

    # Pressure
    pressure_pa: Optional[float] = None
    differential_pressure_pa: Optional[float] = None

    # Vibration spectrum
    vibration_frequencies_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    vibration_amplitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    dominant_frequency_hz: float = 0.0

    # Strain
    strain_measurements: np.ndarray = field(default_factory=lambda: np.array([]))
    strain_gauge_positions: List[str] = field(default_factory=list)

    # Seismic
    seismic_magnitude: Optional[float] = None
    seismic_waveform_digest: np.ndarray = field(default_factory=lambda: np.array([]))

    def vector(self) -> np.ndarray:
        parts = []
        scalars = []
        for v in [self.pressure_pa, self.differential_pressure_pa,
                   self.dominant_frequency_hz, self.seismic_magnitude]:
            if v is not None:
                scalars.append(v)
        if scalars:
            parts.append(np.array(scalars))
        for arr in [self.vibration_amplitudes, self.strain_measurements,
                     self.seismic_waveform_digest]:
            if len(arr) > 0:
                parts.append(arr)
        if parts:
            return np.concatenate(parts).astype(np.float32)
        return np.array([], dtype=np.float32)
