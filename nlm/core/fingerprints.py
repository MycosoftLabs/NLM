"""
Sensory Fingerprint Types
=========================

Non-language representations of measured physical reality.
The NLM thinks in wavelengths, spectra, waveforms, concentrations,
gradients, voltages, and state transitions — not text.

Each fingerprint captures a specific sensory modality in its native
physical units before any lossy projection into language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


@dataclass
class SpectralFingerprint:
    """Electromagnetic spectral decomposition.

    Represents light, infrared, UV, or radio signals as energy
    distributed across wavelength/frequency bins.
    """

    wavelength_bins: List[float]  # nm or Hz bin edges
    energy_values: List[float]  # intensity per bin
    source_type: str  # "optical", "infrared", "uv", "radio", "microwave"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    device_id: str = ""
    integration_time_ms: float = 0.0
    calibration_ref: str = ""

    def num_bins(self) -> int:
        return len(self.wavelength_bins)

    def peak_wavelength(self) -> float:
        if not self.energy_values:
            return 0.0
        idx = max(range(len(self.energy_values)), key=lambda i: self.energy_values[i])
        return self.wavelength_bins[idx] if idx < len(self.wavelength_bins) else 0.0

    def total_energy(self) -> float:
        return sum(self.energy_values)

    def to_dict(self) -> Dict:
        return {
            "type": "spectral",
            "wavelength_bins": self.wavelength_bins,
            "energy_values": self.energy_values,
            "source_type": self.source_type,
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
            "peak_wavelength": self.peak_wavelength(),
            "total_energy": self.total_energy(),
        }


@dataclass
class AcousticFingerprint:
    """Sound / vibration in frequency-energy space.

    Stores STFT-style frequency decomposition, not transcripts.
    """

    frequency_bins: List[float]  # Hz bin edges
    magnitude: List[float]  # dB per bin
    waveform_ref: str = ""  # URI to raw PCM data
    duration_ms: float = 0.0
    sample_rate_hz: int = 44100
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    device_id: str = ""

    def dominant_frequency(self) -> float:
        if not self.magnitude:
            return 0.0
        idx = max(range(len(self.magnitude)), key=lambda i: self.magnitude[i])
        return self.frequency_bins[idx] if idx < len(self.frequency_bins) else 0.0

    def bandwidth(self) -> float:
        if len(self.frequency_bins) < 2:
            return 0.0
        return self.frequency_bins[-1] - self.frequency_bins[0]

    def to_dict(self) -> Dict:
        return {
            "type": "acoustic",
            "frequency_bins": self.frequency_bins,
            "magnitude": self.magnitude,
            "waveform_ref": self.waveform_ref,
            "duration_ms": self.duration_ms,
            "sample_rate_hz": self.sample_rate_hz,
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
        }


@dataclass
class BioelectricFingerprint:
    """Electrical signature from biological sensors.

    Captures voltage/current time-series from FCI probes,
    bioelectric sensors, and impedance measurements.
    """

    voltage_series: List[float]  # mV time-series
    current_series: List[float]  # μA time-series
    impedance: float = 0.0  # Ω
    sample_rate_hz: int = 1000
    electrode_config: str = "bipolar"  # "bipolar", "monopolar", "tetrapolar"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    device_id: str = ""

    def mean_voltage(self) -> float:
        if not self.voltage_series:
            return 0.0
        return sum(self.voltage_series) / len(self.voltage_series)

    def voltage_range(self) -> float:
        if not self.voltage_series:
            return 0.0
        return max(self.voltage_series) - min(self.voltage_series)

    def duration_ms(self) -> float:
        if not self.voltage_series or self.sample_rate_hz == 0:
            return 0.0
        return (len(self.voltage_series) / self.sample_rate_hz) * 1000.0

    def to_dict(self) -> Dict:
        return {
            "type": "bioelectric",
            "voltage_samples": len(self.voltage_series),
            "current_samples": len(self.current_series),
            "impedance_ohm": self.impedance,
            "sample_rate_hz": self.sample_rate_hz,
            "electrode_config": self.electrode_config,
            "mean_voltage_mv": self.mean_voltage(),
            "voltage_range_mv": self.voltage_range(),
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
        }


@dataclass
class ThermalFingerprint:
    """Temperature field and heat transport.

    Not a single temperature reading — captures spatial gradients,
    heat flux, and radiative properties.
    """

    temperature_field: List[List[float]]  # 2D grid in °C
    gradient_magnitude: float = 0.0  # °C/m
    gradient_direction: float = 0.0  # radians
    heat_flux: float = 0.0  # W/m²
    emissivity: float = 0.95
    ambient_temperature: float = 20.0  # °C
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    device_id: str = ""

    def mean_temperature(self) -> float:
        if not self.temperature_field:
            return 0.0
        values = [v for row in self.temperature_field for v in row]
        return sum(values) / len(values) if values else 0.0

    def max_temperature(self) -> float:
        if not self.temperature_field:
            return 0.0
        return max(v for row in self.temperature_field for v in row)

    def min_temperature(self) -> float:
        if not self.temperature_field:
            return 0.0
        return min(v for row in self.temperature_field for v in row)

    def thermal_contrast(self) -> float:
        return self.max_temperature() - self.min_temperature()

    def to_dict(self) -> Dict:
        return {
            "type": "thermal",
            "grid_shape": [len(self.temperature_field), len(self.temperature_field[0])] if self.temperature_field else [0, 0],
            "gradient_magnitude": self.gradient_magnitude,
            "heat_flux_w_m2": self.heat_flux,
            "emissivity": self.emissivity,
            "mean_temp_c": self.mean_temperature(),
            "thermal_contrast_c": self.thermal_contrast(),
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
        }


@dataclass
class ChemicalFingerprint:
    """Chemical composition and concentration vector.

    Encodes VOC/VSC gas concentrations, pH, conductivity, ion profiles,
    and molecular embeddings from ChemistryEncoder.
    """

    compound_vector: List[float] = field(default_factory=list)  # from ChemistryEncoder (128D)
    voc_concentrations: Dict[str, float] = field(default_factory=dict)  # ppb per compound
    vsc_concentrations: Dict[str, float] = field(default_factory=dict)  # ppb per compound
    ph: float = 7.0
    conductivity: float = 0.0  # μS/cm
    dissolved_oxygen: float = 0.0  # mg/L
    ion_concentrations: Dict[str, float] = field(default_factory=dict)  # mg/L per ion
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    device_id: str = ""

    def total_voc(self) -> float:
        return sum(self.voc_concentrations.values())

    def total_vsc(self) -> float:
        return sum(self.vsc_concentrations.values())

    def is_acidic(self) -> bool:
        return self.ph < 7.0

    def to_dict(self) -> Dict:
        return {
            "type": "chemical",
            "compound_vector_dim": len(self.compound_vector),
            "voc_species": list(self.voc_concentrations.keys()),
            "total_voc_ppb": self.total_voc(),
            "ph": self.ph,
            "conductivity_us_cm": self.conductivity,
            "dissolved_oxygen_mg_l": self.dissolved_oxygen,
            "ion_species": list(self.ion_concentrations.keys()),
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
        }


@dataclass
class MechanicalFingerprint:
    """Pressure, vibration, force, and strain.

    Captures physical contact, substrate conditions, and mechanical
    response fields from pressure sensors and accelerometers.
    """

    pressure_pa: float = 101325.0  # atmospheric default
    vibration_spectrum: List[float] = field(default_factory=list)  # magnitude per freq bin
    vibration_freq_bins: List[float] = field(default_factory=list)  # Hz bin edges
    force_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # N (x, y, z)
    strain: float = 0.0  # dimensionless
    moisture_content: float = 0.0  # % by weight
    substrate_density: float = 0.0  # kg/m³
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    device_id: str = ""

    def force_magnitude(self) -> float:
        return (self.force_vector[0] ** 2 + self.force_vector[1] ** 2 + self.force_vector[2] ** 2) ** 0.5

    def dominant_vibration_freq(self) -> float:
        if not self.vibration_spectrum:
            return 0.0
        idx = max(range(len(self.vibration_spectrum)), key=lambda i: self.vibration_spectrum[i])
        return self.vibration_freq_bins[idx] if idx < len(self.vibration_freq_bins) else 0.0

    def to_dict(self) -> Dict:
        return {
            "type": "mechanical",
            "pressure_pa": self.pressure_pa,
            "vibration_bins": len(self.vibration_spectrum),
            "force_magnitude_n": self.force_magnitude(),
            "strain": self.strain,
            "moisture_pct": self.moisture_content,
            "substrate_density_kg_m3": self.substrate_density,
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
        }
