"""
NLM Deterministic Preconditioning Stack

Orchestrates physics, chemistry, and biology preconditioners.
These are deterministic scientific transforms applied BEFORE any
learned model — the key architectural distinction from an LLM.

Pipeline:
  raw sensor data
    -> physics preconditioner (geomagnetic, lunar, atmospheric)
    -> chemistry preconditioner (molecular fingerprints, reaction context)
    -> biology preconditioner (lifecycle state, symbiosis context)
    -> calibrated + derived fields ready for fingerprint extraction
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from nlm.core.frames import PhysicalValue

logger = logging.getLogger(__name__)


class PhysicsPreconditioner:
    """
    Deterministic physics transforms.

    Refactored from FieldPhysicsModel — same calculations but:
    - No random noise injection
    - Returns PhysicalValue objects with SI units
    - Pure functions, no side effects
    """

    # IGRF-13 coefficients (epoch 2020)
    IGRF_G10 = -29404.8  # nT
    IGRF_G11 = -1450.9
    IGRF_H11 = 4652.5

    EARTH_RADIUS_KM = 6371.0
    G_CONST = 6.674e-11
    LUNAR_SEMI_MAJOR_KM = 384400
    LUNAR_MASS_KG = 7.342e22
    SYNODIC_PERIOD_DAYS = 29.53
    J2000_UNIX = 946728000

    def compute_geomagnetic_field(
        self, lat: float, lon: float, alt_m: float
    ) -> Dict[str, PhysicalValue]:
        """Compute geomagnetic field vector. No noise — deterministic."""
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        r = self.EARTH_RADIUS_KM + alt_m / 1000
        scale = (self.EARTH_RADIUS_KM / r) ** 3

        cos_lat = math.cos(lat_rad)
        sin_lat = math.sin(lat_rad)
        cos_lon = math.cos(lon_rad)
        sin_lon = math.sin(lon_rad)

        Br = scale * (2 * self.IGRF_G10 * sin_lat +
                      (self.IGRF_G11 * cos_lon + self.IGRF_H11 * sin_lon) * cos_lat)
        Btheta = scale * (self.IGRF_G10 * cos_lat -
                          (self.IGRF_G11 * cos_lon + self.IGRF_H11 * sin_lon) * sin_lat)
        Bphi = scale * (self.IGRF_G11 * sin_lon - self.IGRF_H11 * cos_lon)

        Bx = -Btheta  # North
        By = Bphi     # East
        Bz = -Br      # Down

        Bh = math.sqrt(Bx**2 + By**2)
        Bf = math.sqrt(Bx**2 + By**2 + Bz**2)
        inclination = math.degrees(math.atan2(Bz, Bh))
        declination = math.degrees(math.atan2(By, Bx))

        return {
            "geomag_Bx_nT": PhysicalValue(value=round(Bx, 1), unit="nT"),
            "geomag_By_nT": PhysicalValue(value=round(By, 1), unit="nT"),
            "geomag_Bz_nT": PhysicalValue(value=round(Bz, 1), unit="nT"),
            "geomag_total_nT": PhysicalValue(value=round(Bf, 1), unit="nT"),
            "geomag_inclination_deg": PhysicalValue(value=round(inclination, 2), unit="degrees"),
            "geomag_declination_deg": PhysicalValue(value=round(declination, 2), unit="degrees"),
        }

    def compute_lunar_influence(
        self, lat: float, timestamp: float
    ) -> Dict[str, PhysicalValue]:
        """Compute lunar phase, gravitational acceleration, and tidal potential."""
        days_since_j2000 = (timestamp - self.J2000_UNIX) / 86400
        phase_angle = (days_since_j2000 % self.SYNODIC_PERIOD_DAYS) / self.SYNODIC_PERIOD_DAYS * 360
        illumination = (1 - math.cos(math.radians(phase_angle))) / 2

        moon_distance_m = self.LUNAR_SEMI_MAJOR_KM * 1000
        g_moon = self.G_CONST * self.LUNAR_MASS_KG / moon_distance_m**2
        lat_rad = math.radians(lat)
        tidal_factor = math.cos(2 * lat_rad)
        tidal_potential = g_moon * self.EARTH_RADIUS_KM * 1000 * tidal_factor

        return {
            "lunar_phase_angle_deg": PhysicalValue(value=round(phase_angle, 1), unit="degrees"),
            "lunar_illumination": PhysicalValue(value=round(illumination, 3), unit="fraction"),
            "lunar_g_accel_m_s2": PhysicalValue(value=g_moon, unit="m/s²"),
            "lunar_tidal_potential": PhysicalValue(value=round(tidal_potential, 6), unit="m²/s²"),
        }

    def compute_atmospheric_baseline(
        self, lat: float, alt_m: float, timestamp: float
    ) -> Dict[str, PhysicalValue]:
        """
        Compute baseline atmospheric conditions from location and time.

        Deterministic — no random noise. Real sensor data overrides these.
        """
        base_temp = 25 - abs(lat) * 0.5
        day_of_year = (timestamp / 86400) % 365.25
        seasonal = 10 * math.sin(2 * math.pi * (day_of_year - 80) / 365.25)
        if lat < 0:
            seasonal *= -1
        altitude_effect = -6.5 * alt_m / 1000
        temperature = base_temp + seasonal + altitude_effect

        sea_level_pressure = 1013.25
        pressure = sea_level_pressure * math.exp(-alt_m / 8500)

        base_humidity = 70 - abs(lat - 15) * 0.5
        humidity = max(30.0, min(100.0, base_humidity))

        dew_point = temperature - (100 - humidity) / 5

        return {
            "atmo_baseline_temp_c": PhysicalValue(value=round(temperature, 1), unit="celsius"),
            "atmo_baseline_pressure_hpa": PhysicalValue(value=round(pressure, 1), unit="hPa"),
            "atmo_baseline_humidity_pct": PhysicalValue(value=round(humidity, 1), unit="percent"),
            "atmo_baseline_dewpoint_c": PhysicalValue(value=round(dew_point, 1), unit="celsius"),
        }

    def precondition(
        self, lat: float, lon: float, alt_m: float, timestamp: Optional[float] = None
    ) -> Dict[str, PhysicalValue]:
        """Run all physics preconditioners. Returns merged dict of PhysicalValues."""
        ts = timestamp or time.time()
        result: Dict[str, PhysicalValue] = {}
        result.update(self.compute_geomagnetic_field(lat, lon, alt_m))
        result.update(self.compute_lunar_influence(lat, ts))
        result.update(self.compute_atmospheric_baseline(lat, alt_m, ts))
        return result


class ChemistryPreconditioner:
    """
    Deterministic chemistry transforms.

    Wraps the existing ChemistryEncoder for use in the preconditioning stack.
    Produces derived chemical fields from raw concentration data.
    """

    def precondition(
        self, raw_chemistry: Dict[str, float]
    ) -> Dict[str, PhysicalValue]:
        """Derive chemical fields from raw concentration data."""
        result: Dict[str, PhysicalValue] = {}

        ph = raw_chemistry.get("ph")
        if ph is not None:
            result["ph"] = PhysicalValue(value=ph, unit="pH")
            # Derive H+ concentration
            h_plus = 10 ** (-ph)
            result["h_plus_mol_l"] = PhysicalValue(value=h_plus, unit="mol/L")

        ec = raw_chemistry.get("electrical_conductivity")
        if ec is not None:
            result["electrical_conductivity"] = PhysicalValue(value=ec, unit="µS/cm")
            # Approximate TDS from EC
            tds = ec * 0.64
            result["tds_approx_mg_l"] = PhysicalValue(value=round(tds, 1), unit="mg/L")

        co2 = raw_chemistry.get("co2_ppm")
        if co2 is not None:
            result["co2_ppm"] = PhysicalValue(value=co2, unit="ppm")

        do = raw_chemistry.get("dissolved_oxygen")
        if do is not None:
            result["dissolved_oxygen_mg_l"] = PhysicalValue(value=do, unit="mg/L")

        return result


class BiologyPreconditioner:
    """
    Deterministic biology transforms.

    Wraps lifecycle and symbiosis modules. Produces derived biological
    fields from environmental state.
    """

    def precondition(
        self,
        temperature_c: Optional[float] = None,
        humidity_pct: Optional[float] = None,
        substrate_moisture_pct: Optional[float] = None,
    ) -> Dict[str, PhysicalValue]:
        """Derive biological context fields."""
        result: Dict[str, PhysicalValue] = {}

        if temperature_c is not None:
            # Optimal fungal growth range assessment
            if 18 <= temperature_c <= 26:
                growth_potential = 1.0
            elif 10 <= temperature_c <= 35:
                # Linear ramp outside optimal
                if temperature_c < 18:
                    growth_potential = (temperature_c - 10) / 8
                else:
                    growth_potential = (35 - temperature_c) / 9
            else:
                growth_potential = 0.0
            result["fungal_growth_potential"] = PhysicalValue(
                value=round(growth_potential, 3), unit="fraction"
            )

        if humidity_pct is not None and temperature_c is not None:
            # Vapor pressure deficit (simplified)
            sat_vp = 0.6108 * math.exp(17.27 * temperature_c / (temperature_c + 237.3))
            actual_vp = sat_vp * humidity_pct / 100
            vpd = sat_vp - actual_vp
            result["vapor_pressure_deficit_kpa"] = PhysicalValue(
                value=round(vpd, 3), unit="kPa"
            )

        if substrate_moisture_pct is not None:
            # Water activity approximation
            water_activity = min(1.0, substrate_moisture_pct / 100 * 1.05)
            result["water_activity"] = PhysicalValue(
                value=round(water_activity, 3), unit="aw"
            )

        return result


class HydroacousticPreconditioner:
    """Deterministic physics transforms for underwater sound propagation.

    Applied before NLM encoding for TAC-O maritime acoustic processing.
    """

    @staticmethod
    def compute_sound_speed(temp_c: float, salinity_psu: float, depth_m: float) -> float:
        """Mackenzie equation for sound speed in seawater (m/s)."""
        c = (1448.96 + 4.591 * temp_c - 0.05304 * temp_c**2
             + 2.374e-4 * temp_c**3 + 1.340 * (salinity_psu - 35)
             + 1.630e-2 * depth_m + 1.675e-7 * depth_m**2
             - 1.025e-2 * temp_c * (salinity_psu - 35)
             - 7.139e-13 * temp_c * depth_m**3)
        return c

    @staticmethod
    def transmission_loss(range_m: float, frequency_hz: float) -> float:
        """Spherical spreading + Thorp absorption loss in dB."""
        spreading = 20 * math.log10(max(range_m, 1.0))
        f_khz = frequency_hz / 1000.0
        alpha = (0.11 * f_khz**2 / (1 + f_khz**2)
                + 44 * f_khz**2 / (4100 + f_khz**2)
                + 2.75e-4 * f_khz**2 + 0.003)
        absorption = alpha * range_m / 1000.0
        return spreading + absorption

    @staticmethod
    def ray_trace_simple(
        ssp: List[Tuple[float, float]], source_depth: float,
        initial_angle_deg: float, max_range: float, step_m: float = 10.0
    ) -> List[Tuple[float, float]]:
        """Simple ray tracing through a sound speed profile using Snell's law.

        Args:
            ssp: List of (depth_m, speed_m_s) pairs (sorted by depth).
            source_depth: Source depth in meters.
            initial_angle_deg: Initial ray angle in degrees from horizontal.
            max_range: Maximum horizontal range in meters.
            step_m: Step size in meters.

        Returns:
            List of (range_m, depth_m) points along the ray path.
        """
        if len(ssp) < 2:
            return [(0.0, source_depth)]

        angle_rad = math.radians(initial_angle_deg)
        depths = [d for d, _ in ssp]
        speeds = [s for _, s in ssp]

        def interp_speed(d: float) -> float:
            if d <= depths[0]:
                return speeds[0]
            if d >= depths[-1]:
                return speeds[-1]
            for i in range(len(depths) - 1):
                if depths[i] <= d <= depths[i + 1]:
                    frac = (d - depths[i]) / (depths[i + 1] - depths[i])
                    return speeds[i] + frac * (speeds[i + 1] - speeds[i])
            return speeds[-1]

        c0 = interp_speed(source_depth)
        cos_theta_over_c = math.cos(angle_rad) / c0

        path = [(0.0, source_depth)]
        r, d = 0.0, source_depth
        while r < max_range and 0 < d < (depths[-1] + 100):
            c_local = interp_speed(d)
            cos_theta = cos_theta_over_c * c_local
            cos_theta = max(-1.0, min(1.0, cos_theta))
            sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta**2))
            dr = step_m * cos_theta
            dd = step_m * sin_theta
            if angle_rad > 0:
                dd = -dd
            r += abs(dr)
            d += dd
            if d < 0:
                d = 0.0
                angle_rad = -angle_rad
            path.append((r, d))
        return path

    @staticmethod
    def calibrate_magnetometer(
        raw_bx: float, raw_by: float, raw_bz: float,
        hard_iron: Tuple[float, float, float],
        soft_iron_matrix: Optional[List[List[float]]] = None,
    ) -> Tuple[float, float, float]:
        """Hard/soft iron calibration for magnetic sensor data."""
        bx = raw_bx - hard_iron[0]
        by = raw_by - hard_iron[1]
        bz = raw_bz - hard_iron[2]
        if soft_iron_matrix is not None:
            si = soft_iron_matrix
            cal_bx = si[0][0] * bx + si[0][1] * by + si[0][2] * bz
            cal_by = si[1][0] * bx + si[1][1] * by + si[1][2] * bz
            cal_bz = si[2][0] * bx + si[2][1] * by + si[2][2] * bz
            return (cal_bx, cal_by, cal_bz)
        return (bx, by, bz)

    def precondition(
        self, raw_data: Dict[str, Any]
    ) -> Dict[str, PhysicalValue]:
        """Run hydroacoustic preconditioning on raw maritime sensor data."""
        result: Dict[str, PhysicalValue] = {}

        temp = raw_data.get("temperature_c", 15.0)
        sal = raw_data.get("salinity_psu", 35.0)
        depth = raw_data.get("depth_m", 0.0)

        ss = self.compute_sound_speed(float(temp), float(sal), float(depth))
        result["sound_speed_m_s"] = PhysicalValue(value=round(ss, 2), unit="m/s")

        freq = raw_data.get("frequency_hz")
        rng = raw_data.get("range_m")
        if freq is not None and rng is not None:
            tl = self.transmission_loss(float(rng), float(freq))
            result["transmission_loss_dB"] = PhysicalValue(value=round(tl, 2), unit="dB")

        return result


class DeterministicPreconditioningStack:
    """
    Orchestrates all preconditioners in the correct order.

    Usage:
        stack = DeterministicPreconditioningStack()
        derived = stack.precondition(raw_data, location)
    """

    def __init__(self):
        self.physics = PhysicsPreconditioner()
        self.chemistry = ChemistryPreconditioner()
        self.biology = BiologyPreconditioner()
        self.hydroacoustic = HydroacousticPreconditioner()

    def precondition(
        self,
        raw_data: Dict[str, Any],
        lat: float = 0.0,
        lon: float = 0.0,
        alt_m: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> Dict[str, PhysicalValue]:
        """
        Run the full deterministic preconditioning stack.

        Returns all derived PhysicalValues from physics + chemistry + biology.
        """
        result: Dict[str, PhysicalValue] = {}

        # 1. Physics
        result.update(self.physics.precondition(lat, lon, alt_m, timestamp))

        # 2. Chemistry
        chem_data: Dict[str, float] = {}
        for key in ["ph", "electrical_conductivity", "co2_ppm", "dissolved_oxygen"]:
            if key in raw_data and raw_data[key] is not None:
                try:
                    chem_data[key] = float(raw_data[key])
                except (TypeError, ValueError):
                    pass
        if chem_data:
            result.update(self.chemistry.precondition(chem_data))

        # 3. Biology
        temp = raw_data.get("temperature_c") or raw_data.get("temperature")
        hum = raw_data.get("humidity_pct") or raw_data.get("humidity")
        moist = raw_data.get("moisture_pct") or raw_data.get("substrate_moisture")
        result.update(self.biology.precondition(
            temperature_c=float(temp) if temp is not None else None,
            humidity_pct=float(hum) if hum is not None else None,
            substrate_moisture_pct=float(moist) if moist is not None else None,
        ))

        # 4. Hydroacoustic (if maritime data present)
        maritime_keys = {"temperature_c", "salinity_psu", "depth_m", "frequency_hz", "range_m"}
        if maritime_keys & set(raw_data.keys()):
            result.update(self.hydroacoustic.precondition(raw_data))

        return result
