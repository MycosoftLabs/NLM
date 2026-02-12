"""
Field Physics Model
===================

Models environmental physics fields that influence fungal behavior:
- Geomagnetic field (IGRF model approximation)
- Lunar gravitational influence
- Atmospheric conditions
- Barometric pressure effects

These fields correlate with fungal fruiting events and can be used
for predictive cultivation timing.

Usage:
    model = FieldPhysicsModel()
    geo = model.get_geomagnetic_field((lat, lon, alt), timestamp)
    lunar = model.get_lunar_gravitational_influence((lat, lon, alt), timestamp)
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np


class FieldPhysicsModel:
    """
    Models environmental physics fields for fungal cultivation optimization.
    
    Integrates geomagnetic, lunar, and atmospheric data to predict
    optimal fruiting conditions.
    """
    
    # Geomagnetic reference values (IGRF-13 epoch 2020)
    IGRF_COEFFICIENTS = {
        "g10": -29404.8,  # nT
        "g11": -1450.9,
        "h11": 4652.5,
        "g20": -2499.6,
    }
    
    # Lunar orbital parameters
    LUNAR_PARAMS = {
        "semi_major_axis_km": 384400,
        "eccentricity": 0.0549,
        "synodic_period_days": 29.53,
        "mass_kg": 7.342e22,
    }
    
    def __init__(self):
        """Initialize the Field Physics Model."""
        self.earth_radius_km = 6371.0
        self.G = 6.674e-11  # Gravitational constant
        print("Initialized FieldPhysicsModel")
    
    def get_geomagnetic_field(
        self,
        location: Tuple[float, float, float],
        timestamp: float,
    ) -> Dict[str, Any]:
        """
        Calculate geomagnetic field vector at a given location and time.
        
        Uses a simplified IGRF model for demonstration.
        
        Args:
            location: (latitude, longitude, altitude) in degrees and meters
            timestamp: Unix timestamp
        
        Returns:
            Dictionary with Bx, By, Bz components, total field, inclination, declination
        """
        lat, lon, alt = location
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        # Simplified dipole field calculation
        r = self.earth_radius_km + alt / 1000  # Convert alt to km
        
        # Get coefficients
        g10 = self.IGRF_COEFFICIENTS["g10"]
        g11 = self.IGRF_COEFFICIENTS["g11"]
        h11 = self.IGRF_COEFFICIENTS["h11"]
        
        # Calculate field components in local coordinates
        # North (X), East (Y), Down (Z) convention
        cos_lat = math.cos(lat_rad)
        sin_lat = math.sin(lat_rad)
        cos_lon = math.cos(lon_rad)
        sin_lon = math.sin(lon_rad)
        
        # Radial scaling
        scale = (self.earth_radius_km / r) ** 3
        
        # Field components (simplified dipole + small multipole)
        Br = scale * (2 * g10 * sin_lat + (g11 * cos_lon + h11 * sin_lon) * cos_lat)
        Btheta = scale * (g10 * cos_lat - (g11 * cos_lon + h11 * sin_lon) * sin_lat)
        Bphi = scale * (g11 * sin_lon - h11 * cos_lon)
        
        # Convert to North-East-Down
        Bx = -Btheta  # North
        By = Bphi     # East
        Bz = -Br      # Down
        
        # Add some realistic noise/variation
        noise_scale = 50  # nT
        Bx += np.random.normal(0, noise_scale)
        By += np.random.normal(0, noise_scale)
        Bz += np.random.normal(0, noise_scale)
        
        # Calculate derived quantities
        Bh = math.sqrt(Bx**2 + By**2)  # Horizontal intensity
        Bf = math.sqrt(Bx**2 + By**2 + Bz**2)  # Total field
        
        inclination = math.degrees(math.atan2(Bz, Bh))
        declination = math.degrees(math.atan2(By, Bx))
        
        return {
            "Bx": round(Bx, 1),
            "By": round(By, 1),
            "Bz": round(Bz, 1),
            "TotalField": round(Bf, 1),
            "HorizontalIntensity": round(Bh, 1),
            "Inclination": round(inclination, 2),
            "Declination": round(declination, 2),
            "unit": "nT",
            "model": "IGRF-13 (simplified)",
            "message": "Geomagnetic field calculated",
        }
    
    def get_lunar_gravitational_influence(
        self,
        location: Tuple[float, float, float],
        timestamp: float,
    ) -> Dict[str, Any]:
        """
        Calculate lunar gravitational force and tidal potential.
        
        Args:
            location: (latitude, longitude, altitude)
            timestamp: Unix timestamp
        
        Returns:
            Dictionary with gravitational force, tidal potential, phase info
        """
        lat, lon, alt = location
        
        # Calculate lunar phase from timestamp
        # Reference: New Moon on Jan 6, 2000 (J2000)
        j2000 = 946728000  # Unix timestamp for J2000
        days_since_j2000 = (timestamp - j2000) / 86400
        
        synodic_period = self.LUNAR_PARAMS["synodic_period_days"]
        phase_angle = (days_since_j2000 % synodic_period) / synodic_period * 360
        
        # Determine phase name
        if phase_angle < 45:
            phase_name = "new_moon"
        elif phase_angle < 90:
            phase_name = "waxing_crescent"
        elif phase_angle < 135:
            phase_name = "first_quarter"
        elif phase_angle < 180:
            phase_name = "waxing_gibbous"
        elif phase_angle < 225:
            phase_name = "full_moon"
        elif phase_angle < 270:
            phase_name = "waning_gibbous"
        elif phase_angle < 315:
            phase_name = "third_quarter"
        else:
            phase_name = "waning_crescent"
        
        illumination = (1 - math.cos(math.radians(phase_angle))) / 2
        
        # Calculate approximate gravitational acceleration
        moon_distance = self.LUNAR_PARAMS["semi_major_axis_km"] * 1000  # meters
        moon_mass = self.LUNAR_PARAMS["mass_kg"]
        
        g_moon = self.G * moon_mass / moon_distance**2  # m/s²
        
        # Tidal force varies with position on Earth
        lat_rad = math.radians(lat)
        tidal_factor = math.cos(2 * lat_rad)  # Maximum at equator
        
        # Tidal potential (simplified)
        tidal_potential = g_moon * self.earth_radius_km * 1000 * tidal_factor
        
        return {
            "phase": phase_name,
            "phase_angle": round(phase_angle, 1),
            "illumination": round(illumination, 3),
            "gravitational_acceleration": f"{g_moon:.2e}",
            "tidal_potential": round(tidal_potential, 6),
            "tidal_factor": round(tidal_factor, 3),
            "moon_distance_km": round(moon_distance / 1000),
            "message": "Lunar gravitational influence calculated",
        }
    
    def get_atmospheric_conditions(
        self,
        location: Tuple[float, float, float],
        timestamp: float,
    ) -> Dict[str, Any]:
        """
        Get or simulate atmospheric conditions.
        
        Args:
            location: (latitude, longitude, altitude)
            timestamp: Unix timestamp
        
        Returns:
            Dictionary with temperature, pressure, humidity, wind
        """
        lat, lon, alt = location
        
        # Simulate based on location and time
        # Base temperature varies with latitude
        base_temp = 25 - abs(lat) * 0.5
        
        # Seasonal variation (Northern Hemisphere)
        day_of_year = (timestamp / 86400) % 365.25
        seasonal_variation = 10 * math.sin(2 * math.pi * (day_of_year - 80) / 365.25)
        if lat < 0:
            seasonal_variation *= -1
        
        # Altitude effect (-6.5°C per 1000m)
        altitude_effect = -6.5 * alt / 1000
        
        temperature = base_temp + seasonal_variation + altitude_effect + np.random.normal(0, 2)
        
        # Pressure (barometric formula)
        sea_level_pressure = 1013.25  # hPa
        pressure = sea_level_pressure * math.exp(-alt / 8500)
        pressure += np.random.normal(0, 5)
        
        # Humidity (higher in tropical regions)
        base_humidity = 70 - abs(lat - 15) * 0.5
        humidity = max(30, min(100, base_humidity + np.random.normal(0, 10)))
        
        # Wind
        wind_speed = abs(np.random.normal(5, 3))
        wind_direction = np.random.uniform(0, 360)
        
        return {
            "temperature_celsius": round(temperature, 1),
            "pressure_hpa": round(pressure, 1),
            "humidity_percent": round(humidity, 1),
            "wind_speed_mps": round(wind_speed, 1),
            "wind_direction_deg": round(wind_direction, 0),
            "dew_point_celsius": round(temperature - (100 - humidity) / 5, 1),
            "altitude_m": round(alt),
            "message": "Atmospheric conditions simulated",
        }
    
    def predict_fruiting_conditions(
        self,
        location: Tuple[float, float, float],
        timestamp: float,
        species: str = "generic",
    ) -> Dict[str, Any]:
        """
        Predict fruiting probability based on environmental conditions.
        
        Args:
            location: (latitude, longitude, altitude)
            timestamp: Current timestamp
            species: Species name for species-specific parameters
        
        Returns:
            Dictionary with fruiting probability and optimal date prediction
        """
        # Get all environmental data
        geo = self.get_geomagnetic_field(location, timestamp)
        lunar = self.get_lunar_gravitational_influence(location, timestamp)
        atmo = self.get_atmospheric_conditions(location, timestamp)
        
        # Calculate fruiting probability based on factors
        probability = 0.3  # Base probability
        
        # Temperature factor (optimal around 18-24°C for most species)
        temp = atmo["temperature_celsius"]
        if 18 <= temp <= 24:
            probability += 0.2
        elif 15 <= temp <= 28:
            probability += 0.1
        
        # Humidity factor (optimal above 80%)
        humidity = atmo["humidity_percent"]
        if humidity >= 85:
            probability += 0.2
        elif humidity >= 75:
            probability += 0.1
        
        # Pressure factor (dropping pressure often triggers fruiting)
        pressure = atmo["pressure_hpa"]
        if pressure < 1010:
            probability += 0.1
        
        # Lunar factor (full moon and new moon often associated with flushes)
        phase = lunar["phase"]
        if phase in ["full_moon", "new_moon"]:
            probability += 0.15
        elif phase in ["waxing_gibbous", "waning_gibbous"]:
            probability += 0.05
        
        # Geomagnetic factor (stable field is favorable)
        total_field = geo["TotalField"]
        if 45000 <= total_field <= 55000:
            probability += 0.05
        
        # Cap probability
        probability = min(0.95, max(0.05, probability))
        
        # Predict optimal date (simplified - next favorable moon phase)
        days_to_full = (180 - lunar["phase_angle"]) / 360 * 29.53
        if days_to_full < 0:
            days_to_full += 29.53
        
        optimal_timestamp = timestamp + days_to_full * 86400
        optimal_date = datetime.fromtimestamp(optimal_timestamp).isoformat()
        
        return {
            "probability": round(probability, 2),
            "confidence": 0.7,
            "optimal_date": optimal_date,
            "days_to_optimal": round(days_to_full, 1),
            "factors": {
                "temperature_score": 0.2 if 18 <= temp <= 24 else 0.1 if 15 <= temp <= 28 else 0,
                "humidity_score": 0.2 if humidity >= 85 else 0.1 if humidity >= 75 else 0,
                "pressure_score": 0.1 if pressure < 1010 else 0,
                "lunar_score": 0.15 if phase in ["full_moon", "new_moon"] else 0.05,
                "geomagnetic_score": 0.05 if 45000 <= total_field <= 55000 else 0,
            },
            "environmental_summary": {
                "temperature": temp,
                "humidity": humidity,
                "pressure": pressure,
                "lunar_phase": phase,
                "geomagnetic_field": total_field,
            },
            "message": "Fruiting conditions predicted",
        }


def analyze_field_conditions(
    lat: float,
    lon: float,
    alt: float = 0,
) -> Dict[str, Any]:
    """
    Convenience function to analyze all field conditions at a location.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters
    
    Returns:
        Complete environmental analysis
    """
    model = FieldPhysicsModel()
    location = (lat, lon, alt)
    timestamp = time.time()
    
    return {
        "geomagnetic": model.get_geomagnetic_field(location, timestamp),
        "lunar": model.get_lunar_gravitational_influence(location, timestamp),
        "atmospheric": model.get_atmospheric_conditions(location, timestamp),
        "fruiting_prediction": model.predict_fruiting_conditions(location, timestamp),
    }
