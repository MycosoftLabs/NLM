"""
Nature-Aware Data Augmentation
==============================

Augmentation strategies that respect natural constraints:
- Seasonal shifts stay within seasonal bounds
- Geographic jitter preserves locality
- Sensor noise matches real measurement characteristics
- Token dropout mirrors masked-language-modeling for nature signals
"""

from __future__ import annotations

import math
import random
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np


class SeasonalShift:
    """Shift timestamps within seasonal bounds.

    Does not cross season boundaries — a summer sample stays in summer.
    """

    SEASONS = {
        "spring": (80, 172),   # Mar 21 – Jun 20 (day of year)
        "summer": (172, 266),  # Jun 21 – Sep 22
        "autumn": (266, 355),  # Sep 23 – Dec 20
        "winter_a": (355, 366),  # Dec 21 – Dec 31
        "winter_b": (1, 80),     # Jan 1 – Mar 20
    }

    def __init__(self, max_shift_days: int = 14):
        self.max_shift_days = max_shift_days

    def get_season(self, day_of_year: int) -> Tuple[int, int]:
        for bounds in self.SEASONS.values():
            if bounds[0] <= day_of_year < bounds[1]:
                return bounds
        return (1, 366)

    def __call__(self, timestamp):
        from datetime import datetime
        doy = timestamp.timetuple().tm_yday
        season_start, season_end = self.get_season(doy)

        max_forward = min(self.max_shift_days, season_end - doy - 1)
        max_backward = min(self.max_shift_days, doy - season_start)

        shift = random.randint(-max_backward, max(0, max_forward))
        return timestamp + timedelta(days=shift)


class GeographicJitter:
    """Small random perturbation to coordinates.

    Keeps samples within a local neighborhood to preserve
    spatial autocorrelation structure.
    """

    def __init__(self, max_offset_km: float = 1.0):
        # ~0.009 degrees latitude per km
        self.max_offset_deg = max_offset_km * 0.009

    def __call__(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        lat_offset = random.gauss(0, self.max_offset_deg / 3.0)
        lon_offset = random.gauss(0, self.max_offset_deg / 3.0)
        alt_offset = random.gauss(0, 5.0)  # ±5m

        new_lat = max(-90.0, min(90.0, lat + lat_offset))
        new_lon = ((lon + lon_offset + 180.0) % 360.0) - 180.0
        new_alt = max(0.0, alt + alt_offset)

        return (new_lat, new_lon, new_alt)


class SensorNoise:
    """Add realistic measurement noise to environmental readings.

    Noise levels are calibrated to typical sensor specifications.
    """

    # Typical sensor noise standard deviations
    NOISE_LEVELS: Dict[str, float] = {
        "temperature_c": 0.5,      # ±0.5°C
        "humidity_pct": 2.0,       # ±2%
        "co2_ppm": 50.0,           # ±50 ppm
        "pressure_hpa": 1.0,       # ±1 hPa
        "light_lux": 50.0,         # ±50 lux
        "ph": 0.1,                 # ±0.1 pH
        "moisture_pct": 3.0,       # ±3%
        "wind_speed_mps": 0.5,     # ±0.5 m/s
    }

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, readings: Dict[str, float]) -> Dict[str, float]:
        noisy = {}
        for key, value in readings.items():
            noise_std = self.NOISE_LEVELS.get(key, abs(value) * 0.01)
            noisy[key] = value + random.gauss(0, noise_std * self.scale)
        return noisy


class TokenDropout:
    """Randomly mask bio-tokens.

    Like masked language modeling but for nature signals —
    forces the model to predict missing sensor information
    from context.
    """

    def __init__(self, dropout_rate: float = 0.15, mask_token: str = "[MASK]"):
        self.dropout_rate = dropout_rate
        self.mask_token = mask_token

    def __call__(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        """Returns (masked_tokens, masked_indices)."""
        masked = list(tokens)
        indices = []
        for i in range(len(masked)):
            if random.random() < self.dropout_rate:
                indices.append(i)
                r = random.random()
                if r < 0.8:
                    masked[i] = self.mask_token
                elif r < 0.9:
                    # Replace with random token
                    masked[i] = random.choice(tokens) if tokens else self.mask_token
                # else: keep original (10% of time)
        return masked, indices


class NatureAugmentor:
    """Composes all augmentation strategies."""

    def __init__(
        self,
        seasonal_shift: bool = True,
        geo_jitter: bool = True,
        sensor_noise: bool = True,
        token_dropout: bool = True,
        noise_scale: float = 1.0,
        dropout_rate: float = 0.15,
        max_shift_days: int = 14,
        max_jitter_km: float = 1.0,
    ):
        self.seasonal_shift = SeasonalShift(max_shift_days) if seasonal_shift else None
        self.geo_jitter = GeographicJitter(max_jitter_km) if geo_jitter else None
        self.sensor_noise = SensorNoise(noise_scale) if sensor_noise else None
        self.token_dropout = TokenDropout(dropout_rate) if token_dropout else None

    def augment(
        self,
        timestamp=None,
        geolocation: Optional[Tuple[float, float, float]] = None,
        readings: Optional[Dict[str, float]] = None,
        bio_tokens: Optional[List[str]] = None,
    ) -> Dict:
        result = {}

        if timestamp is not None and self.seasonal_shift:
            result["timestamp"] = self.seasonal_shift(timestamp)
        elif timestamp is not None:
            result["timestamp"] = timestamp

        if geolocation is not None and self.geo_jitter:
            result["geolocation"] = self.geo_jitter(*geolocation)
        elif geolocation is not None:
            result["geolocation"] = geolocation

        if readings is not None and self.sensor_noise:
            result["readings"] = self.sensor_noise(readings)
        elif readings is not None:
            result["readings"] = readings

        if bio_tokens is not None and self.token_dropout:
            masked, indices = self.token_dropout(bio_tokens)
            result["bio_tokens"] = masked
            result["masked_indices"] = indices
        elif bio_tokens is not None:
            result["bio_tokens"] = bio_tokens
            result["masked_indices"] = []

        return result
