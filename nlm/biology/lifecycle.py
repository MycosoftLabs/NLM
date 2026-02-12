"""
Spore Lifecycle Simulator
=========================

Simulates the complete fungal lifecycle from spore germination through
fruiting body development and sporulation. Integrates environmental
factors and genetic predispositions.

Lifecycle Stages:
1. Spore (dormant)
2. Germination
3. Hyphal Growth
4. Mycelial Network
5. Primordial Formation
6. Fruiting Body Development
7. Sporulation

Usage:
    profile = SPECIES_PROFILES["psilocybe_cubensis"]
    simulator = SporeLifecycleSimulator(profile, initial_conditions)
    state = simulator.advance_stage(hours=24)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class LifecycleStage(Enum):
    """Enumeration of fungal lifecycle stages."""
    SPORE = "spore"
    GERMINATION = "germination"
    HYPHAL_GROWTH = "hyphal_growth"
    MYCELIAL_NETWORK = "mycelial_network"
    PRIMORDIAL = "primordial"
    FRUITING_BODY = "fruiting_body"
    SPORULATION = "sporulation"
    DECAY = "decay"
    FINISHED = "finished"


@dataclass
class SpeciesProfile:
    """Species-specific lifecycle parameters."""
    name: str
    scientific_name: str
    
    # Temperature requirements (°C)
    germination_temp_optimal: float = 24.0
    germination_temp_min: float = 18.0
    germination_temp_max: float = 30.0
    fruiting_temp_optimal: float = 22.0
    fruiting_temp_min: float = 15.0
    fruiting_temp_max: float = 27.0
    
    # Humidity requirements (%)
    humidity_optimal: float = 90.0
    humidity_min: float = 70.0
    
    # CO2 requirements (ppm)
    mycelium_co2_optimal: float = 2000.0
    fruiting_co2_max: float = 800.0
    
    # Light requirements (hours)
    fruiting_light_hours: float = 12.0
    
    # Timing (days)
    germination_days: float = 3.0
    colonization_days: float = 14.0
    pinning_days: float = 5.0
    fruiting_days: float = 7.0
    
    # Growth characteristics
    growth_rate_mm_per_day: float = 5.0


# Pre-defined species profiles
SPECIES_PROFILES = {
    "psilocybe_cubensis": SpeciesProfile(
        name="Golden Teacher",
        scientific_name="Psilocybe cubensis",
        germination_temp_optimal=24.0,
        fruiting_temp_optimal=22.0,
        germination_days=2.0,
        colonization_days=14.0,
        pinning_days=5.0,
        fruiting_days=7.0,
        growth_rate_mm_per_day=7.0,
    ),
    "hericium_erinaceus": SpeciesProfile(
        name="Lion's Mane",
        scientific_name="Hericium erinaceus",
        germination_temp_optimal=22.0,
        fruiting_temp_optimal=18.0,
        germination_days=5.0,
        colonization_days=21.0,
        pinning_days=7.0,
        fruiting_days=10.0,
        growth_rate_mm_per_day=4.0,
    ),
    "pleurotus_ostreatus": SpeciesProfile(
        name="Oyster Mushroom",
        scientific_name="Pleurotus ostreatus",
        germination_temp_optimal=24.0,
        fruiting_temp_optimal=15.0,
        germination_days=2.0,
        colonization_days=10.0,
        pinning_days=3.0,
        fruiting_days=5.0,
        growth_rate_mm_per_day=10.0,
    ),
    "ganoderma_lucidum": SpeciesProfile(
        name="Reishi",
        scientific_name="Ganoderma lucidum",
        germination_temp_optimal=28.0,
        fruiting_temp_optimal=25.0,
        germination_days=7.0,
        colonization_days=30.0,
        pinning_days=14.0,
        fruiting_days=21.0,
        growth_rate_mm_per_day=3.0,
    ),
    "cordyceps_militaris": SpeciesProfile(
        name="Cordyceps",
        scientific_name="Cordyceps militaris",
        germination_temp_optimal=22.0,
        fruiting_temp_optimal=18.0,
        germination_days=5.0,
        colonization_days=28.0,
        pinning_days=7.0,
        fruiting_days=14.0,
        growth_rate_mm_per_day=2.0,
    ),
    "agaricus_bisporus": SpeciesProfile(
        name="Button Mushroom",
        scientific_name="Agaricus bisporus",
        germination_temp_optimal=24.0,
        fruiting_temp_optimal=16.0,
        germination_days=3.0,
        colonization_days=14.0,
        pinning_days=5.0,
        fruiting_days=7.0,
        growth_rate_mm_per_day=6.0,
    ),
}


class SporeLifecycleSimulator:
    """
    Simulates the complete lifecycle of fungal organisms.
    
    Tracks stage progression, environmental responses, and
    provides predictions for cultivation planning.
    """
    
    STAGE_ORDER = [
        LifecycleStage.SPORE,
        LifecycleStage.GERMINATION,
        LifecycleStage.HYPHAL_GROWTH,
        LifecycleStage.MYCELIAL_NETWORK,
        LifecycleStage.PRIMORDIAL,
        LifecycleStage.FRUITING_BODY,
        LifecycleStage.SPORULATION,
        LifecycleStage.DECAY,
        LifecycleStage.FINISHED,
    ]
    
    def __init__(
        self,
        species_profile: SpeciesProfile,
        initial_conditions: Dict[str, Any],
    ):
        """
        Initialize the lifecycle simulator.
        
        Args:
            species_profile: Species-specific parameters
            initial_conditions: Starting environment (temperature, humidity, etc.)
        """
        self.profile = species_profile
        self.current_stage = LifecycleStage.SPORE
        self.stage_progress = 0.0  # 0.0 to 1.0
        self.day_count = 0.0
        self.biomass = 0.1  # grams
        self.health = 100.0
        
        self.environment = {
            "temperature": initial_conditions.get("temperature", 22.0),
            "humidity": initial_conditions.get("humidity", 85.0),
            "co2": initial_conditions.get("co2", 800.0),
            "light_hours": initial_conditions.get("light_hours", 0.0),
        }
        
        self.history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
        print(f"Initialized SporeLifecycleSimulator for {self.profile.scientific_name}")
    
    def advance_stage(self, hours: float = 24.0) -> Dict[str, Any]:
        """
        Advance the lifecycle simulation by specified hours.
        
        Args:
            hours: Number of hours to advance
        
        Returns:
            Dictionary with current lifecycle state
        """
        self.day_count += hours / 24.0
        
        # Calculate environmental factors
        temp_factor = self._temperature_factor()
        humidity_factor = self._humidity_factor()
        
        # Combined growth modifier
        growth_modifier = temp_factor * humidity_factor
        
        # Update health based on conditions
        if growth_modifier < 0.5:
            self.health = max(0, self.health - (1 - growth_modifier) * 5)
        elif growth_modifier > 0.8:
            self.health = min(100, self.health + 1)
        
        # Stage-specific progression
        stage_duration_days = self._get_stage_duration()
        progress_increment = (hours / 24.0) / stage_duration_days * growth_modifier
        
        self.stage_progress += progress_increment
        
        # Update biomass
        if self.current_stage in [
            LifecycleStage.GERMINATION,
            LifecycleStage.HYPHAL_GROWTH,
            LifecycleStage.MYCELIAL_NETWORK,
        ]:
            growth_rate = self.profile.growth_rate_mm_per_day * growth_modifier
            self.biomass *= (1 + growth_rate * 0.05 * (hours / 24.0))
        
        # Check for stage transition
        if self.stage_progress >= 1.0:
            self._transition_to_next_stage()
        
        # Record history
        state = self._get_current_state()
        self.history.append(state)
        
        return state
    
    def _temperature_factor(self) -> float:
        """Calculate temperature response factor (0.0 to 1.0)."""
        temp = self.environment["temperature"]
        
        if self.current_stage in [LifecycleStage.PRIMORDIAL, LifecycleStage.FRUITING_BODY]:
            optimal = self.profile.fruiting_temp_optimal
            min_temp = self.profile.fruiting_temp_min
            max_temp = self.profile.fruiting_temp_max
        else:
            optimal = self.profile.germination_temp_optimal
            min_temp = self.profile.germination_temp_min
            max_temp = self.profile.germination_temp_max
        
        if temp < min_temp or temp > max_temp:
            return 0.1
        
        # Gaussian response curve
        factor = math.exp(-((temp - optimal) ** 2) / 30)
        return max(0.1, min(1.0, factor))
    
    def _humidity_factor(self) -> float:
        """Calculate humidity response factor (0.0 to 1.0)."""
        humidity = self.environment["humidity"]
        optimal = self.profile.humidity_optimal
        
        if humidity < self.profile.humidity_min:
            return 0.1
        
        if humidity >= optimal:
            return 1.0
        
        # Linear scaling
        return (humidity - self.profile.humidity_min) / (optimal - self.profile.humidity_min)
    
    def _get_stage_duration(self) -> float:
        """Get the expected duration of the current stage in days."""
        stage_durations = {
            LifecycleStage.SPORE: 1.0,
            LifecycleStage.GERMINATION: self.profile.germination_days,
            LifecycleStage.HYPHAL_GROWTH: self.profile.colonization_days * 0.3,
            LifecycleStage.MYCELIAL_NETWORK: self.profile.colonization_days * 0.7,
            LifecycleStage.PRIMORDIAL: self.profile.pinning_days,
            LifecycleStage.FRUITING_BODY: self.profile.fruiting_days,
            LifecycleStage.SPORULATION: 3.0,
            LifecycleStage.DECAY: 10.0,
            LifecycleStage.FINISHED: float("inf"),
        }
        return stage_durations.get(self.current_stage, 7.0)
    
    def _transition_to_next_stage(self) -> None:
        """Transition to the next lifecycle stage."""
        current_index = self.STAGE_ORDER.index(self.current_stage)
        
        if current_index < len(self.STAGE_ORDER) - 1:
            self.current_stage = self.STAGE_ORDER[current_index + 1]
            self.stage_progress = 0.0
            print(f"  Lifecycle: Transitioned to {self.current_stage.value}")
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the simulation."""
        return {
            "stage": self.current_stage.value,
            "stage_index": self.STAGE_ORDER.index(self.current_stage),
            "stage_progress": round(self.stage_progress, 3),
            "day_count": round(self.day_count, 1),
            "biomass_grams": round(self.biomass, 2),
            "health": round(self.health, 1),
            "environment": self.environment.copy(),
            "timestamp": time.time(),
        }
    
    def update_environment(
        self,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        co2: Optional[float] = None,
        light_hours: Optional[float] = None,
    ) -> None:
        """Update environmental conditions."""
        if temperature is not None:
            self.environment["temperature"] = temperature
        if humidity is not None:
            self.environment["humidity"] = humidity
        if co2 is not None:
            self.environment["co2"] = co2
        if light_hours is not None:
            self.environment["light_hours"] = light_hours
    
    def predict_harvest_date(self) -> Optional[datetime]:
        """
        Predict the estimated harvest date.
        
        Returns:
            Datetime of predicted harvest or None if past harvest
        """
        if self.current_stage in [
            LifecycleStage.SPORULATION,
            LifecycleStage.DECAY,
            LifecycleStage.FINISHED,
        ]:
            return None
        
        # Calculate remaining days
        remaining_days = 0.0
        
        stage_index = self.STAGE_ORDER.index(self.current_stage)
        
        # Current stage remaining time
        current_duration = self._get_stage_duration()
        remaining_days += current_duration * (1 - self.stage_progress)
        
        # Add time for future stages until fruiting
        fruiting_index = self.STAGE_ORDER.index(LifecycleStage.FRUITING_BODY)
        
        for i in range(stage_index + 1, fruiting_index + 1):
            stage = self.STAGE_ORDER[i]
            remaining_days += self._get_stage_duration()
        
        harvest_date = datetime.now() + timedelta(days=remaining_days)
        return harvest_date
    
    def get_recommendations(self) -> List[str]:
        """Get cultivation recommendations based on current state."""
        recommendations = []
        
        temp = self.environment["temperature"]
        humidity = self.environment["humidity"]
        co2 = self.environment["co2"]
        
        # Temperature recommendations
        if self.current_stage in [LifecycleStage.PRIMORDIAL, LifecycleStage.FRUITING_BODY]:
            if temp > self.profile.fruiting_temp_optimal + 2:
                recommendations.append(f"Lower temperature to {self.profile.fruiting_temp_optimal}°C for optimal fruiting")
            elif temp < self.profile.fruiting_temp_optimal - 2:
                recommendations.append(f"Raise temperature to {self.profile.fruiting_temp_optimal}°C for optimal fruiting")
        else:
            if temp > self.profile.germination_temp_optimal + 2:
                recommendations.append(f"Lower temperature to {self.profile.germination_temp_optimal}°C for optimal colonization")
        
        # Humidity recommendations
        if humidity < self.profile.humidity_min + 5:
            recommendations.append(f"Increase humidity to at least {self.profile.humidity_optimal}%")
        
        # CO2 recommendations
        if self.current_stage in [LifecycleStage.PRIMORDIAL, LifecycleStage.FRUITING_BODY]:
            if co2 > self.profile.fruiting_co2_max:
                recommendations.append(f"Increase FAE to reduce CO2 below {self.profile.fruiting_co2_max}ppm")
        
        # Light recommendations
        if self.current_stage in [LifecycleStage.PRIMORDIAL, LifecycleStage.FRUITING_BODY]:
            if self.environment["light_hours"] < self.profile.fruiting_light_hours:
                recommendations.append(f"Provide {self.profile.fruiting_light_hours} hours of light for pinning")
        
        if not recommendations:
            recommendations.append("Conditions are optimal. Maintain current parameters.")
        
        return recommendations


def run_lifecycle_simulation(
    species_key: str = "psilocybe_cubensis",
    days: int = 30,
) -> Dict[str, Any]:
    """
    Convenience function to run a complete lifecycle simulation.
    
    Args:
        species_key: Key from SPECIES_PROFILES
        days: Number of days to simulate
    
    Returns:
        Final simulation state and history
    """
    profile = SPECIES_PROFILES.get(species_key, SPECIES_PROFILES["psilocybe_cubensis"])
    
    simulator = SporeLifecycleSimulator(profile, {
        "temperature": 24.0,
        "humidity": 90.0,
        "co2": 800.0,
        "light_hours": 12.0,
    })
    
    for day in range(days):
        simulator.advance_stage(hours=24)
    
    return {
        "final_state": simulator._get_current_state(),
        "history": simulator.history,
        "harvest_prediction": simulator.predict_harvest_date(),
        "recommendations": simulator.get_recommendations(),
    }
