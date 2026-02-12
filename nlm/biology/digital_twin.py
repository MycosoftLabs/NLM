"""
Digital Twin Mycelium
=====================

Creates real-time digital representations of mycelial networks that
mirror physical cultivations. Integrates with MycoBrain sensor data
for dynamic state updates and predictive modeling.

Features:
- Network topology simulation
- Sensor data integration
- Growth prediction
- Anomaly detection
- GeoJSON export for Earth Simulator

Usage:
    dtm = DigitalTwinMycelium(initial_state)
    dtm.update_from_mycobrain_data(sensor_data)
    prediction = dtm.predict_growth(hours=24)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class NetworkNode:
    """Represents a node in the mycelial network."""
    id: str
    x: float
    y: float
    node_type: str  # "tip", "junction", "fruiting"
    age_hours: float = 0.0
    resource_level: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "type": self.node_type,
            "age_hours": self.age_hours,
            "resource_level": self.resource_level,
        }


@dataclass
class NetworkEdge:
    """Represents a hyphal connection between nodes."""
    source_id: str
    target_id: str
    strength: float = 1.0
    signal_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "strength": self.strength,
            "signal_active": self.signal_active,
        }


class DigitalTwinMycelium:
    """
    Digital Twin for real-time mycelial network simulation.
    
    Tracks network topology, integrates sensor data, and provides
    growth predictions based on environmental conditions.
    """
    
    def __init__(self, initial_state: Dict[str, Any]):
        """
        Initialize the Digital Twin.
        
        Args:
            initial_state: Dictionary with initial biomass, species, etc.
        """
        self.state = initial_state.copy()
        self.nodes: List[NetworkNode] = []
        self.edges: List[NetworkEdge] = []
        self.history: List[Dict[str, Any]] = []
        self.last_update_time = time.time()
        
        # Initialize state values
        self.state.setdefault("biomass_grams", 1.0)
        self.state.setdefault("network_density", 0.1)
        self.state.setdefault("health", 100.0)
        self.state.setdefault("temperature", 22.0)
        self.state.setdefault("humidity", 85.0)
        
        # Generate initial network
        self._generate_initial_network()
        
        print(f"Initialized DigitalTwinMycelium with {len(self.nodes)} nodes")
    
    def _generate_initial_network(self, num_tips: int = 15) -> None:
        """Generate an initial random network structure."""
        self.nodes = []
        self.edges = []
        
        # Create central node (inoculation point)
        center = NetworkNode(
            id="node_0",
            x=400,
            y=300,
            node_type="junction",
            age_hours=24,
        )
        self.nodes.append(center)
        
        # Create radiating tips
        for i in range(1, num_tips + 1):
            angle = 2 * math.pi * i / num_tips + np.random.normal(0, 0.2)
            distance = np.random.uniform(50, 150)
            
            node = NetworkNode(
                id=f"node_{i}",
                x=400 + distance * math.cos(angle),
                y=300 + distance * math.sin(angle),
                node_type="tip",
                age_hours=np.random.uniform(0, 12),
            )
            self.nodes.append(node)
            
            # Connect to center or nearest junction
            self.edges.append(NetworkEdge(
                source_id="node_0",
                target_id=f"node_{i}",
                strength=np.random.uniform(0.5, 1.0),
            ))
        
        # Add some intermediate junctions
        for i in range(3):
            angle = 2 * math.pi * i / 3 + np.random.normal(0, 0.3)
            distance = np.random.uniform(70, 100)
            
            junction = NetworkNode(
                id=f"junction_{i}",
                x=400 + distance * math.cos(angle),
                y=300 + distance * math.sin(angle),
                node_type="junction",
                age_hours=18,
            )
            self.nodes.append(junction)
            
            self.edges.append(NetworkEdge(
                source_id="node_0",
                target_id=f"junction_{i}",
                strength=0.9,
            ))
    
    def update_from_mycobrain_data(self, sensor_data: Dict[str, Any]) -> None:
        """
        Update the digital twin state from MycoBrain sensor readings.
        
        Args:
            sensor_data: Dictionary with temperature, humidity, CO2, etc.
        """
        print(f"  DTM: Updating with sensor data: {list(sensor_data.keys())}")
        
        # Update environmental conditions
        self.state["temperature"] = sensor_data.get("temperature_celsius", self.state["temperature"])
        self.state["humidity"] = sensor_data.get("humidity_percent", self.state["humidity"])
        self.state["co2"] = sensor_data.get("co2_ppm", 800)
        self.state["last_sensor_update"] = time.time()
        
        # Calculate growth rate based on conditions
        growth_rate = self._calculate_growth_rate()
        
        # Update biomass
        hours_elapsed = (time.time() - self.last_update_time) / 3600
        self.state["biomass_grams"] *= (1 + growth_rate * hours_elapsed)
        
        # Update network density
        self.state["network_density"] = min(1.0, self.state["network_density"] + 0.01 * hours_elapsed)
        
        # Check for anomalies
        anomalies = self._detect_anomalies(sensor_data)
        if anomalies:
            self.state["anomalies"] = anomalies
            self.state["health"] = max(0, self.state["health"] - len(anomalies) * 5)
        
        # Update network visualization
        self._update_network()
        
        # Record history
        self.history.append({
            "timestamp": time.time(),
            "state": self.state.copy(),
            "sensor_data": sensor_data,
        })
        
        self.last_update_time = time.time()
    
    def _calculate_growth_rate(self) -> float:
        """Calculate growth rate based on environmental conditions."""
        temp = self.state.get("temperature", 22)
        humidity = self.state.get("humidity", 85)
        
        # Temperature response (optimal 22°C)
        temp_factor = math.exp(-((temp - 22) ** 2) / 50)
        
        # Humidity response (optimal 85-95%)
        if humidity >= 85:
            humidity_factor = 1.0
        elif humidity >= 70:
            humidity_factor = (humidity - 70) / 15
        else:
            humidity_factor = 0.3
        
        # Base growth rate: 1-2% per hour under optimal conditions
        base_rate = 0.015
        
        return base_rate * temp_factor * humidity_factor
    
    def _detect_anomalies(self, sensor_data: Dict[str, Any]) -> List[str]:
        """Detect anomalies in sensor data."""
        anomalies = []
        
        temp = sensor_data.get("temperature_celsius", 22)
        humidity = sensor_data.get("humidity_percent", 85)
        co2 = sensor_data.get("co2_ppm", 800)
        
        if temp < 15 or temp > 30:
            anomalies.append(f"temperature_out_of_range: {temp}°C")
        
        if humidity < 60:
            anomalies.append(f"humidity_low: {humidity}%")
        
        if co2 > 2000:
            anomalies.append(f"co2_high: {co2}ppm")
        
        return anomalies
    
    def _update_network(self) -> None:
        """Update network visualization based on current state."""
        # Extend tips based on growth
        for node in self.nodes:
            if node.node_type == "tip":
                # Move tip outward slightly
                dx = node.x - 400
                dy = node.y - 300
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance > 0:
                    growth_rate = self._calculate_growth_rate()
                    extension = growth_rate * 5  # pixels per update
                    node.x += dx / distance * extension
                    node.y += dy / distance * extension
                
                node.age_hours += 0.5
        
        # Randomly trigger signal propagation
        if np.random.random() < 0.1:
            for edge in self.edges:
                edge.signal_active = np.random.random() < 0.2
    
    def predict_growth(self, duration_hours: float = 24) -> Dict[str, Any]:
        """
        Predict future growth based on current conditions.
        
        Args:
            duration_hours: Hours to predict ahead
        
        Returns:
            Dictionary with predicted biomass, density, etc.
        """
        print(f"  DTM: Predicting growth for {duration_hours} hours")
        
        current_biomass = self.state.get("biomass_grams", 1.0)
        current_density = self.state.get("network_density", 0.5)
        
        growth_rate = self._calculate_growth_rate()
        
        # Predict biomass with exponential growth (limited by carrying capacity)
        carrying_capacity = 500  # grams
        predicted_biomass = carrying_capacity / (
            1 + ((carrying_capacity / current_biomass) - 1) * 
            math.exp(-growth_rate * duration_hours)
        )
        
        # Predict network density
        predicted_density = min(1.0, current_density + 0.01 * duration_hours * growth_rate)
        
        # Estimate fruiting probability
        fruiting_probability = 0.0
        if predicted_density > 0.8 and self.state.get("humidity", 85) > 80:
            fruiting_probability = (predicted_density - 0.8) * 5
            fruiting_probability = min(0.9, max(0, fruiting_probability))
        
        # Generate recommendations
        recommendations = []
        if self.state.get("humidity", 85) < 80:
            recommendations.append("Increase humidity above 80%")
        if self.state.get("temperature", 22) > 25:
            recommendations.append("Lower temperature to optimal range (20-24°C)")
        if self.state.get("co2", 800) > 1500:
            recommendations.append("Increase FAE to reduce CO2")
        if not recommendations:
            recommendations.append("Conditions are optimal, maintain current parameters")
        
        return {
            "current_biomass_grams": round(current_biomass, 2),
            "predicted_biomass_grams": round(predicted_biomass, 2),
            "predicted_network_density": round(predicted_density, 3),
            "fruiting_probability": round(fruiting_probability, 2),
            "growth_rate_per_hour": round(growth_rate, 4),
            "prediction_window_hours": duration_hours,
            "recommendations": recommendations,
            "message": f"Growth predicted for {duration_hours} hours",
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Return the current state of the digital twin."""
        return {
            **self.state,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }
    
    def export_geojson(self) -> Dict[str, Any]:
        """
        Export network as GeoJSON for Earth Simulator integration.
        
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        # Export nodes as points
        for node in self.nodes:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [node.x / 10, node.y / 10],  # Scale to degrees
                },
                "properties": {
                    "id": node.id,
                    "type": node.node_type,
                    "age_hours": node.age_hours,
                    "resource_level": node.resource_level,
                },
            })
        
        # Export edges as lines
        node_map = {n.id: n for n in self.nodes}
        for edge in self.edges:
            source = node_map.get(edge.source_id)
            target = node_map.get(edge.target_id)
            if source and target:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [source.x / 10, source.y / 10],
                            [target.x / 10, target.y / 10],
                        ],
                    },
                    "properties": {
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "strength": edge.strength,
                        "signal_active": edge.signal_active,
                    },
                })
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "biomass_grams": self.state.get("biomass_grams"),
                "network_density": self.state.get("network_density"),
                "export_timestamp": time.time(),
            },
        }


def create_digital_twin(species: str = "P. cubensis") -> DigitalTwinMycelium:
    """
    Convenience function to create a digital twin.
    
    Args:
        species: Species name for the twin
    
    Returns:
        Initialized DigitalTwinMycelium instance
    """
    return DigitalTwinMycelium({
        "species": species,
        "biomass_grams": 5.0,
        "network_density": 0.3,
        "health": 100.0,
    })
