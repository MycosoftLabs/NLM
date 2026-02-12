"""
Symbiosis Network Mapper
========================

Maps and analyzes symbiotic relationships between fungi and other organisms
including plants, bacteria, insects, and other fungi.

Relationship Types:
- Mycorrhizal (mutualistic root association)
- Parasitic (one organism harmed)
- Saprotrophic (decomposition)
- Endophytic (living within plant tissues)
- Lichen (fungi-algae partnership)
- Predatory (capturing other organisms)

Usage:
    mapper = SymbiosisNetworkMapper()
    mapper.load_from_mindex(data)
    analysis = mapper.analyze_network()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class Organism:
    """Represents an organism in the symbiosis network."""
    id: str
    name: str
    organism_type: str  # "fungus", "plant", "bacteria", "animal", "algae"
    x: float = 0.0
    y: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """Represents a symbiotic relationship between organisms."""
    source_id: str
    target_id: str
    relationship_type: str
    strength: float = 1.0
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SymbiosisNetworkMapper:
    """
    Maps symbiotic relationships and performs network analysis.
    
    Provides methods for adding organisms, relationships, and
    analyzing network properties like keystone species and communities.
    """
    
    RELATIONSHIP_COLORS = {
        "mycorrhizal": "#22c55e",
        "parasitic": "#ef4444",
        "saprotrophic": "#f59e0b",
        "endophytic": "#3b82f6",
        "lichen": "#8b5cf6",
        "predatory": "#dc2626",
        "commensal": "#14b8a6",
        "mutualistic": "#22c55e",
    }
    
    ORGANISM_COLORS = {
        "fungus": "#22c55e",
        "plant": "#84cc16",
        "bacteria": "#06b6d4",
        "animal": "#ef4444",
        "algae": "#10b981",
    }
    
    def __init__(self):
        """Initialize the Symbiosis Network Mapper."""
        self.organisms: Dict[str, Organism] = {}
        self.relationships: List[Relationship] = []
        print("Initialized SymbiosisNetworkMapper")
    
    def add_organism(
        self,
        organism_id: str,
        name: str,
        organism_type: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or update an organism in the network."""
        if x is None:
            x = np.random.uniform(50, 750)
        if y is None:
            y = np.random.uniform(50, 550)
        
        organism = Organism(
            id=organism_id,
            name=name,
            organism_type=organism_type,
            x=x,
            y=y,
            metadata=metadata or {},
        )
        
        self.organisms[organism_id] = organism
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        strength: float = 1.0,
        bidirectional: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a symbiotic relationship between two organisms."""
        if source_id not in self.organisms or target_id not in self.organisms:
            return
        
        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            strength=strength,
            bidirectional=bidirectional,
            metadata=metadata or {},
        )
        
        self.relationships.append(relationship)
    
    def generate_sample_network(self, num_organisms: int = 30) -> None:
        """Generate a sample symbiosis network for demonstration."""
        organism_types = ["fungus", "plant", "bacteria", "animal", "algae"]
        type_weights = [0.4, 0.3, 0.15, 0.1, 0.05]
        
        # Generate organisms
        for i in range(num_organisms):
            org_type = np.random.choice(organism_types, p=type_weights)
            
            # Position in circular layout with some randomness
            angle = 2 * math.pi * i / num_organisms
            radius = 200 + np.random.uniform(-50, 50)
            x = 400 + radius * math.cos(angle)
            y = 300 + radius * math.sin(angle)
            
            self.add_organism(
                organism_id=f"org_{i}",
                name=f"{org_type.title()} {i}",
                organism_type=org_type,
                x=x,
                y=y,
            )
        
        # Generate relationships
        relationship_types = [
            "mycorrhizal", "parasitic", "saprotrophic",
            "endophytic", "lichen", "predatory",
        ]
        rel_weights = [0.35, 0.15, 0.2, 0.15, 0.1, 0.05]
        
        # More relationships for denser network
        for _ in range(int(num_organisms * 1.5)):
            source_idx = np.random.randint(0, num_organisms)
            target_idx = np.random.randint(0, num_organisms)
            
            if source_idx == target_idx:
                continue
            
            rel_type = np.random.choice(relationship_types, p=rel_weights)
            strength = np.random.uniform(0.3, 1.0)
            
            self.add_relationship(
                source_id=f"org_{source_idx}",
                target_id=f"org_{target_idx}",
                relationship_type=rel_type,
                strength=strength,
            )
    
    def analyze_network(self) -> Dict[str, Any]:
        """
        Perform network analysis to identify key metrics and species.
        
        Returns:
            Dictionary with network statistics and analysis results
        """
        num_organisms = len(self.organisms)
        num_relationships = len(self.relationships)
        
        if num_organisms == 0:
            return {"error": "No organisms in network"}
        
        # Calculate degree centrality
        degree_counts: Dict[str, int] = {org_id: 0 for org_id in self.organisms}
        
        for rel in self.relationships:
            degree_counts[rel.source_id] = degree_counts.get(rel.source_id, 0) + 1
            degree_counts[rel.target_id] = degree_counts.get(rel.target_id, 0) + 1
            if rel.bidirectional:
                degree_counts[rel.source_id] += 1
                degree_counts[rel.target_id] += 1
        
        # Find keystone species (highest degree)
        sorted_by_degree = sorted(
            degree_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        keystone_species = []
        for org_id, degree in sorted_by_degree[:5]:
            organism = self.organisms.get(org_id)
            if organism:
                keystone_species.append({
                    "id": org_id,
                    "name": organism.name,
                    "type": organism.organism_type,
                    "degree": degree,
                    "betweenness": degree / max(1, num_relationships) * 100,
                })
        
        # Relationship type breakdown
        rel_type_counts: Dict[str, int] = {}
        for rel in self.relationships:
            rel_type_counts[rel.relationship_type] = rel_type_counts.get(rel.relationship_type, 0) + 1
        
        # Organism type breakdown
        org_type_counts: Dict[str, int] = {}
        for org in self.organisms.values():
            org_type_counts[org.organism_type] = org_type_counts.get(org.organism_type, 0) + 1
        
        # Calculate average degree
        avg_degree = sum(degree_counts.values()) / max(1, num_organisms)
        
        # Simple community detection (by organism type)
        communities = []
        for org_type in set(o.organism_type for o in self.organisms.values()):
            members = [org_id for org_id, org in self.organisms.items() if org.organism_type == org_type]
            if members:
                # Find dominant relationship type for this community
                rel_counts: Dict[str, int] = {}
                for rel in self.relationships:
                    if rel.source_id in members or rel.target_id in members:
                        rel_counts[rel.relationship_type] = rel_counts.get(rel.relationship_type, 0) + 1
                
                dominant_rel = max(rel_counts, key=rel_counts.get) if rel_counts else "none"
                
                communities.append({
                    "id": len(communities) + 1,
                    "organism_type": org_type,
                    "size": len(members),
                    "dominant_relationship": dominant_rel,
                })
        
        return {
            "num_organisms": num_organisms,
            "num_relationships": num_relationships,
            "average_degree": round(avg_degree, 2),
            "density": round(num_relationships / max(1, num_organisms * (num_organisms - 1) / 2), 4),
            "keystone_species": keystone_species,
            "relationship_breakdown": rel_type_counts,
            "organism_breakdown": org_type_counts,
            "communities": communities,
        }
    
    def get_relationships_for_organism(self, organism_id: str) -> List[Dict[str, Any]]:
        """Get all relationships involving a specific organism."""
        relationships = []
        
        for rel in self.relationships:
            if rel.source_id == organism_id or rel.target_id == organism_id:
                other_id = rel.target_id if rel.source_id == organism_id else rel.source_id
                other = self.organisms.get(other_id)
                
                relationships.append({
                    "partner_id": other_id,
                    "partner_name": other.name if other else "Unknown",
                    "partner_type": other.organism_type if other else "unknown",
                    "relationship_type": rel.relationship_type,
                    "strength": rel.strength,
                    "direction": "outgoing" if rel.source_id == organism_id else "incoming",
                })
        
        return relationships
    
    def export_network(self) -> Dict[str, Any]:
        """Export the network as a dictionary suitable for visualization."""
        organisms = []
        for org_id, org in self.organisms.items():
            organisms.append({
                "id": org_id,
                "name": org.name,
                "type": org.organism_type,
                "x": org.x,
                "y": org.y,
                "color": self.ORGANISM_COLORS.get(org.organism_type, "#9ca3af"),
            })
        
        relationships = []
        for rel in self.relationships:
            relationships.append({
                "source": rel.source_id,
                "target": rel.target_id,
                "type": rel.relationship_type,
                "strength": rel.strength,
                "color": self.RELATIONSHIP_COLORS.get(rel.relationship_type, "#6b7280"),
            })
        
        return {
            "organisms": organisms,
            "relationships": relationships,
        }
    
    def export_geojson(self, base_lat: float = 0.0, base_lon: float = 0.0) -> Dict[str, Any]:
        """
        Export network as GeoJSON for Earth Simulator integration.
        
        Args:
            base_lat: Base latitude for positioning
            base_lon: Base longitude for positioning
        
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        # Export organisms as points
        for org_id, org in self.organisms.items():
            # Scale visualization coordinates to geographic
            lat = base_lat + (org.y - 300) / 1000
            lon = base_lon + (org.x - 400) / 1000
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": {
                    "id": org_id,
                    "name": org.name,
                    "organism_type": org.organism_type,
                    "color": self.ORGANISM_COLORS.get(org.organism_type, "#9ca3af"),
                },
            })
        
        # Export relationships as lines
        for rel in self.relationships:
            source = self.organisms.get(rel.source_id)
            target = self.organisms.get(rel.target_id)
            
            if source and target:
                source_lat = base_lat + (source.y - 300) / 1000
                source_lon = base_lon + (source.x - 400) / 1000
                target_lat = base_lat + (target.y - 300) / 1000
                target_lon = base_lon + (target.x - 400) / 1000
                
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [source_lon, source_lat],
                            [target_lon, target_lat],
                        ],
                    },
                    "properties": {
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "relationship_type": rel.relationship_type,
                        "strength": rel.strength,
                        "color": self.RELATIONSHIP_COLORS.get(rel.relationship_type, "#6b7280"),
                    },
                })
        
        return {
            "type": "FeatureCollection",
            "features": features,
        }
    
    def load_from_mindex(self, mindex_data: Dict[str, Any]) -> None:
        """
        Load symbiosis data from MINDEX database format.
        
        Args:
            mindex_data: Dictionary with taxa and relationships from MINDEX
        """
        print("Loading symbiosis data from MINDEX...")
        
        # Load organisms
        for taxon in mindex_data.get("taxa", []):
            self.add_organism(
                organism_id=str(taxon.get("id")),
                name=taxon.get("canonical_name", "Unknown"),
                organism_type=taxon.get("kingdom", "fungus").lower(),
                metadata={"taxon_rank": taxon.get("rank")},
            )
        
        # Load relationships
        for rel in mindex_data.get("relationships", []):
            self.add_relationship(
                source_id=str(rel.get("source_taxon_id")),
                target_id=str(rel.get("target_taxon_id")),
                relationship_type=rel.get("relationship_type", "mutualistic"),
                strength=rel.get("strength", 1.0),
                metadata={"evidence": rel.get("evidence_level")},
            )
        
        print(f"Loaded {len(self.organisms)} organisms and {len(self.relationships)} relationships")


def create_sample_network(num_organisms: int = 30) -> SymbiosisNetworkMapper:
    """
    Convenience function to create a sample network.
    
    Args:
        num_organisms: Number of organisms to generate
    
    Returns:
        SymbiosisNetworkMapper with sample data
    """
    mapper = SymbiosisNetworkMapper()
    mapper.generate_sample_network(num_organisms)
    return mapper
