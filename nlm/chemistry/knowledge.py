"""
Chemistry Knowledge Graph
=========================

Manages relationships between compounds, species, and biological activities.
Provides graph-based queries for compound discovery and pathway analysis.

Node Types:
- Compound: Chemical compounds with structure and properties
- Species: Fungal and other organisms
- Activity: Biological activities (antimicrobial, neuroactive, etc.)
- Pathway: Biosynthetic pathways

Edge Types:
- PRODUCES: Species produces compound
- HAS_ACTIVITY: Compound has biological activity
- PRECURSOR_OF: Compound is precursor of another
- INHIBITS/ACTIVATES: Compound affects biological target

Usage:
    kg = ChemistryKnowledgeGraph()
    kg.add_compound("psilocybin", {...})
    kg.add_relationship("psilocybin", "psychedelic", "HAS_ACTIVITY")
    results = kg.query_compounds_by_activity("antimicrobial")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np


@dataclass
class Node:
    """Represents a node in the knowledge graph."""
    id: str
    node_type: str  # "compound", "species", "activity", "pathway"
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


class ChemistryKnowledgeGraph:
    """
    Graph database for chemistry relationships.
    
    Stores compounds, species, activities, and their relationships
    for querying and discovery.
    """
    
    def __init__(self):
        """Initialize the knowledge graph."""
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency: Dict[str, List[Edge]] = {}
        print("Initialized ChemistryKnowledgeGraph")
    
    def add_compound(
        self,
        compound_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Add a compound node to the graph."""
        node = Node(
            id=compound_id,
            node_type="compound",
            name=data.get("name", compound_id),
            properties=data,
        )
        self.nodes[compound_id] = node
        
        if compound_id not in self.adjacency:
            self.adjacency[compound_id] = []
    
    def add_species(
        self,
        species_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Add a species node to the graph."""
        node = Node(
            id=species_id,
            node_type="species",
            name=data.get("name", species_id),
            properties=data,
        )
        self.nodes[species_id] = node
        
        if species_id not in self.adjacency:
            self.adjacency[species_id] = []
    
    def add_activity(
        self,
        activity_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Add an activity node to the graph."""
        node = Node(
            id=activity_id,
            node_type="activity",
            name=data.get("name", activity_id),
            properties=data,
        )
        self.nodes[activity_id] = node
        
        if activity_id not in self.adjacency:
            self.adjacency[activity_id] = []
    
    def add_pathway(
        self,
        pathway_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Add a biosynthetic pathway node."""
        node = Node(
            id=pathway_id,
            node_type="pathway",
            name=data.get("name", pathway_id),
            properties=data,
        )
        self.nodes[pathway_id] = node
        
        if pathway_id not in self.adjacency:
            self.adjacency[pathway_id] = []
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a relationship between nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return
        
        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            properties=properties or {},
        )
        
        self.edges.append(edge)
        
        if source_id not in self.adjacency:
            self.adjacency[source_id] = []
        self.adjacency[source_id].append(edge)
    
    def query_compounds_by_activity(
        self,
        activity_name: str,
        min_weight: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find compounds with a specific biological activity.
        
        Args:
            activity_name: Name or ID of the activity
            min_weight: Minimum relationship weight
        
        Returns:
            List of compound data dictionaries
        """
        compounds = []
        
        # Find activity node
        activity_id = None
        for node_id, node in self.nodes.items():
            if node.node_type == "activity":
                if activity_name.lower() in node.name.lower():
                    activity_id = node_id
                    break
        
        if not activity_id:
            return compounds
        
        # Find compounds linked to this activity
        for edge in self.edges:
            if edge.target_id == activity_id and edge.edge_type == "HAS_ACTIVITY":
                if edge.weight >= min_weight:
                    node = self.nodes.get(edge.source_id)
                    if node and node.node_type == "compound":
                        compounds.append({
                            "id": node.id,
                            "name": node.name,
                            "weight": edge.weight,
                            **node.properties,
                        })
        
        # Sort by weight
        compounds.sort(key=lambda x: x.get("weight", 0), reverse=True)
        return compounds
    
    def query_species_compounds(
        self,
        species_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get all compounds produced by a species.
        
        Args:
            species_id: Species identifier
        
        Returns:
            List of compound data dictionaries
        """
        compounds = []
        
        for edge in self.adjacency.get(species_id, []):
            if edge.edge_type == "PRODUCES":
                node = self.nodes.get(edge.target_id)
                if node and node.node_type == "compound":
                    compounds.append({
                        "id": node.id,
                        "name": node.name,
                        "evidence": edge.properties.get("evidence", "unknown"),
                        **node.properties,
                    })
        
        return compounds
    
    def query_compound_activities(
        self,
        compound_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get all activities for a compound.
        
        Args:
            compound_id: Compound identifier
        
        Returns:
            List of activity data dictionaries
        """
        activities = []
        
        for edge in self.adjacency.get(compound_id, []):
            if edge.edge_type == "HAS_ACTIVITY":
                node = self.nodes.get(edge.target_id)
                if node and node.node_type == "activity":
                    activities.append({
                        "id": node.id,
                        "name": node.name,
                        "strength": edge.weight,
                        **node.properties,
                    })
        
        return activities
    
    def find_similar_compounds(
        self,
        compound_id: str,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Find similar compounds based on shared activities.
        
        Args:
            compound_id: Query compound
            max_hops: Maximum graph distance
        
        Returns:
            List of similar compound data
        """
        if compound_id not in self.nodes:
            return []
        
        # Get activities of the query compound
        query_activities = set()
        for edge in self.adjacency.get(compound_id, []):
            if edge.edge_type == "HAS_ACTIVITY":
                query_activities.add(edge.target_id)
        
        if not query_activities:
            return []
        
        # Find compounds sharing activities
        similar = {}
        for activity_id in query_activities:
            for edge in self.edges:
                if edge.target_id == activity_id and edge.edge_type == "HAS_ACTIVITY":
                    other_id = edge.source_id
                    if other_id != compound_id and other_id in self.nodes:
                        if other_id not in similar:
                            similar[other_id] = {"shared": 0, "node": self.nodes[other_id]}
                        similar[other_id]["shared"] += 1
        
        # Calculate similarity scores
        results = []
        for comp_id, data in similar.items():
            # Jaccard-like similarity
            other_activities = set()
            for edge in self.adjacency.get(comp_id, []):
                if edge.edge_type == "HAS_ACTIVITY":
                    other_activities.add(edge.target_id)
            
            intersection = len(query_activities & other_activities)
            union = len(query_activities | other_activities)
            similarity = intersection / union if union > 0 else 0
            
            node = data["node"]
            results.append({
                "id": node.id,
                "name": node.name,
                "shared_activities": data["shared"],
                "similarity": round(similarity, 3),
                **node.properties,
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:10]
    
    def get_biosynthetic_precursors(
        self,
        compound_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get biosynthetic precursors for a compound.
        
        Args:
            compound_id: Target compound
        
        Returns:
            List of precursor compounds
        """
        precursors = []
        
        for edge in self.edges:
            if edge.target_id == compound_id and edge.edge_type == "PRECURSOR_OF":
                node = self.nodes.get(edge.source_id)
                if node:
                    precursors.append({
                        "id": node.id,
                        "name": node.name,
                        "reaction": edge.properties.get("reaction"),
                        **node.properties,
                    })
        
        return precursors
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        node_types = {}
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        
        edge_types = {}
        for edge in self.edges:
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_types,
            "edge_types": edge_types,
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary format."""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "type": node.node_type,
                    "name": node.name,
                    "properties": node.properties,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type,
                    "weight": edge.weight,
                    "properties": edge.properties,
                }
                for edge in self.edges
            ],
        }
    
    def load_from_mindex(self, mindex_data: Dict[str, Any]) -> None:
        """
        Load graph data from MINDEX API response.
        
        Args:
            mindex_data: Dictionary with taxa, compounds, activities
        """
        print("Loading ChemistryKnowledgeGraph from MINDEX...")
        
        # Load species
        for taxon in mindex_data.get("taxa", []):
            self.add_species(str(taxon["id"]), {
                "name": taxon.get("canonical_name"),
                "common_name": taxon.get("common_name"),
                "rank": taxon.get("rank"),
            })
        
        # Load compounds
        for compound in mindex_data.get("compounds", []):
            self.add_compound(str(compound["id"]), {
                "name": compound.get("name"),
                "formula": compound.get("formula"),
                "smiles": compound.get("smiles"),
                "molecular_weight": compound.get("molecular_weight"),
            })
            
            # Link to species
            taxon_id = compound.get("taxon_id")
            if taxon_id:
                self.add_relationship(
                    str(taxon_id),
                    str(compound["id"]),
                    "PRODUCES",
                    properties={"evidence": compound.get("evidence_level")},
                )
        
        # Load activities
        for activity in mindex_data.get("activities", []):
            self.add_activity(str(activity["id"]), {
                "name": activity.get("name"),
                "category": activity.get("category"),
            })
        
        # Load compound-activity links
        for link in mindex_data.get("compound_activities", []):
            self.add_relationship(
                str(link["compound_id"]),
                str(link["activity_id"]),
                "HAS_ACTIVITY",
                weight=link.get("strength", 1.0),
            )
        
        stats = self.get_graph_stats()
        print(f"Loaded {stats['total_nodes']} nodes and {stats['total_edges']} edges")


def create_sample_knowledge_graph() -> ChemistryKnowledgeGraph:
    """
    Create a sample knowledge graph for demonstration.
    
    Returns:
        Populated ChemistryKnowledgeGraph
    """
    kg = ChemistryKnowledgeGraph()
    
    # Add sample compounds
    compounds = [
        {"id": "psilocybin", "name": "Psilocybin", "formula": "C12H17N2O4P"},
        {"id": "psilocin", "name": "Psilocin", "formula": "C12H16N2O"},
        {"id": "hericenone_a", "name": "Hericenone A", "formula": "C35H54O4"},
        {"id": "erinacine_a", "name": "Erinacine A", "formula": "C25H36O5"},
        {"id": "ganoderic_acid_a", "name": "Ganoderic Acid A", "formula": "C30H44O7"},
        {"id": "cordycepin", "name": "Cordycepin", "formula": "C10H13N5O3"},
    ]
    
    for comp in compounds:
        kg.add_compound(comp["id"], comp)
    
    # Add species
    species = [
        {"id": "p_cubensis", "name": "Psilocybe cubensis"},
        {"id": "h_erinaceus", "name": "Hericium erinaceus"},
        {"id": "g_lucidum", "name": "Ganoderma lucidum"},
        {"id": "c_militaris", "name": "Cordyceps militaris"},
    ]
    
    for sp in species:
        kg.add_species(sp["id"], sp)
    
    # Add activities
    activities = [
        {"id": "psychedelic", "name": "Psychedelic", "category": "Neuroactive"},
        {"id": "neurotrophic", "name": "Neurotrophic", "category": "Neuroactive"},
        {"id": "anti_inflammatory", "name": "Anti-inflammatory", "category": "Immunomodulatory"},
        {"id": "anticancer", "name": "Anticancer", "category": "Health"},
        {"id": "antiviral", "name": "Antiviral", "category": "Antimicrobial"},
    ]
    
    for act in activities:
        kg.add_activity(act["id"], act)
    
    # Add relationships
    kg.add_relationship("p_cubensis", "psilocybin", "PRODUCES")
    kg.add_relationship("p_cubensis", "psilocin", "PRODUCES")
    kg.add_relationship("h_erinaceus", "hericenone_a", "PRODUCES")
    kg.add_relationship("h_erinaceus", "erinacine_a", "PRODUCES")
    kg.add_relationship("g_lucidum", "ganoderic_acid_a", "PRODUCES")
    kg.add_relationship("c_militaris", "cordycepin", "PRODUCES")
    
    kg.add_relationship("psilocybin", "psychedelic", "HAS_ACTIVITY", weight=1.0)
    kg.add_relationship("psilocin", "psychedelic", "HAS_ACTIVITY", weight=1.0)
    kg.add_relationship("hericenone_a", "neurotrophic", "HAS_ACTIVITY", weight=0.9)
    kg.add_relationship("erinacine_a", "neurotrophic", "HAS_ACTIVITY", weight=0.95)
    kg.add_relationship("ganoderic_acid_a", "anti_inflammatory", "HAS_ACTIVITY", weight=0.8)
    kg.add_relationship("ganoderic_acid_a", "anticancer", "HAS_ACTIVITY", weight=0.7)
    kg.add_relationship("cordycepin", "antiviral", "HAS_ACTIVITY", weight=0.85)
    kg.add_relationship("cordycepin", "anticancer", "HAS_ACTIVITY", weight=0.6)
    
    kg.add_relationship("psilocybin", "psilocin", "PRECURSOR_OF", properties={"reaction": "Dephosphorylation"})
    
    return kg
