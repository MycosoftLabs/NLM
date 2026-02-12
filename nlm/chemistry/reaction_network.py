"""
Reaction Network Graph
======================

Graph database for biochemical reactions and metabolic pathways.
Stores compounds, reactions, enzymes, and their relationships
for pathway analysis and flux prediction.

Node Types:
- Compound: Metabolites and intermediates
- Reaction: Enzymatic reactions
- Enzyme: Catalytic proteins

Usage:
    network = ReactionNetworkGraph()
    network.add_reaction("rxn_001", "Tryptophan decarboxylation", ...)
    pathway = network.find_pathway("L-Tryptophan", "Psilocybin")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import deque


@dataclass
class ReactionNode:
    """Represents a reaction in the network."""
    id: str
    name: str
    enzyme: str
    reaction_type: str
    reversible: bool = False
    rate_constant: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompoundNode:
    """Represents a compound in the network."""
    id: str
    name: str
    formula: str = ""
    smiles: str = ""
    is_primary: bool = False  # Primary metabolite
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReactionNetworkGraph:
    """
    Graph representation of biochemical reaction networks.
    
    Provides methods for pathway finding, flux analysis, and
    network visualization.
    """
    
    def __init__(self):
        """Initialize the reaction network."""
        self.compounds: Dict[str, CompoundNode] = {}
        self.reactions: Dict[str, ReactionNode] = {}
        
        # Edges: compound -> list of reactions that consume it
        self.compound_to_reactions: Dict[str, List[str]] = {}
        # Edges: reaction -> list of products
        self.reaction_to_products: Dict[str, List[str]] = {}
        # Edges: reaction -> list of substrates
        self.reaction_to_substrates: Dict[str, List[str]] = {}
        
        print("Initialized ReactionNetworkGraph")
    
    def add_compound(
        self,
        compound_id: str,
        name: str,
        formula: str = "",
        smiles: str = "",
        is_primary: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a compound node to the network."""
        compound = CompoundNode(
            id=compound_id,
            name=name,
            formula=formula,
            smiles=smiles,
            is_primary=is_primary,
            metadata=metadata or {},
        )
        self.compounds[compound_id] = compound
        
        if compound_id not in self.compound_to_reactions:
            self.compound_to_reactions[compound_id] = []
    
    def add_reaction(
        self,
        reaction_id: str,
        name: str,
        substrates: List[str],
        products: List[str],
        enzyme: str = "",
        reaction_type: str = "enzymatic",
        reversible: bool = False,
        rate_constant: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a reaction to the network.
        
        Args:
            reaction_id: Unique reaction identifier
            name: Reaction name
            substrates: List of substrate compound IDs
            products: List of product compound IDs
            enzyme: Enzyme name
            reaction_type: Type of reaction
            reversible: Whether reaction is reversible
            rate_constant: Reaction rate constant
            metadata: Additional metadata
        """
        reaction = ReactionNode(
            id=reaction_id,
            name=name,
            enzyme=enzyme,
            reaction_type=reaction_type,
            reversible=reversible,
            rate_constant=rate_constant,
            metadata=metadata or {},
        )
        self.reactions[reaction_id] = reaction
        
        # Add edges
        self.reaction_to_substrates[reaction_id] = substrates
        self.reaction_to_products[reaction_id] = products
        
        # Link compounds to reactions
        for substrate in substrates:
            if substrate not in self.compound_to_reactions:
                self.compound_to_reactions[substrate] = []
            self.compound_to_reactions[substrate].append(reaction_id)
    
    def find_pathway(
        self,
        start_compound: str,
        end_compound: str,
        max_steps: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """
        Find a pathway between two compounds using BFS.
        
        Args:
            start_compound: Starting compound ID or name
            end_compound: Target compound ID or name
            max_steps: Maximum pathway length
        
        Returns:
            Pathway dictionary or None if not found
        """
        # Resolve compound names to IDs
        start_id = self._resolve_compound(start_compound)
        end_id = self._resolve_compound(end_compound)
        
        if not start_id or not end_id:
            return None
        
        # BFS for shortest path
        queue = deque([(start_id, [], 0)])
        visited = set()
        
        while queue:
            current, path, depth = queue.popleft()
            
            if depth > max_steps:
                continue
            
            if current == end_id:
                return {
                    "start": start_compound,
                    "end": end_compound,
                    "pathway": path,
                    "steps": len(path),
                    "found": True,
                }
            
            if current in visited:
                continue
            visited.add(current)
            
            # Explore reactions that consume this compound
            for reaction_id in self.compound_to_reactions.get(current, []):
                reaction = self.reactions.get(reaction_id)
                if not reaction:
                    continue
                
                # Get products of this reaction
                for product in self.reaction_to_products.get(reaction_id, []):
                    if product not in visited:
                        new_path = path + [{
                            "reaction_id": reaction_id,
                            "reaction_name": reaction.name,
                            "enzyme": reaction.enzyme,
                            "substrate": current,
                            "product": product,
                        }]
                        queue.append((product, new_path, depth + 1))
        
        return {
            "start": start_compound,
            "end": end_compound,
            "pathway": [],
            "steps": 0,
            "found": False,
            "message": "No pathway found within step limit",
        }
    
    def _resolve_compound(self, compound: str) -> Optional[str]:
        """Resolve compound name or ID."""
        # Direct ID match
        if compound in self.compounds:
            return compound
        
        # Name match
        for comp_id, comp in self.compounds.items():
            if comp.name.lower() == compound.lower():
                return comp_id
        
        return None
    
    def get_all_pathways(
        self,
        start_compound: str,
        end_compound: str,
        max_steps: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Find all pathways between two compounds.
        
        Args:
            start_compound: Starting compound
            end_compound: Target compound
            max_steps: Maximum pathway length
        
        Returns:
            List of pathway dictionaries
        """
        start_id = self._resolve_compound(start_compound)
        end_id = self._resolve_compound(end_compound)
        
        if not start_id or not end_id:
            return []
        
        all_pathways = []
        
        def dfs(current: str, path: List[Dict], visited: Set[str]):
            if len(path) > max_steps:
                return
            
            if current == end_id:
                all_pathways.append({
                    "pathway": path.copy(),
                    "steps": len(path),
                })
                return
            
            for reaction_id in self.compound_to_reactions.get(current, []):
                reaction = self.reactions.get(reaction_id)
                if not reaction:
                    continue
                
                for product in self.reaction_to_products.get(reaction_id, []):
                    if product not in visited:
                        visited.add(product)
                        path.append({
                            "reaction_id": reaction_id,
                            "reaction_name": reaction.name,
                            "enzyme": reaction.enzyme,
                            "substrate": current,
                            "product": product,
                        })
                        dfs(product, path, visited)
                        path.pop()
                        visited.remove(product)
        
        dfs(start_id, [], {start_id})
        
        # Sort by path length
        all_pathways.sort(key=lambda x: x["steps"])
        
        return all_pathways
    
    def calculate_flux(
        self,
        pathway: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate theoretical flux through a pathway.
        
        Args:
            pathway: List of pathway steps
        
        Returns:
            Estimated flux value
        """
        if not pathway:
            return 0.0
        
        # Flux is limited by slowest step (rate-limiting)
        min_rate = float("inf")
        
        for step in pathway:
            reaction_id = step.get("reaction_id")
            reaction = self.reactions.get(reaction_id)
            if reaction:
                min_rate = min(min_rate, reaction.rate_constant)
        
        return min_rate if min_rate != float("inf") else 0.0
    
    def get_compound_neighbors(
        self,
        compound_id: str,
        direction: str = "both",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get neighboring compounds connected by reactions.
        
        Args:
            compound_id: Compound to query
            direction: "forward", "backward", or "both"
        
        Returns:
            Dictionary with upstream and downstream compounds
        """
        result = {"upstream": [], "downstream": []}
        
        if direction in ("forward", "both"):
            # Find reactions consuming this compound
            for reaction_id in self.compound_to_reactions.get(compound_id, []):
                reaction = self.reactions.get(reaction_id)
                if reaction:
                    for product in self.reaction_to_products.get(reaction_id, []):
                        comp = self.compounds.get(product)
                        if comp:
                            result["downstream"].append({
                                "compound_id": product,
                                "name": comp.name,
                                "reaction": reaction.name,
                            })
        
        if direction in ("backward", "both"):
            # Find reactions producing this compound
            for reaction_id, products in self.reaction_to_products.items():
                if compound_id in products:
                    reaction = self.reactions.get(reaction_id)
                    if reaction:
                        for substrate in self.reaction_to_substrates.get(reaction_id, []):
                            comp = self.compounds.get(substrate)
                            if comp:
                                result["upstream"].append({
                                    "compound_id": substrate,
                                    "name": comp.name,
                                    "reaction": reaction.name,
                                })
        
        return result
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the reaction network."""
        return {
            "num_compounds": len(self.compounds),
            "num_reactions": len(self.reactions),
            "num_primary_metabolites": sum(1 for c in self.compounds.values() if c.is_primary),
            "reversible_reactions": sum(1 for r in self.reactions.values() if r.reversible),
        }
    
    def export_for_visualization(self) -> Dict[str, Any]:
        """Export network for visualization."""
        nodes = []
        edges = []
        
        # Add compound nodes
        for comp_id, comp in self.compounds.items():
            nodes.append({
                "id": comp_id,
                "label": comp.name,
                "type": "compound",
                "is_primary": comp.is_primary,
            })
        
        # Add reaction nodes and edges
        for rxn_id, rxn in self.reactions.items():
            nodes.append({
                "id": rxn_id,
                "label": rxn.name,
                "type": "reaction",
                "enzyme": rxn.enzyme,
            })
            
            # Substrate -> Reaction edges
            for substrate in self.reaction_to_substrates.get(rxn_id, []):
                edges.append({
                    "source": substrate,
                    "target": rxn_id,
                    "type": "substrate",
                })
            
            # Reaction -> Product edges
            for product in self.reaction_to_products.get(rxn_id, []):
                edges.append({
                    "source": rxn_id,
                    "target": product,
                    "type": "product",
                })
        
        return {"nodes": nodes, "edges": edges}


def create_psilocybin_network() -> ReactionNetworkGraph:
    """Create a sample psilocybin biosynthesis network."""
    network = ReactionNetworkGraph()
    
    # Add compounds
    network.add_compound("trp", "L-Tryptophan", "C11H12N2O2", is_primary=True)
    network.add_compound("tryptamine", "Tryptamine", "C10H12N2")
    network.add_compound("4ht", "4-Hydroxytryptamine", "C10H12N2O")
    network.add_compound("norbae", "Norbaeocystin", "C10H13N2O4P")
    network.add_compound("bae", "Baeocystin", "C11H15N2O4P")
    network.add_compound("psilocybin", "Psilocybin", "C12H17N2O4P")
    network.add_compound("psilocin", "Psilocin", "C12H16N2O")
    
    # Add reactions
    network.add_reaction(
        "rxn_1", "Tryptophan decarboxylation",
        substrates=["trp"], products=["tryptamine"],
        enzyme="PsiD (Tryptophan decarboxylase)"
    )
    network.add_reaction(
        "rxn_2", "4-Hydroxylation",
        substrates=["tryptamine"], products=["4ht"],
        enzyme="PsiH (P450 hydroxylase)"
    )
    network.add_reaction(
        "rxn_3", "Phosphorylation",
        substrates=["4ht"], products=["norbae"],
        enzyme="PsiK (Kinase)"
    )
    network.add_reaction(
        "rxn_4", "First methylation",
        substrates=["norbae"], products=["bae"],
        enzyme="PsiM (Methyltransferase)"
    )
    network.add_reaction(
        "rxn_5", "Second methylation",
        substrates=["bae"], products=["psilocybin"],
        enzyme="PsiM (Methyltransferase)"
    )
    network.add_reaction(
        "rxn_6", "Dephosphorylation",
        substrates=["psilocybin"], products=["psilocin"],
        enzyme="Alkaline phosphatase",
        reversible=False
    )
    
    return network
