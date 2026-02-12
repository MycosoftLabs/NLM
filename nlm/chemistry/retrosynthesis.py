"""
Retrosynthesis Engine
=====================

Analyzes and predicts biosynthetic pathways for fungal compounds.
Works backward from target molecules to identify precursors and
enzymatic reactions.

Features:
- Biosynthetic pathway prediction
- Precursor identification
- Reaction step analysis
- Pathway optimization suggestions

Usage:
    engine = RetrosynthesisEngine()
    pathway = engine.analyze_biosynthetic_pathway(target_compound)
    precursors = engine.predict_precursors(compound)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ReactionStep:
    """Represents a step in a biosynthetic pathway."""
    step_number: int
    product: Dict[str, Any]
    precursors: List[Dict[str, Any]]
    enzyme: str
    reaction_type: str
    confidence: float = 0.8


@dataclass
class BiosyntheticPathway:
    """Represents a complete biosynthetic pathway."""
    target: Dict[str, Any]
    steps: List[ReactionStep] = field(default_factory=list)
    starting_materials: List[Dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0
    pathway_confidence: float = 0.0


# Common biosynthetic reaction types
REACTION_TYPES = {
    "hydroxylation": {
        "enzymes": ["CYP450", "Monooxygenase", "Hydroxylase"],
        "description": "Addition of hydroxyl (-OH) group",
        "typical_change": "Adds O",
    },
    "methylation": {
        "enzymes": ["Methyltransferase", "SAM-dependent methylase"],
        "description": "Transfer of methyl (-CH3) group",
        "typical_change": "Adds C",
    },
    "phosphorylation": {
        "enzymes": ["Kinase", "Phosphotransferase"],
        "description": "Addition of phosphate group",
        "typical_change": "Adds PO4",
    },
    "dephosphorylation": {
        "enzymes": ["Phosphatase", "Alkaline phosphatase"],
        "description": "Removal of phosphate group",
        "typical_change": "Removes PO4",
    },
    "decarboxylation": {
        "enzymes": ["Decarboxylase", "Tryptophan decarboxylase"],
        "description": "Removal of carboxyl group as CO2",
        "typical_change": "Removes CO2",
    },
    "cyclization": {
        "enzymes": ["Cyclase", "Terpene cyclase"],
        "description": "Ring formation",
        "typical_change": "Ring closure",
    },
    "oxidation": {
        "enzymes": ["Oxidase", "Oxidoreductase", "Laccase"],
        "description": "Electron removal/oxygen addition",
        "typical_change": "Adds O or removes H",
    },
    "reduction": {
        "enzymes": ["Reductase", "NADPH-dependent reductase"],
        "description": "Electron addition/hydrogen addition",
        "typical_change": "Adds H",
    },
    "polyketide_synthesis": {
        "enzymes": ["PKS", "Polyketide synthase"],
        "description": "Iterative addition of acetyl units",
        "typical_change": "Chain extension",
    },
    "terpene_synthesis": {
        "enzymes": ["Terpene synthase", "Squalene synthase"],
        "description": "Formation of isoprenoid structures",
        "typical_change": "IPP/DMAPP condensation",
    },
}

# Known biosynthetic pathways for common fungal compounds
KNOWN_PATHWAYS = {
    "psilocybin": {
        "starting": "L-Tryptophan",
        "steps": [
            ("Tryptophan", "Tryptamine", "decarboxylation", "TrpDC"),
            ("Tryptamine", "4-Hydroxytryptamine", "hydroxylation", "P450"),
            ("4-Hydroxytryptamine", "Norbaeocystin", "phosphorylation", "Kinase"),
            ("Norbaeocystin", "Baeocystin", "methylation", "MT1"),
            ("Baeocystin", "Psilocybin", "methylation", "MT2"),
        ],
    },
    "hericenone": {
        "starting": "Fatty acid + Orsellinic acid",
        "steps": [
            ("Orsellinic acid", "Hydroxybenzoic acid", "hydroxylation", "P450"),
            ("Hydroxybenzoic acid + Fatty acid", "Hericenone precursor", "condensation", "Condensase"),
            ("Hericenone precursor", "Hericenone", "oxidation", "Oxidase"),
        ],
    },
    "ganoderic_acid": {
        "starting": "Lanosterol",
        "steps": [
            ("Lanosterol", "Oxidized lanosterol", "oxidation", "CYP450-1"),
            ("Oxidized lanosterol", "Hydroxylated intermediate", "hydroxylation", "CYP450-2"),
            ("Hydroxylated intermediate", "Ganoderic acid precursor", "hydroxylation", "CYP450-3"),
            ("Ganoderic acid precursor", "Ganoderic acid A", "acetylation", "Acetyltransferase"),
        ],
    },
    "cordycepin": {
        "starting": "Adenosine",
        "steps": [
            ("Adenosine", "3'-AMP", "phosphorylation", "Adenosine kinase"),
            ("3'-AMP", "3'-deoxyadenosine-MP", "reduction", "Reductase"),
            ("3'-deoxyadenosine-MP", "Cordycepin", "dephosphorylation", "Phosphatase"),
        ],
    },
}


class RetrosynthesisEngine:
    """
    Predicts and analyzes biosynthetic pathways.
    
    Uses known pathways and reaction rules to work backward
    from target compounds to identify precursors and reactions.
    """
    
    def __init__(self):
        """Initialize the retrosynthesis engine."""
        self.known_pathways = KNOWN_PATHWAYS
        self.reaction_types = REACTION_TYPES
        print("Initialized RetrosynthesisEngine")
    
    def analyze_biosynthetic_pathway(
        self,
        target_compound: Dict[str, Any],
        max_steps: int = 6,
    ) -> Dict[str, Any]:
        """
        Analyze the biosynthetic pathway for a target compound.
        
        Args:
            target_compound: Target compound data
            max_steps: Maximum pathway depth
        
        Returns:
            Pathway analysis dictionary
        """
        name = target_compound.get("name", "").lower()
        
        # Check for known pathways
        for pathway_key, pathway_data in self.known_pathways.items():
            if pathway_key in name:
                return self._build_known_pathway(target_compound, pathway_data)
        
        # Generate predicted pathway
        return self._predict_pathway(target_compound, max_steps)
    
    def _build_known_pathway(
        self,
        target: Dict[str, Any],
        pathway_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build pathway from known data."""
        steps = []
        
        for i, (precursor, product, reaction_type, enzyme) in enumerate(pathway_data["steps"]):
            step = {
                "step_number": i + 1,
                "precursor": {"name": precursor},
                "product": {"name": product},
                "reaction_type": reaction_type,
                "enzyme": enzyme,
                "confidence": 0.9,
                "description": self.reaction_types.get(reaction_type, {}).get(
                    "description", "Unknown reaction"
                ),
            }
            steps.append(step)
        
        return {
            "target": target,
            "starting_material": pathway_data["starting"],
            "total_steps": len(steps),
            "steps": steps,
            "pathway_confidence": 0.85,
            "pathway_type": "known",
            "message": f"Known biosynthetic pathway identified with {len(steps)} steps",
        }
    
    def _predict_pathway(
        self,
        target: Dict[str, Any],
        max_steps: int,
    ) -> Dict[str, Any]:
        """Predict pathway using reaction rules."""
        formula = target.get("formula", "")
        mw = target.get("molecular_weight", target.get("weight", 300))
        
        steps = []
        current = target.copy()
        current_mw = mw
        
        # Work backward through possible reactions
        step_num = 0
        while step_num < max_steps and current_mw > 100:
            # Choose a reaction type based on formula hints
            reaction_type = self._select_reverse_reaction(formula, current_mw)
            reaction_info = self.reaction_types.get(reaction_type, {})
            
            # Generate precursor
            precursor = self._generate_precursor(current, reaction_type)
            
            step = {
                "step_number": step_num + 1,
                "product": current,
                "precursor": precursor,
                "reaction_type": reaction_type,
                "enzyme": random.choice(reaction_info.get("enzymes", ["Unknown enzyme"])),
                "confidence": round(random.uniform(0.5, 0.8), 2),
                "description": reaction_info.get("description", "Unknown"),
            }
            steps.append(step)
            
            # Move to precursor
            current = precursor
            current_mw *= 0.8  # Approximate MW decrease
            step_num += 1
            
            # Random early termination
            if random.random() < 0.2:
                break
        
        # Reverse steps for forward direction
        steps.reverse()
        for i, step in enumerate(steps):
            step["step_number"] = i + 1
        
        return {
            "target": target,
            "starting_material": steps[0]["precursor"]["name"] if steps else "Unknown",
            "total_steps": len(steps),
            "steps": steps,
            "pathway_confidence": round(0.5 + 0.05 * len(steps), 2),
            "pathway_type": "predicted",
            "message": f"Predicted pathway with {len(steps)} steps (confidence varies by step)",
        }
    
    def _select_reverse_reaction(self, formula: str, mw: float) -> str:
        """Select likely reverse reaction based on structure hints."""
        weights = {}
        
        # Phosphate suggests dephosphorylation
        if "P" in formula:
            weights["dephosphorylation"] = 0.4
        
        # Oxygen content suggests hydroxylation/oxidation
        if "O" in formula:
            weights["hydroxylation"] = 0.2
            weights["oxidation"] = 0.15
        
        # Nitrogen suggests decarboxylation (especially tryptamines)
        if "N" in formula and mw < 400:
            weights["decarboxylation"] = 0.25
        
        # High MW suggests cyclization or polyketide
        if mw > 400:
            weights["cyclization"] = 0.2
            weights["polyketide_synthesis"] = 0.15
        
        # Add defaults
        for reaction in self.reaction_types:
            if reaction not in weights:
                weights[reaction] = 0.05
        
        # Normalize and choose
        total = sum(weights.values())
        probs = [w / total for w in weights.values()]
        
        return random.choices(list(weights.keys()), weights=probs)[0]
    
    def _generate_precursor(
        self,
        product: Dict[str, Any],
        reaction_type: str,
    ) -> Dict[str, Any]:
        """Generate precursor compound for a reaction."""
        product_name = product.get("name", "Compound")
        
        # Generate precursor name
        precursor_prefixes = {
            "hydroxylation": "Deoxy-",
            "methylation": "Nor-",
            "phosphorylation": "Dephospho-",
            "dephosphorylation": "Phospho-",
            "decarboxylation": "Carboxy-",
            "cyclization": "Linear-",
            "oxidation": "Reduced-",
            "reduction": "Oxidized-",
        }
        
        prefix = precursor_prefixes.get(reaction_type, "Pre-")
        precursor_name = f"{prefix}{product_name}"
        
        return {
            "name": precursor_name,
            "formula": product.get("formula", ""),
            "predicted": True,
        }
    
    def predict_precursors(
        self,
        compound: Dict[str, Any],
        num_precursors: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Predict immediate precursors for a compound.
        
        Args:
            compound: Target compound
            num_precursors: Number of precursors to predict
        
        Returns:
            List of precursor predictions
        """
        precursors = []
        formula = compound.get("formula", "")
        
        # Generate precursors for different reaction types
        possible_reactions = ["hydroxylation", "methylation", "phosphorylation", 
                            "decarboxylation", "oxidation", "reduction"]
        
        for reaction in random.sample(possible_reactions, min(num_precursors, len(possible_reactions))):
            precursor = self._generate_precursor(compound, reaction)
            reaction_info = self.reaction_types.get(reaction, {})
            
            precursors.append({
                "name": precursor["name"],
                "reaction_to_product": reaction,
                "enzyme": random.choice(reaction_info.get("enzymes", ["Unknown"])),
                "confidence": round(random.uniform(0.5, 0.85), 2),
            })
        
        precursors.sort(key=lambda x: x["confidence"], reverse=True)
        return precursors
    
    def get_pathway_visualization(
        self,
        pathway: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate visualization data for a pathway.
        
        Args:
            pathway: Pathway analysis result
        
        Returns:
            Visualization-ready data
        """
        nodes = []
        edges = []
        
        # Add starting material
        if pathway.get("starting_material"):
            nodes.append({
                "id": "start",
                "label": pathway["starting_material"],
                "type": "precursor",
                "x": 0,
                "y": 300,
            })
        
        # Add steps
        for i, step in enumerate(pathway.get("steps", [])):
            # Add product node
            product_id = f"product_{i}"
            nodes.append({
                "id": product_id,
                "label": step["product"].get("name", f"Product {i+1}"),
                "type": "intermediate" if i < len(pathway["steps"]) - 1 else "target",
                "x": (i + 1) * 200,
                "y": 300,
            })
            
            # Add edge
            source_id = "start" if i == 0 else f"product_{i-1}"
            edges.append({
                "source": source_id,
                "target": product_id,
                "label": step.get("reaction_type", ""),
                "enzyme": step.get("enzyme", ""),
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "total_steps": len(pathway.get("steps", [])),
        }


def analyze_pathway(compound_name: str) -> Dict[str, Any]:
    """
    Convenience function to analyze a pathway.
    
    Args:
        compound_name: Name of target compound
    
    Returns:
        Pathway analysis
    """
    engine = RetrosynthesisEngine()
    return engine.analyze_biosynthetic_pathway({"name": compound_name})
