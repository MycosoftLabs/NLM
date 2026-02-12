"""
Computational Alchemy Laboratory
================================

Design and simulate novel chemical compounds with predicted properties
and biological activities. Combines generative chemistry with property
prediction for rational drug/compound design.

Features:
- Compound structure generation
- Property prediction
- Activity optimization
- Virtual screening
- Molecular modification suggestions

Usage:
    lab = ComputationalAlchemyLab()
    design = lab.design_compound(design_parameters)
    optimized = lab.optimize_for_activity(compound, "anticancer")
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .encoder import ChemistryEncoder
from .predictor import BioactivityPredictor


@dataclass
class DesignedCompound:
    """Represents a computationally designed compound."""
    id: str
    name: str
    formula: str
    molecular_weight: float
    predicted_smiles: str
    predicted_properties: Dict[str, Any] = field(default_factory=dict)
    predicted_activities: List[Dict[str, Any]] = field(default_factory=list)
    design_parameters: Dict[str, Any] = field(default_factory=dict)
    optimization_score: float = 0.0


class ComputationalAlchemyLab:
    """
    Computational laboratory for designing novel compounds.
    
    Uses a combination of rule-based generation and ML-guided
    optimization to create compounds with desired properties.
    """
    
    # Molecular building blocks
    SCAFFOLDS = {
        "tryptamine": {
            "base_formula": "C10H12N2",
            "base_mw": 160.22,
            "activities": ["psychedelic", "neuroactive"],
        },
        "steroid": {
            "base_formula": "C27H44O",
            "base_mw": 388.65,
            "activities": ["anti_inflammatory", "hormonal"],
        },
        "triterpene": {
            "base_formula": "C30H48O3",
            "base_mw": 456.70,
            "activities": ["anticancer", "hepatoprotective"],
        },
        "polyketide": {
            "base_formula": "C20H24O6",
            "base_mw": 360.40,
            "activities": ["antimicrobial", "cytotoxic"],
        },
        "peptide": {
            "base_formula": "C12H22N4O4",
            "base_mw": 286.33,
            "activities": ["antimicrobial", "immunomodulatory"],
        },
        "polysaccharide": {
            "base_formula": "C24H42O21",
            "base_mw": 666.58,
            "activities": ["immunomodulatory", "antioxidant"],
        },
    }
    
    # Modification groups
    MODIFICATIONS = {
        "hydroxyl": {"formula_add": "O", "mw_add": 16.0, "effect": "hydrophilicity"},
        "methyl": {"formula_add": "C", "mw_add": 14.0, "effect": "lipophilicity"},
        "amino": {"formula_add": "N", "mw_add": 15.0, "effect": "basicity"},
        "phosphate": {"formula_add": "PO4", "mw_add": 95.0, "effect": "prodrug"},
        "acetyl": {"formula_add": "C2O", "mw_add": 42.0, "effect": "stability"},
        "fluorine": {"formula_add": "F", "mw_add": 19.0, "effect": "metabolic_stability"},
    }
    
    def __init__(
        self,
        encoder: Optional[ChemistryEncoder] = None,
        predictor: Optional[BioactivityPredictor] = None,
    ):
        """
        Initialize the Alchemy Lab.
        
        Args:
            encoder: ChemistryEncoder for embeddings
            predictor: BioactivityPredictor for activity prediction
        """
        self.encoder = encoder or ChemistryEncoder()
        self.predictor = predictor or BioactivityPredictor()
        self.designed_compounds: List[DesignedCompound] = []
        print("Initialized ComputationalAlchemyLab")
    
    def design_compound(
        self,
        design_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Design a novel compound based on parameters.
        
        Args:
            design_parameters: Dictionary with:
                - scaffold: Base scaffold type
                - target_activities: Desired activities
                - mw_range: (min, max) molecular weight
                - modifications: List of modifications
        
        Returns:
            Designed compound data
        """
        scaffold = design_parameters.get("scaffold", random.choice(list(self.SCAFFOLDS.keys())))
        target_activities = design_parameters.get("target_activities", [])
        mw_range = design_parameters.get("mw_range", (200, 600))
        modifications = design_parameters.get("modifications", [])
        
        # Get base scaffold
        scaffold_data = self.SCAFFOLDS.get(scaffold, self.SCAFFOLDS["tryptamine"])
        
        # Generate compound ID
        compound_id = f"MYCO-{random.randint(10000, 99999)}"
        
        # Start with base properties
        formula = scaffold_data["base_formula"]
        mw = scaffold_data["base_mw"]
        
        # Apply modifications
        applied_mods = []
        for mod_name in modifications:
            mod = self.MODIFICATIONS.get(mod_name)
            if mod:
                mw += mod["mw_add"]
                applied_mods.append({
                    "name": mod_name,
                    "effect": mod["effect"],
                })
        
        # Adjust to target MW range
        while mw < mw_range[0]:
            mod = random.choice(list(self.MODIFICATIONS.values()))
            mw += mod["mw_add"]
        
        while mw > mw_range[1] and mw > scaffold_data["base_mw"]:
            mw -= random.uniform(10, 30)
        
        mw = max(mw_range[0], min(mw_range[1], mw))
        
        # Generate predicted SMILES (simplified)
        smiles = self._generate_smiles(scaffold, applied_mods)
        
        # Predict properties
        compound_data = {
            "name": f"{scaffold.title()} Derivative {compound_id}",
            "formula": formula,
            "molecular_weight": mw,
            "smiles": smiles,
        }
        
        predicted_properties = self._predict_properties(compound_data)
        predicted_activities = self.predictor.predict_activity(compound_data)
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            predicted_activities,
            target_activities,
        )
        
        # Create compound object
        designed = DesignedCompound(
            id=compound_id,
            name=compound_data["name"],
            formula=formula,
            molecular_weight=round(mw, 2),
            predicted_smiles=smiles,
            predicted_properties=predicted_properties,
            predicted_activities=predicted_activities,
            design_parameters=design_parameters,
            optimization_score=optimization_score,
        )
        
        self.designed_compounds.append(designed)
        
        return {
            "id": compound_id,
            "name": designed.name,
            "formula": formula,
            "molecular_weight": round(mw, 2),
            "smiles": smiles,
            "scaffold": scaffold,
            "modifications": applied_mods,
            "predicted_properties": predicted_properties,
            "predicted_activities": predicted_activities,
            "optimization_score": round(optimization_score, 3),
            "message": f"Successfully designed compound {compound_id}",
        }
    
    def _generate_smiles(
        self,
        scaffold: str,
        modifications: List[Dict[str, Any]],
    ) -> str:
        """Generate a simplified SMILES representation."""
        base_smiles = {
            "tryptamine": "NCCc1c[nH]c2ccccc12",
            "steroid": "CC12CCC3C(CCC4CC(O)CCC43C)C1CCC2O",
            "triterpene": "CC(C)C1CCC2(C)CCC3(C)C(CCC4C5(C)CCC(O)C(C)(C)C5CCC34C)C12",
            "polyketide": "CC1OC(=O)C2=C(O)C(O)=CC=C2C1",
            "peptide": "NCC(=O)NC(C)C(=O)O",
            "polysaccharide": "OCC1OC(O)C(O)C(O)C1O",
        }
        
        smiles = base_smiles.get(scaffold, "CCCC")
        
        # Add modification markers (simplified)
        for mod in modifications:
            if mod["name"] == "hydroxyl":
                smiles = smiles.replace("C", "C(O)", 1)
            elif mod["name"] == "methyl":
                smiles = smiles.replace("C", "C(C)", 1)
            elif mod["name"] == "amino":
                smiles = smiles.replace("C", "C(N)", 1)
        
        return smiles
    
    def _predict_properties(
        self,
        compound_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Predict physicochemical properties."""
        mw = compound_data.get("molecular_weight", 300)
        
        # Rule of 5 predictions
        logp = (mw - 200) / 100 + random.uniform(-1, 1)
        hbd = max(0, int((mw - 100) / 50) + random.randint(-1, 1))
        hba = max(0, int((mw - 50) / 40) + random.randint(-1, 2))
        tpsa = mw * 0.15 + random.uniform(0, 20)
        
        # Lipinski's Rule of 5 compliance
        ro5_violations = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ])
        
        return {
            "molecular_weight": round(mw, 2),
            "logP": round(logp, 2),
            "h_bond_donors": hbd,
            "h_bond_acceptors": hba,
            "tpsa": round(tpsa, 1),
            "rotatable_bonds": max(0, int(mw / 50) + random.randint(-2, 2)),
            "ro5_violations": ro5_violations,
            "drug_likeness": "Good" if ro5_violations <= 1 else "Moderate" if ro5_violations == 2 else "Poor",
        }
    
    def _calculate_optimization_score(
        self,
        predicted_activities: List[Dict[str, Any]],
        target_activities: List[str],
    ) -> float:
        """Calculate how well compound matches target activities."""
        if not target_activities:
            return 0.5
        
        score = 0.0
        for pred in predicted_activities:
            activity_id = pred.get("activity_id", "")
            confidence = pred.get("confidence", 0)
            
            if activity_id in target_activities:
                score += confidence
        
        return score / len(target_activities)
    
    def optimize_for_activity(
        self,
        compound_data: Dict[str, Any],
        target_activity: str,
        iterations: int = 5,
    ) -> Dict[str, Any]:
        """
        Optimize a compound for a specific activity.
        
        Args:
            compound_data: Starting compound
            target_activity: Target activity to optimize for
            iterations: Number of optimization rounds
        
        Returns:
            Optimized compound data
        """
        best = compound_data.copy()
        best_score = 0.0
        
        for i in range(iterations):
            # Generate variations
            variations = self._generate_variations(best)
            
            for variant in variations:
                activities = self.predictor.predict_activity(variant)
                
                for activity in activities:
                    if activity["activity_id"] == target_activity:
                        if activity["confidence"] > best_score:
                            best_score = activity["confidence"]
                            best = variant.copy()
        
        return {
            "original": compound_data,
            "optimized": best,
            "target_activity": target_activity,
            "achieved_confidence": round(best_score, 2),
            "iterations": iterations,
            "modifications_applied": best.get("modifications", []),
        }
    
    def _generate_variations(
        self,
        compound_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate structural variations of a compound."""
        variations = []
        
        for mod_name, mod_data in self.MODIFICATIONS.items():
            variant = compound_data.copy()
            variant["molecular_weight"] = compound_data.get("molecular_weight", 300) + mod_data["mw_add"]
            variant["modifications"] = compound_data.get("modifications", []) + [mod_name]
            variant["name"] = f"{compound_data.get('name', 'Compound')}-{mod_name}"
            variations.append(variant)
        
        return variations
    
    def virtual_screening(
        self,
        target_activity: str,
        num_compounds: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Screen multiple designed compounds for activity.
        
        Args:
            target_activity: Target activity to screen for
            num_compounds: Number of compounds to generate and screen
        
        Returns:
            List of compounds sorted by predicted activity
        """
        results = []
        
        scaffolds = list(self.SCAFFOLDS.keys())
        
        for _ in range(num_compounds):
            scaffold = random.choice(scaffolds)
            mods = random.sample(
                list(self.MODIFICATIONS.keys()),
                random.randint(0, 3),
            )
            
            design_params = {
                "scaffold": scaffold,
                "target_activities": [target_activity],
                "modifications": mods,
            }
            
            compound = self.design_compound(design_params)
            
            # Find score for target activity
            target_score = 0.0
            for activity in compound["predicted_activities"]:
                if activity["activity_id"] == target_activity:
                    target_score = activity["confidence"]
                    break
            
            compound["target_score"] = target_score
            results.append(compound)
        
        # Sort by target score
        results.sort(key=lambda x: x["target_score"], reverse=True)
        
        return results
    
    def suggest_modifications(
        self,
        compound_data: Dict[str, Any],
        target_property: str,
    ) -> List[Dict[str, Any]]:
        """
        Suggest modifications to improve a specific property.
        
        Args:
            compound_data: Current compound
            target_property: Property to improve
        
        Returns:
            List of suggested modifications
        """
        suggestions = []
        
        property_mods = {
            "solubility": ["hydroxyl", "amino"],
            "stability": ["fluorine", "acetyl"],
            "bioavailability": ["methyl", "phosphate"],
            "activity": ["hydroxyl", "amino", "methyl"],
        }
        
        mods = property_mods.get(target_property, list(self.MODIFICATIONS.keys()))
        
        for mod_name in mods:
            mod_data = self.MODIFICATIONS.get(mod_name, {})
            suggestions.append({
                "modification": mod_name,
                "expected_effect": mod_data.get("effect", "unknown"),
                "mw_change": f"+{mod_data.get('mw_add', 0):.1f} Da",
                "rationale": f"Adding {mod_name} group may improve {target_property}",
            })
        
        return suggestions
    
    def get_design_history(self) -> List[Dict[str, Any]]:
        """Get history of designed compounds."""
        return [
            {
                "id": c.id,
                "name": c.name,
                "formula": c.formula,
                "mw": c.molecular_weight,
                "score": c.optimization_score,
            }
            for c in self.designed_compounds
        ]


def run_virtual_screening(
    target_activity: str = "anticancer",
    num_compounds: int = 5,
) -> List[Dict[str, Any]]:
    """
    Convenience function for virtual screening.
    
    Args:
        target_activity: Activity to screen for
        num_compounds: Number of compounds to generate
    
    Returns:
        Screened compounds
    """
    lab = ComputationalAlchemyLab()
    return lab.virtual_screening(target_activity, num_compounds)
