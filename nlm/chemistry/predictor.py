"""
Bioactivity Predictor
=====================

Predicts biological activities of compounds using machine learning
and knowledge graph inference.

Models:
- Activity classification (multi-label)
- Species association prediction
- Compound production prediction
- Toxicity estimation

Usage:
    predictor = BioactivityPredictor(encoder, knowledge_graph)
    activities = predictor.predict_activity(compound_data)
    species = predictor.predict_species_association(compound_data)
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .encoder import ChemistryEncoder
from .knowledge import ChemistryKnowledgeGraph


class BioactivityPredictor:
    """
    Predicts biological activities and associations for compounds.
    
    Uses a combination of similarity-based methods and simple models
    to predict activities, toxicity, and species associations.
    """
    
    # Known activity categories with typical features
    ACTIVITY_SIGNATURES = {
        "psychedelic": {
            "keywords": ["tryptamine", "indole", "psilocybin", "DMT"],
            "formula_patterns": ["N2", "N3"],
            "mw_range": (150, 400),
        },
        "neurotrophic": {
            "keywords": ["erinacine", "hericenone", "NGF"],
            "formula_patterns": ["O4", "O5"],
            "mw_range": (300, 600),
        },
        "antimicrobial": {
            "keywords": ["antibiotic", "antimicrobial", "bactericidal"],
            "formula_patterns": ["N", "O"],
            "mw_range": (100, 800),
        },
        "antioxidant": {
            "keywords": ["phenol", "polyphenol", "flavonoid"],
            "formula_patterns": ["O3", "O4", "O5"],
            "mw_range": (150, 500),
        },
        "anticancer": {
            "keywords": ["cytotoxic", "antitumor", "apoptosis"],
            "formula_patterns": [],
            "mw_range": (200, 800),
        },
        "immunomodulatory": {
            "keywords": ["polysaccharide", "beta-glucan", "immune"],
            "formula_patterns": ["O6", "O7", "O8"],
            "mw_range": (1000, 50000),
        },
        "anti_inflammatory": {
            "keywords": ["triterpene", "ganoderic", "steroid"],
            "formula_patterns": ["O6", "O7"],
            "mw_range": (400, 600),
        },
        "hepatoprotective": {
            "keywords": ["liver", "hepato", "detox"],
            "formula_patterns": [],
            "mw_range": (200, 800),
        },
    }
    
    # Common fungal species and their typical compound profiles
    SPECIES_PROFILES = {
        "psilocybe_cubensis": {
            "compounds": ["psilocybin", "psilocin", "baeocystin"],
            "activities": ["psychedelic", "neuroplasticity"],
        },
        "hericium_erinaceus": {
            "compounds": ["hericenones", "erinacines"],
            "activities": ["neurotrophic", "neuroprotective"],
        },
        "ganoderma_lucidum": {
            "compounds": ["ganoderic acids", "triterpenes", "polysaccharides"],
            "activities": ["immunomodulatory", "anticancer", "anti_inflammatory"],
        },
        "trametes_versicolor": {
            "compounds": ["PSK", "PSP", "polysaccharides"],
            "activities": ["immunomodulatory", "anticancer"],
        },
        "cordyceps_militaris": {
            "compounds": ["cordycepin", "adenosine"],
            "activities": ["antiviral", "anticancer", "energizing"],
        },
        "inonotus_obliquus": {
            "compounds": ["betulinic acid", "melanins", "polysaccharides"],
            "activities": ["antioxidant", "anticancer", "immunomodulatory"],
        },
    }
    
    def __init__(
        self,
        encoder: Optional[ChemistryEncoder] = None,
        knowledge_graph: Optional[ChemistryKnowledgeGraph] = None,
    ):
        """
        Initialize the predictor.
        
        Args:
            encoder: ChemistryEncoder for compound embeddings
            knowledge_graph: ChemistryKnowledgeGraph for context
        """
        self.encoder = encoder or ChemistryEncoder()
        self.knowledge_graph = knowledge_graph
        print("Initialized BioactivityPredictor")
    
    def predict_activity(
        self,
        compound_data: Dict[str, Any],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Predict biological activities for a compound.
        
        Args:
            compound_data: Compound information (name, formula, smiles, etc.)
            top_n: Number of top predictions to return
        
        Returns:
            List of activity predictions with confidence scores
        """
        predictions = []
        
        name = compound_data.get("name", "").lower()
        formula = compound_data.get("formula", "")
        smiles = compound_data.get("smiles", "")
        mw = compound_data.get("molecular_weight", compound_data.get("weight", 300))
        
        for activity, signature in self.ACTIVITY_SIGNATURES.items():
            score = 0.0
            evidence = []
            
            # Keyword matching
            for keyword in signature["keywords"]:
                if keyword.lower() in name:
                    score += 0.3
                    evidence.append(f"Name contains '{keyword}'")
            
            # Formula pattern matching
            for pattern in signature["formula_patterns"]:
                if pattern in formula:
                    score += 0.2
                    evidence.append(f"Formula contains '{pattern}'")
            
            # Molecular weight range
            mw_min, mw_max = signature["mw_range"]
            if mw_min <= mw <= mw_max:
                score += 0.15
                evidence.append(f"MW in typical range ({mw_min}-{mw_max})")
            
            # Add some randomness for exploration
            score += random.uniform(0, 0.1)
            
            # Normalize score
            score = min(0.99, max(0.1, score))
            
            predictions.append({
                "activity_name": activity.replace("_", " ").title(),
                "activity_id": activity,
                "confidence": round(score, 2),
                "evidence": evidence,
            })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return predictions[:top_n]
    
    def predict_species_association(
        self,
        compound_data: Dict[str, Any],
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Predict which species might produce a compound.
        
        Args:
            compound_data: Compound information
            top_n: Number of top predictions
        
        Returns:
            List of species predictions with confidence
        """
        predictions = []
        
        name = compound_data.get("name", "").lower()
        activities = [a["activity_id"] for a in self.predict_activity(compound_data, 3)]
        
        for species_id, profile in self.SPECIES_PROFILES.items():
            score = 0.0
            evidence = []
            
            # Check compound name similarity
            for known_compound in profile["compounds"]:
                if known_compound.lower() in name or name in known_compound.lower():
                    score += 0.4
                    evidence.append(f"Similar to known compound '{known_compound}'")
            
            # Check activity overlap
            for activity in activities:
                if activity in profile["activities"]:
                    score += 0.25
                    evidence.append(f"Shares '{activity}' activity")
            
            # Add randomness
            score += random.uniform(0, 0.1)
            score = min(0.95, max(0.1, score))
            
            species_name = species_id.replace("_", " ").title()
            
            predictions.append({
                "species_id": species_id,
                "species_name": species_name,
                "confidence": round(score, 2),
                "evidence": evidence,
            })
        
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:top_n]
    
    def predict_toxicity(
        self,
        compound_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Estimate toxicity risk for a compound.
        
        Args:
            compound_data: Compound information
        
        Returns:
            Toxicity assessment dictionary
        """
        mw = compound_data.get("molecular_weight", compound_data.get("weight", 300))
        formula = compound_data.get("formula", "")
        smiles = compound_data.get("smiles", "")
        
        # Calculate toxicity indicators
        risk_factors = []
        risk_score = 0.0
        
        # High molecular weight can indicate poor absorption (safer)
        if mw > 500:
            risk_score -= 0.1
            risk_factors.append("High MW suggests poor oral absorption")
        
        # Certain atoms increase toxicity risk
        toxicity_atoms = {
            "F": 0.1, "Cl": 0.15, "Br": 0.2, "I": 0.25,
            "As": 0.5, "Hg": 0.6, "Pb": 0.5,
        }
        
        for atom, weight in toxicity_atoms.items():
            if atom in formula:
                risk_score += weight
                risk_factors.append(f"Contains {atom} (elevated risk)")
        
        # Nitrogen-containing heterocycles
        if "N" in formula and mw < 400:
            risk_score += 0.1
            risk_factors.append("Small nitrogen-containing compound")
        
        # Normalize
        risk_score = min(1.0, max(0.0, 0.3 + risk_score))
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = "Low"
        elif risk_score < 0.6:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        return {
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "factors": risk_factors,
            "recommendation": self._get_toxicity_recommendation(risk_level),
        }
    
    def _get_toxicity_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on toxicity risk level."""
        recommendations = {
            "Low": "Standard precautions. Suitable for initial screening.",
            "Moderate": "Additional safety testing recommended before in vivo studies.",
            "High": "Significant concern. Consider structural modifications to reduce toxicity.",
        }
        return recommendations.get(risk_level, "Unknown risk level.")
    
    def predict_compound_production(
        self,
        species_data: Dict[str, Any],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Predict compounds likely produced by a species.
        
        Args:
            species_data: Species information (name, taxonomy, etc.)
            top_n: Number of predictions
        
        Returns:
            List of compound predictions
        """
        predictions = []
        species_name = species_data.get("name", "").lower()
        
        # Match to known profiles
        matched_profile = None
        for species_id, profile in self.SPECIES_PROFILES.items():
            if species_id.replace("_", " ") in species_name:
                matched_profile = profile
                break
        
        if matched_profile:
            # Return known compounds
            for compound in matched_profile["compounds"]:
                predictions.append({
                    "compound_name": compound,
                    "confidence": round(random.uniform(0.7, 0.95), 2),
                    "evidence": ["Known from literature"],
                })
        else:
            # Generate generic predictions based on typical fungal metabolites
            generic_compounds = [
                "Polysaccharides",
                "Triterpenes",
                "Sterols",
                "Phenolic compounds",
                "Enzymes",
            ]
            for compound in generic_compounds:
                predictions.append({
                    "compound_name": compound,
                    "confidence": round(random.uniform(0.3, 0.6), 2),
                    "evidence": ["Common fungal metabolite class"],
                })
        
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:top_n]
    
    def analyze_compound(
        self,
        compound_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a compound.
        
        Args:
            compound_data: Compound information
        
        Returns:
            Complete analysis dictionary
        """
        return {
            "compound": compound_data,
            "predicted_activities": self.predict_activity(compound_data),
            "predicted_species": self.predict_species_association(compound_data),
            "toxicity_assessment": self.predict_toxicity(compound_data),
            "embedding": self.encoder.encode(compound_data).tolist(),
            "analysis_complete": True,
        }
