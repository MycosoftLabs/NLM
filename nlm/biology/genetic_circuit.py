"""
Genetic Circuit Simulator
=========================

Simulates gene regulatory networks and biosynthetic pathways in fungi.
Models gene expression dynamics, protein production, and metabolite
accumulation.

Features:
- Hill function kinetics
- Gene interaction networks
- Environmental stress response
- Metabolite production optimization

Usage:
    circuit = CIRCUITS["psilocybin_pathway"]
    simulator = GeneticCircuitSimulator(circuit)
    trajectory = simulator.run_simulation(steps=100)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GeneComponent:
    """Represents a gene in the circuit."""
    id: str
    name: str
    initial_expression: float = 50.0
    max_expression: float = 100.0
    basal_rate: float = 0.1
    degradation_rate: float = 0.05
    color: str = "#22c55e"


@dataclass
class Interaction:
    """Represents an interaction between components."""
    source_id: str
    target_id: str
    interaction_type: str  # "activates", "represses", "produces"
    strength: float = 1.0
    hill_coefficient: float = 2.0
    threshold: float = 50.0


@dataclass
class GeneticCircuit:
    """Defines a complete genetic circuit."""
    id: str
    name: str
    species: str
    product: str
    genes: List[GeneComponent] = field(default_factory=list)
    interactions: List[Interaction] = field(default_factory=list)


# Pre-defined genetic circuits
CIRCUITS = {
    "psilocybin_pathway": GeneticCircuit(
        id="psilocybin_pathway",
        name="Psilocybin Biosynthesis",
        species="Psilocybe cubensis",
        product="Psilocybin",
        genes=[
            GeneComponent("psiD", "Tryptophan Decarboxylase", 60, color="#22c55e"),
            GeneComponent("psiK", "Kinase", 45, color="#3b82f6"),
            GeneComponent("psiM", "Methyltransferase", 55, color="#8b5cf6"),
            GeneComponent("psiH", "Hydroxylase", 40, color="#f59e0b"),
        ],
        interactions=[
            Interaction("psiD", "psiK", "produces", 0.8),
            Interaction("psiK", "psiM", "produces", 0.9),
            Interaction("psiM", "psiH", "activates", 0.7),
            Interaction("psiH", "psiD", "activates", 0.5),  # Feedback
        ],
    ),
    "hericenone_pathway": GeneticCircuit(
        id="hericenone_pathway",
        name="Hericenone Production",
        species="Hericium erinaceus",
        product="Hericenone A",
        genes=[
            GeneComponent("herA", "PKS", 55, color="#22c55e"),
            GeneComponent("herB", "Oxidoreductase", 50, color="#3b82f6"),
            GeneComponent("herC", "Cyclase", 45, color="#8b5cf6"),
            GeneComponent("herD", "Transferase", 60, color="#f59e0b"),
        ],
        interactions=[
            Interaction("herA", "herB", "produces", 0.85),
            Interaction("herB", "herC", "produces", 0.75),
            Interaction("herC", "herD", "produces", 0.9),
        ],
    ),
    "ganoderic_pathway": GeneticCircuit(
        id="ganoderic_pathway",
        name="Ganoderic Acid Synthesis",
        species="Ganoderma lucidum",
        product="Ganoderic Acid A",
        genes=[
            GeneComponent("lanS", "Lanosterol Synthase", 50, color="#22c55e"),
            GeneComponent("cyp1", "CYP450-1", 45, color="#3b82f6"),
            GeneComponent("cyp2", "CYP450-2", 40, color="#8b5cf6"),
            GeneComponent("ganA", "Acetyltransferase", 55, color="#f59e0b"),
        ],
        interactions=[
            Interaction("lanS", "cyp1", "produces", 0.8),
            Interaction("cyp1", "cyp2", "produces", 0.7),
            Interaction("cyp2", "ganA", "produces", 0.85),
            Interaction("ganA", "lanS", "represses", 0.3),  # Feedback inhibition
        ],
    ),
    "cordycepin_pathway": GeneticCircuit(
        id="cordycepin_pathway",
        name="Cordycepin Biosynthesis",
        species="Cordyceps militaris",
        product="Cordycepin",
        genes=[
            GeneComponent("cns1", "Adenosine Kinase", 55, color="#22c55e"),
            GeneComponent("cns2", "Reductase", 50, color="#3b82f6"),
            GeneComponent("cns3", "Phosphorylase", 45, color="#8b5cf6"),
            GeneComponent("cns4", "3'-dA Synthase", 60, color="#f59e0b"),
        ],
        interactions=[
            Interaction("cns1", "cns2", "produces", 0.9),
            Interaction("cns2", "cns3", "produces", 0.8),
            Interaction("cns3", "cns4", "produces", 0.85),
        ],
    ),
}


class GeneticCircuitSimulator:
    """
    Simulates gene regulatory network dynamics.
    
    Uses Hill function kinetics to model gene expression changes
    over time in response to regulatory interactions.
    """
    
    def __init__(self, circuit: GeneticCircuit):
        """
        Initialize the simulator with a genetic circuit.
        
        Args:
            circuit: GeneticCircuit definition
        """
        self.circuit = circuit
        self.state: Dict[str, float] = {}
        self.modifications: Dict[str, float] = {}
        self.stress_level = 0.0
        self.nutrient_level = 50.0
        
        # Initialize gene expression levels
        for gene in circuit.genes:
            self.state[gene.id] = gene.initial_expression
        
        # Add metabolite accumulator
        self.metabolite_level = 0.0
        
        print(f"Initialized GeneticCircuitSimulator for {circuit.name}")
    
    def apply_modification(self, gene_id: str, delta: float) -> None:
        """
        Apply a gene expression modification.
        
        Args:
            gene_id: Gene to modify
            delta: Change in expression (-50 to +50)
        """
        self.modifications[gene_id] = delta
    
    def set_conditions(
        self,
        stress_level: Optional[float] = None,
        nutrient_level: Optional[float] = None,
    ) -> None:
        """Set environmental conditions."""
        if stress_level is not None:
            self.stress_level = max(0, min(100, stress_level))
        if nutrient_level is not None:
            self.nutrient_level = max(0, min(100, nutrient_level))
    
    def run_simulation(
        self,
        steps: int = 100,
        timestep: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run the genetic circuit simulation.
        
        Args:
            steps: Number of simulation steps
            timestep: Time increment per step
        
        Returns:
            Dictionary with trajectory, final state, and metrics
        """
        start_time = time.time()
        trajectory = []
        
        for step in range(steps):
            # Calculate new expression levels
            new_state = self._compute_next_state(timestep)
            
            # Apply modifications
            for gene_id, delta in self.modifications.items():
                if gene_id in new_state:
                    new_state[gene_id] = max(0, min(100, new_state[gene_id] + delta * 0.1))
            
            # Apply stress effects
            if self.stress_level > 0:
                for gene_id in new_state:
                    new_state[gene_id] *= (1 - self.stress_level * 0.003)
            
            # Apply nutrient effects
            nutrient_factor = self.nutrient_level / 50.0
            for gene_id in new_state:
                new_state[gene_id] *= (0.8 + 0.4 * nutrient_factor)
            
            # Clamp values
            for gene_id in new_state:
                new_state[gene_id] = max(0, min(100, new_state[gene_id]))
            
            self.state = new_state
            
            # Update metabolite level
            self._update_metabolite()
            
            # Record trajectory
            trajectory.append({**self.state})
        
        # Analyze results
        final_state = self.state.copy()
        bottleneck = min(final_state, key=final_state.get)
        average_expression = sum(final_state.values()) / len(final_state)
        
        execution_time = int((time.time() - start_time) * 1000)
        
        return {
            "trajectory": trajectory,
            "final_state": final_state,
            "final_metabolite": round(self.metabolite_level, 2),
            "bottleneck_gene": bottleneck,
            "average_expression": round(average_expression, 2),
            "flux_rate": round(self.metabolite_level / steps, 4),
            "execution_time_ms": execution_time,
            "message": f"Circuit simulation completed for {self.circuit.name}",
        }
    
    def _compute_next_state(self, timestep: float) -> Dict[str, float]:
        """Compute the next state using Hill kinetics."""
        new_state = {}
        gene_map = {g.id: g for g in self.circuit.genes}
        
        for gene in self.circuit.genes:
            current = self.state[gene.id]
            
            # Basal production
            production = gene.basal_rate * gene.max_expression
            
            # Sum regulatory effects
            for interaction in self.circuit.interactions:
                if interaction.target_id == gene.id:
                    source_level = self.state.get(interaction.source_id, 0)
                    
                    # Hill function
                    hill_term = (source_level ** interaction.hill_coefficient) / (
                        interaction.threshold ** interaction.hill_coefficient +
                        source_level ** interaction.hill_coefficient
                    )
                    
                    if interaction.interaction_type == "activates":
                        production += interaction.strength * gene.max_expression * hill_term
                    elif interaction.interaction_type == "represses":
                        production *= (1 - interaction.strength * hill_term)
            
            # Degradation
            degradation = gene.degradation_rate * current
            
            # Update level
            new_level = current + (production - degradation) * timestep
            new_state[gene.id] = new_level
        
        return new_state
    
    def _update_metabolite(self) -> None:
        """Update the final metabolite level based on pathway flux."""
        # Get the last gene in the pathway (product)
        last_gene = self.circuit.genes[-1]
        expression = self.state.get(last_gene.id, 0)
        
        # Metabolite production rate depends on expression
        production_rate = expression * 0.01
        
        # Accumulate metabolite
        self.metabolite_level += production_rate
    
    def analyze_pathway(self) -> Dict[str, Any]:
        """Analyze the genetic pathway for bottlenecks and optimization targets."""
        expression_levels = [(g.id, self.state.get(g.id, 0)) for g in self.circuit.genes]
        expression_levels.sort(key=lambda x: x[1])
        
        bottleneck = expression_levels[0] if expression_levels else ("none", 0)
        rate_limiting = expression_levels[:2] if len(expression_levels) >= 2 else expression_levels
        
        # Suggest overexpression targets
        suggestions = []
        for gene_id, level in rate_limiting:
            if level < 50:
                suggestions.append({
                    "gene": gene_id,
                    "current_expression": round(level, 1),
                    "recommendation": "Overexpress to increase flux",
                    "expected_improvement": f"{(50 - level) / 50 * 100:.0f}%",
                })
        
        return {
            "bottleneck_gene": bottleneck[0],
            "bottleneck_expression": round(bottleneck[1], 1),
            "rate_limiting_steps": [
                {"gene": g, "expression": round(e, 1)} for g, e in rate_limiting
            ],
            "optimization_suggestions": suggestions,
            "pathway_efficiency": round(self.metabolite_level / 100, 2),
        }


def run_circuit_simulation(
    circuit_id: str = "psilocybin_pathway",
    modifications: Optional[Dict[str, float]] = None,
    steps: int = 100,
) -> Dict[str, Any]:
    """
    Convenience function to run a circuit simulation.
    
    Args:
        circuit_id: Key from CIRCUITS dictionary
        modifications: Gene expression modifications
        steps: Simulation steps
    
    Returns:
        Simulation results
    """
    circuit = CIRCUITS.get(circuit_id, CIRCUITS["psilocybin_pathway"])
    simulator = GeneticCircuitSimulator(circuit)
    
    if modifications:
        for gene_id, delta in modifications.items():
            simulator.apply_modification(gene_id, delta)
    
    return simulator.run_simulation(steps=steps)
