"""
NLM Biology Layer
=================

Provides biological simulation capabilities for fungal systems including
digital twins, lifecycle modeling, genetic circuits, and symbiosis networks.

Modules:
- digital_twin: Real-time mycelium network simulation
- lifecycle: Spore-to-sporulation lifecycle modeling
- genetic_circuit: Gene regulatory network simulation
- symbiosis: Inter-species relationship mapping
"""

from .digital_twin import DigitalTwinMycelium
from .lifecycle import SporeLifecycleSimulator, LifecycleStage
from .genetic_circuit import GeneticCircuitSimulator
from .symbiosis import SymbiosisNetworkMapper

__all__ = [
    "DigitalTwinMycelium",
    "SporeLifecycleSimulator",
    "LifecycleStage",
    "GeneticCircuitSimulator",
    "SymbiosisNetworkMapper",
]
