"""
NLM Physics Layer
=================

Provides physics-based simulation capabilities for molecular dynamics,
quantum-inspired calculations, tensor networks, and environmental field physics.

Modules:
- qise: Quantum-Inspired Simulation Engine
- tensor_network: Tensor Network Simulator for large molecular systems
- molecular_dynamics: Classical molecular dynamics engine
- field_physics: Environmental field calculations (geomagnetic, lunar, atmospheric)
"""

from .qise import QISE
from .tensor_network import TensorNetworkSimulator
from .molecular_dynamics import MolecularDynamicsEngine
from .field_physics import FieldPhysicsModel

__all__ = [
    "QISE",
    "TensorNetworkSimulator",
    "MolecularDynamicsEngine",
    "FieldPhysicsModel",
]
