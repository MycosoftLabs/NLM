"""
NLM - Nature Learning Model

A grounded sensory world model that thinks in fields, spectra, voltages,
concentrations, gradients, and state transitions. Language is a lossy
projection of this deeper grounded state.

The NLM is not a language model. It processes:
- Spatial data (geographic, geomagnetic, terrain)
- Temporal data (cyclical physics-derived, not linear positions)
- Spectral/Sensory data (6 fingerprint types in native physical units)
- World state (environmental + entity graph + external feeds)
- Self state (MYCA/MAS internal condition)
- Action/Intent (recent + planned interventions)

Architecture: SSM/Mamba temporal core + Graph/Hypergraph backbone +
Sparse Attention fusion. Governed by AVANI guardian layer.
"""

__version__ = "0.1.0"

from nlm.client import NLMClient

__all__ = ["NLMClient", "__version__"]

