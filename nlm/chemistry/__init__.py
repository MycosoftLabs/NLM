"""
NLM Chemistry Layer
===================

Provides chemistry and biochemistry simulation capabilities including
molecular encoding, knowledge graphs, bioactivity prediction, and
retrosynthesis analysis.

Modules:
- encoder: Chemical compound vector encoding
- knowledge: Chemistry knowledge graph
- predictor: Bioactivity and species prediction
- retrosynthesis: Biosynthetic pathway analysis
- reaction_network: Biochemical reaction networks
- alchemy: Computational compound design
"""

from .encoder import ChemistryEncoder, encode_compound
from .knowledge import ChemistryKnowledgeGraph
from .predictor import BioactivityPredictor
from .retrosynthesis import RetrosynthesisEngine
from .reaction_network import ReactionNetworkGraph
from .alchemy import ComputationalAlchemyLab

__all__ = [
    "ChemistryEncoder",
    "encode_compound",
    "ChemistryKnowledgeGraph",
    "BioactivityPredictor",
    "RetrosynthesisEngine",
    "ReactionNetworkGraph",
    "ComputationalAlchemyLab",
]
