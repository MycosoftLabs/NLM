"""
NLM Model — Grounded Sensory World Model.

Hybrid architecture: SSM/Mamba temporal core + Graph encoders +
Sparse attention fusion. Not transformer-first. Language is secondary.
"""

from nlm.model.config import NLMConfig
from nlm.model.nlm_model import NatureLearningModel

__all__ = ["NLMConfig", "NatureLearningModel"]
