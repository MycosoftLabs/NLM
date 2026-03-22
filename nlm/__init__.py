"""
NLM — Nature Learning Model

A grounded sensory world model that learns from raw physical reality —
wavelengths, waveforms, voltages, gas concentrations, temperature gradients,
pressure fields — and predicts what happens next.

NLM is not an LLM. It does not start from language. It starts from raw
physical reality and builds upward through deterministic scientific transforms,
sensory fingerprint extraction, Merkle-rooted state assembly, and a hybrid
learned model (SSM + Graph + Sparse Attention).
"""

__version__ = "0.2.0"

from nlm.client import NLMClient

__all__ = ["NLMClient", "__version__"]

