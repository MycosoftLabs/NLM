"""
TurboQuant — Extreme Compression for NLM
==========================================

Implementation of TurboQuant (arxiv 2504.19874), combining:
- TurboQuant_mse: random rotation + optimal scalar quantization
- QJL: Quantized Johnson-Lindenstrauss 1-bit sign projection
- TurboQuant_prod: MSE + QJL residual for unbiased inner products
- PolarQuant: recursive polar coordinate quantization

Applied to NLM for weight compression, KV cache quantization,
and SSM state compression.
"""

from nlm.quantization.config import QuantConfig
from nlm.quantization.rotation import generate_projection_matrix, generate_rotation_matrix
from nlm.quantization.codebook import compute_codebook, get_known_codebook
from nlm.quantization.turbo_mse import TurboQuantMSE
from nlm.quantization.qjl import QJLTransform
from nlm.quantization.turbo_prod import TurboQuantProd
from nlm.quantization.linear import QuantizedLinear
from nlm.quantization.kv_cache import QuantizedKVCache
from nlm.quantization.api import quantize_model, estimate_memory_savings

__all__ = [
    "QuantConfig",
    "generate_rotation_matrix",
    "generate_projection_matrix",
    "compute_codebook",
    "get_known_codebook",
    "TurboQuantMSE",
    "QJLTransform",
    "TurboQuantProd",
    "QuantizedLinear",
    "QuantizedKVCache",
    "quantize_model",
    "estimate_memory_savings",
]
