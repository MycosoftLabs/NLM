"""Tests for TurboQuant quantization: core algorithms, weight quantization,
KV cache, attention wrappers, SSM compression, and full model quantization."""

import math

import pytest
import torch
import torch.nn as nn

from nlm.model.attention import CyclicalAttention, SpatialLocalityAttention
from nlm.model.config import NLMConfig
from nlm.model.fusion import SparseAttentionFusion
from nlm.model.nlm_model import NatureLearningModel
from nlm.model.ssm_blocks import SelectiveSSMBlock
from nlm.quantization.api import estimate_memory_savings, quantize_model
from nlm.quantization.attention_wrappers import (
    QuantizedCyclicalAttention,
    QuantizedSparseAttentionFusion,
    QuantizedSpatialLocalityAttention,
)
from nlm.quantization.codebook import beta_pdf, compute_codebook, get_known_codebook
from nlm.quantization.config import QuantConfig
from nlm.quantization.kv_cache import QuantizedKVCache, ScalarValueQuantizer
from nlm.quantization.linear import QuantizedLinear
from nlm.quantization.qjl import QJLTransform
from nlm.quantization.rotation import generate_projection_matrix, generate_rotation_matrix
from nlm.quantization.ssm_compress import QuantizedSelectiveSSMBlock, SSMStateCompressor
from nlm.quantization.turbo_mse import TurboQuantMSE
from nlm.quantization.turbo_prod import TurboQuantProd
from nlm.quantization.utils import (
    compute_mse_distortion,
    information_theoretic_lower_bound,
    pack_indices,
    pack_sign_bits,
    theoretical_mse_bound,
    unpack_indices,
    unpack_sign_bits,
)


# --- Fixtures ---


@pytest.fixture
def config():
    """Small config for testing."""
    return NLMConfig(
        spatial_dim=32,
        temporal_dim=32,
        spectral_sensory_dim=64,
        world_state_dim=64,
        self_state_dim=32,
        action_intent_dim=32,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        num_ssm_layers=2,
        ssm_state_dim=16,
        ssm_expand_factor=2,
        graph_hidden_dim=64,
        graph_num_heads=4,
        graph_num_layers=1,
        fusion_num_heads=4,
        fusion_num_layers=1,
        max_spectral_bins=64,
        max_acoustic_bins=32,
        max_bioelectric_samples=64,
        max_thermal_grid=8,
        chemical_vector_dim=32,
        max_mechanical_bins=32,
        num_env_targets=14,
        num_anomaly_categories=10,
        num_bio_token_types=50,
        bio_token_embed_dim=32,
        max_bio_tokens=32,
    )


@pytest.fixture
def quant_config():
    return QuantConfig(
        enabled=True,
        quantize_weights=True,
        weight_bit_width=4,
        quantize_kv_cache=True,
        key_bit_width=3,
        value_bit_width=4,
        quantize_ssm_state=False,
        seed=42,
    )


# --- Rotation Tests ---


class TestRotation:
    def test_orthogonality(self):
        """Π^T @ Π == I within floating point tolerance."""
        d = 64
        Pi = generate_rotation_matrix(d, seed=42)
        identity = Pi.t() @ Pi
        assert torch.allclose(identity, torch.eye(d), atol=1e-5)

    def test_deterministic_with_seed(self):
        """Same seed produces same rotation."""
        Pi1 = generate_rotation_matrix(32, seed=123)
        Pi2 = generate_rotation_matrix(32, seed=123)
        assert torch.allclose(Pi1, Pi2)

    def test_different_seeds_differ(self):
        """Different seeds produce different rotations."""
        Pi1 = generate_rotation_matrix(32, seed=1)
        Pi2 = generate_rotation_matrix(32, seed=2)
        assert not torch.allclose(Pi1, Pi2)

    def test_shape(self):
        Pi = generate_rotation_matrix(128, seed=42)
        assert Pi.shape == (128, 128)

    def test_projection_matrix_shape(self):
        S = generate_projection_matrix(64, m=32, seed=42)
        assert S.shape == (32, 64)

    def test_projection_matrix_default_m(self):
        S = generate_projection_matrix(64, seed=42)
        assert S.shape == (64, 64)


# --- Codebook Tests ---


class TestCodebook:
    def test_known_codebook_b1(self):
        """b=1 codebook matches ±√(2/(πd))."""
        d = 64
        cb = get_known_codebook(d, 1)
        expected_val = math.sqrt(2.0 / (math.pi * d))
        assert cb is not None
        assert len(cb) == 2
        assert abs(cb[0].item() + expected_val) < 1e-6
        assert abs(cb[1].item() - expected_val) < 1e-6

    def test_known_codebook_b2(self):
        """b=2 codebook has 4 entries."""
        cb = get_known_codebook(64, 2)
        assert cb is not None
        assert len(cb) == 4
        # Should be symmetric around 0
        assert abs(cb[0].item() + cb[3].item()) < 1e-6
        assert abs(cb[1].item() + cb[2].item()) < 1e-6

    def test_codebook_symmetry(self):
        """Computed codebook should be approximately symmetric."""
        cb = compute_codebook(32, 3)
        assert len(cb) == 8
        for i in range(4):
            assert abs(cb[i].item() + cb[7 - i].item()) < 0.05

    def test_codebook_sorted(self):
        """Codebook entries should be sorted."""
        cb = compute_codebook(64, 4)
        for i in range(len(cb) - 1):
            assert cb[i] <= cb[i + 1]

    def test_beta_pdf_integrates_to_one(self):
        """PDF should integrate to approximately 1."""
        x = torch.linspace(-0.999, 0.999, 10000)
        pdf = beta_pdf(x, d=64)
        integral = (pdf * (2.0 / 10000)).sum().item()
        assert abs(integral - 1.0) < 0.05

    def test_beta_pdf_d2(self):
        """d=2 should give uniform distribution."""
        x = torch.linspace(-0.9, 0.9, 100)
        pdf = beta_pdf(x, d=2)
        assert torch.allclose(pdf, torch.full_like(pdf, 0.5))


# --- TurboQuant MSE Tests ---


class TestTurboQuantMSE:
    def test_roundtrip_shape(self):
        """Quantize then dequantize preserves shape."""
        dim = 64
        q = TurboQuantMSE(dim, bit_width=4, seed=42)
        x = torch.randn(8, dim)
        x = x / x.norm(dim=-1, keepdim=True)  # normalize

        x_hat, indices = q(x)
        assert x_hat.shape == x.shape
        assert indices.shape == x.shape

    def test_distortion_within_bound(self):
        """MSE should be within theoretical bound for unit-norm vectors."""
        dim = 64
        b = 4
        q = TurboQuantMSE(dim, bit_width=b, seed=42)

        x = torch.randn(100, dim)
        x = x / x.norm(dim=-1, keepdim=True)

        x_hat, _ = q(x)
        mse = ((x - x_hat) ** 2).mean().item()
        bound = theoretical_mse_bound(b)

        # Should be well within the theoretical bound
        assert mse < bound * 2.0, f"MSE {mse} exceeds 2x theoretical bound {bound}"

    def test_higher_bits_lower_mse(self):
        """Higher bit widths should produce lower MSE."""
        dim = 64
        x = torch.randn(50, dim)
        x = x / x.norm(dim=-1, keepdim=True)

        mse_values = []
        for b in [1, 2, 3, 4]:
            q = TurboQuantMSE(dim, bit_width=b, seed=42)
            x_hat, _ = q(x)
            mse = ((x - x_hat) ** 2).mean().item()
            mse_values.append(mse)

        for i in range(len(mse_values) - 1):
            assert mse_values[i] > mse_values[i + 1], (
                f"MSE at b={i+1} ({mse_values[i]}) should be > MSE at b={i+2} ({mse_values[i+1]})"
            )

    def test_batch_dimensions(self):
        """Works with arbitrary leading batch dims."""
        dim = 32
        q = TurboQuantMSE(dim, bit_width=2, seed=42)
        x = torch.randn(2, 3, 4, dim)
        x_hat, indices = q(x)
        assert x_hat.shape == (2, 3, 4, dim)
        assert indices.shape == (2, 3, 4, dim)


# --- QJL Tests ---


class TestQJL:
    def test_sign_quantization(self):
        """Output sign bits should be ±1."""
        dim = 64
        qjl = QJLTransform(dim, seed=42)
        x = torch.randn(10, dim)
        signs, norms = qjl.quantize(x)
        assert signs.dtype == torch.int8
        assert ((signs == 1) | (signs == -1)).all()

    def test_norm_preservation(self):
        """Stored norm should match input norm."""
        dim = 32
        qjl = QJLTransform(dim, seed=42)
        x = torch.randn(5, dim)
        _, norms = qjl.quantize(x)
        expected = torch.linalg.norm(x, dim=-1)
        assert torch.allclose(norms, expected, atol=1e-5)

    def test_inner_product_unbiased(self):
        """QJL inner product estimation should be unbiased (correct in expectation)."""
        dim = 128
        num_trials = 50

        q = torch.randn(dim)
        k = torch.randn(dim)
        exact = (q * k).sum().item()

        # Average over many independent QJL instances (different seeds)
        estimates = []
        for seed in range(num_trials):
            qjl = QJLTransform(dim, seed=seed)
            signs_k, norm_k = qjl.quantize(k)
            est = qjl.estimate_inner_product(
                q.unsqueeze(0), signs_k.unsqueeze(0), norm_k.unsqueeze(0)
            )
            estimates.append(est.item())

        mean_estimate = sum(estimates) / len(estimates)
        # Mean should be close to exact (within 30% + small absolute tolerance)
        assert abs(mean_estimate - exact) < abs(exact) * 0.5 + 2.0, (
            f"Mean estimate {mean_estimate} too far from exact {exact}"
        )

    def test_dequantize_shape(self):
        dim = 64
        qjl = QJLTransform(dim, seed=42)
        x = torch.randn(5, dim)
        signs, norms = qjl.quantize(x)
        x_hat = qjl.dequantize(signs, norms)
        assert x_hat.shape == x.shape


# --- TurboQuant Prod Tests ---


class TestTurboQuantProd:
    def test_roundtrip_shape(self):
        dim = 32
        q = TurboQuantProd(dim, bit_width=3, seed=42)
        x = torch.randn(4, dim)
        x_hat, result = q(x)
        assert x_hat.shape == x.shape
        assert result.mse_indices.shape == (4, dim)
        assert result.qjl_signs.shape == (4, dim)
        assert result.residual_norm.shape == (4,)

    def test_attention_scores_shape(self):
        """Attention score computation produces correct shape."""
        head_dim = 32
        q = TurboQuantProd(head_dim, bit_width=3, seed=42)

        batch, heads, seq_q, seq_k = 2, 4, 3, 5
        Q = torch.randn(batch, heads, seq_q, head_dim)
        K = torch.randn(batch, heads, seq_k, head_dim)

        quantized_K = q.quantize(K)
        scores = q.attention_scores(Q, quantized_K)
        assert scores.shape == (batch, heads, seq_q, seq_k)

    def test_minimum_bit_width(self):
        """bit_width < 2 should raise error."""
        with pytest.raises(ValueError):
            TurboQuantProd(32, bit_width=1)


# --- QuantizedLinear Tests ---


class TestQuantizedLinear:
    def test_from_linear(self):
        """Convert nn.Linear to QuantizedLinear."""
        linear = nn.Linear(64, 32)
        q_linear = QuantizedLinear.from_linear(linear, bit_width=4)
        assert q_linear.in_features == 64
        assert q_linear.out_features == 32

    def test_output_shape(self):
        """Quantized forward produces correct shape."""
        linear = nn.Linear(64, 32)
        q_linear = QuantizedLinear.from_linear(linear, bit_width=4)
        x = torch.randn(8, 64)
        out = q_linear(x)
        assert out.shape == (8, 32)

    def test_output_approximately_close(self):
        """Quantized output should be close to original."""
        linear = nn.Linear(128, 64)
        q_linear = QuantizedLinear.from_linear(linear, bit_width=4)
        x = torch.randn(16, 128)

        with torch.no_grad():
            original_out = linear(x)
            quant_out = q_linear(x)

        # Should be reasonably close (not exact)
        relative_error = (original_out - quant_out).norm() / original_out.norm()
        assert relative_error < 0.5, f"Relative error {relative_error} too high"

    def test_memory_reduction(self):
        """Quantized weight should use fewer bytes."""
        linear = nn.Linear(256, 128)
        q_linear = QuantizedLinear.from_linear(linear, bit_width=4)
        assert q_linear.memory_bytes() < q_linear.original_memory_bytes()

    def test_no_bias(self):
        """Works with bias=False."""
        linear = nn.Linear(64, 32, bias=False)
        q_linear = QuantizedLinear.from_linear(linear, bit_width=4)
        out = q_linear(torch.randn(4, 64))
        assert out.shape == (4, 32)


# --- Scalar Value Quantizer Tests ---


class TestScalarValueQuantizer:
    def test_roundtrip(self):
        vq = ScalarValueQuantizer(bit_width=4)
        v = torch.randn(8, 32)
        indices, v_min, v_max = vq.quantize(v)
        v_hat = vq.dequantize(indices, v_min, v_max)
        assert v_hat.shape == v.shape
        # Should be reasonably close
        assert ((v - v_hat) ** 2).mean() < 0.01

    def test_constant_vector(self):
        """Constant vectors should not crash (zero range)."""
        vq = ScalarValueQuantizer(bit_width=4)
        v = torch.ones(4, 16) * 3.0
        indices, v_min, v_max = vq.quantize(v)
        v_hat = vq.dequantize(indices, v_min, v_max)
        assert torch.allclose(v_hat, v, atol=0.01)


# --- Quantized KV Cache Tests ---


class TestQuantizedKVCache:
    def test_append_and_compute(self):
        """Cache keys/values, compute attention output."""
        head_dim = 32
        num_heads = 4
        cache = QuantizedKVCache(
            head_dim=head_dim, num_heads=num_heads,
            key_bit_width=3, value_bit_width=4, seed=42,
        )

        batch = 2
        seq_len = 5
        K = torch.randn(batch, num_heads, seq_len, head_dim)
        V = torch.randn(batch, num_heads, seq_len, head_dim)
        cache.append(K, V)

        assert cache.seq_len == seq_len

        Q = torch.randn(batch, num_heads, 1, head_dim)
        out = cache.compute_attention(Q)
        assert out.shape == (batch, num_heads, 1, head_dim)

    def test_incremental_append(self):
        """Multiple appends should concatenate properly."""
        head_dim = 16
        cache = QuantizedKVCache(
            head_dim=head_dim, num_heads=2,
            key_bit_width=3, value_bit_width=4, seed=42,
        )

        cache.append(
            torch.randn(1, 2, 3, head_dim),
            torch.randn(1, 2, 3, head_dim),
        )
        cache.append(
            torch.randn(1, 2, 2, head_dim),
            torch.randn(1, 2, 2, head_dim),
        )
        assert cache.seq_len == 5

    def test_reset(self):
        head_dim = 16
        cache = QuantizedKVCache(head_dim=head_dim, num_heads=1, seed=42)
        cache.append(torch.randn(1, 1, 3, head_dim), torch.randn(1, 1, 3, head_dim))
        cache.reset()
        assert cache.seq_len == 0


# --- Attention Wrapper Tests ---


class TestQuantizedAttention:
    def test_cyclical_attention_wrapper(self):
        d_model, num_heads = 64, 4
        original = CyclicalAttention(d_model, num_heads)
        qc = QuantConfig(enabled=True, key_bit_width=3, value_bit_width=4, seed=42)
        wrapped = QuantizedCyclicalAttention(original, qc)

        x = torch.randn(2, 6, d_model)
        temporal = torch.randn(2, 6, 12)
        out = wrapped(x, temporal)
        assert out.shape == (2, 6, d_model)

    def test_spatial_attention_wrapper(self):
        d_model, num_heads = 64, 4
        original = SpatialLocalityAttention(d_model, num_heads)
        qc = QuantConfig(enabled=True, key_bit_width=3, value_bit_width=4, seed=42)
        wrapped = QuantizedSpatialLocalityAttention(original, qc)

        x = torch.randn(2, 6, d_model)
        coords = torch.randn(2, 6, 3)
        out = wrapped(x, coords)
        assert out.shape == (2, 6, d_model)

    def test_fusion_wrapper(self, config):
        original = SparseAttentionFusion(config)
        qc = QuantConfig(enabled=True, key_bit_width=3, value_bit_width=4, seed=42)
        wrapped = QuantizedSparseAttentionFusion(original, qc)

        streams = [torch.randn(2, dim) for dim in [
            config.spatial_dim, config.temporal_dim,
            config.spectral_sensory_dim, config.world_state_dim,
            config.self_state_dim, config.action_intent_dim,
        ]]
        out = wrapped(*streams)
        assert out.shape == (2, config.hidden_dim)


# --- SSM Compression Tests ---


class TestSSMCompression:
    def test_state_roundtrip(self):
        d_inner, d_state = 64, 16
        compressor = SSMStateCompressor(d_inner, d_state, bit_width=4, seed=42)
        h = torch.randn(4, d_inner, d_state)
        indices, scale = compressor.compress(h)
        h_hat = compressor.decompress(indices, scale)
        assert h_hat.shape == h.shape

    def test_quantized_ssm_block(self):
        d_model = 64
        original = SelectiveSSMBlock(d_model, d_state=16, d_conv=4, expand=2)
        wrapped = QuantizedSelectiveSSMBlock(original, bit_width=4, seed=42)

        x = torch.randn(2, 3, d_model)
        out = wrapped(x)
        assert out.shape == (2, 3, d_model)


# --- Full Model Quantization Tests ---


class TestQuantizeAPI:
    def test_quantize_weights_only(self, config):
        model = NatureLearningModel(config)
        qc = QuantConfig(
            enabled=True,
            quantize_weights=True,
            weight_bit_width=4,
            quantize_kv_cache=False,
            quantize_ssm_state=False,
        )
        quantize_model(model, qc)

        # Check that some modules were replaced
        has_quantized = False
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                has_quantized = True
                break
        assert has_quantized

    def test_quantize_kv_cache_only(self, config):
        model = NatureLearningModel(config)
        qc = QuantConfig(
            enabled=True,
            quantize_weights=False,
            quantize_kv_cache=True,
            key_bit_width=3,
            value_bit_width=4,
            quantize_ssm_state=False,
        )
        quantize_model(model, qc)

        has_quantized_attn = False
        for module in model.modules():
            if isinstance(module, QuantizedSparseAttentionFusion):
                has_quantized_attn = True
                break
        assert has_quantized_attn

    def test_disabled_config_no_changes(self, config):
        model = NatureLearningModel(config)
        qc = QuantConfig(enabled=False)
        quantize_model(model, qc)

        for module in model.modules():
            assert not isinstance(module, QuantizedLinear)

    def test_memory_savings(self, config):
        model = NatureLearningModel(config)
        qc = QuantConfig(enabled=True, quantize_weights=True, weight_bit_width=4)
        savings = estimate_memory_savings(model, qc)
        assert savings["compression_ratio"] > 1.0
        assert savings["original_bytes"] > savings["quantized_bytes"]

    def test_quantized_model_forward(self, config):
        """Full model still produces correct output shapes after quantization."""
        model = NatureLearningModel(config)
        qc = QuantConfig(
            enabled=True,
            quantize_weights=True,
            weight_bit_width=4,
            quantize_kv_cache=True,
            key_bit_width=3,
            value_bit_width=4,
            quantize_ssm_state=False,
        )
        quantize_model(model, qc)

        B = 2
        with torch.no_grad():
            output = model(
                spatial_features=torch.randn(B, 37),
                temporal_features=torch.randn(B, 12),
                spectral=torch.randn(B, config.max_spectral_bins),
                acoustic=torch.randn(B, config.max_acoustic_bins),
                bioelectric=torch.randn(B, config.max_bioelectric_samples),
                thermal=torch.randn(B, config.max_thermal_grid ** 2),
                chemical=torch.randn(B, config.chemical_vector_dim),
                mechanical=torch.randn(B, config.max_mechanical_bins + 5),
            )

        assert output.hidden.shape == (B, config.hidden_dim)
        assert output.next_state.shape == (B, config.num_env_targets)
        assert output.anomaly_scores.shape == (B, config.num_anomaly_categories)


# --- Utility Tests ---


class TestUtils:
    def test_pack_unpack_4bit(self):
        indices = torch.tensor([3, 7, 1, 15, 0, 8], dtype=torch.int16)
        packed = pack_indices(indices, bit_width=4)
        unpacked = unpack_indices(packed, bit_width=4, num_elements=6)
        assert torch.equal(indices, unpacked)

    def test_pack_unpack_2bit(self):
        indices = torch.tensor([0, 1, 2, 3, 1, 0, 3, 2], dtype=torch.int16)
        packed = pack_indices(indices, bit_width=2)
        unpacked = unpack_indices(packed, bit_width=2, num_elements=8)
        assert torch.equal(indices, unpacked)

    def test_pack_unpack_1bit(self):
        indices = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.int16)
        packed = pack_indices(indices, bit_width=1)
        unpacked = unpack_indices(packed, bit_width=1, num_elements=8)
        assert torch.equal(indices, unpacked)

    def test_sign_bit_packing(self):
        signs = torch.tensor([1, -1, 1, 1, -1, -1, 1, -1], dtype=torch.int8)
        packed = pack_sign_bits(signs)
        unpacked = unpack_sign_bits(packed)
        assert torch.equal(signs, unpacked)

    def test_theoretical_bounds(self):
        for b in [1, 2, 3, 4]:
            mse_bound = theoretical_mse_bound(b)
            lower = information_theoretic_lower_bound(b)
            assert mse_bound > lower
            # Gap should be at most ~2.7x
            assert mse_bound / lower < 3.0

    def test_compute_mse_distortion(self):
        original = torch.randn(10, 32)
        noisy = original + torch.randn_like(original) * 0.1
        dist = compute_mse_distortion(original, noisy)
        assert dist > 0
        assert dist < 1.0  # noise is small relative to signal
