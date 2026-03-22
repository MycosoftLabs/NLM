"""Tests for NLM model: forward pass, shapes, heads, AVANI."""

import torch
import pytest

from nlm.model.config import NLMConfig
from nlm.model.nlm_model import NatureLearningModel, NLMOutput
from nlm.model.encoders import (
    SpatialEncoder,
    TemporalEncoder,
    SpectralSensoryEncoder,
    WorldStateGraphEncoder,
    SelfStateGraphEncoder,
    ActionIntentEncoder,
)
from nlm.model.ssm_blocks import SelectiveSSMBlock, TemporalSSMStack
from nlm.model.heads import (
    NextStatePredictionHead,
    AnomalyDetectionHead,
    EcologicalImpactHead,
    GroundingConfidenceHead,
    CausalConsistencyHead,
)
from nlm.model.fusion import SparseAttentionFusion
from nlm.guardian.avani import AVANIGuardian, VerdictAction


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
def batch(config):
    B = 4
    return {
        "spatial_features": torch.randn(B, 37),
        "temporal_features": torch.randn(B, 12),
        "spectral": torch.randn(B, config.max_spectral_bins),
        "acoustic": torch.randn(B, config.max_acoustic_bins),
        "bioelectric": torch.randn(B, config.max_bioelectric_samples),
        "thermal": torch.randn(B, config.max_thermal_grid ** 2),
        "chemical": torch.randn(B, config.chemical_vector_dim),
        "mechanical": torch.randn(B, config.max_mechanical_bins + 5),
        "modality_mask": torch.ones(B, 6),
        "env_features": torch.randn(B, config.num_env_targets + 14),
        "bio_token_ids": torch.randint(0, config.num_bio_token_types, (B, config.max_bio_tokens)),
        "graph_features": torch.randn(B, config.graph_hidden_dim),
        "self_state_features": torch.randn(B, 69),
        "recent_actions": torch.randn(B, 8, 64),
        "intended_actions": torch.randn(B, 4, 64),
    }


class TestEncoders:
    def test_spatial_encoder(self, config):
        enc = SpatialEncoder(config)
        out = enc(torch.randn(2, 37))
        assert out.shape == (2, config.spatial_dim)

    def test_temporal_encoder(self, config):
        enc = TemporalEncoder(config)
        out = enc(torch.randn(2, 12))
        assert out.shape == (2, config.temporal_dim)

    def test_spectral_sensory_encoder(self, config):
        enc = SpectralSensoryEncoder(config)
        out = enc(
            torch.randn(2, config.max_spectral_bins),
            torch.randn(2, config.max_acoustic_bins),
            torch.randn(2, config.max_bioelectric_samples),
            torch.randn(2, config.max_thermal_grid ** 2),
            torch.randn(2, config.chemical_vector_dim),
            torch.randn(2, config.max_mechanical_bins + 5),
        )
        assert out.shape == (2, config.spectral_sensory_dim)

    def test_world_state_encoder(self, config):
        enc = WorldStateGraphEncoder(config)
        out = enc(
            torch.randn(2, config.num_env_targets + 14),
            torch.randint(0, 50, (2, config.max_bio_tokens)),
            torch.randn(2, config.graph_hidden_dim),
        )
        assert out.shape == (2, config.world_state_dim)

    def test_self_state_encoder(self, config):
        enc = SelfStateGraphEncoder(config)
        out = enc(torch.randn(2, 69))
        assert out.shape == (2, config.self_state_dim)

    def test_action_intent_encoder(self, config):
        enc = ActionIntentEncoder(config)
        out = enc(torch.randn(2, 3, 64), torch.randn(2, 2, 64))
        assert out.shape == (2, config.action_intent_dim)


class TestSSMBlocks:
    def test_selective_ssm_block(self):
        block = SelectiveSSMBlock(d_model=64, d_state=16, d_conv=4, expand=2)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_temporal_ssm_stack(self, config):
        stack = TemporalSSMStack(config)
        x = torch.randn(2, 5, config.hidden_dim)
        out = stack(x)
        assert out.shape == (2, 5, config.hidden_dim)


class TestFusion:
    def test_sparse_attention_fusion(self, config):
        fusion = SparseAttentionFusion(config)
        out = fusion(
            torch.randn(2, config.spatial_dim),
            torch.randn(2, config.temporal_dim),
            torch.randn(2, config.spectral_sensory_dim),
            torch.randn(2, config.world_state_dim),
            torch.randn(2, config.self_state_dim),
            torch.randn(2, config.action_intent_dim),
        )
        assert out.shape == (2, config.hidden_dim)


class TestHeads:
    def test_next_state_head(self, config):
        head = NextStatePredictionHead(config)
        out = head(torch.randn(2, config.hidden_dim))
        assert out.shape == (2, config.num_env_targets)

    def test_anomaly_head(self, config):
        head = AnomalyDetectionHead(config)
        out = head(torch.randn(2, config.hidden_dim))
        assert out.shape == (2, config.num_anomaly_categories)
        assert (out >= 0).all() and (out <= 1).all()

    def test_ecological_head(self, config):
        head = EcologicalImpactHead(config)
        out = head(torch.randn(2, config.hidden_dim))
        assert "harm_score" in out
        assert out["harm_score"].shape == (2,)

    def test_grounding_head(self, config):
        head = GroundingConfidenceHead(config)
        out = head(torch.randn(2, config.hidden_dim))
        assert out.shape == (2,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_causal_head(self, config):
        head = CausalConsistencyHead(config)
        out = head(torch.randn(2, config.hidden_dim), torch.randn(2, config.hidden_dim))
        assert out.shape == (2,)


class TestNatureLearningModel:
    def test_forward_pass(self, config, batch):
        model = NatureLearningModel(config)
        output = model(**batch)
        assert isinstance(output, NLMOutput)
        assert output.hidden.shape == (4, config.hidden_dim)
        assert output.next_state.shape == (4, config.num_env_targets)
        assert output.anomaly_scores.shape == (4, config.num_anomaly_categories)
        assert output.grounding_confidence.shape == (4,)

    def test_parameter_count(self, config):
        model = NatureLearningModel(config)
        count = model.count_parameters()
        assert count > 0
        by_module = model.count_parameters_by_module()
        assert "spatial_encoder" in by_module
        assert "temporal_ssm" in by_module
        assert "fusion" in by_module

    def test_gradient_flow(self, config, batch):
        model = NatureLearningModel(config)
        output = model(**batch)
        loss = output.next_state.sum() + output.hidden.sum()
        loss.backward()
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                break  # Just check first param


class TestAVANIGuardian:
    def test_allow_safe_action(self):
        guardian = AVANIGuardian()
        verdict = guardian.evaluate(harm_score=0.05, grounding_confidence=0.95)
        assert verdict.action == VerdictAction.ALLOW

    def test_veto_harmful_action(self):
        guardian = AVANIGuardian()
        verdict = guardian.evaluate(harm_score=0.9, biosphere_risk=0.85, reversibility=0.1)
        assert verdict.action == VerdictAction.VETO

    def test_escalate_uncertain(self):
        guardian = AVANIGuardian()
        verdict = guardian.evaluate(grounding_confidence=0.2)
        assert verdict.action == VerdictAction.ESCALATE
        assert verdict.uncertainty_escalation

    def test_attenuate_moderate_harm(self):
        guardian = AVANIGuardian()
        verdict = guardian.evaluate(harm_score=0.35, grounding_confidence=0.8)
        assert verdict.action == VerdictAction.ATTENUATE

    def test_scope_check(self):
        guardian = AVANIGuardian()
        verdict = guardian.evaluate(
            harm_score=0.1,
            proposed_action={"type": "release_organism"},
        )
        assert verdict.out_of_scope
        assert verdict.action == VerdictAction.ESCALATE

    def test_physics_violation(self):
        guardian = AVANIGuardian()
        verdict = guardian.evaluate(
            environmental_context={"temperature_c": -100},
        )
        assert verdict.physics_violation
        assert verdict.action == VerdictAction.VETO

    def test_strict_mode(self):
        guardian = AVANIGuardian(strict_mode=True)
        verdict = guardian.evaluate(harm_score=0.35)
        assert verdict.action in (VerdictAction.ESCALATE, VerdictAction.ATTENUATE, VerdictAction.VETO)
