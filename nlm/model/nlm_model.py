"""
NatureLearningModel — The Main Model Class
============================================

A grounded sensory world model, not a language model.
Language is a downstream translation surface.

Architecture:
1. DeterministicPreconditioner (no gradients — physics/chemistry/biology)
2. Six stream encoders (Spatial, Temporal, Spectral/Sensory, World State, Self State, Action/Intent)
3. TemporalSSMStack (Mamba-style — linear-time temporal processing)
4. HyperDAGEncoder (graph/hypergraph — structural reasoning)
5. SparseAttentionFusion (cross-stream binding — selective, not default)
6. Prediction heads (next-state, intervention, anomaly, ecological, grounding, causal)

The order of cognition:
    raw reality → deterministic transforms → rooted state → predictive model → heads
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from nlm.model.config import NLMConfig
from nlm.model.encoders import (
    ActionIntentEncoder,
    SelfStateGraphEncoder,
    SpatialEncoder,
    SpectralSensoryEncoder,
    TemporalEncoder,
    WorldStateGraphEncoder,
)
from nlm.model.fusion import SparseAttentionFusion
from nlm.model.heads import (
    AnomalyDetectionHead,
    CausalConsistencyHead,
    CompoundClassificationHead,
    EcologicalImpactHead,
    GrowthPredictionHead,
    GroundingConfidenceHead,
    InterventionOutcomeHead,
    NextStatePredictionHead,
    SpeciesClassificationHead,
)
from nlm.model.ssm_blocks import TemporalSSMStack


@dataclass
class NLMOutput:
    """Output from the NatureLearningModel."""

    # Fused hidden representation
    hidden: torch.Tensor  # (batch, hidden_dim)

    # Primary head outputs (always computed)
    next_state: torch.Tensor  # (batch, num_env_targets)
    anomaly_scores: torch.Tensor  # (batch, num_anomaly_categories)
    ecological_impact: Dict[str, torch.Tensor]  # harm_score, biosphere_risk, reversibility
    grounding_confidence: torch.Tensor  # (batch,)

    # Optional head outputs
    intervention_outcome: Optional[torch.Tensor] = None
    causal_consistency: Optional[torch.Tensor] = None
    species_logits: Optional[torch.Tensor] = None
    compound_logits: Optional[torch.Tensor] = None
    growth_prediction: Optional[torch.Tensor] = None

    # Temporal hidden states (for sequence processing)
    temporal_hidden: Optional[torch.Tensor] = None


class NatureLearningModel(nn.Module):
    """The Nature Learning Model.

    A grounded sensory world model that thinks in fields, spectra,
    voltages, concentrations, gradients, and state transitions.
    Language is a lossy projection of this deeper state.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        self.config = config

        # --- 6 Stream Encoders ---
        self.spatial_encoder = SpatialEncoder(config)
        self.temporal_encoder = TemporalEncoder(config)
        self.spectral_sensory_encoder = SpectralSensoryEncoder(config)
        self.world_state_encoder = WorldStateGraphEncoder(config)
        self.self_state_encoder = SelfStateGraphEncoder(config)
        self.action_intent_encoder = ActionIntentEncoder(config)

        # --- Temporal SSM Core ---
        self.temporal_ssm = TemporalSSMStack(config)

        # --- Sparse Attention Fusion ---
        self.fusion = SparseAttentionFusion(config)

        # --- Primary Prediction Heads ---
        self.next_state_head = NextStatePredictionHead(config)
        self.anomaly_head = AnomalyDetectionHead(config)
        self.ecological_head = EcologicalImpactHead(config)
        self.grounding_head = GroundingConfidenceHead(config)
        self.intervention_head = InterventionOutcomeHead(config)
        self.causal_head = CausalConsistencyHead(config)

        # --- Secondary Prediction Heads ---
        self.species_head = SpeciesClassificationHead(config)
        self.compound_head = CompoundClassificationHead(config)
        self.growth_head = GrowthPredictionHead(config)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        # Stream 1: Spatial
        spatial_features: torch.Tensor,        # (batch, 37)
        # Stream 2: Temporal
        temporal_features: torch.Tensor,       # (batch, 12)
        # Stream 3: Spectral/Sensory (6 sub-modalities)
        spectral: torch.Tensor,                # (batch, max_spectral_bins)
        acoustic: torch.Tensor,                # (batch, max_acoustic_bins)
        bioelectric: torch.Tensor,             # (batch, max_bioelectric_samples)
        thermal: torch.Tensor,                 # (batch, max_thermal_grid²)
        chemical: torch.Tensor,                # (batch, chemical_vector_dim)
        mechanical: torch.Tensor,              # (batch, max_mechanical_bins+5)
        modality_mask: Optional[torch.Tensor] = None,  # (batch, 6)
        # Stream 4: World State
        env_features: torch.Tensor = None,     # (batch, env_dim)
        bio_token_ids: torch.Tensor = None,    # (batch, max_bio_tokens) int
        graph_features: torch.Tensor = None,   # (batch, graph_hidden_dim)
        # Stream 5: Self State
        self_state_features: torch.Tensor = None,  # (batch, self_state_raw_dim)
        # Stream 6: Action/Intent
        recent_actions: torch.Tensor = None,   # (batch, num_recent, action_dim)
        intended_actions: torch.Tensor = None,  # (batch, num_intended, action_dim)
        # Optional: for causal consistency head
        prev_hidden: Optional[torch.Tensor] = None,
        # Optional: for intervention head
        proposed_action: Optional[torch.Tensor] = None,
    ) -> NLMOutput:
        batch_size = spatial_features.size(0)
        device = spatial_features.device

        # --- Encode 6 streams ---
        enc_spatial = self.spatial_encoder(spatial_features)

        enc_temporal = self.temporal_encoder(temporal_features)

        enc_spectral = self.spectral_sensory_encoder(
            spectral, acoustic, bioelectric, thermal, chemical, mechanical, modality_mask,
        )

        # World state with defaults
        if env_features is None:
            env_features = torch.zeros(batch_size, self.config.num_env_targets + 14, device=device)
        if bio_token_ids is None:
            bio_token_ids = torch.zeros(batch_size, self.config.max_bio_tokens, dtype=torch.long, device=device)
        if graph_features is None:
            graph_features = torch.zeros(batch_size, self.config.graph_hidden_dim, device=device)
        enc_world = self.world_state_encoder(env_features, bio_token_ids, graph_features)

        # Self state
        if self_state_features is None:
            self_state_features = torch.zeros(batch_size, 69, device=device)
        enc_self = self.self_state_encoder(self_state_features)

        # Action/Intent
        if recent_actions is None:
            recent_actions = torch.zeros(batch_size, 1, 64, device=device)
        if intended_actions is None:
            intended_actions = torch.zeros(batch_size, 1, 64, device=device)
        enc_action = self.action_intent_encoder(recent_actions, intended_actions)

        # --- Sparse Attention Fusion ---
        hidden = self.fusion(
            enc_spatial, enc_temporal, enc_spectral,
            enc_world, enc_self, enc_action,
        )  # (batch, hidden_dim)

        # --- Temporal SSM (process as single-step sequence) ---
        temporal_out = self.temporal_ssm(hidden.unsqueeze(1))  # (batch, 1, hidden_dim)
        hidden = temporal_out.squeeze(1)  # (batch, hidden_dim)

        # --- Primary Heads ---
        next_state = self.next_state_head(hidden)
        anomaly_scores = self.anomaly_head(hidden)
        eco_impact = self.ecological_head(hidden)
        grounding = self.grounding_head(hidden)

        # --- Optional Heads ---
        intervention_out = None
        if proposed_action is not None:
            intervention_out = self.intervention_head(hidden, proposed_action)

        causal_out = None
        if prev_hidden is not None:
            causal_out = self.causal_head(prev_hidden, hidden)

        return NLMOutput(
            hidden=hidden,
            next_state=next_state,
            anomaly_scores=anomaly_scores,
            ecological_impact=eco_impact,
            grounding_confidence=grounding,
            intervention_outcome=intervention_out,
            causal_consistency=causal_out,
            species_logits=self.species_head(hidden),
            compound_logits=self.compound_head(hidden),
            growth_prediction=self.growth_head(hidden),
            temporal_hidden=temporal_out,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_module(self) -> Dict[str, int]:
        counts = {}
        for name, module in self.named_children():
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return counts
