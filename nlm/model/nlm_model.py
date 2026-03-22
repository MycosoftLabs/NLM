"""
NLM World Model — Top-Level Model Class

The NLMWorldModel is the complete forward pass:

    RootedNatureFrame
      -> 6 stream encoders
      -> sparse attention fusion
      -> prediction heads
      -> outputs (gated by AVANI before reaching agents/language)

This is NOT a transformer. It is a hybrid:
  SSM/Mamba (temporal) + GNN (graph) + Sparse Attention (fusion only)

Language enters at the very end, if at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from nlm.model.encoders import (
    ActionIntentEncoder,
    SpatialEncoder,
    SpectralSensoryEncoder,
    TemporalEncoder,
)
from nlm.model.fusion import HybridFusionCore
from nlm.model.graph_encoders import SelfStateGraphEncoder, WorldStateGraphEncoder
from nlm.model.heads import (
    AnomalyDetectionHead,
    CausalConsistencyHead,
    ClassificationHead,
    CompoundPredictionHead,
    EcologicalImpactHead,
    GrowthPredictionHead,
    GroundingConfidenceHead,
    InterventionOutcomeHead,
    NextStatePredictionHead,
    SpeciesPredictionHead,
)
from nlm.model.ssm_blocks import SSMStack


@dataclass
class NLMConfig:
    """Configuration for the NLM World Model."""

    d_model: int = 256
    d_state: int = 16
    n_ssm_layers: int = 4
    n_fusion_layers: int = 3
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1

    # Graph encoder settings
    d_graph_node: int = 128
    d_graph_edge: int = 64
    n_graph_mp_layers: int = 3

    # Encoder settings
    n_spatial_frequencies: int = 32
    d_per_modality: int = 64

    # Head settings
    n_species: int = 1000
    n_compounds: int = 500
    n_classes: int = 100
    n_anomaly_types: int = 8


class NLMWorldModel(nn.Module):
    """
    Nature Learning Model — Grounded Sensory World Model.

    Forward pass:
        stream inputs -> 6 encoders -> fusion -> prediction heads

    The model does NOT start from language. It starts from physical
    reality: wavelengths, waveforms, voltages, gas concentrations,
    temperature gradients, pressure fields.
    """

    def __init__(self, config: Optional[NLMConfig] = None):
        super().__init__()
        self.config = config or NLMConfig()
        d = self.config.d_model

        # ── Stream Encoders ─────────────────────────────────────
        self.spatial_encoder = SpatialEncoder(
            d_output=d,
            n_frequencies=self.config.n_spatial_frequencies,
        )
        self.temporal_encoder = TemporalEncoder(d_output=d)
        self.spectral_sensory_encoder = SpectralSensoryEncoder(
            d_output=d,
            d_per_modality=self.config.d_per_modality,
        )
        self.world_state_encoder = WorldStateGraphEncoder(
            d_node=self.config.d_graph_node,
            d_edge=self.config.d_graph_edge,
            d_output=d,
            n_layers=self.config.n_graph_mp_layers,
        )
        self.self_state_encoder = SelfStateGraphEncoder(
            d_node=self.config.d_graph_node // 2,
            d_edge=self.config.d_graph_edge // 2,
            d_output=d,
            n_layers=2,
        )
        self.action_intent_encoder = ActionIntentEncoder(d_output=d)

        # ── Temporal SSM Backbone ───────────────────────────────
        self.ssm_backbone = SSMStack(
            d_model=d,
            n_layers=self.config.n_ssm_layers,
            d_state=self.config.d_state,
        )

        # ── Fusion Core ─────────────────────────────────────────
        self.fusion = HybridFusionCore(
            d_model=d,
            n_heads=self.config.n_heads,
            n_streams=6,
            n_layers=self.config.n_fusion_layers,
            d_ff=self.config.d_ff,
            dropout=self.config.dropout,
        )

        # ── Primary Prediction Heads ────────────────────────────
        self.next_state_head = NextStatePredictionHead(d_input=d)
        self.intervention_head = InterventionOutcomeHead(d_state=d)
        self.anomaly_head = AnomalyDetectionHead(
            d_input=d, n_anomaly_types=self.config.n_anomaly_types,
        )
        self.eco_impact_head = EcologicalImpactHead(d_input=d)
        self.grounding_head = GroundingConfidenceHead(d_input=d)
        self.causal_consistency_head = CausalConsistencyHead(d_input=d)

        # ── Secondary Prediction Heads ──────────────────────────
        self.species_head = SpeciesPredictionHead(
            d_input=d, n_species=self.config.n_species,
        )
        self.compound_head = CompoundPredictionHead(
            d_input=d, n_compounds=self.config.n_compounds,
        )
        self.growth_head = GrowthPredictionHead(d_input=d)
        self.classification_head = ClassificationHead(
            d_input=d, n_classes=self.config.n_classes,
        )

    def forward(
        self,
        *,
        # Stream 1: Spatial
        lat: torch.Tensor,
        lon: torch.Tensor,
        alt: torch.Tensor,
        # Stream 2: Temporal
        timestamps: torch.Tensor,
        # Stream 3: Spectral/Sensory (all optional)
        spectral: Optional[torch.Tensor] = None,
        acoustic: Optional[torch.Tensor] = None,
        bioelectric: Optional[torch.Tensor] = None,
        thermal: Optional[torch.Tensor] = None,
        chemical: Optional[torch.Tensor] = None,
        mechanical: Optional[torch.Tensor] = None,
        # Stream 4: World State Graph
        world_node_features: Optional[torch.Tensor] = None,
        world_edge_index: Optional[torch.Tensor] = None,
        world_edge_features: Optional[torch.Tensor] = None,
        # Stream 5: Self State Graph
        self_node_features: Optional[torch.Tensor] = None,
        self_edge_index: Optional[torch.Tensor] = None,
        # Stream 6: Action/Intent
        action_features: Optional[torch.Tensor] = None,
        # Intervention (for counterfactual prediction)
        intervention_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Full forward pass through the NLM World Model.

        Returns dict of all head outputs.
        """
        d = self.config.d_model
        batch = lat.shape[0]
        device = lat.device

        # ── Encode each stream ──────────────────────────────────

        # Stream 1: Spatial
        spatial_emb = self.spatial_encoder(lat, lon, alt)  # (batch, d)

        # Stream 2: Temporal
        temporal_emb = self.temporal_encoder(timestamps)  # (batch, d)

        # Stream 3: Spectral/Sensory
        if any(t is not None for t in [spectral, acoustic, bioelectric, thermal, chemical, mechanical]):
            sensory_emb = self.spectral_sensory_encoder(
                spectral=spectral, acoustic=acoustic,
                bioelectric=bioelectric, thermal=thermal,
                chemical=chemical, mechanical=mechanical,
            )
        else:
            sensory_emb = torch.zeros(batch, d, device=device)

        # Stream 4: World State Graph
        if world_node_features is not None and world_edge_index is not None:
            world_emb = self.world_state_encoder(
                world_node_features, world_edge_index, world_edge_features,
            )
            if world_emb.shape[0] == 1 and batch > 1:
                world_emb = world_emb.expand(batch, -1)
        else:
            world_emb = torch.zeros(batch, d, device=device)

        # Stream 5: Self State Graph
        if self_node_features is not None and self_edge_index is not None:
            self_emb = self.self_state_encoder(
                self_node_features, self_edge_index,
            )
            if self_emb.shape[0] == 1 and batch > 1:
                self_emb = self_emb.expand(batch, -1)
        else:
            self_emb = torch.zeros(batch, d, device=device)

        # Stream 6: Action/Intent
        if action_features is not None:
            action_emb = self.action_intent_encoder(action_features)
        else:
            action_emb = torch.zeros(batch, d, device=device)

        # ── Stack streams and fuse ──────────────────────────────
        stream_embeddings = torch.stack([
            spatial_emb,
            temporal_emb,
            sensory_emb,
            world_emb,
            self_emb,
            action_emb,
        ], dim=1)  # (batch, 6, d)

        fused = self.fusion(stream_embeddings)  # (batch, d)

        # ── Prediction heads ────────────────────────────────────
        outputs: Dict[str, Any] = {}

        # Primary heads
        outputs["next_state"] = self.next_state_head(fused)
        outputs["anomaly"] = self.anomaly_head(fused)
        outputs["eco_impact"] = self.eco_impact_head(fused)
        outputs["grounding_confidence"] = self.grounding_head(fused)
        outputs["causal_consistency"] = self.causal_consistency_head(fused)

        # Counterfactual prediction (if intervention is provided)
        if intervention_embedding is not None:
            outputs["intervention_outcome"] = self.intervention_head(
                fused, intervention_embedding,
            )

        # Secondary heads
        outputs["species_logits"] = self.species_head(fused)
        outputs["compound_logits"] = self.compound_head(fused)
        outputs["growth"] = self.growth_head(fused)
        outputs["classification_logits"] = self.classification_head(fused)

        # Also return the fused representation for downstream use
        outputs["fused_state"] = fused

        return outputs

    def reset_graph_states(self, batch_size: int = 1, device: str = "cpu"):
        """Reset stateful graph encoder hidden states."""
        self.world_state_encoder.reset_state(batch_size, device)
        self.self_state_encoder.reset_state(batch_size, device)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
