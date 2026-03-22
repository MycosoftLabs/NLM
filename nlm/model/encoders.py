"""
Six Stream Encoders
===================

Each encoder processes one modality of the RootedNatureFrame:
1. SpatialEncoder — geographic coordinates + geomagnetic fields
2. TemporalEncoder — multi-scale cyclical time encoding
3. SpectralSensoryEncoder — all 6 fingerprint types
4. WorldStateGraphEncoder — environmental state + entity graph (graph-aware)
5. SelfStateGraphEncoder — MYCA/MAS internal state (graph-aware)
6. ActionIntentEncoder — recent/intended actions

Bio-tokens are one derived modality within WorldStateGraphEncoder,
not the primary substrate.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlm.model.config import NLMConfig


class SpatialEncoder(nn.Module):
    """Encodes geographic position and geomagnetic field topology.

    Input features: sinusoidal position encoding (24D) + geomagnetic (6D) +
    physics-derived atmospheric (4D) = ~34D raw → projected to spatial_dim.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        # Sinusoidal positional: 6 frequencies × 4 (sin/cos for lat/lon) + alt + lat/lon = 27
        # Geomagnetic: Bx, By, Bz, inclination, declination, field_strength = 6
        # Atmospheric: temp, pressure, humidity, wind = 4
        raw_dim = 27 + 6 + 4
        self.projection = nn.Sequential(
            nn.Linear(raw_dim, config.spatial_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.spatial_dim * 2, config.spatial_dim),
            nn.LayerNorm(config.spatial_dim),
        )

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """Args: spatial_features: (batch, raw_dim) from NaturePreprocessor.encode_spatial + physics"""
        return self.projection(spatial_features)


class TemporalEncoder(nn.Module):
    """Encodes time as physics-derived cyclical features.

    Not learned positions — deterministic cycles:
    time-of-day, day-of-year, lunar phase, solar declination, week cycle.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        # Temporal features: 12D from NaturePreprocessor.encode_temporal
        raw_dim = 12
        self.projection = nn.Sequential(
            nn.Linear(raw_dim, config.temporal_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.temporal_dim * 2, config.temporal_dim),
            nn.LayerNorm(config.temporal_dim),
        )

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """Args: temporal_features: (batch, 12) cyclical time encoding"""
        return self.projection(temporal_features)


class SpectralSensoryEncoder(nn.Module):
    """Encodes all 6 sensory fingerprint types into a unified representation.

    Processes: spectral, acoustic, bioelectric, thermal, chemical, mechanical
    fingerprints through modality-specific sub-encoders, then fuses.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        sub_dim = config.spectral_sensory_dim // 4

        # Modality-specific encoders (1D conv for sequence data, MLP for fixed)
        self.spectral_enc = nn.Sequential(
            nn.Linear(config.max_spectral_bins, sub_dim),
            nn.GELU(),
            nn.LayerNorm(sub_dim),
        )
        self.acoustic_enc = nn.Sequential(
            nn.Linear(config.max_acoustic_bins, sub_dim),
            nn.GELU(),
            nn.LayerNorm(sub_dim),
        )
        self.bioelectric_enc = nn.Sequential(
            nn.Linear(config.max_bioelectric_samples, sub_dim),
            nn.GELU(),
            nn.LayerNorm(sub_dim),
        )
        self.thermal_enc = nn.Sequential(
            nn.Linear(config.max_thermal_grid * config.max_thermal_grid, sub_dim),
            nn.GELU(),
            nn.LayerNorm(sub_dim),
        )
        self.chemical_enc = nn.Sequential(
            nn.Linear(config.chemical_vector_dim, sub_dim),
            nn.GELU(),
            nn.LayerNorm(sub_dim),
        )
        self.mechanical_enc = nn.Sequential(
            nn.Linear(config.max_mechanical_bins + 5, sub_dim),  # +5 for pressure, force_xyz, strain
            nn.GELU(),
            nn.LayerNorm(sub_dim),
        )

        # Modality presence flags (learned)
        self.modality_embeddings = nn.Embedding(6, sub_dim)

        # Fusion across modalities
        self.fusion = nn.Sequential(
            nn.Linear(sub_dim * 6, config.spectral_sensory_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.spectral_sensory_dim, config.spectral_sensory_dim),
            nn.LayerNorm(config.spectral_sensory_dim),
        )

    def _pad_or_truncate(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(-1) >= target_len:
            return x[..., :target_len]
        pad_size = target_len - x.size(-1)
        return F.pad(x, (0, pad_size))

    def forward(
        self,
        spectral: torch.Tensor,      # (batch, max_spectral_bins) or zeros
        acoustic: torch.Tensor,      # (batch, max_acoustic_bins) or zeros
        bioelectric: torch.Tensor,   # (batch, max_bioelectric_samples) or zeros
        thermal: torch.Tensor,       # (batch, max_thermal_grid²) or zeros
        chemical: torch.Tensor,      # (batch, chemical_vector_dim) or zeros
        mechanical: torch.Tensor,    # (batch, max_mechanical_bins+5) or zeros
        modality_mask: Optional[torch.Tensor] = None,  # (batch, 6) bool — which modalities are present
    ) -> torch.Tensor:
        batch_size = spectral.size(0)
        device = spectral.device

        enc_s = self.spectral_enc(spectral)
        enc_a = self.acoustic_enc(acoustic)
        enc_b = self.bioelectric_enc(bioelectric)
        enc_t = self.thermal_enc(thermal)
        enc_c = self.chemical_enc(chemical)
        enc_m = self.mechanical_enc(mechanical)

        # Add modality-specific embeddings
        mod_ids = torch.arange(6, device=device)
        mod_embs = self.modality_embeddings(mod_ids)  # (6, sub_dim)

        enc_s = enc_s + mod_embs[0]
        enc_a = enc_a + mod_embs[1]
        enc_b = enc_b + mod_embs[2]
        enc_t = enc_t + mod_embs[3]
        enc_c = enc_c + mod_embs[4]
        enc_m = enc_m + mod_embs[5]

        # Zero out missing modalities
        if modality_mask is not None:
            enc_s = enc_s * modality_mask[:, 0:1]
            enc_a = enc_a * modality_mask[:, 1:2]
            enc_b = enc_b * modality_mask[:, 2:3]
            enc_t = enc_t * modality_mask[:, 3:4]
            enc_c = enc_c * modality_mask[:, 4:5]
            enc_m = enc_m * modality_mask[:, 5:6]

        concat = torch.cat([enc_s, enc_a, enc_b, enc_t, enc_c, enc_m], dim=-1)
        return self.fusion(concat)


class WorldStateGraphEncoder(nn.Module):
    """Encodes complete external world state.

    Graph-aware and stateful — not just a flat vector of env scalars.
    Combines:
    - Environmental scalar features
    - Bio-token embeddings (as one derived modality)
    - Physics-derived fields
    - Entity-relation graph features (from HyperDAG L2)
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        # Environmental scalars + physics fields
        env_dim = config.num_env_targets + 14  # 14 physics context values

        # Bio-token embedding
        self.bio_token_embed = nn.Embedding(
            config.num_bio_token_types + 2,  # +2 for PAD and MASK
            config.bio_token_embed_dim,
            padding_idx=0,
        )
        self.token_pool = nn.AdaptiveAvgPool1d(1)

        # Graph features (flattened node/edge summaries)
        graph_summary_dim = config.graph_hidden_dim

        # Combine all
        combined_dim = env_dim + config.bio_token_embed_dim + graph_summary_dim
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, config.world_state_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.world_state_dim * 2, config.world_state_dim),
            nn.LayerNorm(config.world_state_dim),
        )

        # Graph summary encoder
        self.graph_summary = nn.Sequential(
            nn.Linear(config.graph_hidden_dim, graph_summary_dim),
            nn.GELU(),
            nn.LayerNorm(graph_summary_dim),
        )

    def forward(
        self,
        env_features: torch.Tensor,       # (batch, env_dim)
        bio_token_ids: torch.Tensor,       # (batch, seq_len) int
        graph_features: torch.Tensor,      # (batch, graph_hidden_dim) - precomputed graph summary
    ) -> torch.Tensor:
        # Bio-token pooled embedding
        token_emb = self.bio_token_embed(bio_token_ids)  # (batch, seq_len, embed_dim)
        token_pooled = token_emb.mean(dim=1)  # (batch, embed_dim)

        # Graph summary
        graph_enc = self.graph_summary(graph_features)

        # Combine
        combined = torch.cat([env_features, token_pooled, graph_enc], dim=-1)
        return self.projection(combined)


class SelfStateGraphEncoder(nn.Module):
    """Encodes complete MYCA/MAS internal state.

    Graph-aware — represents agent relationships, service dependencies,
    and resource topology, not just flat scalars.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        # Self-state features: safety_mode (one-hot 3), num_agents, num_tools,
        # service_health (one-hot per service), resource_levels, embodiment_readiness
        raw_dim = 3 + 1 + 1 + 32 + 16 + 16  # approximate feature budget
        self.projection = nn.Sequential(
            nn.Linear(raw_dim, config.self_state_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.self_state_dim * 2, config.self_state_dim),
            nn.LayerNorm(config.self_state_dim),
        )

    def forward(self, self_state_features: torch.Tensor) -> torch.Tensor:
        """Args: self_state_features: (batch, raw_dim) flattened self-state"""
        return self.projection(self_state_features)


class ActionIntentEncoder(nn.Module):
    """Encodes recent and intended actions for active inference.

    Without this stream, the model is purely observational.
    This enables counterfactual reasoning and intervention prediction.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        # Action features: recent_action embeddings + intent embeddings
        # Each action is represented as a type embedding + parameter vector
        self.action_type_embed = nn.Embedding(64, 32)  # 64 action types
        raw_dim = 32 + 32  # action_type_embed + action_params
        self.recent_encoder = nn.GRU(raw_dim, config.action_intent_dim // 2, batch_first=True)
        self.intent_encoder = nn.GRU(raw_dim, config.action_intent_dim // 2, batch_first=True)
        self.output_norm = nn.LayerNorm(config.action_intent_dim)

    def forward(
        self,
        recent_actions: torch.Tensor,   # (batch, num_recent, raw_dim)
        intended_actions: torch.Tensor,  # (batch, num_intended, raw_dim)
    ) -> torch.Tensor:
        # Process recent actions sequence
        _, recent_hidden = self.recent_encoder(recent_actions)
        recent_out = recent_hidden.squeeze(0)  # (batch, dim//2)

        # Process intended actions sequence
        _, intent_hidden = self.intent_encoder(intended_actions)
        intent_out = intent_hidden.squeeze(0)  # (batch, dim//2)

        combined = torch.cat([recent_out, intent_out], dim=-1)
        return self.output_norm(combined)
