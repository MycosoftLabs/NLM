"""
NLM Model Configuration
========================

Defines all hyperparameters for the Nature Learning Model.
Medium scale (~100-300M params): hidden_dim=512, 12 layers, 16 heads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NLMConfig:
    """Configuration for the Nature Learning Model.

    The NLM is a grounded sensory world model with 6 learned streams:
    Spatial, Temporal, Spectral/Sensory, World State, Self State, Action/Intent.
    """

    # --- Stream encoder dimensions ---
    spatial_dim: int = 128
    temporal_dim: int = 128
    spectral_sensory_dim: int = 256
    world_state_dim: int = 256
    self_state_dim: int = 128
    action_intent_dim: int = 128

    # --- Fused representation ---
    hidden_dim: int = 512
    num_heads: int = 16
    num_layers: int = 12
    ff_dim: int = 2048
    dropout: float = 0.1

    # --- SSM / Mamba temporal core ---
    ssm_state_dim: int = 64        # SSM latent state dimension
    ssm_conv_width: int = 4        # Local convolution width
    ssm_expand_factor: int = 2     # Expansion factor for inner dim
    ssm_dt_rank: str = "auto"      # Discretization rank
    num_ssm_layers: int = 6        # SSM layers for temporal processing

    # --- Graph encoder ---
    graph_hidden_dim: int = 256
    graph_num_heads: int = 8
    graph_num_layers: int = 4
    max_graph_nodes: int = 512     # Max nodes in a subgraph batch

    # --- Sparse attention fusion ---
    fusion_num_heads: int = 16
    fusion_num_layers: int = 4
    fusion_sparsity: float = 0.9   # Fraction of attention weights zeroed

    # --- Bio-tokens ---
    num_bio_token_types: int = 160  # From bio_tokens.py vocabulary + special tokens
    bio_token_embed_dim: int = 128
    max_bio_tokens: int = 256

    # --- Fingerprint processing ---
    max_spectral_bins: int = 512
    max_acoustic_bins: int = 256
    max_bioelectric_samples: int = 1024
    max_thermal_grid: int = 32     # Max thermal grid dimension
    chemical_vector_dim: int = 128  # From ChemistryEncoder
    max_mechanical_bins: int = 128

    # --- Prediction heads ---
    num_env_targets: int = 14      # Environmental forecast dimensions
    num_anomaly_categories: int = 20
    num_species_classes: int = 1000
    num_compound_classes: int = 500

    # --- Preconditioning ---
    use_physics_precondition: bool = True
    use_chemistry_precondition: bool = True
    use_biology_precondition: bool = True

    # --- Training ---
    max_sequence_length: int = 4096  # Max frames in a sequence
    gradient_checkpointing: bool = False

    def total_input_dim(self) -> int:
        """Total dimension after concatenating all 6 stream encoders."""
        return (
            self.spatial_dim + self.temporal_dim + self.spectral_sensory_dim
            + self.world_state_dim + self.self_state_dim + self.action_intent_dim
        )
