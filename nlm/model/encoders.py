"""
NLM Stream Encoders

The four non-graph stream encoders:
1. SpatialEncoder    — geolocation, spatial relations, site topology
2. TemporalEncoder   — timestamps, time-series, periodicity
3. SpectralSensoryEncoder — all 6 fingerprint types
4. ActionIntentEncoder — recent actions, intended actions

(Streams 4 and 5 — WorldState and SelfState — use graph encoders
defined in graph_encoders.py)
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlm.model.ssm_blocks import SSMBlock


class SpatialEncoder(nn.Module):
    """
    Stream 1: Spatial Encoder.

    Encodes geolocation (lat, lon, alt), spatial relations, and
    site topology into a fixed embedding.

    Uses sinusoidal positional encoding for coordinates (inspired by
    NeRF-style spatial encoding) for smooth spatial representation.
    """

    def __init__(self, d_output: int = 256, n_frequencies: int = 32):
        super().__init__()
        self.d_output = d_output
        self.n_frequencies = n_frequencies

        # Input: lat, lon, alt + sinusoidal features
        d_spatial_features = 3 + 3 * 2 * n_frequencies  # raw + sin/cos for each coord
        self.proj = nn.Sequential(
            nn.Linear(d_spatial_features, d_output),
            nn.SiLU(),
            nn.Linear(d_output, d_output),
            nn.LayerNorm(d_output),
        )

    def _sinusoidal_encode(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply sinusoidal positional encoding to spatial coordinates."""
        freqs = torch.pow(2.0, torch.arange(self.n_frequencies, device=coords.device, dtype=coords.dtype))
        # coords: (..., 3), freqs: (n_freq,)
        angles = coords.unsqueeze(-1) * freqs  # (..., 3, n_freq)
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        # Flatten: (..., 3 * 2 * n_freq)
        encoded = torch.cat([sin_enc, cos_enc], dim=-1).flatten(start_dim=-2)
        return encoded

    def forward(self, lat: torch.Tensor, lon: torch.Tensor, alt: torch.Tensor) -> torch.Tensor:
        """
        Encode spatial coordinates.

        Args:
            lat, lon, alt: each (batch,) or (batch, 1)

        Returns:
            (batch, d_output)
        """
        if lat.dim() == 1:
            lat = lat.unsqueeze(-1)
            lon = lon.unsqueeze(-1)
            alt = alt.unsqueeze(-1)

        # Normalize coordinates
        lat_norm = lat / 90.0
        lon_norm = lon / 180.0
        alt_norm = alt / 10000.0  # normalize altitude to ~10km scale

        coords = torch.cat([lat_norm, lon_norm, alt_norm], dim=-1)  # (batch, 3)
        sinusoidal = self._sinusoidal_encode(coords)  # (batch, 3*2*n_freq)
        features = torch.cat([coords, sinusoidal], dim=-1)

        return self.proj(features)


class TemporalEncoder(nn.Module):
    """
    Stream 2: Temporal Encoder.

    Encodes timestamps and time-series windows using:
    - Calendar features (hour, day, month, day-of-year)
    - Periodic encoding (sin/cos for cyclic features)
    - SSM block for sequence-level temporal patterns
    """

    def __init__(self, d_output: int = 256, d_model: int = 128):
        super().__init__()
        self.d_output = d_output

        # Calendar feature projection
        # Features: hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos,
        #           day_of_year_sin, day_of_year_cos, unix_normalized
        n_calendar = 9
        self.calendar_proj = nn.Sequential(
            nn.Linear(n_calendar, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # SSM for temporal sequence patterns
        self.ssm = SSMBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_output),
            nn.LayerNorm(d_output),
        )

    def encode_timestamp(self, unix_ts: torch.Tensor) -> torch.Tensor:
        """
        Encode a single timestamp into calendar features.

        Args:
            unix_ts: (batch,) unix timestamps

        Returns:
            (batch, 9) calendar feature vector
        """
        # Normalize to [0, 1] range (seconds since 2020-01-01)
        epoch_2020 = 1577836800.0
        normalized = (unix_ts - epoch_2020) / (365.25 * 86400 * 10)  # ~10 year window

        # Extract cyclical features
        seconds_in_day = unix_ts % 86400
        hour_frac = seconds_in_day / 86400
        day_frac = (unix_ts / 86400) % 365.25 / 365.25

        # Monthly cycle (approximate)
        month_frac = (unix_ts / 86400) % 30.44 / 30.44

        features = torch.stack([
            torch.sin(2 * math.pi * hour_frac),
            torch.cos(2 * math.pi * hour_frac),
            torch.sin(2 * math.pi * day_frac),
            torch.cos(2 * math.pi * day_frac),
            torch.sin(2 * math.pi * month_frac),
            torch.cos(2 * math.pi * month_frac),
            torch.sin(2 * math.pi * day_frac * 4),  # quarterly
            torch.cos(2 * math.pi * day_frac * 4),
            normalized,
        ], dim=-1)

        return features

    def forward(
        self,
        timestamps: torch.Tensor,
        sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode temporal information.

        Args:
            timestamps: (batch,) or (batch, seq_len) unix timestamps
            sequence: optional (batch, seq_len, d_model) time-series features

        Returns:
            (batch, d_output)
        """
        if timestamps.dim() == 1:
            # Single timestamp
            cal = self.encode_timestamp(timestamps)  # (batch, 9)
            h = self.calendar_proj(cal)  # (batch, d_model)
            return self.out_proj(h)

        # Sequence of timestamps
        batch, seq_len = timestamps.shape
        cal = self.encode_timestamp(timestamps.reshape(-1)).reshape(batch, seq_len, -1)
        h = self.calendar_proj(cal)  # (batch, seq_len, d_model)

        if sequence is not None:
            h = h + sequence  # combine with input features

        # SSM for temporal patterns
        h = self.ssm(h)  # (batch, seq_len, d_model)

        # Take last hidden state
        h = h[:, -1, :]  # (batch, d_model)
        return self.out_proj(h)


class SpectralSensoryEncoder(nn.Module):
    """
    Stream 3: Spectral/Sensory Encoder.

    Encodes all 6 fingerprint types into a unified embedding.
    Each modality has its own sub-encoder, then outputs are fused.
    """

    def __init__(self, d_output: int = 256, d_per_modality: int = 64):
        super().__init__()
        self.d_output = d_output
        self.d_per_modality = d_per_modality

        # Per-modality encoders (handle variable-length inputs)
        self.spectral_enc = self._make_modality_encoder(d_per_modality)
        self.acoustic_enc = self._make_modality_encoder(d_per_modality)
        self.bioelectric_enc = self._make_modality_encoder(d_per_modality)
        self.thermal_enc = self._make_modality_encoder(d_per_modality)
        self.chemical_enc = self._make_modality_encoder(d_per_modality)
        self.mechanical_enc = self._make_modality_encoder(d_per_modality)

        # Presence flags for each modality
        self.modality_embed = nn.Embedding(6, d_per_modality)

        # Fusion: cross-modality attention then projection
        self.fusion = nn.Sequential(
            nn.Linear(d_per_modality * 6, d_output),
            nn.SiLU(),
            nn.Linear(d_output, d_output),
            nn.LayerNorm(d_output),
        )

    def _make_modality_encoder(self, d_out: int) -> nn.Module:
        """Create a modality-specific encoder that handles variable input."""
        return nn.Sequential(
            nn.LazyLinear(d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(
        self,
        spectral: Optional[torch.Tensor] = None,
        acoustic: Optional[torch.Tensor] = None,
        bioelectric: Optional[torch.Tensor] = None,
        thermal: Optional[torch.Tensor] = None,
        chemical: Optional[torch.Tensor] = None,
        mechanical: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode sensory fingerprints.

        Each input is (batch, variable_dim) from the fingerprint.vector().
        Missing modalities get zero embeddings.

        Returns:
            (batch, d_output)
        """
        # Determine batch size and device from first non-None input
        ref = next((t for t in [spectral, acoustic, bioelectric, thermal, chemical, mechanical] if t is not None), None)
        if ref is None:
            raise ValueError("At least one sensory modality must be provided")

        batch = ref.shape[0]
        device = ref.device
        d = self.d_per_modality

        embeddings = []
        encoders = [self.spectral_enc, self.acoustic_enc, self.bioelectric_enc,
                     self.thermal_enc, self.chemical_enc, self.mechanical_enc]
        inputs = [spectral, acoustic, bioelectric, thermal, chemical, mechanical]

        for i, (enc, inp) in enumerate(zip(encoders, inputs)):
            if inp is not None and inp.numel() > 0:
                emb = enc(inp)  # (batch, d_per_modality)
                emb = emb + self.modality_embed(torch.tensor(i, device=device))
                embeddings.append(emb)
            else:
                embeddings.append(torch.zeros(batch, d, device=device))

        # Concatenate all modality embeddings
        fused = torch.cat(embeddings, dim=-1)  # (batch, 6 * d_per_modality)
        return self.fusion(fused)


class ActionIntentEncoder(nn.Module):
    """
    Stream 6: Action/Intent Encoder.

    Encodes recent actions and intended actions into a fixed embedding.
    Uses a simple MLP over action features (action_type one-hot + parameters).
    """

    def __init__(self, d_input: int = 64, d_output: int = 256, max_actions: int = 16):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.max_actions = max_actions

        self.action_proj = nn.Sequential(
            nn.Linear(d_input, d_output),
            nn.SiLU(),
            nn.Linear(d_output, d_output),
        )

        # Attention over actions
        self.attn = nn.Linear(d_output, 1)

        self.out_proj = nn.Sequential(
            nn.Linear(d_output, d_output),
            nn.LayerNorm(d_output),
        )

    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        """
        Encode action/intent features.

        Args:
            action_features: (batch, n_actions, d_input)

        Returns:
            (batch, d_output)
        """
        h = self.action_proj(action_features)  # (batch, n_actions, d_output)

        # Attention-weighted aggregation
        attn_weights = F.softmax(self.attn(h), dim=1)  # (batch, n_actions, 1)
        pooled = (h * attn_weights).sum(dim=1)  # (batch, d_output)

        return self.out_proj(pooled)
