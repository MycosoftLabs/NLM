"""
Deterministic Preconditioning Stack
====================================

Before any learned weights fire, raw data passes through deterministic
scientific transforms. The model learns residuals on top of physics.

Stack:
1. Physics — FieldPhysicsModel (geomagnetic, lunar, atmospheric)
2. Chemistry — ChemistryEncoder (compound embeddings), QISE (molecular properties)
3. Biology/Mycology — SporeLifecycleSimulator, GeneticCircuitSimulator, DigitalTwinMycelium
4. Calibration/Normalization — TranslationLayer (raw→normalized→bio-tokens)
5. Fingerprint extraction — spectral, acoustic, bioelectric, thermal, chemical, mechanical
6. State assembly — all above composed into PreconditionedInput
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PreconditionedInput:
    """Output of the deterministic preconditioning stack.

    All physics/chemistry/biology computations are done.
    Ready for the learned encoders.
    """

    # Spatial features (from physics)
    spatial_features: np.ndarray = field(default_factory=lambda: np.zeros(37, dtype=np.float32))

    # Temporal features (from physics-derived cycles)
    temporal_features: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))

    # Sensory fingerprint features (padded/flattened)
    spectral_features: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))
    acoustic_features: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float32))
    bioelectric_features: np.ndarray = field(default_factory=lambda: np.zeros(1024, dtype=np.float32))
    thermal_features: np.ndarray = field(default_factory=lambda: np.zeros(1024, dtype=np.float32))
    chemical_features: np.ndarray = field(default_factory=lambda: np.zeros(128, dtype=np.float32))
    mechanical_features: np.ndarray = field(default_factory=lambda: np.zeros(133, dtype=np.float32))
    modality_mask: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))

    # World state features
    env_features: np.ndarray = field(default_factory=lambda: np.zeros(28, dtype=np.float32))
    bio_token_ids: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.int64))
    graph_features: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float32))

    # Self state features
    self_state_features: np.ndarray = field(default_factory=lambda: np.zeros(69, dtype=np.float32))

    # Action features
    recent_action_features: np.ndarray = field(default_factory=lambda: np.zeros((8, 64), dtype=np.float32))
    intent_action_features: np.ndarray = field(default_factory=lambda: np.zeros((4, 64), dtype=np.float32))

    # Metadata
    physics_context: Dict[str, float] = field(default_factory=dict)
    bio_tokens: List[str] = field(default_factory=list)


class DeterministicPreconditioner:
    """Multi-layer deterministic transform stack.

    Wraps existing domain modules to compute physics/chemistry/biology
    context BEFORE the neural network sees data.

    No gradients flow through this — all computations are deterministic.
    """

    def __init__(self, config=None):
        self._physics = None
        self._chemistry_encoder = None
        self._preprocessor = None

        # Bio-token vocabulary lookup
        self._token_to_id: Dict[str, int] = {}
        self._build_token_vocab()

    def _build_token_vocab(self):
        """Build token→id mapping from bio_tokens vocabulary."""
        try:
            from nlm.telemetry.bio_tokens import BIO_TOKEN_VOCABULARY
            # Reserve 0 for PAD, 1 for MASK
            for i, token_code in enumerate(sorted(BIO_TOKEN_VOCABULARY.keys())):
                self._token_to_id[token_code] = i + 2
        except ImportError:
            pass

    def _get_physics(self):
        if self._physics is None:
            from nlm.physics.field_physics import FieldPhysicsModel
            self._physics = FieldPhysicsModel()
        return self._physics

    def _get_chemistry_encoder(self):
        if self._chemistry_encoder is None:
            from nlm.chemistry.encoder import ChemistryEncoder
            self._chemistry_encoder = ChemistryEncoder(embedding_dim=128)
        return self._chemistry_encoder

    def _get_preprocessor(self):
        if self._preprocessor is None:
            from nlm.data.preprocessor import NaturePreprocessor
            self._preprocessor = NaturePreprocessor()
        return self._preprocessor

    def precondition(
        self,
        readings: Dict[str, Any],
        location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        timestamp=None,
        fingerprints: Optional[Dict[str, list]] = None,
        self_state_dict: Optional[Dict[str, Any]] = None,
        action_history: Optional[List[Dict]] = None,
        intended_actions: Optional[List[Dict]] = None,
        compound_data: Optional[Dict[str, Any]] = None,
    ) -> PreconditionedInput:
        """Run the full deterministic preconditioning stack.

        Args:
            readings: Raw sensor readings dict
            location: (lat, lon, alt) tuple
            timestamp: Observation time
            fingerprints: Pre-extracted fingerprints dict
            self_state_dict: MYCA/MAS self state
            action_history: Recent actions
            intended_actions: Planned actions
            compound_data: Chemical compound data for embedding
        """
        preprocessor = self._get_preprocessor()
        result = PreconditionedInput()

        # 1. Physics preconditioning
        physics_ctx = preprocessor.compute_physics_context(location, timestamp)
        result.physics_context = physics_ctx

        # Spatial features = sinusoidal position + geomagnetic + atmospheric
        spatial = preprocessor.encode_spatial(*location)
        geo_atmo = np.array([
            physics_ctx.get("geo_bx", 0), physics_ctx.get("geo_by", 0),
            physics_ctx.get("geo_bz", 0), physics_ctx.get("geo_inclination", 0),
            physics_ctx.get("geo_declination", 0), physics_ctx.get("geo_field_strength", 0),
            physics_ctx.get("atmo_temperature", 0), physics_ctx.get("atmo_pressure", 0),
            physics_ctx.get("atmo_humidity", 0), physics_ctx.get("atmo_wind_speed", 0),
        ], dtype=np.float32)
        result.spatial_features = np.concatenate([spatial, geo_atmo])

        # 2. Temporal features
        result.temporal_features = preprocessor.encode_temporal(timestamp, physics_ctx)

        # 3. Chemistry preconditioning (if compound data)
        if compound_data:
            encoder = self._get_chemistry_encoder()
            result.chemical_features = encoder.encode(compound_data).astype(np.float32)
            result.modality_mask[4] = 1.0

        # 4. Bio-token extraction
        bio_tokens = preprocessor.extract_bio_tokens(readings)
        result.bio_tokens = bio_tokens
        token_ids = np.zeros(256, dtype=np.int64)
        for i, token in enumerate(bio_tokens[:256]):
            token_ids[i] = self._token_to_id.get(token, 1)  # 1 = MASK for unknown
        result.bio_token_ids = token_ids

        # 5. Environmental features
        normalized = preprocessor.normalize_readings(readings)
        env_vals = list(normalized.values())
        physics_vals = list(physics_ctx.values())
        all_env = env_vals + physics_vals
        env_arr = np.zeros(28, dtype=np.float32)
        env_arr[:min(len(all_env), 28)] = all_env[:28]
        result.env_features = env_arr

        # 6. Fingerprint features (if provided)
        if fingerprints:
            if fingerprints.get("spectral"):
                fp = fingerprints["spectral"][0]
                vals = fp.energy_values[:512]
                result.spectral_features[:len(vals)] = vals
                result.modality_mask[0] = 1.0

            if fingerprints.get("acoustic"):
                fp = fingerprints["acoustic"][0]
                vals = fp.magnitude[:256]
                result.acoustic_features[:len(vals)] = vals
                result.modality_mask[1] = 1.0

            if fingerprints.get("bioelectric"):
                fp = fingerprints["bioelectric"][0]
                vals = fp.voltage_series[:1024]
                result.bioelectric_features[:len(vals)] = vals
                result.modality_mask[2] = 1.0

            if fingerprints.get("thermal"):
                fp = fingerprints["thermal"][0]
                flat = [v for row in fp.temperature_field for v in row][:1024]
                result.thermal_features[:len(flat)] = flat
                result.modality_mask[3] = 1.0

            if fingerprints.get("mechanical"):
                fp = fingerprints["mechanical"][0]
                vals = fp.vibration_spectrum[:128]
                result.mechanical_features[:len(vals)] = vals
                result.mechanical_features[128] = fp.pressure_pa
                result.mechanical_features[129:132] = list(fp.force_vector)
                result.mechanical_features[132] = fp.strain
                result.modality_mask[5] = 1.0

        # 7. Self-state features
        if self_state_dict:
            self_arr = np.zeros(69, dtype=np.float32)
            safety_modes = {"normal": 0, "cautious": 1, "lockdown": 2}
            mode = self_state_dict.get("safety_mode", "normal")
            mode_idx = safety_modes.get(mode, 0)
            self_arr[mode_idx] = 1.0
            self_arr[3] = len(self_state_dict.get("active_agents", []))
            self_arr[4] = len(self_state_dict.get("available_tools", []))
            result.self_state_features = self_arr

        # 8. Action features
        if action_history:
            for i, action in enumerate(action_history[:8]):
                action_type_id = hash(action.get("type", "")) % 64
                result.recent_action_features[i, action_type_id % 32] = 1.0

        if intended_actions:
            for i, action in enumerate(intended_actions[:4]):
                action_type_id = hash(action.get("type", "")) % 64
                result.intent_action_features[i, action_type_id % 32] = 1.0

        return result
