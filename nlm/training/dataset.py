"""
NLM Dataset & DataLoader
=========================

Loads RootedNatureFrames and converts them to model-ready tensor batches.
Supports loading from:
- JSON/JSONL files of serialized frames
- MINDEX frame store (via MINDEXClient)
- Synthetic generation for testing

Every training example is linked to provenance in MINDEX.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nlm.model.config import NLMConfig
from nlm.model.preconditioner import DeterministicPreconditioner, PreconditionedInput


class RootedFrameDataset(Dataset):
    """Dataset of preconditioned NLM training samples.

    Each sample is a PreconditionedInput ready for the model.
    """

    def __init__(
        self,
        frames: Optional[List[Dict[str, Any]]] = None,
        frame_dir: Optional[str] = None,
        config: Optional[NLMConfig] = None,
        augment: bool = False,
    ):
        self.config = config or NLMConfig()
        self.preconditioner = DeterministicPreconditioner(self.config)
        self.augment = augment
        self.augmentor = None

        if augment:
            from nlm.data.augmentation import NatureAugmentor
            self.augmentor = NatureAugmentor()

        # Load frames
        self.frames: List[Dict[str, Any]] = []
        if frames:
            self.frames = frames
        elif frame_dir:
            self._load_from_dir(frame_dir)

    def _load_from_dir(self, frame_dir: str) -> None:
        path = Path(frame_dir)
        for f in sorted(path.glob("*.json")):
            with open(f) as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    self.frames.extend(data)
                else:
                    self.frames.append(data)
        for f in sorted(path.glob("*.jsonl")):
            with open(f) as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        self.frames.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame = self.frames[idx]

        # Extract raw data from frame
        observation = frame.get("observation", {})
        readings = observation.get("normalized_physical", {})
        bio_tokens = observation.get("bio_tokens", [])
        geo = frame.get("geolocation", [0.0, 0.0, 0.0])
        if isinstance(geo, list):
            geo = tuple(geo)

        timestamp = frame.get("timestamp")
        self_state = frame.get("self_state", {})
        action_ctx = frame.get("action_context", {})

        # Augment if enabled
        if self.augment and self.augmentor:
            aug = self.augmentor.augment(
                geolocation=geo, readings=readings, bio_tokens=bio_tokens,
            )
            geo = aug.get("geolocation", geo)
            readings = aug.get("readings", readings)
            bio_tokens = aug.get("bio_tokens", bio_tokens)

        # Precondition
        preconditioned = self.preconditioner.precondition(
            readings=readings,
            location=geo,
            timestamp=timestamp,
            self_state_dict=self_state,
            action_history=action_ctx.get("recent_actions", []),
            intended_actions=action_ctx.get("intended_actions", []),
        )

        return self._to_tensors(preconditioned)

    def _to_tensors(self, p: PreconditionedInput) -> Dict[str, torch.Tensor]:
        return {
            "spatial_features": torch.from_numpy(p.spatial_features),
            "temporal_features": torch.from_numpy(p.temporal_features),
            "spectral": torch.from_numpy(p.spectral_features),
            "acoustic": torch.from_numpy(p.acoustic_features),
            "bioelectric": torch.from_numpy(p.bioelectric_features),
            "thermal": torch.from_numpy(p.thermal_features),
            "chemical": torch.from_numpy(p.chemical_features),
            "mechanical": torch.from_numpy(p.mechanical_features),
            "modality_mask": torch.from_numpy(p.modality_mask),
            "env_features": torch.from_numpy(p.env_features),
            "bio_token_ids": torch.from_numpy(p.bio_token_ids),
            "graph_features": torch.from_numpy(p.graph_features),
            "self_state_features": torch.from_numpy(p.self_state_features),
            "recent_actions": torch.from_numpy(p.recent_action_features),
            "intended_actions": torch.from_numpy(p.intent_action_features),
        }


class SyntheticNatureDataset(Dataset):
    """Generates synthetic training data for testing/bootstrapping.

    Produces physically plausible random nature observations.
    """

    def __init__(self, num_samples: int = 1000, config: Optional[NLMConfig] = None):
        self.num_samples = num_samples
        self.config = config or NLMConfig()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.RandomState(idx)

        return {
            "spatial_features": torch.randn(37),
            "temporal_features": torch.randn(12),
            "spectral": torch.randn(self.config.max_spectral_bins).abs(),
            "acoustic": torch.randn(self.config.max_acoustic_bins).abs(),
            "bioelectric": torch.randn(self.config.max_bioelectric_samples),
            "thermal": torch.randn(self.config.max_thermal_grid ** 2),
            "chemical": torch.randn(self.config.chemical_vector_dim),
            "mechanical": torch.randn(self.config.max_mechanical_bins + 5),
            "modality_mask": torch.ones(6),
            "env_features": torch.randn(self.config.num_env_targets + 14),
            "bio_token_ids": torch.randint(0, self.config.num_bio_token_types, (self.config.max_bio_tokens,)),
            "graph_features": torch.randn(self.config.graph_hidden_dim),
            "self_state_features": torch.randn(69),
            "recent_actions": torch.randn(8, 64),
            "intended_actions": torch.randn(4, 64),
            # Targets
            "next_state_target": torch.randn(self.config.num_env_targets),
            "anomaly_target": torch.zeros(self.config.num_anomaly_categories),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


# --- Maritime / TAC-O Dataset ---


class MaritimeAcousticDataset(Dataset):
    """Dataset for maritime acoustic training data loaded from MINDEX.

    Loads acoustic signatures, ocean environments, and TAC-O observations
    from the MINDEX maritime tables for NLM maritime head training.
    """

    CATEGORIES = [
        'submarine', 'surface_vessel', 'torpedo', 'uuv',
        'mine', 'marine_mammal', 'fish_school', 'seismic',
        'weather_noise', 'shipping_noise', 'ambient', 'unknown',
    ]

    def __init__(
        self,
        data_path: Optional[str] = None,
        mindex_url: str = "http://192.168.0.189:8000",
        config: Optional[NLMConfig] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.config = config or NLMConfig()
        self.mindex_url = mindex_url
        self.samples: List[Dict[str, Any]] = []

        if data_path:
            self._load_from_file(data_path, max_samples)

    def _load_from_file(self, path: str, max_samples: Optional[int] = None):
        """Load training data from JSON/JSONL file."""
        p = Path(path)
        if not p.exists():
            return

        if p.suffix == ".jsonl":
            with open(p) as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    try:
                        self.samples.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        elif p.suffix == ".json":
            with open(p) as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.samples = data[:max_samples] if max_samples else data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        hydroacoustic = self._extract_hydroacoustic(sample)
        magnetic = self._extract_magnetic(sample)
        ocean_env = self._extract_ocean_env(sample)

        category = sample.get("category", "unknown")
        category_idx = self.CATEGORIES.index(category) if category in self.CATEGORIES else len(self.CATEGORIES) - 1

        is_marine_mammal = 1.0 if category == "marine_mammal" else 0.0

        return {
            "hydroacoustic": torch.tensor(hydroacoustic, dtype=torch.float32),
            "magnetic": torch.tensor(magnetic, dtype=torch.float32),
            "ocean_env": torch.tensor(ocean_env, dtype=torch.float32),
            "target_category": torch.tensor(category_idx, dtype=torch.long),
            "is_marine_mammal": torch.tensor(is_marine_mammal, dtype=torch.float32),
        }

    def _extract_hydroacoustic(self, sample: Dict) -> List[float]:
        """Extract hydroacoustic feature vector from sample."""
        fingerprint = sample.get("fingerprint", {})
        spectral_energy = fingerprint.get("spectral_energy", [])
        target_len = self.config.max_hydroacoustic_bins
        vec = spectral_energy[:target_len]
        vec.extend([0.0] * (target_len - len(vec)))
        return vec

    def _extract_magnetic(self, sample: Dict) -> List[float]:
        """Extract magnetic anomaly feature vector."""
        mag = sample.get("magnetic", {})
        return [
            mag.get("Bx", 0.0), mag.get("By", 0.0), mag.get("Bz", 0.0),
            mag.get("total_field", 0.0), mag.get("inclination", 0.0),
            mag.get("declination", 0.0), mag.get("anomaly_magnitude", 0.0),
            mag.get("gradient_x", 0.0), mag.get("gradient_y", 0.0),
            mag.get("dipole_moment_estimate", 0.0),
        ]

    def _extract_ocean_env(self, sample: Dict) -> List[float]:
        """Extract ocean environment feature vector."""
        env = sample.get("environment", {})
        ssp = env.get("sound_speed_profile", [])
        ssp_flat = []
        for depth, speed in ssp[:30]:
            ssp_flat.extend([depth, speed])
        target_len = self.config.max_ocean_env_features
        vec = [
            env.get("sea_surface_temp", 0.0),
            env.get("salinity", 35.0),
            env.get("sea_state", 0),
            env.get("current_speed", 0.0),
        ] + ssp_flat
        vec = vec[:target_len]
        vec.extend([0.0] * (target_len - len(vec)))
        return vec
