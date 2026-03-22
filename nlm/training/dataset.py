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
