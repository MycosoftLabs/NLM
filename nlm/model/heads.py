"""
NLM Prediction Heads

Primary heads (core world model outputs):
- NextStatePredictionHead: what happens next?
- InterventionOutcomeHead: what if we do X?
- AnomalyDetectionHead: is something unusual?
- EcologicalImpactHead: what is the ecological cost?
- GroundingConfidenceHead: how grounded is this prediction?
- CausalConsistencyHead: does this respect causal structure?

Secondary heads (domain-specific):
- SpeciesPredictionHead
- CompoundPredictionHead
- GrowthPredictionHead
- ClassificationHead
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Primary Heads ───────────────────────────────────────────────────


class NextStatePredictionHead(nn.Module):
    """
    Predict the next RootedNatureFrame state.

    The primary objective of the NLM world model: given the current
    fused state representation, predict the state at the next timestep.
    """

    def __init__(self, d_input: int = 256, d_state: int = 512):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_input, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_input),
        )
        self.uncertainty_head = nn.Linear(d_input, d_input)  # predicts per-dim uncertainty

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused: (batch, d_input)

        Returns:
            dict with 'predicted_state' and 'uncertainty'
        """
        predicted = self.predictor(fused)
        uncertainty = F.softplus(self.uncertainty_head(fused))
        return {
            "predicted_state": predicted,
            "uncertainty": uncertainty,
        }


class InterventionOutcomeHead(nn.Module):
    """
    Predict outcome given a proposed action.

    Counterfactual prediction: what would happen if we take action X
    from the current state?
    """

    def __init__(self, d_state: int = 256, d_action: int = 64):
        super().__init__()
        self.merger = nn.Sequential(
            nn.Linear(d_state + d_action, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_state),
        )
        self.confidence = nn.Linear(d_state, 1)

    def forward(
        self, fused: torch.Tensor, action_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused: (batch, d_state)
            action_embedding: (batch, d_action)
        """
        combined = torch.cat([fused, action_embedding], dim=-1)
        outcome = self.merger(combined)
        conf = torch.sigmoid(self.confidence(outcome))
        return {"predicted_outcome": outcome, "confidence": conf}


class AnomalyDetectionHead(nn.Module):
    """Detect anomalies in the current state."""

    def __init__(self, d_input: int = 256, n_anomaly_types: int = 8):
        super().__init__()
        self.score_head = nn.Sequential(
            nn.Linear(d_input, d_input // 2),
            nn.SiLU(),
            nn.Linear(d_input // 2, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(d_input, d_input // 2),
            nn.SiLU(),
            nn.Linear(d_input // 2, n_anomaly_types),
        )

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        anomaly_score = torch.sigmoid(self.score_head(fused))
        anomaly_type_logits = self.type_head(fused)
        return {
            "anomaly_score": anomaly_score,
            "anomaly_type_logits": anomaly_type_logits,
        }


class EcologicalImpactHead(nn.Module):
    """Score the ecological impact of the current or predicted state."""

    def __init__(self, d_input: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_input, d_input // 2),
            nn.SiLU(),
            nn.Linear(d_input // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.head(fused)  # (batch, 1) in [0, 1]


class GroundingConfidenceHead(nn.Module):
    """Estimate how well-grounded a prediction is."""

    def __init__(self, d_input: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_input, d_input // 2),
            nn.SiLU(),
            nn.Linear(d_input // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.head(fused)


class CausalConsistencyHead(nn.Module):
    """Score whether a prediction respects known causal structure."""

    def __init__(self, d_input: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_input, d_input // 2),
            nn.SiLU(),
            nn.Linear(d_input // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.head(fused)


# ── Secondary Heads ─────────────────────────────────────────────────


class SpeciesPredictionHead(nn.Module):
    """Predict species from fused state."""

    def __init__(self, d_input: int = 256, n_species: int = 1000):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.SiLU(),
            nn.Linear(d_input, n_species),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.head(fused)  # logits


class CompoundPredictionHead(nn.Module):
    """Predict compound/chemical from fused state."""

    def __init__(self, d_input: int = 256, n_compounds: int = 500):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.SiLU(),
            nn.Linear(d_input, n_compounds),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.head(fused)


class GrowthPredictionHead(nn.Module):
    """Predict growth/fruiting probability and timing."""

    def __init__(self, d_input: int = 256):
        super().__init__()
        self.probability = nn.Sequential(
            nn.Linear(d_input, d_input // 2),
            nn.SiLU(),
            nn.Linear(d_input // 2, 1),
            nn.Sigmoid(),
        )
        self.timing = nn.Sequential(
            nn.Linear(d_input, d_input // 2),
            nn.SiLU(),
            nn.Linear(d_input // 2, 1),  # days to event
            nn.Softplus(),
        )

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "probability": self.probability(fused),
            "days_to_event": self.timing(fused),
        }


class ClassificationHead(nn.Module):
    """General-purpose classification head."""

    def __init__(self, d_input: int = 256, n_classes: int = 100):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_input, n_classes),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.head(fused)
