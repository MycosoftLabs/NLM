"""
Prediction Heads
=================

Primary heads define the NLM's purpose as a grounded world model:
- NextStatePrediction: predict the next RootedNatureFrame state
- InterventionOutcome: predict consequences of proposed actions
- AnomalyDetection: score anomaly probability per modality
- EcologicalImpact: score environmental consequence (feeds AVANI)
- GroundingConfidence: score how well-grounded current state is
- CausalConsistency: detect violations of causal coherence

Secondary heads provide task-specific outputs.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlm.model.config import NLMConfig


class NextStatePredictionHead(nn.Module):
    """Predict the next observation state.

    The core self-supervised objective: given current grounded state,
    predict what the sensors will read next.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_env_targets),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Args: hidden (batch, hidden_dim). Returns: (batch, num_env_targets)"""
        return self.predictor(hidden)


class InterventionOutcomeHead(nn.Module):
    """Predict the outcome of a proposed intervention.

    Given current state + proposed action, predict the resulting state change.
    Enables counterfactual reasoning and action planning.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        # Action embedding is concatenated with hidden state
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim + config.action_intent_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_env_targets),
        )

    def forward(self, hidden: torch.Tensor, action_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (batch, hidden_dim) — current state
            action_embed: (batch, action_intent_dim) — proposed action
        Returns: (batch, num_env_targets) — predicted state change
        """
        combined = torch.cat([hidden, action_embed], dim=-1)
        return self.predictor(combined)


class AnomalyDetectionHead(nn.Module):
    """Score anomaly probability per modality/category.

    Outputs probability that each sensor category is anomalous.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim // 2, config.num_anomaly_categories),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns: (batch, num_anomaly_categories) in [0, 1]"""
        return self.scorer(hidden)


class EcologicalImpactHead(nn.Module):
    """Score environmental/ecological consequence.

    Feeds directly into AVANI guardian layer.
    Outputs: harm_score, biosphere_risk, reversibility.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim // 2, 128),
            nn.GELU(),
        )
        self.harm_score = nn.Linear(128, 1)
        self.biosphere_risk = nn.Linear(128, 1)
        self.reversibility = nn.Linear(128, 1)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.scorer(hidden)
        return {
            "harm_score": torch.sigmoid(self.harm_score(features)).squeeze(-1),
            "biosphere_risk": torch.sigmoid(self.biosphere_risk(features)).squeeze(-1),
            "reversibility": torch.sigmoid(self.reversibility(features)).squeeze(-1),
        }


class GroundingConfidenceHead(nn.Module):
    """Score how well-grounded the current state representation is.

    Low grounding = high uncertainty, missing data, stale sensors.
    AVANI uses this to decide when to escalate or defer.
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns: (batch,) in [0, 1] — 1.0 = fully grounded"""
        return self.scorer(hidden).squeeze(-1)


class CausalConsistencyHead(nn.Module):
    """Detect violations of causal coherence.

    Given two consecutive states, score whether the transition is
    causally consistent (respects physical laws, temporal ordering).
    """

    def __init__(self, config: NLMConfig):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.ff_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim // 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_t: torch.Tensor, hidden_t1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_t: (batch, hidden_dim) — state at time t
            hidden_t1: (batch, hidden_dim) — state at time t+1
        Returns: (batch,) in [0, 1] — 1.0 = causally consistent
        """
        combined = torch.cat([hidden_t, hidden_t1], dim=-1)
        return self.scorer(combined).squeeze(-1)


# --- Secondary Heads ---

class SpeciesClassificationHead(nn.Module):
    def __init__(self, config: NLMConfig):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.num_species_classes),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden)


class CompoundClassificationHead(nn.Module):
    def __init__(self, config: NLMConfig):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.num_compound_classes),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden)


class GrowthPredictionHead(nn.Module):
    def __init__(self, config: NLMConfig):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 4),  # biomass_delta, fruiting_prob, health_score, days_to_harvest
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.predictor(hidden)
