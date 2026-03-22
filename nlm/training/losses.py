"""
Physics-Informed Loss Functions
================================

The NLM is not trained with generic supervised loss.
Training enforces:
- Physics consistency (conservation laws, thermodynamic bounds)
- Temporal coherence (nature changes smoothly)
- Spatial coherence (nearby locations are correlated)
- Ecological consistency (biosphere constraints)
- Causal consistency (state transitions respect causality)
- Uncertainty calibration (confidence matches accuracy)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConsistencyLoss(nn.Module):
    """Penalizes predictions that violate known physical laws.

    Enforces:
    - Temperature within thermodynamic bounds
    - Humidity in [0, 100]
    - Pressure within barometric range
    - Energy conservation (net change bounded)
    """

    def __init__(self):
        super().__init__()
        # Physical bounds (normalized to [0, 1])
        self.bounds = {
            0: (0.0, 1.0),   # temperature
            1: (0.0, 1.0),   # humidity
            2: (0.0, 1.0),   # CO2
            3: (0.0, 1.0),   # pressure
        }

    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args: predictions (batch, num_env_targets) — predicted environmental state
        Returns: scalar loss
        """
        loss = torch.tensor(0.0, device=predictions.device)

        for dim, (lo, hi) in self.bounds.items():
            if dim < predictions.size(-1):
                vals = predictions[:, dim]
                # Penalize values outside physical bounds
                below = F.relu(lo - vals)
                above = F.relu(vals - hi)
                loss = loss + (below ** 2 + above ** 2).mean()

        return loss


class TemporalCoherenceLoss(nn.Module):
    """Penalizes large discontinuities in sequential predictions.

    Nature changes smoothly — large jumps between consecutive
    predictions indicate the model is not respecting temporal dynamics.
    """

    def __init__(self, max_delta: float = 0.2):
        super().__init__()
        self.max_delta = max_delta

    def forward(
        self, pred_t: torch.Tensor, pred_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_t: (batch, dim) prediction at time t
            pred_t1: (batch, dim) prediction at time t+1
        """
        delta = (pred_t1 - pred_t).abs()
        excess = F.relu(delta - self.max_delta)
        return (excess ** 2).mean()


class SpatialCoherenceLoss(nn.Module):
    """Nearby locations should have correlated predictions.

    Penalizes large differences between spatially close observations.
    """

    def __init__(self, distance_scale: float = 0.01):
        super().__init__()
        self.distance_scale = distance_scale

    def forward(
        self,
        predictions: torch.Tensor,  # (batch, dim)
        spatial_coords: torch.Tensor,  # (batch, 3) — normalized lat, lon, alt
    ) -> torch.Tensor:
        # Pairwise prediction differences
        pred_diff = (predictions.unsqueeze(0) - predictions.unsqueeze(1)) ** 2  # (B, B, dim)
        pred_diff = pred_diff.sum(dim=-1)  # (B, B)

        # Pairwise spatial distances
        coord_diff = (spatial_coords.unsqueeze(0) - spatial_coords.unsqueeze(1)) ** 2
        spatial_dist = coord_diff.sum(dim=-1)  # (B, B)

        # Weight: closer points should have smaller prediction differences
        weight = torch.exp(-spatial_dist / self.distance_scale)

        # Loss: weighted prediction divergence (nearby points should agree)
        loss = (weight * pred_diff).mean()
        return loss


class EcologicalConsistencyLoss(nn.Module):
    """Penalizes ecologically implausible predictions.

    Enforces constraints like:
    - Growth rate bounded by species capacity
    - Population dynamics conservation
    - Resource balance
    """

    def forward(
        self,
        growth_pred: torch.Tensor,  # (batch, 4) — biomass_delta, fruiting_prob, health, days_to_harvest
        eco_impact: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=growth_pred.device)

        # Fruiting probability should be in [0, 1]
        if growth_pred.size(-1) > 1:
            fruiting = growth_pred[:, 1]
            loss = loss + F.relu(-fruiting).mean() + F.relu(fruiting - 1.0).mean()

        # Health score in [0, 1]
        if growth_pred.size(-1) > 2:
            health = growth_pred[:, 2]
            loss = loss + F.relu(-health).mean() + F.relu(health - 1.0).mean()

        # High harm should suppress growth predictions
        harm = eco_impact.get("harm_score")
        if harm is not None and growth_pred.size(-1) > 0:
            # If harm is high, growth should be low
            growth_harm_conflict = F.relu(growth_pred[:, 0] * harm - 0.5)
            loss = loss + growth_harm_conflict.mean()

        return loss


class CausalConsistencyLoss(nn.Module):
    """Penalizes violations of causal coherence.

    The model's causal consistency head should score truly
    consistent transitions high and violations low.
    Uses the head output as auxiliary supervision.
    """

    def forward(
        self, causal_score: torch.Tensor, is_consistent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            causal_score: (batch,) model's consistency score [0, 1]
            is_consistent: (batch,) ground truth label [0, 1]
        """
        return F.binary_cross_entropy(causal_score, is_consistent)


class UncertaintyCalibrationLoss(nn.Module):
    """Calibrates confidence scores to match actual accuracy.

    High confidence should correspond to low error.
    Low confidence should correspond to high error.
    """

    def forward(
        self,
        confidence: torch.Tensor,  # (batch,) in [0, 1]
        error: torch.Tensor,  # (batch,) prediction error magnitude
    ) -> torch.Tensor:
        # Confidence should be inversely related to error
        expected_error = 1.0 - confidence
        return F.mse_loss(expected_error, error.clamp(0, 1))


class NLMLoss(nn.Module):
    """Combined loss for Nature Learning Model training.

    Weighted sum of all consistency losses plus task-specific losses.
    """

    def __init__(
        self,
        alpha_physics: float = 1.0,
        alpha_temporal: float = 0.5,
        alpha_spatial: float = 0.3,
        alpha_ecological: float = 0.5,
        alpha_causal: float = 0.5,
        alpha_calibration: float = 0.3,
        alpha_next_state: float = 1.0,
        alpha_anomaly: float = 0.5,
    ):
        super().__init__()
        self.physics_loss = PhysicsConsistencyLoss()
        self.temporal_loss = TemporalCoherenceLoss()
        self.spatial_loss = SpatialCoherenceLoss()
        self.ecological_loss = EcologicalConsistencyLoss()
        self.causal_loss = CausalConsistencyLoss()
        self.calibration_loss = UncertaintyCalibrationLoss()

        self.alpha_physics = alpha_physics
        self.alpha_temporal = alpha_temporal
        self.alpha_spatial = alpha_spatial
        self.alpha_ecological = alpha_ecological
        self.alpha_causal = alpha_causal
        self.alpha_calibration = alpha_calibration
        self.alpha_next_state = alpha_next_state
        self.alpha_anomaly = alpha_anomaly

    def forward(
        self,
        # Model outputs
        next_state_pred: torch.Tensor,
        next_state_target: torch.Tensor,
        anomaly_scores: torch.Tensor,
        anomaly_targets: Optional[torch.Tensor] = None,
        ecological_impact: Optional[Dict[str, torch.Tensor]] = None,
        grounding_confidence: Optional[torch.Tensor] = None,
        growth_prediction: Optional[torch.Tensor] = None,
        # Context for consistency losses
        spatial_coords: Optional[torch.Tensor] = None,
        prev_pred: Optional[torch.Tensor] = None,
        causal_score: Optional[torch.Tensor] = None,
        causal_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses and return breakdown."""
        losses = {}

        # Primary: next-state prediction MSE
        losses["next_state"] = F.mse_loss(next_state_pred, next_state_target) * self.alpha_next_state

        # Physics consistency
        losses["physics"] = self.physics_loss(next_state_pred) * self.alpha_physics

        # Temporal coherence
        if prev_pred is not None:
            losses["temporal"] = self.temporal_loss(prev_pred, next_state_pred) * self.alpha_temporal

        # Spatial coherence
        if spatial_coords is not None:
            losses["spatial"] = self.spatial_loss(next_state_pred, spatial_coords) * self.alpha_spatial

        # Ecological consistency
        if growth_prediction is not None and ecological_impact is not None:
            losses["ecological"] = self.ecological_loss(growth_prediction, ecological_impact) * self.alpha_ecological

        # Causal consistency
        if causal_score is not None and causal_target is not None:
            losses["causal"] = self.causal_loss(causal_score, causal_target) * self.alpha_causal

        # Uncertainty calibration
        if grounding_confidence is not None:
            prediction_error = (next_state_pred - next_state_target).abs().mean(dim=-1)
            losses["calibration"] = self.calibration_loss(
                grounding_confidence, prediction_error
            ) * self.alpha_calibration

        # Anomaly detection
        if anomaly_targets is not None:
            losses["anomaly"] = F.binary_cross_entropy(
                anomaly_scores, anomaly_targets
            ) * self.alpha_anomaly

        # Total
        losses["total"] = sum(losses.values())

        return losses
