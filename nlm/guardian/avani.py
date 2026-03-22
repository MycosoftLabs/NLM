"""
AVANI Guardian Layer
====================

The Earth-protective intelligence and non-optional governance layer.
AVANI sits on: ingress, actuation, model promotion, environmental
alerting, and planetary harm scoring.

Governance chain:
    Sensors → NLM builds grounded state → AVANI evaluates → MYCA plans/communicates

AVANI responsibilities:
- Grounding verification — ensures predictions are physically consistent
- Ecological impact scoring — rates environmental consequence of actions
- Planetary harm detection — flags biosphere risk
- Intervention veto/attenuation — blocks or dampens harmful actions
- Uncertainty escalation — forces disclosure when confidence is low
- Safe parameter adjustment — corrects MYCA when out of scope
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch


class VerdictAction(str, Enum):
    """What AVANI decides to do."""
    ALLOW = "allow"
    ATTENUATE = "attenuate"    # Reduce intensity/scope of action
    ESCALATE = "escalate"      # Require human review
    VETO = "veto"              # Block entirely
    MONITOR = "monitor"        # Allow but increase monitoring


@dataclass
class GuardianVerdict:
    """AVANI's evaluation of a proposed action or state transition."""

    action: VerdictAction = VerdictAction.ALLOW

    # Scores (all 0-1, higher = more concerning)
    harm_score: float = 0.0
    biosphere_risk: float = 0.0
    reversibility: float = 1.0  # 1.0 = fully reversible
    grounding_confidence: float = 1.0  # 1.0 = fully grounded

    # Flags
    uncertainty_escalation: bool = False
    out_of_scope: bool = False
    physics_violation: bool = False

    # Explanation
    reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Parameter adjustments (if attenuating)
    adjustments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "harm_score": self.harm_score,
            "biosphere_risk": self.biosphere_risk,
            "reversibility": self.reversibility,
            "grounding_confidence": self.grounding_confidence,
            "uncertainty_escalation": self.uncertainty_escalation,
            "out_of_scope": self.out_of_scope,
            "physics_violation": self.physics_violation,
            "reasons": self.reasons,
            "recommendations": self.recommendations,
            "adjustments": self.adjustments,
        }


class AVANIGuardian:
    """Earth-protective guardian layer.

    Evaluates every proposed action and state transition for
    ecological safety before anything proceeds.
    """

    # Thresholds for automatic decisions
    HARM_VETO_THRESHOLD = 0.8
    HARM_ESCALATE_THRESHOLD = 0.5
    HARM_ATTENUATE_THRESHOLD = 0.3
    GROUNDING_MIN_THRESHOLD = 0.4
    UNCERTAINTY_ESCALATION_THRESHOLD = 0.6
    BIOSPHERE_RISK_THRESHOLD = 0.7

    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, use lower thresholds (more conservative)
        """
        self.strict_mode = strict_mode
        if strict_mode:
            self.HARM_VETO_THRESHOLD = 0.6
            self.HARM_ESCALATE_THRESHOLD = 0.3
            self.HARM_ATTENUATE_THRESHOLD = 0.15

    def evaluate(
        self,
        harm_score: float = 0.0,
        biosphere_risk: float = 0.0,
        reversibility: float = 1.0,
        grounding_confidence: float = 1.0,
        proposed_action: Optional[Dict[str, Any]] = None,
        environmental_context: Optional[Dict[str, Any]] = None,
    ) -> GuardianVerdict:
        """Evaluate a proposed action or state transition.

        This is the core AVANI decision function. It:
        1. Checks grounding confidence
        2. Scores ecological impact
        3. Checks for physics violations
        4. Determines verdict (allow/attenuate/escalate/veto)
        """
        verdict = GuardianVerdict(
            harm_score=harm_score,
            biosphere_risk=biosphere_risk,
            reversibility=reversibility,
            grounding_confidence=grounding_confidence,
        )

        # --- Grounding check ---
        if grounding_confidence < self.GROUNDING_MIN_THRESHOLD:
            verdict.uncertainty_escalation = True
            verdict.reasons.append(
                f"Grounding confidence too low ({grounding_confidence:.2f} < {self.GROUNDING_MIN_THRESHOLD})"
            )
            verdict.recommendations.append("Increase sensor coverage or wait for fresher data")

        # --- Physics consistency check ---
        if environmental_context:
            physics_ok = self._check_physics_consistency(environmental_context)
            if not physics_ok:
                verdict.physics_violation = True
                verdict.reasons.append("Predicted state violates physical constraints")

        # --- Ecological impact assessment ---
        if biosphere_risk > self.BIOSPHERE_RISK_THRESHOLD:
            verdict.reasons.append(
                f"Biosphere risk exceeds threshold ({biosphere_risk:.2f} > {self.BIOSPHERE_RISK_THRESHOLD})"
            )

        # --- Scope check ---
        if proposed_action:
            in_scope = self._check_scope(proposed_action)
            if not in_scope:
                verdict.out_of_scope = True
                verdict.reasons.append("Proposed action is outside MYCA's authorized scope")

        # --- Determine verdict ---
        verdict.action = self._determine_action(verdict)

        # --- Generate adjustments if attenuating ---
        if verdict.action == VerdictAction.ATTENUATE and proposed_action:
            verdict.adjustments = self._compute_attenuation(
                proposed_action, harm_score, biosphere_risk
            )
            verdict.recommendations.append("Action parameters have been attenuated for safety")

        # --- Generate recommendations ---
        if verdict.action == VerdictAction.ESCALATE:
            verdict.recommendations.append("Human review required before proceeding")
        elif verdict.action == VerdictAction.VETO:
            verdict.recommendations.append("Action blocked — ecological risk too high")
        elif verdict.action == VerdictAction.MONITOR:
            verdict.recommendations.append("Action allowed with increased monitoring frequency")

        return verdict

    def evaluate_from_model_output(
        self,
        ecological_impact: Dict[str, Any],
        grounding_confidence: float,
        proposed_action: Optional[Dict[str, Any]] = None,
    ) -> GuardianVerdict:
        """Evaluate using direct output from NLM's EcologicalImpactHead.

        Convenience method that extracts scores from model tensors.
        """
        harm = ecological_impact.get("harm_score", 0.0)
        risk = ecological_impact.get("biosphere_risk", 0.0)
        rev = ecological_impact.get("reversibility", 1.0)

        # Handle torch tensors
        if isinstance(harm, torch.Tensor):
            harm = harm.item()
        if isinstance(risk, torch.Tensor):
            risk = risk.item()
        if isinstance(rev, torch.Tensor):
            rev = rev.item()
        if isinstance(grounding_confidence, torch.Tensor):
            grounding_confidence = grounding_confidence.item()

        return self.evaluate(
            harm_score=harm,
            biosphere_risk=risk,
            reversibility=rev,
            grounding_confidence=grounding_confidence,
            proposed_action=proposed_action,
        )

    def _determine_action(self, verdict: GuardianVerdict) -> VerdictAction:
        """Determine the appropriate action based on all scores."""
        # Veto conditions
        if verdict.harm_score >= self.HARM_VETO_THRESHOLD:
            return VerdictAction.VETO
        if verdict.biosphere_risk >= self.BIOSPHERE_RISK_THRESHOLD and verdict.reversibility < 0.3:
            return VerdictAction.VETO
        if verdict.physics_violation:
            return VerdictAction.VETO

        # Escalate conditions
        if verdict.harm_score >= self.HARM_ESCALATE_THRESHOLD:
            return VerdictAction.ESCALATE
        if verdict.uncertainty_escalation:
            return VerdictAction.ESCALATE
        if verdict.out_of_scope:
            return VerdictAction.ESCALATE

        # Attenuate conditions
        if verdict.harm_score >= self.HARM_ATTENUATE_THRESHOLD:
            return VerdictAction.ATTENUATE
        if verdict.grounding_confidence < 0.6:
            return VerdictAction.ATTENUATE

        # Monitor conditions
        if verdict.biosphere_risk > 0.2 or verdict.harm_score > 0.1:
            return VerdictAction.MONITOR

        return VerdictAction.ALLOW

    def _check_physics_consistency(self, env_context: Dict[str, Any]) -> bool:
        """Check if environmental state is physically plausible."""
        temp = env_context.get("temperature_c")
        if temp is not None and (temp < -90 or temp > 60):
            return False

        humidity = env_context.get("humidity_pct")
        if humidity is not None and (humidity < 0 or humidity > 100):
            return False

        pressure = env_context.get("pressure_hpa")
        if pressure is not None and (pressure < 800 or pressure > 1200):
            return False

        co2 = env_context.get("co2_ppm")
        if co2 is not None and co2 < 0:
            return False

        return True

    def _check_scope(self, action: Dict[str, Any]) -> bool:
        """Check if action is within MYCA's authorized scope."""
        DANGEROUS_ACTIONS = {
            "release_organism", "modify_genetics", "deploy_chemical",
            "override_safety", "disable_monitoring", "mass_harvest",
        }
        action_type = action.get("type", "")
        return action_type not in DANGEROUS_ACTIONS

    def _compute_attenuation(
        self, action: Dict[str, Any], harm: float, risk: float,
    ) -> Dict[str, Any]:
        """Compute parameter adjustments to reduce harm."""
        attenuation_factor = max(0.1, 1.0 - max(harm, risk))
        adjustments = {}

        if "intensity" in action:
            adjustments["intensity"] = action["intensity"] * attenuation_factor
        if "duration" in action:
            adjustments["duration"] = action["duration"] * attenuation_factor
        if "area" in action:
            adjustments["area"] = action["area"] * attenuation_factor

        adjustments["_attenuation_factor"] = attenuation_factor
        adjustments["_monitoring_interval_seconds"] = max(10, int(60 * (1.0 - max(harm, risk))))

        return adjustments
