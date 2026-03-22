"""
NLM AVANI Guardian Layer

AVANI is mandatory on all ingress paths. It sits between prediction
heads and any output/action.

Responsibilities:
- Grounding verification: is this prediction grounded in observed reality?
- Ecological impact scoring: ecological cost of proposed actions
- Planetary harm detection: risk to ecosystems/species/environments
- Intervention veto/attenuation: block actions that fail safety thresholds
- Uncertainty escalation: escalate to human if confidence too low

AVANI governance logic is owned by MAS. This module contains the
runtime client/interface. NLM calls AVANI; it does not define policies.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from nlm.core.frames import RootedNatureFrame

logger = logging.getLogger(__name__)


class GroundingLevel(str, Enum):
    """How well-grounded a prediction is in observed reality."""

    FULLY_GROUNDED = "fully_grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    EXTRAPOLATED = "extrapolated"
    UNGROUNDED = "ungrounded"


class VetoDecision(str, Enum):
    """AVANI's decision on whether to allow an action."""

    ALLOW = "allow"
    ATTENUATE = "attenuate"  # allow with reduced parameters
    ESCALATE = "escalate"    # require human approval
    VETO = "veto"            # block entirely


class EscalationReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    HIGH_ECOLOGICAL_IMPACT = "high_ecological_impact"
    HARM_DETECTED = "harm_detected"
    OUT_OF_SCOPE = "out_of_scope"
    NOVEL_SITUATION = "novel_situation"


@dataclass
class GroundingScore:
    """Result of grounding verification."""

    level: GroundingLevel = GroundingLevel.UNGROUNDED
    score: float = 0.0  # [0, 1]
    supporting_evidence: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)


@dataclass
class EcoScore:
    """Ecological impact assessment."""

    impact_score: float = 0.0  # [0, 1], 0 = no impact, 1 = catastrophic
    affected_species: List[str] = field(default_factory=list)
    affected_ecosystems: List[str] = field(default_factory=list)
    reversibility: float = 1.0  # [0, 1], 1 = fully reversible
    confidence: float = 0.5


@dataclass
class HarmAssessment:
    """Assessment of potential harm from an action."""

    harm_detected: bool = False
    harm_types: List[str] = field(default_factory=list)
    severity: float = 0.0  # [0, 1]
    affected_entities: List[str] = field(default_factory=list)
    mitigation_suggestions: List[str] = field(default_factory=list)


@dataclass
class ProposedAction:
    """An action proposed by the prediction heads or agent."""

    action_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_entity: str = ""
    expected_outcome: str = ""
    confidence: float = 0.5
    source_frame_root: str = ""  # hex of the frame that produced this


@dataclass
class GatedAction:
    """An action after AVANI gating."""

    original_action: ProposedAction = field(default_factory=ProposedAction)
    decision: VetoDecision = VetoDecision.ALLOW
    modified_parameters: Dict[str, Any] = field(default_factory=dict)
    grounding: GroundingScore = field(default_factory=GroundingScore)
    eco_score: EcoScore = field(default_factory=EcoScore)
    harm: HarmAssessment = field(default_factory=HarmAssessment)
    reason: str = ""


@dataclass
class UncertaintyReport:
    """Report on prediction uncertainty for escalation decisions."""

    overall_confidence: float = 0.5
    per_sensor_confidence: Dict[str, float] = field(default_factory=dict)
    missing_sensors: List[str] = field(default_factory=list)
    stale_sensors: List[str] = field(default_factory=list)
    novel_conditions: bool = False


@dataclass
class EscalationDecision:
    """AVANI's decision on whether to escalate to human."""

    should_escalate: bool = False
    reason: EscalationReason = EscalationReason.LOW_CONFIDENCE
    message: str = ""
    fallback_action: Optional[str] = None


@dataclass
class PredictionOutput:
    """Output from prediction heads, to be verified by AVANI."""

    prediction_type: str = ""
    values: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    source_frame_root: str = ""


class AVANIGuardian:
    """
    AVANI guardian client/interface.

    Calls MAS-hosted AVANI services for governance decisions.
    Falls back to conservative local heuristics if MAS is unreachable.
    """

    def __init__(self, mas_url: Optional[str] = None, timeout: float = 5.0):
        self.mas_url = (
            mas_url
            or os.getenv("MAS_API_URL", "http://localhost:8001")
        ).rstrip("/")
        self.timeout = timeout
        logger.info(f"AVANIGuardian initialized: {self.mas_url}")

    async def verify_grounding(
        self,
        frame: RootedNatureFrame,
        prediction: PredictionOutput,
    ) -> GroundingScore:
        """
        Verify that a prediction is grounded in observed reality.

        Checks that the prediction's source frame has valid Merkle roots
        and that the prediction is consistent with the frame's observations.
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.mas_url}/api/avani/grounding",
                    json={
                        "frame": frame.to_dict(),
                        "prediction": {
                            "type": prediction.prediction_type,
                            "values": prediction.values,
                            "confidence": prediction.confidence,
                        },
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return GroundingScore(
                        level=GroundingLevel(data.get("level", "ungrounded")),
                        score=data.get("score", 0.0),
                        supporting_evidence=data.get("evidence", []),
                        gaps=data.get("gaps", []),
                    )
        except Exception as e:
            logger.warning(f"AVANI grounding check failed, using local fallback: {e}")

        # Local fallback: check frame has valid roots
        score = 0.0
        gaps = []
        evidence = []

        if frame.frame_root:
            score += 0.3
            evidence.append("frame_root present")
        else:
            gaps.append("no frame_root")

        if frame.event_root and frame.event_root != b"\x00" * 32:
            score += 0.3
            evidence.append("event_root non-genesis")
        else:
            gaps.append("no observations in frame")

        if prediction.confidence > 0.5:
            score += 0.2
            evidence.append(f"prediction confidence {prediction.confidence:.2f}")

        if frame.observation.fingerprints:
            score += 0.2
            evidence.append(f"{len(frame.observation.fingerprints)} fingerprints present")
        else:
            gaps.append("no sensory fingerprints")

        level = GroundingLevel.UNGROUNDED
        if score >= 0.8:
            level = GroundingLevel.FULLY_GROUNDED
        elif score >= 0.5:
            level = GroundingLevel.PARTIALLY_GROUNDED
        elif score >= 0.3:
            level = GroundingLevel.EXTRAPOLATED

        return GroundingScore(
            level=level, score=score,
            supporting_evidence=evidence, gaps=gaps,
        )

    async def score_ecological_impact(
        self,
        action: ProposedAction,
        world_state: Dict[str, Any],
    ) -> EcoScore:
        """Score the ecological impact of a proposed action."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.mas_url}/api/avani/eco-impact",
                    json={
                        "action": {
                            "type": action.action_type,
                            "parameters": action.parameters,
                            "target": action.target_entity,
                        },
                        "world_state": world_state,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return EcoScore(
                        impact_score=data.get("impact_score", 0.0),
                        affected_species=data.get("affected_species", []),
                        affected_ecosystems=data.get("affected_ecosystems", []),
                        reversibility=data.get("reversibility", 1.0),
                        confidence=data.get("confidence", 0.5),
                    )
        except Exception as e:
            logger.warning(f"AVANI eco-impact check failed, using fallback: {e}")

        # Conservative fallback: assume moderate impact
        return EcoScore(impact_score=0.3, confidence=0.3, reversibility=0.8)

    async def detect_harm(self, action: ProposedAction) -> HarmAssessment:
        """Detect potential harm from a proposed action."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.mas_url}/api/avani/harm-detection",
                    json={
                        "action": {
                            "type": action.action_type,
                            "parameters": action.parameters,
                            "target": action.target_entity,
                        },
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return HarmAssessment(
                        harm_detected=data.get("harm_detected", False),
                        harm_types=data.get("harm_types", []),
                        severity=data.get("severity", 0.0),
                        affected_entities=data.get("affected_entities", []),
                        mitigation_suggestions=data.get("mitigations", []),
                    )
        except Exception as e:
            logger.warning(f"AVANI harm detection failed, using fallback: {e}")

        return HarmAssessment(harm_detected=False, severity=0.0)

    async def gate_action(
        self,
        action: ProposedAction,
        frame: RootedNatureFrame,
    ) -> GatedAction:
        """
        Full AVANI gate: grounding + eco-impact + harm -> decision.

        This is the main entry point for all action gating.
        """
        prediction = PredictionOutput(
            prediction_type=action.action_type,
            values=action.parameters,
            confidence=action.confidence,
            source_frame_root=action.source_frame_root,
        )

        grounding = await self.verify_grounding(frame, prediction)
        eco_score = await self.score_ecological_impact(
            action,
            frame.to_dict().get("world_state", {}),
        )
        harm = await self.detect_harm(action)

        # Decision logic
        decision = VetoDecision.ALLOW
        reason = ""

        if harm.harm_detected and harm.severity > 0.7:
            decision = VetoDecision.VETO
            reason = f"Harm detected: {', '.join(harm.harm_types)}"
        elif eco_score.impact_score > 0.8:
            decision = VetoDecision.VETO
            reason = f"Ecological impact too high: {eco_score.impact_score:.2f}"
        elif grounding.level == GroundingLevel.UNGROUNDED:
            decision = VetoDecision.ESCALATE
            reason = "Prediction is ungrounded"
        elif eco_score.impact_score > 0.5 or harm.severity > 0.3:
            decision = VetoDecision.ATTENUATE
            reason = "Moderate risk detected"

        return GatedAction(
            original_action=action,
            decision=decision,
            modified_parameters=action.parameters if decision == VetoDecision.ALLOW else {},
            grounding=grounding,
            eco_score=eco_score,
            harm=harm,
            reason=reason,
        )

    async def escalate_uncertainty(
        self, uncertainty: UncertaintyReport
    ) -> EscalationDecision:
        """Decide whether to escalate based on uncertainty levels."""
        if uncertainty.novel_conditions:
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.NOVEL_SITUATION,
                message="Novel conditions detected — human review recommended",
                fallback_action="safe_defaults",
            )

        if uncertainty.overall_confidence < 0.3:
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.LOW_CONFIDENCE,
                message=f"Overall confidence too low: {uncertainty.overall_confidence:.2f}",
                fallback_action="safe_defaults",
            )

        if len(uncertainty.missing_sensors) > 3:
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.LOW_CONFIDENCE,
                message=f"{len(uncertainty.missing_sensors)} sensors missing",
                fallback_action="wait_for_data",
            )

        return EscalationDecision(should_escalate=False)
