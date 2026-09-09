"""
Confidence-Gated Handoff Logic for Stage 3 to Stage 4 Integration.
Evaluates model confidence and rare-class uncertainty to safely route payloads
to either automated Stage 4 treatment optimization or an oncology human-review queue.
"""

from typing import Dict, Any, List, Optional, Literal, Tuple
from dataclasses import dataclass, field
import logging
from contract_validation import Stage3OutputPayload

logger = logging.getLogger("stage3_stage4.confidence_gate")

DEFAULT_RARE_HAZARD_CLASSES: Tuple[str, ...] = ("CARDIAC", "DERMATOLOGIC", "NEUROPATHIC")


@dataclass(frozen=True)
class ConfidenceGateConfig:
    """
    Configuration for Stage 3 -> Stage 4 confidence gating.
    
    Attributes:
        urgency_min_confidence: Minimum posterior confidence required for automated urgency routing. Default 0.65.
        hazard_min_confidence: Minimum posterior confidence required for automated standard hazard routing. Default 0.65.
        rare_hazard_min_confidence: Elevated threshold (0.70) for rare toxicity hazard classes 
            (CARDIAC, DERMATOLOGIC, NEUROPATHIC) justified by their wide Clopper-Pearson 95% CIs.
        critical_override_threshold: Probability threshold P(CRITICAL) that rescues borderline ambulatory notes.
            Set to 0.30 for Config D safety gating; None or 0.50 for Config C pure argmax.
        mode: Operating mode ('CONFIG_C_PROMOTED' or 'CONFIG_D_GATED').
        rare_hazard_classes: Tuple of organ toxicity classes considered rare / high-estimation-variance.
    """
    urgency_min_confidence: float = 0.65
    hazard_min_confidence: float = 0.65
    rare_hazard_min_confidence: float = 0.70
    critical_override_threshold: Optional[float] = 0.30
    mode: Literal["CONFIG_C_PROMOTED", "CONFIG_D_GATED"] = "CONFIG_C_PROMOTED"
    rare_hazard_classes: Tuple[str, ...] = DEFAULT_RARE_HAZARD_CLASSES

    def __post_init__(self):
        for name, val in [
            ("urgency_min_confidence", self.urgency_min_confidence),
            ("hazard_min_confidence", self.hazard_min_confidence),
            ("rare_hazard_min_confidence", self.rare_hazard_min_confidence),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")
        if self.critical_override_threshold is not None:
            if not (0.0 <= self.critical_override_threshold <= 1.0):
                raise ValueError(f"critical_override_threshold must be in [0.0, 1.0], got {self.critical_override_threshold}")


@dataclass
class RoutingResult:
    """
    Structured outcome of the confidence gate evaluation.
    """
    destination: Literal["STAGE_4_AUTOMATED", "HUMAN_REVIEW"]
    is_automated: bool
    routing_reasons: List[str]
    effective_urgency_class: str
    effective_hazard_class: str
    evaluated_metrics: Dict[str, Any] = field(default_factory=dict)


class ConfidenceGate:
    """
    Clinical decision safety gate enforcing triage urgency and toxicity hazard thresholds.
    """

    def __init__(self, config: Optional[ConfidenceGateConfig] = None):
        self.config = config or ConfidenceGateConfig()

    def evaluate(self, payload: Stage3OutputPayload) -> RoutingResult:
        """
        Evaluate a validated Stage 3 output payload against confidence thresholds.
        Returns RoutingResult with destination 'STAGE_4_AUTOMATED' or 'HUMAN_REVIEW'.
        """
        urgency = payload.triage_urgency
        hazard = payload.toxicity_hazard
        reasons: List[str] = []

        urg_class = urgency.predicted_class
        urg_conf = urgency.confidence
        haz_class = hazard.predicted_class
        haz_conf = hazard.confidence

        effective_urgency = urg_class
        effective_hazard = haz_class

        # 1. Critical Urgency Safety Rescue (Config D Gating mode or override enabled)
        if self.config.critical_override_threshold is not None and urgency.class_probabilities:
            p_crit = urgency.class_probabilities.get("CRITICAL", 0.0)
            if urg_class != "CRITICAL" and p_crit >= self.config.critical_override_threshold:
                reasons.append(
                    f"CRITICAL safety override triggered: P(CRITICAL)={p_crit:.4f} >= threshold ({self.config.critical_override_threshold:.2f}). Rescued from {urg_class} to CRITICAL."
                )
                effective_urgency = "CRITICAL"
                # If rescued to CRITICAL with borderline confidence, route to human review for confirmation
                if p_crit < 0.65:
                    reasons.append(
                        f"Rescued CRITICAL note has posterior {p_crit:.4f} < 0.65: requiring clinical oncologist confirmation."
                    )

        # 2. Triage Urgency Confidence Check
        if effective_urgency != "CRITICAL" or self.config.critical_override_threshold is None:
            if urg_conf < self.config.urgency_min_confidence:
                reasons.append(
                    f"Low urgency confidence: {urg_conf:.4f} < floor ({self.config.urgency_min_confidence:.2f}) for class '{effective_urgency}'."
                )

        # 3. Toxicity Hazard Confidence Check (Standard vs. Rare Toxicity Classes)
        if haz_class in self.config.rare_hazard_classes:
            # Elevated threshold for rare classes due to small cohort support (n < 30)
            if haz_conf < self.config.rare_hazard_min_confidence:
                reasons.append(
                    f"Elevated rare hazard threshold triggered: '{haz_class}' confidence {haz_conf:.4f} < rare class floor ({self.config.rare_hazard_min_confidence:.2f}). Small validation cohort requires oncologist review."
                )
        else:
            # Standard threshold for common hazard classes
            if haz_conf < self.config.hazard_min_confidence:
                reasons.append(
                    f"Low hazard confidence: {haz_conf:.4f} < floor ({self.config.hazard_min_confidence:.2f}) for class '{haz_class}'."
                )

        # 4. Final Routing Determination
        if reasons:
            destination = "HUMAN_REVIEW"
            is_automated = False
            logger.info(
                f"Document {payload.document_id} routed to HUMAN_REVIEW. Reasons: {'; '.join(reasons)}"
            )
        else:
            destination = "STAGE_4_AUTOMATED"
            is_automated = True
            reasons.append("Confidence satisfies all clinical safety thresholds for automated Stage 4 optimization.")
            logger.debug(f"Document {payload.document_id} routed to STAGE_4_AUTOMATED.")

        return RoutingResult(
            destination=destination,
            is_automated=is_automated,
            routing_reasons=reasons,
            effective_urgency_class=effective_urgency,
            effective_hazard_class=effective_hazard,
            evaluated_metrics={
                "urgency_predicted_class": urg_class,
                "urgency_confidence": urg_conf,
                "hazard_predicted_class": haz_class,
                "hazard_confidence": haz_conf,
                "is_rare_hazard": haz_class in self.config.rare_hazard_classes,
                "gate_mode": self.config.mode,
            }
        )
