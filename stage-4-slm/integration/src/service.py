"""
Clinical Decision Support Service Orchestrator for Stage 4.
Coordinates SLM inference engine, safety guardrails, telemetry, and fallback routing.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

INTEG_SRC_DIR = Path(__file__).resolve().parent
SLM_SRC_DIR = INTEG_SRC_DIR.parent.parent / "slm" / "src"
sys.path.insert(0, str(INTEG_SRC_DIR))
sys.path.insert(0, str(SLM_SRC_DIR))

from model import ClinicalDecisionSupportEngine
from guardrails import ClinicalSafetyGuardrails
from fallback import ClinicalDecisionFallback
from schemas import (
    DecisionSupportResponse,
    RiskResponse,
    ActionResponse,
    BatchDecisionSupportResponse,
    HealthResponse
)

logger = logging.getLogger("stage4.integration.service")


class ClinicalDecisionService:
    """Core decision support orchestrator."""

    def __init__(self):
        self.start_time = time.time()
        self.engine = ClinicalDecisionSupportEngine()
        self.guardrails = ClinicalSafetyGuardrails()
        self.fallback = ClinicalDecisionFallback()
        self.model_version = "clinical-slm-oncology-v1.0.0"
        logger.info("ClinicalDecisionService initialized and operational.")

    def get_health(self) -> HealthResponse:
        uptime = time.time() - self.start_time
        return HealthResponse(
            status="OK",
            service="stage4-slm-decision-support",
            model_version=self.model_version,
            device=self.engine.device,
            uptime_seconds=round(uptime, 2),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

    def process_decision_support(
        self,
        document_id: str,
        patient_id: str,
        clinical_note: str
    ) -> DecisionSupportResponse:
        t0 = time.perf_counter()
        try:
            triad = self.engine.generate_triad(clinical_note)
            passed, warnings, preserved = self.guardrails.audit(clinical_note, triad)

            if passed:
                routing = "STAGE_4_AUTO"
                final_triad = triad
            else:
                routing = "FALLBACK_STAGE3_BASELINE"
                final_triad = self.fallback.generate_fallback_triad(
                    clinical_note,
                    reason="; ".join(warnings)
                )

        except Exception as e:
            logger.error(f"Inference failure: {e}", exc_info=True)
            routing = "FALLBACK_STAGE3_BASELINE"
            warnings = [f"Exception during SLM generation: {str(e)}"]
            final_triad = self.fallback.generate_fallback_triad(clinical_note, reason=str(e))
            passed = False
            preserved = {"drugs": [], "dosages": []}

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return DecisionSupportResponse(
            document_id=document_id,
            patient_id=patient_id,
            target_risk=final_triad["target_risk"],
            target_key_finding=final_triad["target_key_finding"],
            target_action=final_triad["target_action"],
            preserved_entities=preserved,
            safety_gate_passed=passed,
            routing=routing,
            guardrail_warnings=warnings,
            latency_ms=round(latency_ms, 2)
        )

    def process_risk(self, patient_id: str, clinical_note: str) -> RiskResponse:
        t0 = time.perf_counter()
        el = self.engine.extract_key_elements(clinical_note)
        triad = self.engine.generate_triad(clinical_note)
        latency = (time.perf_counter() - t0) * 1000.0

        tier = "CRITICAL" if el["is_critical"] else ("HIGH" if el["is_high"] else "LOW")
        return RiskResponse(
            patient_id=patient_id,
            target_risk=triad["target_risk"],
            hazard_category=el["hazard_type"],
            urgency_tier=tier,
            latency_ms=round(latency, 2)
        )

    def process_action(self, patient_id: str, clinical_note: str) -> ActionResponse:
        t0 = time.perf_counter()
        el = self.engine.extract_key_elements(clinical_note)
        triad = self.engine.generate_triad(clinical_note)
        latency = (time.perf_counter() - t0) * 1000.0

        mod = "HOLD_IMMEDIATELY" if el["is_critical"] else ("DOSE_REDUCE" if el["is_high"] else "MAINTAIN_CURRENT")
        tier = "CRITICAL" if el["is_critical"] else ("HIGH" if el["is_high"] else "LOW")

        return ActionResponse(
            patient_id=patient_id,
            target_action=triad["target_action"],
            dose_modification=mod,
            urgency_tier=tier,
            latency_ms=round(latency, 2)
        )

    def process_batch(self, notes: List[Any]) -> BatchDecisionSupportResponse:
        t0 = time.perf_counter()
        results = []
        for n in notes:
            res = self.process_decision_support(
                document_id=getattr(n, "document_id", "DOC-UNKNOWN"),
                patient_id=getattr(n, "patient_id", "PT-UNKNOWN"),
                clinical_note=getattr(n, "clinical_note", "")
            )
            results.append(res)
        batch_latency = (time.perf_counter() - t0) * 1000.0
        return BatchDecisionSupportResponse(
            total_processed=len(results),
            successful_count=len(results),
            results=results,
            batch_latency_ms=round(batch_latency, 2)
        )
