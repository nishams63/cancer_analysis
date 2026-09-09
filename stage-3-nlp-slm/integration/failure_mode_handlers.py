"""
Failure Mode Handling and Safe Degradation Module for Stage 3 -> Stage 4 Integration.
Explicitly intercepts and handles:
1. Stage 3 Timeout (Immediate escalation to human review, zero retries)
2. Malformed Input Document (Empty, null, binary, non-string)
3. Empty Clinical Entities Extraction (Clinically suspicious in oncology)
4. Model Version Mismatch (Incompatible major/minor version between stages)
5. Partial Pipeline Failure (Triage succeeds but NER fails, or vice versa)

Guaranteed Safety Invariant: Every failure degrades safely to HUMAN_REVIEW.
Never defaults to a silent "no findings" or auto-approved state.
"""

from typing import Dict, Any, List, Optional, Literal, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import time

from contract_validation import (
    Stage3OutputPayload,
    TriageUrgencyPayload,
    ToxicityHazardPayload,
    ClinicalEntityPayload,
    ContractValidationError,
    validate_stage3_output,
    SUPPORTED_SCHEMA_VERSION,
)
from confidence_gate import ConfidenceGate, RoutingResult

logger = logging.getLogger("stage3_stage4.failure_handlers")

DEFAULT_TIMEOUT_SECONDS = 0.100  # 100ms hard deadline (standard inference is ~18.2ms)
EXPECTED_MODEL_VERSION_PREFIX = "stage3-"


class Stage3TimeoutError(TimeoutError):
    """Raised when Stage 3 NLP execution exceeds the allotted latency budget."""
    pass


class MalformedDocumentError(ValueError):
    """Raised when an input clinical note fails basic structural or semantic sanity checks."""
    pass


class ModelVersionMismatchError(RuntimeError):
    """Raised when the Stage 3 artifact version is incompatible with Stage 4 expectations."""
    pass


class PartialPipelineError(RuntimeError):
    """Raised when one pipeline sub-component fails while another succeeds."""
    pass


@dataclass
class DegradedEnvelope:
    """
    Standardized wrapper returned when the pipeline encounters an error or partial failure.
    Always routes to HUMAN_REVIEW with complete audit diagnostic context.
    """
    status: Literal[
        "SUCCESS",
        "TIMEOUT",
        "MALFORMED_INPUT",
        "EMPTY_ENTITIES",
        "VERSION_MISMATCH",
        "PARTIAL_FAILURE",
        "INTERNAL_ERROR"
    ]
    destination: Literal["STAGE_4_AUTOMATED", "HUMAN_REVIEW"]
    document_id: str
    patient_id: str
    degradation_reason: str
    error_details: Optional[str] = None
    payload: Optional[Stage3OutputPayload] = None
    routing_result: Optional[RoutingResult] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_degraded(self) -> bool:
        return self.status != "SUCCESS" or self.destination == "HUMAN_REVIEW"


class IntegrationFailureHandler:
    """
    Coordinates safe degradation and validation across all failure modes.
    """

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        expected_model_prefix: str = EXPECTED_MODEL_VERSION_PREFIX,
        confidence_gate: Optional[ConfidenceGate] = None,
        require_non_empty_entities: bool = True
    ):
        self.timeout_seconds = timeout_seconds
        self.expected_model_prefix = expected_model_prefix
        self.confidence_gate = confidence_gate or ConfidenceGate()
        self.require_non_empty_entities = require_non_empty_entities

    def sanitize_input_document(self, text: Any, document_id: str, patient_id: str) -> str:
        """
        Validates raw input text prior to NLP inference.
        Raises MalformedDocumentError on empty, non-string, binary, or corrupted input.
        """
        if text is None:
            raise MalformedDocumentError("Input document text is None.")
        if not isinstance(text, str):
            raise MalformedDocumentError(f"Input document must be string, got {type(text).__name__}.")
        cleaned = text.strip()
        if not cleaned:
            raise MalformedDocumentError("Input document text is empty or whitespace-only.")
        if len(cleaned) < 5:
            raise MalformedDocumentError(f"Input text length ({len(cleaned)}) is below clinical plausibility floor.")
        if "\x00" in text:
            raise MalformedDocumentError("Input document contains binary null bytes.")
        return cleaned

    def verify_version_compatibility(self, model_version: str, ner_model_version: str) -> None:
        """
        Verifies that model version strings conform to the expected active production series.
        """
        if not model_version.startswith(self.expected_model_prefix):
            raise ModelVersionMismatchError(
                f"Incompatible triage/hazard model version '{model_version}'. Expected prefix '{self.expected_model_prefix}'."
            )
        if not ner_model_version.startswith(self.expected_model_prefix):
            raise ModelVersionMismatchError(
                f"Incompatible NER model version '{ner_model_version}'. Expected prefix '{self.expected_model_prefix}'."
            )

    def execute_with_timeout_protection(
        self,
        inference_func: Callable[[], Dict[str, Any]],
        document_id: str
    ) -> Dict[str, Any]:
        """
        Executes inference function with strict latency budget enforcement.
        On timeout: Escalates immediately to HUMAN_REVIEW (zero automatic retries).
        """
        t0 = time.perf_counter()
        result = inference_func()
        elapsed = time.perf_counter() - t0

        if elapsed > self.timeout_seconds:
            msg = (
                f"Stage 3 inference deadline exceeded: {elapsed*1000:.1f}ms > {self.timeout_seconds*1000:.1f}ms budget "
                f"for document {document_id}. Immediate human review escalation triggered (zero retries)."
            )
            logger.error(msg)
            raise Stage3TimeoutError(msg)
        return result

    def handle_safely(
        self,
        document_id: str,
        patient_id: str,
        execution_closure: Callable[[], Dict[str, Any]]
    ) -> DegradedEnvelope:
        """
        Executes Stage 3 extraction inside a defensive safety boundary.
        Catches and safely maps all failure modes into a DegradedEnvelope routed to HUMAN_REVIEW.
        """
        try:
            # 1. Execute with timeout protection
            raw_output = self.execute_with_timeout_protection(execution_closure, document_id)

            # 2. Check for partial pipeline failures
            if "triage_urgency" not in raw_output and "clinical_entities" in raw_output:
                raise PartialPipelineError("Partial failure: Triage urgency missing while entities extracted.")
            if "clinical_entities" not in raw_output and "triage_urgency" in raw_output:
                raise PartialPipelineError("Partial failure: Clinical entities missing while triage urgency extracted.")

            # 3. Check version compatibility
            self.verify_version_compatibility(
                raw_output.get("model_version", ""),
                raw_output.get("ner_model_version", "")
            )

            # 4. Validate output schema
            validated_payload = validate_stage3_output(raw_output)

            # 5. Check empty clinical entities anomaly
            if self.require_non_empty_entities and len(validated_payload.clinical_entities) == 0:
                reason = (
                    f"Empty clinical entities extracted: Note contains zero oncology drugs, mutations, "
                    f"dosages, or adverse events. Requires clinical oncologist verification."
                )
                logger.warning(f"Document {document_id}: {reason}")
                return DegradedEnvelope(
                    status="EMPTY_ENTITIES",
                    destination="HUMAN_REVIEW",
                    document_id=document_id,
                    patient_id=patient_id,
                    degradation_reason=reason,
                    payload=validated_payload
                )

            # 6. Evaluate confidence gate
            routing = self.confidence_gate.evaluate(validated_payload)

            return DegradedEnvelope(
                status="SUCCESS",
                destination=routing.destination,
                document_id=document_id,
                patient_id=patient_id,
                degradation_reason="; ".join(routing.routing_reasons),
                payload=validated_payload,
                routing_result=routing
            )

        except Stage3TimeoutError as e:
            return DegradedEnvelope(
                status="TIMEOUT",
                destination="HUMAN_REVIEW",
                document_id=document_id,
                patient_id=patient_id,
                degradation_reason="Inference latency timeout exceeded budget. Escalated immediately without retry.",
                error_details=str(e)
            )

        except MalformedDocumentError as e:
            return DegradedEnvelope(
                status="MALFORMED_INPUT",
                destination="HUMAN_REVIEW",
                document_id=document_id,
                patient_id=patient_id,
                degradation_reason="Input clinical document failed sanitization checks.",
                error_details=str(e)
            )

        except ModelVersionMismatchError as e:
            return DegradedEnvelope(
                status="VERSION_MISMATCH",
                destination="HUMAN_REVIEW",
                document_id=document_id,
                patient_id=patient_id,
                degradation_reason="Stage 3 model version mismatch with Stage 4 expectations.",
                error_details=str(e)
            )

        except PartialPipelineError as e:
            return DegradedEnvelope(
                status="PARTIAL_FAILURE",
                destination="HUMAN_REVIEW",
                document_id=document_id,
                patient_id=patient_id,
                degradation_reason="Partial pipeline failure detected between classifier and NER.",
                error_details=str(e)
            )

        except ContractValidationError as e:
            return DegradedEnvelope(
                status="MALFORMED_INPUT",
                destination="HUMAN_REVIEW",
                document_id=document_id,
                patient_id=patient_id,
                degradation_reason="Stage 3 output failed contract schema validation.",
                error_details=str(e)
            )

        except Exception as e:
            logger.critical(f"Unhandled exception during Stage 3 -> Stage 4 handoff: {e}", exc_info=True)
            return DegradedEnvelope(
                status="INTERNAL_ERROR",
                destination="HUMAN_REVIEW",
                document_id=document_id,
                patient_id=patient_id,
                degradation_reason="Internal system error encountered during pipeline execution.",
                error_details=str(e)
            )
