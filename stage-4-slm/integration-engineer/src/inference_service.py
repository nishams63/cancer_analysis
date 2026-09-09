"""
Inference Service Orchestrator for Stage 4 Integration.
Connects prompt building, local GGUF inference, output parsing,
safety firewall enforcement, provenance tracking, and immutable audit logging.
"""

import time
import uuid
from typing import Dict, Any, Optional

from config import AppConfig
from model_manager import ModelManager
from prompt_builder import PromptBuilder
from output_parser import OutputParser
from safety_gateway import SafetyGateway
from provenance import ProvenanceTracker
from audit_logger import AuditLogger


class InferenceService:
    """End-to-end clinical inference pipeline for offline decision support."""

    def __init__(
        self,
        config: AppConfig,
        model_manager: ModelManager,
        safety_gateway: SafetyGateway,
        audit_logger: AuditLogger
    ):
        self.config = config
        self.model_manager = model_manager
        self.safety_gateway = safety_gateway
        self.audit_logger = audit_logger
        self.prompt_builder = PromptBuilder(max_length_chars=config.service.max_note_length_chars)
        self.output_parser = OutputParser()

    def process_note(
        self,
        clinical_note: str,
        reference_entities: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes complete inference workflow:
        Input -> Prompt -> GGUF Runtime -> Parser -> Safety Firewall -> Provenance -> Audit
        """
        start_time = time.time()
        inference_id = f"inf-{uuid.uuid4().hex[:12]}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # 1. Build prompt
        prompt = self.prompt_builder.build_prompt(clinical_note)

        # 2. Local offline inference
        raw_output, confidence, gen_latency_ms = self.model_manager.generate(
            prompt=prompt,
            clinical_note=clinical_note,
            max_tokens=self.config.model.max_tokens,
            temperature=self.config.model.temperature
        )

        # 3. Output parsing
        parsed_fields, spoken_summary = self.output_parser.parse_generation(raw_output)

        # 4. Safety Gateway (6 gates + calibrated threshold)
        safety_verdict = self.safety_gateway.evaluate(
            clinical_note=clinical_note,
            raw_output=raw_output,
            confidence=confidence,
            reference_entities=reference_entities
        )

        total_latency_ms = (time.time() - start_time) * 1000.0

        # 5. Provenance tracking
        provenance = ProvenanceTracker.build_provenance(
            inference_id=inference_id,
            clinical_note=clinical_note,
            prompt=prompt,
            raw_output=raw_output,
            model_name=self.config.model.name,
            adapter_version="best_model_adapter-r16-alpha32",
            quantization=self.config.model.quantization,
            prompt_version=self.prompt_builder.template_version,
            tokenizer_version="Qwen2TokenizerFast",
            runtime_version="llama.cpp-0.3.35"
        )

        # 6. Immutable audit log
        self.audit_logger.log_inference(
            inference_id=inference_id,
            timestamp=timestamp,
            provenance=provenance,
            confidence=confidence,
            latency_ms=total_latency_ms,
            firewall_verdict=safety_verdict,
            review_required=safety_verdict["review_required"]
        )

        # 7. Assemble standardized clinical response contract
        response = {
            "inference_id": inference_id,
            "timestamp": timestamp,
            "risk": parsed_fields.get("Risk", "Unknown"),
            "key_finding": parsed_fields.get("Key Finding", ""),
            "action": parsed_fields.get("Action", ""),
            "confidence": round(confidence, 4),
            "safety_status": safety_verdict["safety_status"],
            "review_required": safety_verdict["review_required"],
            "model_version": f"{self.config.model.name}-{self.config.model.quantization}",
            "latency_ms": round(total_latency_ms, 2),
            "spoken_summary": spoken_summary,
            "provenance": {
                "input_hash": provenance["input_hash"][:16] + "...",
                "output_hash": provenance["output_hash"][:16] + "...",
                "quantization": provenance["quantization"]
            }
        }

        if safety_verdict["review_required"]:
            response["failure_reason"] = safety_verdict.get("failure_reason", "Safety gate failure")
            response["failed_checks"] = safety_verdict.get("failed_checks", [])
            response["review_status"] = "PENDING_CLINICIAN_REVIEW"

        return response
