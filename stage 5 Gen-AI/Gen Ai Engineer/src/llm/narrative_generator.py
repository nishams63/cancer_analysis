"""Master Narrative Generator with validation feedback and regeneration loop."""
from typing import Dict, Any, Tuple
from .prompt_builder import PromptBuilder
from .response_parser import ResponseParser
from .nvidia_client import get_llm_client, LLMClient
from src.validation.narrative_validator import NarrativeValidator, NarrativeValidationResult


class NarrativeGenerator:
    def __init__(self, llm_client: LLMClient = None, max_attempts: int = 3):
        self.llm_client = llm_client or get_llm_client()
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()
        self.validator = NarrativeValidator()
        self.max_attempts = max_attempts

    def generate_narrative(self, patient: Dict[str, Any], scenario: Dict[str, Any],
                           evidence_context: str) -> Tuple[bool, str, NarrativeValidationResult, Dict[str, Any]]:
        """Generate and iteratively validate clinical narrative until PASS or retry exhaustion."""
        attempts = 0
        last_narrative = ""
        last_val_result = None
        feedback = None

        log_history = []

        while attempts < self.max_attempts:
            attempts += 1
            messages = self.prompt_builder.build_messages(
                patient=patient,
                scenario=scenario,
                evidence_context=evidence_context,
                correction_feedback=feedback
            )

            response = self.llm_client.generate(messages)
            parsed = self.response_parser.parse_narrative(response.content)
            narrative = parsed.get("full_narrative", response.content)
            last_narrative = narrative

            # Validate narrative against ground truth patient
            val_result = self.validator.validate(narrative, patient, scenario)
            last_val_result = val_result

            log_history.append({
                "attempt": attempts,
                "latency_ms": response.latency_ms,
                "tokens": response.total_tokens,
                "valid": val_result.valid,
                "violations": val_result.violations
            })

            if val_result.valid:
                meta = {
                    "attempts": attempts,
                    "model": response.model,
                    "provider": response.provider,
                    "latency_ms": response.latency_ms,
                    "log_history": log_history
                }
                return True, narrative, val_result, meta

            # Prepare correction feedback for next attempt
            feedback = "; ".join([f"{v['field']}: {v['reason']}" for v in val_result.violations])

        meta = {
            "attempts": attempts,
            "status": "FAILED_NARRATIVE_VALIDATION",
            "log_history": log_history
        }
        return False, last_narrative, last_val_result, meta