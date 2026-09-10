"""
Prompt Builder Module for Stage 4 Integration.
Formats clinical notes into canonical prompt pairs per Stage 5 template v1.0.0.
"""

from typing import Dict, Any

PROMPT_TEMPLATE_NAME = "clinical_decision_support_v1"
PROMPT_TEMPLATE_VERSION = "1.0.0"

SYSTEM_INSTRUCTION = (
    "You are a clinical decision-support summarization assistant.\n"
    "Read the clinical note and produce a structured response.\n\n"
    "Return exactly:\n\n"
    "Risk: <Low | Moderate | High>\n\n"
    "Key Finding: <concise clinically relevant finding>\n\n"
    "Action: <appropriate action/monitoring recommendation>"
)


class PromptBuilder:
    """Builds and sanitizes clinical decision-support prompts."""

    def __init__(self, max_length_chars: int = 10000):
        self.max_length_chars = max_length_chars
        self.template_name = PROMPT_TEMPLATE_NAME
        self.template_version = PROMPT_TEMPLATE_VERSION

    def build_prompt(self, clinical_note: str) -> str:
        """Sanitizes note and wraps in canonical instruction prompt."""
        if not clinical_note or not str(clinical_note).strip():
            raise ValueError("Clinical note cannot be empty.")

        clean_note = str(clinical_note).strip()
        if len(clean_note) > self.max_length_chars:
            clean_note = clean_note[: self.max_length_chars]

        return f"{SYSTEM_INSTRUCTION}\n\nClinical Note:\n{clean_note}\n\nResponse:\n"
