"""Prompt Builder constructing strict system instruction and user scenario context."""
import json
from typing import Dict, Any, List


class PromptBuilder:
    SYSTEM_INSTRUCTION = """You are the controlled clinical narrative realization component of an oncology AI stress-testing system.

IMPORTANT:
The supplied structured synthetic patient profile is the source of truth.
The patient is entirely synthetic and is being generated only for AI evaluation and stress testing.
The retrieved oncology evidence is supporting context only.
You MUST NOT alter, remove, contradict or independently decide any core patient facts.

Preserve exactly the supplied:
- age
- sex
- mutations
- biomarkers
- treatment
- dosage
- adverse events
- resistance status
- temporal relationships
- missing information
- required scenario conditions

Do not fill deliberately missing information.
Do not introduce unsupported mutations.
Do not introduce unsupported biomarkers.
Do not introduce unsupported treatments.
Do not introduce unsupported dosage values.
Do not change negative findings into positive findings.
Do not change positive findings into negative findings.
Do not make treatment recommendations.
Do not diagnose a real individual.
Do not imply this is a real patient.
Use retrieved evidence only for terminology, contextual grounding and realistic clinical phrasing.

Generate a realistic synthetic oncology clinical narrative that faithfully represents the structured profile."""

    def build_messages(self, patient: Dict[str, Any], scenario: Dict[str, Any],
                       evidence_context: str, correction_feedback: str = None) -> List[Dict[str, str]]:
        demo = patient.get("demographics", {})
        muts = ", ".join(patient.get("mutations", []))
        treatments = ", ".join([f"{t.get('treatment_name')} (Line {t.get('line', 1)})" for t in patient.get("treatments", [])])
        dosages = ", ".join([f"{k}: {v} mg" for k, v in patient.get("dosages", {}).items()])
        missing = ", ".join(patient.get("missing_fields", [])) or "None"

        user_content_parts = [
            f"SCENARIO TITLE: {scenario.get('title', 'Oncology Stress Test')}",
            f"SCENARIO ID: {scenario.get('scenario_id', 'SYN-S001')}",
            "\n=== STRUCTURED SYNTHETIC PATIENT GROUND TRUTH ===",
            f"- Age: {demo.get('age')}",
            f"- Sex: {demo.get('sex')}",
            f"- Primary Diagnosis: {demo.get('cancer_type')} ({demo.get('cancer_stage')})",
            f"- Confirmed Mutations: {muts}",
            f"- Biomarkers & Labs: {json.dumps(patient.get('biomarkers', {}))}",
            f"- Current Treatment: {treatments}",
            f"- Prescribed Dosages: {dosages}",
            f"- Resistance Status: {patient.get('resistance', {}).get('status', False)} ({patient.get('resistance', {}).get('mechanism', 'None')})",
            f"- Intentionally Missing Clinical Fields: {missing}",
            "\n=== REQUIRED SCENARIO CONDITIONS ===",
            f"{json.dumps(scenario.get('required_entities', []), indent=2)}",
            "\n=== FORBIDDEN MODIFICATIONS ===",
            f"{json.dumps(scenario.get('forbidden_modifications', []), indent=2)}",
            "\n=== RETRIEVED ONCOLOGY EVIDENCE (SUPPORTING CONTEXT ONLY) ===",
            f"{evidence_context}"
        ]

        if correction_feedback:
            user_content_parts.append(
                f"\n!!! CORRECTION INSTRUCTION FROM PREVIOUS ATTEMPT !!!\n"
                f"The previous draft was REJECTED due to the following constraint violations:\n"
                f"{correction_feedback}\n"
                f"You MUST strictly fix these errors in this generation while preserving all facts exactly."
            )

        user_content_parts.append(
            "\nTASK: Generate a complete outpatient oncology clinical encounter progress note faithfully realizing the above ground truth facts. Format with Subjective, Objective, Assessment, and Plan."
        )

        return [
            {"role": "system", "content": self.SYSTEM_INSTRUCTION},
            {"role": "user", "content": "\n".join(user_content_parts)}
        ]