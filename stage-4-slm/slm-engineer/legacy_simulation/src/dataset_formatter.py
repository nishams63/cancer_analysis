"""
Dataset Formatter Module for Stage 5 SLM.
Encodes clinical note and target pairs into formatted prompt-completion examples
with input-masking support for causal language model fine-tuning.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from prompt_template import ClinicalPromptTemplate


class ClinicalDatasetFormatter:
    """Formats clinical datasets for instruction fine-tuning and evaluation."""

    def __init__(self, prompt_template: Optional[ClinicalPromptTemplate] = None):
        self.prompt_template = prompt_template or ClinicalPromptTemplate()

    def format_dataframe(self, df: pd.DataFrame, is_raw: bool = False) -> List[Dict[str, Any]]:
        """
        Transforms a DataFrame into a list of prompt-completion dictionaries.
        """
        formatted_records = []
        for idx, row in df.iterrows():
            note = str(row.get("clinical_note", ""))
            prompt = self.prompt_template.format_input_prompt(note)

            if is_raw:
                # Use raw generator output
                target = str(row.get("raw_target", row.get("raw_summary", ""))).strip()
            else:
                target_risk = str(row.get("target_risk", ""))
                target_kf = str(row.get("target_key_finding", ""))
                target_act = str(row.get("target_action", ""))
                target = self.prompt_template.format_target_output(target_risk, target_kf, target_act)

            full_text = f"{prompt}{target}"

            formatted_records.append({
                "patient_id": row.get("patient_id", ""),
                "note_id": row.get("note_id", row.get("document_id", f"row_{idx}")),
                "prompt": prompt,
                "target": target,
                "full_text": full_text,
                "expected_risk": self.prompt_template._extract_risk_tier(str(row.get("target_risk", ""))),
                "ner_genes": row.get("ner_genes", []),
                "ner_drugs": row.get("ner_drugs", []),
                "ner_dosages": row.get("ner_dosages", []),
                "ner_adverse_events": row.get("ner_adverse_events", [])
            })

        return formatted_records
