"""
Instruction-Tuning Formatter Module for Stage 4.
Formats validated PASS records into canonical SLM instruction-tuning schema
per Section 8 & Section 11.
"""

import logging
from typing import Dict, List, Any
import pandas as pd

logger = logging.getLogger("stage4.formatter")

DEFAULT_INSTRUCTION = "Analyze the clinical note and provide the patient's Risk, Key Finding, and Action."


class InstructionTuningFormatter:
    """Formats clinical records into structured SLM fine-tuning dataset."""

    def __init__(self, instruction: str = DEFAULT_INSTRUCTION):
        self.instruction = instruction

    def format_dataset(self, df_pass: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
        """
        Transforms validated PASS records into the standardized Stage 4 schema.
        """
        logger.info("Formatting %d PASS records for SLM fine-tuning...", len(df_pass))
        formatted_rows = []

        for idx, row in df_pass.iterrows():
            pat_id = str(row.get("patient_id", f"PT-{idx}"))
            note_id = str(row.get("document_id", row.get("note_id", f"DOC-{idx}")))
            text = str(row.get("text", row.get("clinical_note", "")))
            t_risk = str(row.get("target_risk", ""))
            t_kf = str(row.get("target_key_finding", ""))
            t_act = str(row.get("target_action", ""))
            
            genes = list(row.get("ner_genes", []))
            drugs = list(row.get("ner_drugs", []))
            dosages = list(row.get("ner_dosages", []))
            aes = list(row.get("ner_adverse_events", []))

            # Consolidate missing/invented entities into clean lists or strings
            missing_ents = []
            for m in [row.get("missing_genes", []), row.get("missing_drugs", []), row.get("missing_dosages", []), row.get("missing_adverse_events", [])]:
                if isinstance(m, list):
                    missing_ents.extend(m)
            
            invented = list(row.get("invented_entities", []))

            split_val = str(row.get(split_col, row.get("data_split", "TRAIN")))

            record = {
                "patient_id": pat_id,
                "note_id": note_id,
                "clinical_note": text,
                "instruction": self.instruction,
                "target_risk": t_risk,
                "target_key_finding": t_kf,
                "target_action": t_act,
                "ner_genes": genes,
                "ner_drugs": drugs,
                "ner_dosages": dosages,
                "ner_adverse_events": aes,
                "entity_check_status": str(row.get("entity_check_status", "PASS")),
                "entity_coverage": float(row.get("entity_coverage", 1.0)),
                "missing_entities": missing_ents,
                "invented_entities": invented,
                "generation_model_version": str(row.get("generation_model_version", "clinical-draft-target-generator-v1.0.0")),
                "split": split_val
            }
            formatted_rows.append(record)

        df_formatted = pd.DataFrame(formatted_rows)
        logger.info("Dataset formatting complete. Columns: %s", list(df_formatted.columns))
        return df_formatted
