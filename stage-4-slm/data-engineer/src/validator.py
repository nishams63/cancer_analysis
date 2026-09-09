"""
Data Validation Module for Stage 4 Clinical NLP Pipeline.
Validates clinical notes, identifiers, NER integrity, and entity shapes
prior to target generation.
"""

import logging
from typing import Dict, List, Any, Tuple
import pandas as pd

logger = logging.getLogger("stage4.validator")


class DataValidationError(Exception):
    """Raised when critical clinical data validation checks fail."""
    pass


class Stage4DataValidator:
    """Pre-generation input validator for clinical notes and NER structures."""

    def __init__(self, min_char_length: int = 20):
        self.min_char_length = min_char_length

    def validate_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Validates notes and NER records.
        Returns:
            clean_df: Validated records ready for generation
            invalid_df: Records failing validation
            audit_report: Detailed dictionary of checks and failure counts
        """
        audit = {
            "total_evaluated": len(df),
            "clean_count": 0,
            "invalid_count": 0,
            "missing_note_count": 0,
            "empty_note_count": 0,
            "duplicate_note_text_count": 0,
            "missing_patient_id_count": 0,
            "missing_note_id_count": 0,
            "invalid_ner_record_count": 0,
            "duplicate_ner_entity_count": 0,
            "malformed_entity_value_count": 0,
            "failure_reasons": {}
        }

        if df.empty:
            return df.copy(), pd.DataFrame(), audit

        invalid_indices = []
        failure_reasons = {}

        # 1. Identifier checks
        for idx, row in df.iterrows():
            reasons = []

            # Check note ID
            doc_id = str(row.get("document_id", row.get("note_id", ""))).strip()
            if not doc_id or doc_id.lower() in ("none", "nan", "null"):
                audit["missing_note_id_count"] += 1
                reasons.append("Missing/Empty note ID")

            # Check patient ID
            pat_id = str(row.get("patient_id", "")).strip()
            if not pat_id or pat_id.lower() in ("none", "nan", "null"):
                audit["missing_patient_id_count"] += 1
                reasons.append("Missing/Empty patient ID")

            # Check clinical text
            text = row.get("text", row.get("clinical_note", ""))
            if text is None or pd.isna(text):
                audit["missing_note_count"] += 1
                reasons.append("Null clinical note")
            else:
                text_str = str(text).strip()
                if len(text_str) < self.min_char_length:
                    audit["empty_note_count"] += 1
                    reasons.append(f"Empty or overly brief clinical note (< {self.min_char_length} chars)")

            # Check NER structure
            ner_raw = row.get("ner_entities_parsed", row.get("ner_entities", []))
            if isinstance(ner_raw, str):
                import json
                try:
                    ner_list = json.loads(ner_raw)
                except Exception:
                    audit["invalid_ner_record_count"] += 1
                    reasons.append("Malformed JSON in NER entities")
                    ner_list = []
            elif isinstance(ner_raw, list):
                ner_list = ner_raw
            else:
                audit["invalid_ner_record_count"] += 1
                reasons.append("Non-list NER entities structure")
                ner_list = []

            # Check entity validity and duplicates
            seen_entities = set()
            has_malformed = False
            has_duplicate = False

            for ent in ner_list:
                if not isinstance(ent, dict):
                    has_malformed = True
                    continue
                txt = str(ent.get("text", "")).strip()
                lbl = str(ent.get("label", "")).strip()
                if not txt or not lbl:
                    has_malformed = True
                
                key = (lbl.upper(), txt.lower())
                if key in seen_entities:
                    has_duplicate = True
                seen_entities.add(key)

            if has_malformed:
                audit["malformed_entity_value_count"] += 1
                reasons.append("Malformed entity values in NER record")
            if has_duplicate:
                audit["duplicate_ner_entity_count"] += 1
                reasons.append("Duplicate NER entity mentions in record")

            if reasons:
                invalid_indices.append(idx)
                failure_reasons[idx] = "; ".join(reasons)

        # Check for duplicate note text across dataset
        text_col = "text" if "text" in df.columns else "clinical_note"
        text_dups = df.duplicated(subset=[text_col], keep="first")
        audit["duplicate_note_text_count"] = int(text_dups.sum())

        for idx in df[text_dups].index:
            if idx not in failure_reasons:
                invalid_indices.append(idx)
                failure_reasons[idx] = "Duplicate note text"
            else:
                failure_reasons[idx] += "; Duplicate note text"

        invalid_mask = df.index.isin(invalid_indices)
        clean_df = df[~invalid_mask].copy()
        invalid_df = df[invalid_mask].copy()

        if not invalid_df.empty:
            invalid_df["validation_failure_reason"] = [failure_reasons.get(i, "Validation error") for i in invalid_df.index]

        audit["clean_count"] = len(clean_df)
        audit["invalid_count"] = len(invalid_df)
        audit["failure_reasons"] = {str(k): v for k, v in list(failure_reasons.items())[:50]}

        logger.info(
            "Validation complete: Total=%d, Clean=%d, Invalid=%d, DupText=%d",
            audit["total_evaluated"],
            audit["clean_count"],
            audit["invalid_count"],
            audit["duplicate_note_text_count"]
        )

        return clean_df, invalid_df, audit
