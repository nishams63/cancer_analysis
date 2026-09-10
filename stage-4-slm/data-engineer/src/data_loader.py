"""
Robust Data Loader and Joiner for Stage 4 SLM Fine-Tuning Pipeline.
Loads Stage 3 clinical notes and verified NER outputs, joins on stable IDs,
and validates alignment without silently discarding any records.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd

logger = logging.getLogger("stage4.data_loader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class Stage4DataLoader:
    """Loads and joins Stage 3 clinical notes with verified NER outputs."""

    def __init__(self, stage3_parquet_path: str, ner_predictions_path: Optional[str] = None):
        self.stage3_parquet_path = Path(stage3_parquet_path)
        self.ner_predictions_path = Path(ner_predictions_path) if ner_predictions_path else None
        self.load_audit: Dict[str, Any] = {
            "input_records_count": 0,
            "joined_records_count": 0,
            "unmatched_records_count": 0,
            "missing_patient_ids_count": 0,
            "missing_note_ids_count": 0,
            "duplicate_records_count": 0,
            "discarded_records": []
        }

    @staticmethod
    def parse_ner_entities(entities_raw: Any) -> Tuple[List[str], List[str], List[str], List[str], List[Dict[str, Any]]]:
        """
        Parses NER entities into typed categories:
        - GENE_MUTATION -> Genes
        - DRUG_NAME -> Drugs
        - DOSAGE -> Dosages
        - ADVERSE_EVENT -> Adverse Events
        """
        genes: List[str] = []
        drugs: List[str] = []
        dosages: List[str] = []
        adverse_events: List[str] = []
        raw_list: List[Dict[str, Any]] = []

        if not entities_raw:
            return genes, drugs, dosages, adverse_events, raw_list

        if isinstance(entities_raw, str):
            try:
                raw_list = json.loads(entities_raw)
            except Exception as e:
                logger.warning("Failed to parse JSON string for ner_entities: %s", e)
                return genes, drugs, dosages, adverse_events, raw_list
        elif isinstance(entities_raw, list):
            raw_list = entities_raw
        else:
            return genes, drugs, dosages, adverse_events, raw_list

        for ent in raw_list:
            if not isinstance(ent, dict):
                continue
            lbl = ent.get("label", "").upper().strip()
            txt = ent.get("text", "").strip()
            if not txt:
                continue

            if lbl in ("GENE_MUTATION", "GENE", "MUTATION"):
                if txt not in genes:
                    genes.append(txt)
            elif lbl in ("DRUG_NAME", "DRUG"):
                if txt not in drugs:
                    drugs.append(txt)
            elif lbl in ("DOSAGE", "DOSE"):
                if txt not in dosages:
                    dosages.append(txt)
            elif lbl in ("ADVERSE_EVENT", "AE", "TOXICITY"):
                if txt not in adverse_events:
                    adverse_events.append(txt)

        return genes, drugs, dosages, adverse_events, raw_list

    def load_and_join(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads clinical notes from Stage 3 Parquet, joins with NER entities,
        and logs audit metrics.
        """
        if not self.stage3_parquet_path.exists():
            raise FileNotFoundError(f"Stage 3 Parquet dataset not found at: {self.stage3_parquet_path}")

        logger.info("Loading Stage 3 clinical notes from: %s", self.stage3_parquet_path)
        df_notes = pd.read_parquet(self.stage3_parquet_path).copy(deep=True)
        self.load_audit["input_records_count"] = len(df_notes)

        # Validate mandatory identifiers
        if "document_id" not in df_notes.columns and "note_id" in df_notes.columns:
            df_notes["document_id"] = df_notes["note_id"]
        if "note_id" not in df_notes.columns and "document_id" in df_notes.columns:
            df_notes["note_id"] = df_notes["document_id"]

        # Check for missing identifiers
        missing_notes = df_notes["document_id"].isnull() | (df_notes["document_id"].astype(str).str.strip() == "")
        missing_patients = df_notes["patient_id"].isnull() | (df_notes["patient_id"].astype(str).str.strip() == "")

        self.load_audit["missing_note_ids_count"] = int(missing_notes.sum())
        self.load_audit["missing_patient_ids_count"] = int(missing_patients.sum())

        # Check for duplicate document IDs
        dup_docs = df_notes.duplicated(subset=["document_id"], keep="first")
        self.load_audit["duplicate_records_count"] = int(dup_docs.sum())

        valid_mask = ~missing_notes & ~missing_patients & ~dup_docs
        df_valid = df_notes[valid_mask].copy()

        # Log discarded rows
        discarded_df = df_notes[~valid_mask]
        for _, row in discarded_df.iterrows():
            reason = []
            if missing_notes.loc[row.name]:
                reason.append("Missing Note ID")
            if missing_patients.loc[row.name]:
                reason.append("Missing Patient ID")
            if dup_docs.loc[row.name]:
                reason.append("Duplicate Note ID")
            self.load_audit["discarded_records"].append({
                "index": int(row.name),
                "document_id": str(row.get("document_id", "")),
                "patient_id": str(row.get("patient_id", "")),
                "reason": "; ".join(reason)
            })

        # Process NER entities
        parsed_genes: List[List[str]] = []
        parsed_drugs: List[List[str]] = []
        parsed_dosages: List[List[str]] = []
        parsed_aes: List[List[str]] = []
        parsed_raw: List[List[Dict[str, Any]]] = []

        for _, row in df_valid.iterrows():
            g, d, dose, ae, r = self.parse_ner_entities(row.get("ner_entities"))
            parsed_genes.append(g)
            parsed_drugs.append(d)
            parsed_dosages.append(dose)
            parsed_aes.append(ae)
            parsed_raw.append(r)

        df_valid["ner_genes"] = parsed_genes
        df_valid["ner_drugs"] = parsed_drugs
        df_valid["ner_dosages"] = parsed_dosages
        df_valid["ner_adverse_events"] = parsed_aes
        df_valid["ner_entities_parsed"] = parsed_raw

        # If an external NER predictions file is provided, join and reconcile
        if self.ner_predictions_path and self.ner_predictions_path.exists():
            logger.info("Joining external Stage 3 NER outputs from: %s", self.ner_predictions_path)
            df_ner_ext = pd.read_parquet(self.ner_predictions_path)
            # Match on document_id / note_id
            id_col = "document_id" if "document_id" in df_ner_ext.columns else "note_id"
            merged = pd.merge(df_valid, df_ner_ext, on=id_col, how="inner", suffixes=("", "_ext"))
            self.load_audit["joined_records_count"] = len(merged)
            self.load_audit["unmatched_records_count"] = len(df_valid) - len(merged)
            df_valid = merged
        else:
            self.load_audit["joined_records_count"] = len(df_valid)
            self.load_audit["unmatched_records_count"] = 0

        logger.info(
            "DataLoader summary: Input=%d, Joined=%d, Unmatched=%d, Dups=%d, Discarded=%d",
            self.load_audit["input_records_count"],
            self.load_audit["joined_records_count"],
            self.load_audit["unmatched_records_count"],
            self.load_audit["duplicate_records_count"],
            len(self.load_audit["discarded_records"])
        )

        return df_valid, self.load_audit
