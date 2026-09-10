"""
Clinical SLM Inference Pipeline for Stage 4.
Extracts structured decision support triad (Risk, Key Finding, Action)
from clinical notes, verifies entity preservation, and benchmarks latency.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

SLM_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SLM_SRC_DIR))

from model import ClinicalDecisionSupportEngine

logger = logging.getLogger("stage4.slm.inference")


class ClinicalInferencePipeline:
    """Production-grade inference pipeline for clinical oncology decision support."""

    def __init__(self, config_path: Optional[str] = None):
        self.engine = ClinicalDecisionSupportEngine()
        logger.info("ClinicalInferencePipeline initialized successfully.")

    def predict_note(self, clinical_note: str, note_id: str = "DOC-UNKNOWN", patient_id: str = "PT-UNKNOWN") -> Dict[str, Any]:
        """Performs single-note inference and returns structured triad with telemetry."""
        t0 = time.perf_counter()
        triad = self.engine.generate_triad(clinical_note)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Entity verification
        note_upper = clinical_note.upper()
        preserved_drugs = []
        for drug in ["Docetaxel", "Cisplatin", "Erlotinib", "Carboplatin", "Paclitaxel", "Nivolumab", "Pemetrexed"]:
            if drug.upper() in note_upper and drug.upper() in (triad["target_key_finding"] + triad["target_risk"]).upper():
                preserved_drugs.append(drug)

        result = {
            "patient_id": patient_id,
            "note_id": note_id,
            "target_risk": triad["target_risk"],
            "target_key_finding": triad["target_key_finding"],
            "target_action": triad["target_action"],
            "preserved_drugs": preserved_drugs,
            "latency_ms": round(latency_ms, 2),
            "engine_status": "READY"
        }
        return result

    def predict_batch(self, notes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Batch inference over multiple clinical notes."""
        results = []
        for item in notes:
            res = self.predict_note(
                clinical_note=item.get("clinical_note", item.get("text", "")),
                note_id=item.get("note_id", item.get("document_id", "DOC-UNKNOWN")),
                patient_id=item.get("patient_id", "PT-UNKNOWN")
            )
            results.append(res)
        return results


if __name__ == "__main__":
    pipeline = ClinicalInferencePipeline()
    sample_note = (
        "ONCOLOGY CONSULTATION PROGRESS NOTE\n"
        "Patient ID: PT-000001\n"
        "Demographics: 62-year-old male presenting for Cycle 4 evaluation.\n"
        "DIAGNOSIS & MOLECULAR PROFILING:\n"
        "Primary Diagnosis: Stage IV NSCLC. Genomic Profile: Confirmed driver mutation in EGFR.\n"
        "CLINICAL ASSESSMENT: Patient tolerating treatment well, no acute adverse toxicities.\n"
        "TREATMENT PLAN: Administer scheduled therapy with Erlotinib at a dosage of 150 mg."
    )
    res = pipeline.predict_note(sample_note, note_id="DOC-001", patient_id="PT-001")
    print("\n--- Clinical Inference Sample Result ---")
    for k, v in res.items():
        print(f"{k}: {v}")
