"""
Evaluation Orchestrator Module for Stage 5 SLM.
Executes test-set evaluation across clinical, structural, and safety dimensions,
and saves negation failure cases to predictions/negation_failures.parquet.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from clinical_metrics import ClinicalEvaluator
from prompt_template import ClinicalPromptTemplate


class SLMEvaluationOrchestrator:
    """Orchestrates comprehensive test-set evaluation."""

    def __init__(self, predictions_dir: str, prompt_template: Optional[ClinicalPromptTemplate] = None):
        self.predictions_dir = Path(predictions_dir)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_template = prompt_template or ClinicalPromptTemplate()
        self.evaluator = ClinicalEvaluator(self.prompt_template)

    def evaluate_test_set(
        self,
        experiment_id: str,
        test_records: List[Dict[str, Any]],
        predictions: List[str]
    ) -> Dict[str, Any]:
        """
        Runs evaluation on test records and saves any negation failures to Parquet.
        """
        targets = [r["target"] for r in test_records]
        expected_risks = [r["expected_risk"] for r in test_records]
        clinical_notes = [r["prompt"] for r in test_records]
        patient_ids = [r["patient_id"] for r in test_records]
        note_ids = [r["note_id"] for r in test_records]

        ref_entities = [
            {
                "ner_genes": r.get("ner_genes", []),
                "ner_drugs": r.get("ner_drugs", []),
                "ner_dosages": r.get("ner_dosages", []),
                "ner_adverse_events": r.get("ner_adverse_events", [])
            }
            for r in test_records
        ]

        metrics = self.evaluator.evaluate_batch(
            predictions=predictions,
            targets=targets,
            expected_risks=expected_risks,
            clinical_notes=clinical_notes,
            reference_entities_list=ref_entities,
            patient_ids=patient_ids,
            note_ids=note_ids
        )

        # Export negation failures if any occurred
        neg_failures = metrics.get("negation_failures", [])
        if neg_failures:
            df_neg = pd.DataFrame(neg_failures)
            out_p = self.predictions_dir / f"{experiment_id}_negation_failures.parquet"
            df_neg.to_parquet(out_p, index=False)

        return metrics
