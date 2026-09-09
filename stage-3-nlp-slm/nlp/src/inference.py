"""
Deterministic Inference Service for Stage 3 Clinical NLP.
Provides single-document and batch inference returning structured clinical entities,
negation attributions, predicted triage urgency, and toxicity hazard classifications.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from config import ARTIFACTS_DIR
from text_cleaning import clean_clinical_text
from text_normalization import normalize_clinical_text
from clinical_concepts import extract_clinical_concepts
from baseline import ClinicalNLPBaselines


class ClinicalNLPInferenceEngine:
    """End-to-end inference service for oncology clinical text."""

    def __init__(self, baselines: Optional[ClinicalNLPBaselines] = None):
        if baselines is None:
            self.baselines = ClinicalNLPBaselines.load()
        else:
            self.baselines = baselines

    def analyze_document(self, text: str, document_id: str = "INFER-DOC-001") -> Dict[str, Any]:
        """
        Run end-to-end analysis on an unstructured clinical note:
        1. Conservative cleaning & normalization.
        2. Clinical entity extraction & negation scoping.
        3. Feature extraction & model scoring.
        4. Structured JSON response.
        """
        cleaned = clean_clinical_text(text)
        normalized = normalize_clinical_text(cleaned)

        # Extract entities with polarity
        concepts = extract_clinical_concepts(normalized, assign_polarity=True)

        # Feature transformation and classification
        dummy_df = pd.DataFrame([{
            "document_id": document_id,
            "text": normalized
        }])

        X, struct_df = self.baselines.feature_pipeline.transform(dummy_df)
        urg_pred_idx = self.baselines.urgency_model.predict(X)[0]
        urg_probs = self.baselines.urgency_model.predict_proba(X)[0]

        haz_pred_idx = self.baselines.hazard_model.predict(X)[0]
        haz_probs = self.baselines.hazard_model.predict_proba(X)[0]

        urg_class = self.baselines.label_manager.inverse_transform_urgency(np.array([urg_pred_idx]))[0]
        haz_class = self.baselines.label_manager.inverse_transform_hazard(np.array([haz_pred_idx]))[0]

        urg_classes = list(self.baselines.label_manager.urgency_encoder.classes_)
        haz_classes = list(self.baselines.label_manager.hazard_encoder.classes_)

        return {
            "document_id": document_id,
            "cleaned_text_length": len(normalized),
            "word_count": len(normalized.split()),
            "triage_urgency": {
                "predicted_class": urg_class,
                "confidence": round(float(np.max(urg_probs)), 4),
                "class_probabilities": {cls: round(float(p), 4) for cls, p in zip(urg_classes, urg_probs)}
            },
            "toxicity_hazard": {
                "predicted_class": haz_class,
                "confidence": round(float(np.max(haz_probs)), 4),
                "class_probabilities": {cls: round(float(p), 4) for cls, p in zip(haz_classes, haz_probs)}
            },
            "clinical_entities": concepts,
            "structured_summary": {
                "total_entities_found": len(concepts),
                "affirmed_adverse_events": sum(1 for c in concepts if c["label"] == "ADVERSE_EVENT" and c.get("polarity") == "AFFIRMED"),
                "negated_adverse_events": sum(1 for c in concepts if c["label"] == "ADVERSE_EVENT" and c.get("polarity") == "NEGATED"),
                "antineoplastic_agents": [c["text"] for c in concepts if c["label"] == "DRUG_NAME"],
                "driver_mutations": [c["text"] for c in concepts if c["label"] == "GENE_MUTATION"]
            }
        }

    def analyze_batch(self, texts: List[str], doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Run batch analysis over multiple clinical texts."""
        if doc_ids is None:
            doc_ids = [f"BATCH-DOC-{i:04d}" for i in range(len(texts))]
        return [self.analyze_document(t, doc_id=did) for t, did in zip(texts, doc_ids)]


def get_inference_engine() -> ClinicalNLPInferenceEngine:
    """Factory helper to obtain a loaded inference engine."""
    return ClinicalNLPInferenceEngine()
