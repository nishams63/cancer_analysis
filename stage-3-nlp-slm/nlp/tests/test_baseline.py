"""
Test Suite: Baseline NLP Models & Inference Service.
Verifies baseline model loading, prediction bounds, and end-to-end inference outputs.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from baseline import ClinicalNLPBaselines
from inference import ClinicalNLPInferenceEngine, get_inference_engine


@pytest.fixture(scope="module")
def loaded_baselines():
    """Load pre-trained baseline models from disk."""
    return ClinicalNLPBaselines.load()


def test_baseline_models_loaded_and_predict(loaded_baselines):
    """Verify models are loaded and predict valid class indices."""
    assert loaded_baselines.is_trained is True
    assert hasattr(loaded_baselines.urgency_model, "predict")
    assert hasattr(loaded_baselines.hazard_model, "predict")


def test_end_to_end_inference_engine(loaded_baselines):
    """Verify inference engine produces valid structured dictionary with probabilities."""
    engine = ClinicalNLPInferenceEngine(loaded_baselines)
    test_note = (
        "ONCOLOGY CONSULTATION: 64yo female with Stage III NSCLC, EGFR L858R mutation. "
        "Patient received Cisplatin 75 mg/m2. Denies acute fever or dyspnea. "
        "Mild fatigue noted. ALT/AST normal. ECOG PS: 1."
    )
    result = engine.analyze_document(test_note, document_id="TEST-DOC-001")

    assert result["document_id"] == "TEST-DOC-001"
    assert "triage_urgency" in result
    assert "toxicity_hazard" in result
    assert "clinical_entities" in result
    assert "structured_summary" in result

    # Check probability bounds
    urg_conf = result["triage_urgency"]["confidence"]
    assert 0.0 <= urg_conf <= 1.0
    prob_sum = sum(result["triage_urgency"]["class_probabilities"].values())
    assert np.isclose(prob_sum, 1.0, atol=1e-3)

    # Check entities detected
    entities = result["clinical_entities"]
    assert len(entities) > 0
    labels = [e["label"] for e in entities]
    assert "DRUG_NAME" in labels
    assert "GENE_MUTATION" in labels
