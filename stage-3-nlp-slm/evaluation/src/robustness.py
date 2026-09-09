"""
Robustness Evaluation Module for Stage 3 Clinical NLP.
Tests prediction stability and extraction invariance under harmless textual perturbations
(casing, spacing, punctuation) and controlled concept negation probes.
"""

from typing import Dict, Any, List, Callable, Tuple
import re
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from model_loader import FrozenNLPArtifacts, load_frozen_artifacts
from prediction_runner import extract_features
import sys
from config import NLP_ROOT_DIR

nlp_src_dir = (NLP_ROOT_DIR / "src").resolve()
if str(nlp_src_dir) not in sys.path:
    sys.path.append(str(nlp_src_dir))

from clinical_concepts import extract_clinical_concepts


def perturb_upper(text: str) -> str:
    """Convert entire text to uppercase."""
    return text.upper()


def perturb_lower(text: str) -> str:
    """Convert entire text to lowercase."""
    return text.lower()


def perturb_extra_whitespace(text: str) -> str:
    """Inject extra whitespace and padded spaces around words."""
    words = text.split()
    return "  ".join(words)


def perturb_punctuation_commas(text: str) -> str:
    """Replace commas with semicolons (harmless syntactic punctuation substitution)."""
    return text.replace(",", ";")


PERTURBATIONS: Dict[str, Callable[[str], str]] = {
    "uppercase": perturb_upper,
    "lowercase": perturb_lower,
    "extra_whitespace": perturb_extra_whitespace,
    "comma_to_semicolon": perturb_punctuation_commas
}


def evaluate_perturbation_stability(
    df: pd.DataFrame,
    artifacts: FrozenNLPArtifacts,
    sample_size: int = 300
) -> Dict[str, Any]:
    """
    Measure prediction agreement and entity extraction stability under harmless text variations.
    """
    eval_subset = df.head(sample_size).copy(deep=True)

    # 1. Compute baseline (unperturbed) predictions
    X_orig, _ = extract_features(eval_subset, artifacts)
    orig_urg_preds = artifacts.urgency_model.predict(X_orig)
    orig_haz_preds = artifacts.hazard_model.predict(X_orig)

    orig_entity_counts = []
    for t in eval_subset["text"]:
        ents = extract_clinical_concepts(t, assign_polarity=True)
        orig_entity_counts.append(len(ents))

    perturbation_results = {}

    for name, func in PERTURBATIONS.items():
        pert_df = eval_subset.copy(deep=True)
        pert_df["text"] = pert_df["text"].apply(func)

        X_pert, _ = extract_features(pert_df, artifacts)
        pert_urg_preds = artifacts.urgency_model.predict(X_pert)
        pert_haz_preds = artifacts.hazard_model.predict(X_pert)

        urg_agreement = float(np.mean(orig_urg_preds == pert_urg_preds))
        haz_agreement = float(np.mean(orig_haz_preds == pert_haz_preds))

        pert_entity_counts = []
        for t in pert_df["text"]:
            ents = extract_clinical_concepts(t, assign_polarity=True)
            pert_entity_counts.append(len(ents))

        entity_agreement = float(np.mean(np.array(orig_entity_counts) == np.array(pert_entity_counts)))

        perturbation_results[name] = {
            "urgency_prediction_agreement": round(urg_agreement, 4),
            "hazard_prediction_agreement": round(haz_agreement, 4),
            "entity_count_agreement": round(entity_agreement, 4),
            "urgency_failure_rate": round(1.0 - urg_agreement, 4),
            "hazard_failure_rate": round(1.0 - haz_agreement, 4)
        }

    return {
        "sample_size": len(eval_subset),
        "perturbation_evaluations": perturbation_results
    }


def evaluate_controlled_negation_probes(artifacts: FrozenNLPArtifacts) -> Dict[str, Any]:
    """
    Test whether prepending negation cues appropriately shifts model predictions / extractions.
    Uses clinical symptom probes: nausea, dyspnea, rash, fatigue.
    """
    probes = [
        ("affirmative_dyspnea", "Patient is experiencing severe dyspnea and cough.", "CRITICAL"),
        ("negated_dyspnea", "Patient denies dyspnea and cough is absent.", "LOW"),
        ("affirmative_fatigue", "Patient reports manageable mild fatigue.", "LOW"),
        ("affirmative_hepatotoxicity", "Patient presents with elevated transaminases and jaundice.", "HIGH"),
        ("negated_hepatotoxicity", "No evidence of elevated transaminases or hepatotoxicity.", "LOW")
    ]

    probe_records = []
    for name, text, expected_tier in probes:
        dummy_df = pd.DataFrame([{
            "document_id": "PROBE-001",
            "patient_id": "PT-TEST",
            "encounter_id": "ENC-TEST",
            "document_type": "oncology_consultation",
            "text": text
        }])

        X_probe, _ = extract_features(dummy_df, artifacts)
        urg_pred_idx = artifacts.urgency_model.predict(X_probe)[0]
        urg_pred_label = artifacts.urgency_encoder.inverse_transform([urg_pred_idx])[0]
        haz_pred_idx = artifacts.hazard_model.predict(X_probe)[0]
        haz_pred_label = artifacts.hazard_encoder.inverse_transform([haz_pred_idx])[0]

        extracted = extract_clinical_concepts(text, assign_polarity=True)
        polarities = [e.get("polarity", "UNKNOWN") for e in extracted]

        probe_records.append({
            "probe_name": name,
            "text": text,
            "expected_urgency_direction": expected_tier,
            "predicted_urgency": urg_pred_label,
            "predicted_hazard": haz_pred_label,
            "extracted_entities": len(extracted),
            "concept_polarities": polarities
        })

    return {
        "total_probes": len(probes),
        "probe_details": probe_records
    }
