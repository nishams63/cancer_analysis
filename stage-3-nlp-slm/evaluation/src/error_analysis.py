"""
Error Analysis Module for Stage 3 Clinical NLP Evaluation.
Analyzes classification failure modes, high-confidence errors, safety-critical misses,
and entity extraction boundary/type discrepancies.
"""

from typing import Dict, Any, List
import json
import pandas as pd
import numpy as np

import sys
from config import NLP_ROOT_DIR, RESULTS_ERROR_DIR

nlp_src_dir = (NLP_ROOT_DIR / "src").resolve()
if str(nlp_src_dir) not in sys.path:
    sys.path.append(str(nlp_src_dir))

from clinical_concepts import extract_clinical_concepts


def analyze_classification_errors(
    pred_df: pd.DataFrame,
    task_prefix: str = "urgency"
) -> Dict[str, Any]:
    """
    Perform granular taxonomy of classification errors:
    - total misclassifications
    - high-confidence errors (confidence >= 0.70 but wrong)
    - safety-critical misses (e.g., true CRITICAL missed)
    - cross-class confusion pairings
    """
    true_col = f"ground_truth_{task_prefix}"
    pred_col = f"predicted_{task_prefix}"
    conf_col = f"{task_prefix}_confidence"

    is_error = pred_df[true_col] != pred_df[pred_col]
    error_df = pred_df[is_error].copy()

    total_samples = len(pred_df)
    total_errors = len(error_df)
    error_rate = total_errors / total_samples if total_samples > 0 else 0.0

    # High confidence errors
    high_conf_errors = error_df[error_df[conf_col] >= 0.70]

    # Confusion pairings
    confusion_pairs = {}
    for _, row in error_df.iterrows():
        pair_key = f"{row[true_col]} -> {row[pred_col]}"
        confusion_pairs[pair_key] = confusion_pairs.get(pair_key, 0) + 1

    sorted_confusion = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

    # Safety-critical analysis for Urgency
    critical_misses = []
    if task_prefix == "urgency":
        crit_df = error_df[error_df[true_col] == "CRITICAL"]
        for _, row in crit_df.iterrows():
            critical_misses.append({
                "document_id": row["document_id"],
                "true_class": row[true_col],
                "predicted_class": row[pred_col],
                "confidence": float(row[conf_col])
            })

    # Curate sample of top errors for reporting (privacy safe)
    sample_errors = []
    for _, row in error_df.head(10).iterrows():
        sample_errors.append({
            "document_id": row["document_id"],
            "document_type": row.get("document_type", "unknown"),
            "true_label": row[true_col],
            "predicted_label": row[pred_col],
            "confidence": float(row[conf_col])
        })

    return {
        "task_name": task_prefix,
        "total_samples": total_samples,
        "total_errors": total_errors,
        "error_rate": round(error_rate, 4),
        "high_confidence_errors_count": len(high_conf_errors),
        "high_confidence_error_rate": round(len(high_conf_errors) / total_errors, 4) if total_errors > 0 else 0.0,
        "top_confusion_pairs": [{"pair": p, "count": c} for p, c in sorted_confusion[:8]],
        "critical_misses_count": len(critical_misses),
        "critical_misses_details": critical_misses,
        "sample_error_cases": sample_errors
    }


def analyze_extraction_errors(
    df: pd.DataFrame,
    sample_limit: int = 150
) -> Dict[str, Any]:
    """
    Analyze entity extraction error categories:
    - missed entities (false negatives)
    - extraneous entities (false positives)
    - span boundary offsets
    """
    fn_count = 0
    fp_count = 0
    boundary_mismatch_count = 0

    error_examples = []

    for _, row in df.head(sample_limit).iterrows():
        text = row["text"]
        pred_entities = extract_clinical_concepts(text, assign_polarity=True)
        raw_gt = row["ner_entities"]
        gt_entities = json.loads(raw_gt) if isinstance(raw_gt, str) else raw_gt

        pred_spans = {(e["start"], e["end"], e["label"]): e for e in pred_entities}
        gt_spans = {(e["start"], e["end"], e["label"]): e for e in gt_entities}

        # Check for false negatives
        for g_key, g_ent in gt_spans.items():
            if g_key not in pred_spans:
                # Check if there is an overlapping span with same label
                overlap = any(
                    p["label"] == g_ent["label"] and
                    max(p["start"], g_ent["start"]) < min(p["end"], g_ent["end"])
                    for p in pred_entities
                )
                if overlap:
                    boundary_mismatch_count += 1
                else:
                    fn_count += 1
                    if len(error_examples) < 10:
                        error_examples.append({
                            "type": "FALSE_NEGATIVE",
                            "entity_label": g_ent["label"],
                            "text_snippet": g_ent.get("text", text[g_ent["start"]:g_ent["end"]]),
                            "document_id": row["document_id"]
                        })

        # Check for false positives
        for p_key, p_ent in pred_spans.items():
            if p_key not in gt_spans:
                overlap = any(
                    g["label"] == p_ent["label"] and
                    max(p_ent["start"], g["start"]) < min(p_ent["end"], g["end"])
                    for g in gt_entities
                )
                if not overlap:
                    fp_count += 1
                    if len(error_examples) < 10:
                        error_examples.append({
                            "type": "FALSE_POSITIVE",
                            "entity_label": p_ent["label"],
                            "text_snippet": p_ent.get("text", text[p_ent["start"]:p_ent["end"]]),
                            "document_id": row["document_id"]
                        })

    return {
        "documents_sampled": sample_limit,
        "false_negatives_count": fn_count,
        "false_positives_count": fp_count,
        "boundary_discrepancies_count": boundary_mismatch_count,
        "sample_error_details": error_examples
    }
