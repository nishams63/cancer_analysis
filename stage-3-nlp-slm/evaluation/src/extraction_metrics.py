"""
Clinical Entity Extraction (NER) Metrics Module for Stage 3 Clinical NLP Evaluation.
Evaluates exact and relaxed span-level Precision, Recall, and F1 across entity types.
"""

from typing import List, Dict, Any, Tuple
import json
import numpy as np
import pandas as pd
import sys
from config import NLP_ROOT_DIR, VALID_NER_LABELS

nlp_src_dir = (NLP_ROOT_DIR / "src").resolve()
if str(nlp_src_dir) not in sys.path:
    sys.path.append(str(nlp_src_dir))

from clinical_concepts import extract_clinical_concepts, evaluate_entity_spans


def evaluate_dataset_extractions(
    df: pd.DataFrame,
    match_type: str = "relaxed"
) -> Dict[str, Any]:
    """
    Evaluate concept extraction on a complete dataframe against ground-truth `ner_entities`.
    Calculates macro-averaged and micro-averaged span Precision, Recall, F1,
    as well as per-entity-type performance.
    """
    doc_results = []
    entity_counts = {lbl: {"tp": 0, "pred": 0, "gt": 0} for lbl in VALID_NER_LABELS}

    total_tp = 0
    total_pred = 0
    total_gt = 0

    for _, row in df.iterrows():
        text = row["text"]
        pred_entities = extract_clinical_concepts(text, assign_polarity=True)
        raw_gt = row["ner_entities"]
        gt_entities = json.loads(raw_gt) if isinstance(raw_gt, str) else raw_gt

        # Document-level span evaluation
        doc_eval = evaluate_entity_spans(pred_entities, gt_entities, match_type=match_type)
        doc_results.append(doc_eval)

        # Per-entity-type accounting
        matched_gt_indices = set()
        for p in pred_entities:
            p_lbl = p["label"]
            if p_lbl in entity_counts:
                entity_counts[p_lbl]["pred"] += 1
            total_pred += 1

            for i, gt in enumerate(gt_entities):
                if i in matched_gt_indices:
                    continue
                if p_lbl == gt["label"]:
                    is_match = False
                    if match_type == "exact":
                        is_match = (p["start"] == gt["start"] and p["end"] == gt["end"])
                    elif match_type == "relaxed":
                        is_match = (max(p["start"], gt["start"]) < min(p["end"], gt["end"]))

                    if is_match:
                        matched_gt_indices.add(i)
                        total_tp += 1
                        if p_lbl in entity_counts:
                            entity_counts[p_lbl]["tp"] += 1
                        break

        for gt in gt_entities:
            g_lbl = gt["label"]
            if g_lbl in entity_counts:
                entity_counts[g_lbl]["gt"] += 1
            total_gt += 1

    # Macro averages across documents
    macro_p = float(np.mean([r["precision"] for r in doc_results]))
    macro_r = float(np.mean([r["recall"] for r in doc_results]))
    macro_f1 = float(np.mean([r["f1"] for r in doc_results]))

    # Micro averages pooled across dataset
    micro_p = total_tp / total_pred if total_pred > 0 else 0.0
    micro_r = total_tp / total_gt if total_gt > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Per-entity type metrics
    per_entity_type = {}
    for lbl, counts in entity_counts.items():
        tp = counts["tp"]
        p_cnt = counts["pred"]
        g_cnt = counts["gt"]
        p = tp / p_cnt if p_cnt > 0 else 0.0
        r = tp / g_cnt if g_cnt > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        per_entity_type[lbl] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "predicted_count": p_cnt,
            "ground_truth_count": g_cnt,
            "true_positives": tp
        }

    return {
        "match_type": match_type,
        "evaluated_documents": len(df),
        "macro_mean_precision": round(macro_p, 4),
        "macro_mean_recall": round(macro_r, 4),
        "macro_mean_f1": round(macro_f1, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "total_true_positives": total_tp,
        "total_predicted_entities": total_pred,
        "total_ground_truth_entities": total_gt,
        "per_entity_type": per_entity_type
    }
