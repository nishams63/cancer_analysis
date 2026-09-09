"""
Master Experiment Runner for Stage 3 Clinical Concept Model Upgrade.
Executes fine-tuning of ContextualClinicalConceptModel on TRAIN,
evaluates NER performance on VALIDATION, benchmarks Experiments A, B, C, D,
measures latency/efficiency, tests whitespace robustness, and determines model promotion.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import time
import psutil
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib

import sys
src_dir = Path(__file__).resolve().parent
nlp_dir = src_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from token_alignment import canonicalize_text
from contextual_concept_model import ContextualClinicalConceptModel
from upgraded_feature_pipeline import extract_upgraded_concept_counts, UpgradedClinicalFeaturePipeline
from data_loader import load_train_data, load_validation_data
from label_preparation import TargetLabelManager
from clinical_concepts import evaluate_entity_spans

# Target paths for model artifacts, results, and reports
MODELS_DIR = nlp_dir / "models"
IMPROVED_MODEL_DIR = MODELS_DIR / "improved_concept_model"
RESULTS_DIR = nlp_dir / "results"
RESULTS_COMP_DIR = RESULTS_DIR / "comparison"
RESULTS_IMP_DIR = RESULTS_DIR / "improved"
REPORTS_DIR = nlp_dir / "reports"

for d in [IMPROVED_MODEL_DIR, RESULTS_COMP_DIR, RESULTS_IMP_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def evaluate_ner_on_dataframe(
    df: pd.DataFrame,
    concept_model: ContextualClinicalConceptModel,
    match_type: str = "exact"
) -> Dict[str, Any]:
    """Evaluate entity extraction on a dataframe against gold ner_entities."""
    doc_results = []
    entity_counts = {
        "GENE_MUTATION": {"tp": 0, "pred": 0, "gt": 0},
        "DRUG_NAME": {"tp": 0, "pred": 0, "gt": 0},
        "DOSAGE": {"tp": 0, "pred": 0, "gt": 0},
        "ADVERSE_EVENT": {"tp": 0, "pred": 0, "gt": 0}
    }

    for _, row in df.iterrows():
        raw_text = row["text"]
        text = canonicalize_text(raw_text)
        pred_entities = concept_model.predict_entities(text, assign_polarity=True)
        raw_gt = row["ner_entities"]
        gt_entities = json.loads(raw_gt) if isinstance(raw_gt, str) else raw_gt

        doc_eval = evaluate_entity_spans(pred_entities, gt_entities, match_type=match_type)
        doc_results.append(doc_eval)

        # Per-class accounting
        matched_gt = set()
        for p in pred_entities:
            p_lbl = p["label"]
            if p_lbl in entity_counts:
                entity_counts[p_lbl]["pred"] += 1

            for i, gt in enumerate(gt_entities):
                if i in matched_gt:
                    continue
                if p_lbl == gt["label"]:
                    is_match = False
                    if match_type == "exact":
                        is_match = (p["start"] == gt["start"] and p["end"] == gt["end"])
                    elif match_type == "relaxed":
                        is_match = (max(p["start"], gt["start"]) < min(p["end"], gt["end"]))

                    if is_match:
                        matched_gt.add(i)
                        if p_lbl in entity_counts:
                            entity_counts[p_lbl]["tp"] += 1
                        break

        for gt in gt_entities:
            g_lbl = gt["label"]
            if g_lbl in entity_counts:
                entity_counts[g_lbl]["gt"] += 1

    macro_p = float(np.mean([r["precision"] for r in doc_results]))
    macro_r = float(np.mean([r["recall"] for r in doc_results]))
    macro_f1 = float(np.mean([r["f1"] for r in doc_results]))

    per_entity = {}
    for lbl, cnts in entity_counts.items():
        tp = cnts["tp"]
        pred_cnt = cnts["pred"]
        gt_cnt = cnts["gt"]
        p = tp / pred_cnt if pred_cnt > 0 else 0.0
        r = tp / gt_cnt if gt_cnt > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        per_entity[lbl] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "predicted_count": pred_cnt,
            "ground_truth_count": gt_cnt,
            "true_positives": tp
        }

    return {
        "match_type": match_type,
        "evaluated_documents": len(df),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_entity_type": per_entity
    }


def run_experiments() -> Dict[str, Any]:
    print("=" * 70)
    print("STAGE 3 CLINICAL CONCEPT MODEL UPGRADE — EXPERIMENTAL PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/6] Loading TRAIN and VALIDATION partitions...")
    df_train = load_train_data()
    df_val = load_validation_data()
    print(f"Loaded: TRAIN={len(df_train)} docs | VAL={len(df_val)} docs")

    # 2. Fit Contextual Concept Model
    print("\n[2/6] Fine-tuning Contextual Transformer Token Classifier (all-MiniLM-L6-v2)...")
    concept_model = ContextualClinicalConceptModel()
    train_start = time.time()
    # 2 epochs on TRAIN is optimal for fast convergence without overfitting
    concept_model.fit(df_train, epochs=2, batch_size=32, learning_rate=5e-5, verbose=True)
    train_duration = time.time() - train_start

    print(f"Saving improved concept model to {IMPROVED_MODEL_DIR}...")
    concept_model.save(IMPROVED_MODEL_DIR)

    # 3. Evaluate Contextual NER on VALIDATION
    print("\n[3/6] Evaluating Contextual NER on VALIDATION (Exact & Relaxed)...")
    ner_exact = evaluate_ner_on_dataframe(df_val, concept_model, match_type="exact")
    ner_relaxed = evaluate_ner_on_dataframe(df_val, concept_model, match_type="relaxed")

    with open(RESULTS_IMP_DIR / "improved_ner_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"exact": ner_exact, "relaxed": ner_relaxed}, f, indent=2)

    print(f"Upgraded NER Exact F1: {ner_exact['macro_f1']} | Relaxed F1: {ner_relaxed['macro_f1']}")
    print(f"Dosage Precision: {ner_relaxed['per_entity_type']['DOSAGE']['precision']} (Baseline was 0.4000)")
    print(f"Adverse Event Exact F1: {ner_exact['per_entity_type']['ADVERSE_EVENT']['f1']} (Baseline was 0.3746)")

    # 4. Extract Structured Features and Train Feature Pipeline
    print("\n[4/6] Extracting upgraded structured features and fitting feature pipeline...")
    struct_train_df = extract_upgraded_concept_counts(df_train, concept_model)
    struct_val_df = extract_upgraded_concept_counts(df_val, concept_model)

    feat_pipeline = UpgradedClinicalFeaturePipeline(max_tfidf_features=1000)
    feat_pipeline.fit(df_train, struct_train_df)
    feat_pipeline.save(IMPROVED_MODEL_DIR)

    # 5. Label Manager & Targets
    label_mgr = TargetLabelManager().fit(df_train)
    y_urg_train = label_mgr.transform_urgency(df_train["urgency_level"])
    y_urg_val = label_mgr.transform_urgency(df_val["urgency_level"])
    y_haz_train = label_mgr.transform_hazard(df_train["hazard_type"])
    y_haz_val = label_mgr.transform_hazard(df_val["hazard_type"])

    # 6. Execute Experiments A, B, C, D
    print("\n[5/6] Running Downstream Classification Experiments A, B, C, D...")

    # Experiment A: Baseline Control (Frozen reference from validation evaluation)
    exp_a_metrics = {
        "experiment": "Experiment A (Baseline Control)",
        "features": "1,000 TF-IDF + 12 Regex Concept Counts (1,012-dim)",
        "feature_dim": 1012,
        "urgency_accuracy": 0.8559,
        "urgency_macro_f1": 0.7557,
        "urgency_critical_recall": 0.9457,
        "hazard_accuracy": 0.7756,
        "hazard_macro_f1": 0.5214,
        "ner_relaxed_f1": 0.7670,
        "ner_exact_f1": 0.6437
    }

    experiments_results = {"Experiment_A": exp_a_metrics}

    for exp_type, name, desc in [
        ("B", "Experiment_B", "Contextual Concept Counts Only (12-dim)"),
        ("C", "Experiment_C", "Contextual Embeddings + Upgraded Concept Counts (396-dim)"),
        ("D", "Experiment_D", "Hybrid: TF-IDF + Contextual Embeddings + Concept Counts (1,396-dim)")
    ]:
        print(f"\nEvaluating {name}: {desc}...")
        X_tr = feat_pipeline.transform_experiment(df_train, struct_train_df, concept_model, experiment_type=exp_type)
        X_ev = feat_pipeline.transform_experiment(df_val, struct_val_df, concept_model, experiment_type=exp_type)

        # Urgency Model
        urg_clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=42)
        urg_clf.fit(X_tr, y_urg_train)
        urg_preds = urg_clf.predict(X_ev)

        urg_acc = float(accuracy_score(y_urg_val, urg_preds))
        urg_mf1 = float(f1_score(y_urg_val, urg_preds, average="macro", zero_division=0))
        urg_rep = classification_report(y_urg_val, urg_preds, target_names=list(label_mgr.urgency_encoder.classes_), output_dict=True, zero_division=0)
        crit_recall = float(urg_rep.get("CRITICAL", {}).get("recall", 0.0))

        # Hazard Model
        haz_clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=1000, random_state=42)
        haz_clf.fit(X_tr, y_haz_train)
        haz_preds = haz_clf.predict(X_ev)

        haz_acc = float(accuracy_score(y_haz_val, haz_preds))
        haz_mf1 = float(f1_score(y_haz_val, haz_preds, average="macro", zero_division=0))

        exp_res = {
            "experiment": name,
            "description": desc,
            "feature_dim": X_tr.shape[1],
            "urgency_accuracy": round(urg_acc, 4),
            "urgency_macro_f1": round(urg_mf1, 4),
            "urgency_critical_recall": round(crit_recall, 4),
            "hazard_accuracy": round(haz_acc, 4),
            "hazard_macro_f1": round(haz_mf1, 4),
            "ner_relaxed_f1": ner_relaxed["macro_f1"],
            "ner_exact_f1": ner_exact["macro_f1"]
        }
        experiments_results[name] = exp_res
        print(f"-> Urgency Macro F1: {exp_res['urgency_macro_f1']} | Critical Recall: {exp_res['urgency_critical_recall']} | Hazard Macro F1: {exp_res['hazard_macro_f1']}")

    # 7. Efficiency & Latency Benchmarks
    print("\n[6/6] Measuring Efficiency, Memory, and Latency...")
    sample_texts = [canonicalize_text(t) for t in df_val["text"].head(100)]

    t0 = time.time()
    for t in sample_texts:
        _ = concept_model.predict_entities(t, assign_polarity=True)
    ner_latency_ms = ((time.time() - t0) / len(sample_texts)) * 1000.0

    t1 = time.time()
    _ = concept_model.extract_contextual_embeddings(sample_texts, batch_size=32)
    emb_latency_ms = ((time.time() - t1) / len(sample_texts)) * 1000.0

    total_pipeline_latency_ms = ner_latency_ms + emb_latency_ms
    docs_per_sec = 1000.0 / total_pipeline_latency_ms if total_pipeline_latency_ms > 0 else 0.0

    # Model size on disk
    model_size_bytes = sum(f.stat().st_size for f in IMPROVED_MODEL_DIR.glob("**/*") if f.is_file())
    model_size_mb = model_size_bytes / (1024 * 1024)

    total_params = sum(p.numel() for p in concept_model.model.parameters())
    trainable_params = sum(p.numel() for p in concept_model.model.parameters() if p.requires_grad)
    process_mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    efficiency_summary = {
        "model_name": "all-MiniLM-L6-v2 (Contextual Token Classifier)",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_disk_mb": round(model_size_mb, 2),
        "process_memory_mb": round(process_mem_mb, 2),
        "ner_inference_latency_ms": round(ner_latency_ms, 2),
        "embedding_inference_latency_ms": round(emb_latency_ms, 2),
        "total_latency_per_doc_ms": round(total_pipeline_latency_ms, 2),
        "throughput_docs_per_sec": round(docs_per_sec, 2),
        "training_duration_sec": round(train_duration, 2)
    }

    with open(RESULTS_COMP_DIR / "efficiency_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(efficiency_summary, f, indent=2)

    # 8. Whitespace Robustness Verification
    print("Testing Whitespace Robustness (Original vs. Multi-Space)...")
    test_text_single = "Patient presents with severe nausea and acute dyspnea following treatment."
    test_text_multi = "Patient   presents   with   severe   nausea   and   acute   dyspnea   following   treatment."

    ents_single = concept_model.predict_entities(test_text_single)
    ents_multi = concept_model.predict_entities(test_text_multi)

    is_whitespace_identical = (
        len(ents_single) == len(ents_multi) and
        all(s["label"] == m["label"] and s["text"] == m["text"] for s, m in zip(ents_single, ents_multi))
    )

    # 9. Promotion Decision Rule Evaluation
    print("\n" + "=" * 70)
    print("PROMOTION DECISION RULE EVALUATION")
    print("=" * 70)
    exp_d = experiments_results["Experiment_D"]

    criteria_ner = (ner_exact["macro_f1"] >= 0.6437 or ner_relaxed["macro_f1"] >= 0.7670)
    criteria_dosage = (ner_relaxed["per_entity_type"]["DOSAGE"]["precision"] > 0.4000)
    criteria_safety = (exp_d["urgency_critical_recall"] >= 0.9300)
    criteria_urgency = (exp_d["urgency_macro_f1"] >= 0.7557)
    criteria_latency = (total_pipeline_latency_ms <= 50.0)

    is_promoted = (
        criteria_ner and
        criteria_dosage and
        criteria_safety and
        criteria_urgency and
        criteria_latency
    )

    decision_summary = {
        "criteria": {
            "ner_f1_improved": {
                "achieved": bool(criteria_ner),
                "baseline_exact_f1": 0.6437,
                "improved_exact_f1": ner_exact["macro_f1"],
                "baseline_relaxed_f1": 0.7670,
                "improved_relaxed_f1": ner_relaxed["macro_f1"]
            },
            "dosage_precision_improved": {
                "achieved": bool(criteria_dosage),
                "baseline_dosage_p": 0.4000,
                "improved_dosage_p": ner_relaxed["per_entity_type"]["DOSAGE"]["precision"]
            },
            "critical_recall_preserved": {
                "achieved": bool(criteria_safety),
                "baseline_critical_recall": 0.9457,
                "hybrid_critical_recall": exp_d["urgency_critical_recall"]
            },
            "downstream_macro_f1_maintained": {
                "achieved": bool(criteria_urgency),
                "baseline_macro_f1": 0.7557,
                "hybrid_macro_f1": exp_d["urgency_macro_f1"]
            },
            "acceptable_efficiency": {
                "achieved": bool(criteria_latency),
                "latency_per_doc_ms": round(total_pipeline_latency_ms, 2),
                "target_max_ms": 50.0
            },
            "whitespace_robustness_verified": {
                "achieved": bool(is_whitespace_identical)
            }
        },
        "promotion_verdict": "PROMOTED_AS_IMPROVED_CONCEPT_MODEL" if is_promoted else "RETAIN_BASELINE",
        "recommended_architecture": "Experiment D (Hybrid: 1000 TF-IDF + 384 Contextual Embeddings + 12 Upgraded Counts)" if is_promoted else "Experiment A"
    }

    final_comparison = {
        "experiments": experiments_results,
        "efficiency": efficiency_summary,
        "decision": decision_summary
    }

    with open(RESULTS_COMP_DIR / "experiment_comparison.json", "w", encoding="utf-8") as f:
        json.dump(final_comparison, f, indent=2)

    print(f"NER Exact F1: {ner_exact['macro_f1']} (Baseline: 0.6437)")
    print(f"NER Relaxed F1: {ner_relaxed['macro_f1']} (Baseline: 0.7670)")
    print(f"Dosage Precision: {ner_relaxed['per_entity_type']['DOSAGE']['precision']} (Baseline: 0.4000)")
    print(f"Adverse Event Exact F1: {ner_exact['per_entity_type']['ADVERSE_EVENT']['f1']} (Baseline: 0.3746)")
    print(f"Hybrid Urgency Macro F1: {exp_d['urgency_macro_f1']} (Baseline: 0.7557)")
    print(f"Hybrid Critical Recall: {exp_d['urgency_critical_recall']} (Baseline: 0.9457)")
    print(f"Latency per Document: {total_pipeline_latency_ms:.2f} ms")
    print(f"Whitespace Invariance: {is_whitespace_identical}")
    print(f"\nFINAL DECISION: {decision_summary['promotion_verdict']}")
    print("=" * 70)

    return final_comparison


if __name__ == "__main__":
    run_experiments()
