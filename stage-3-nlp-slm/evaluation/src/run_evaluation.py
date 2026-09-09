"""
Master Evaluation Orchestrator for Stage 3 Clinical NLP.
Executes the comprehensive, reproducible evaluation across Validation and Locked Test sets,
generating all metrics, predictions, error analyses, calibration curves, and figures.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from config import (
    EVAL_MODULE_DIR,
    RESULTS_DIR,
    RESULTS_VAL_DIR,
    RESULTS_TEST_DIR,
    RESULTS_METRICS_DIR,
    RESULTS_PRED_DIR,
    RESULTS_CM_DIR,
    RESULTS_ERROR_DIR,
    RESULTS_ROBUST_DIR,
    RESULTS_CALIB_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
    BOOTSTRAP_ITERATIONS,
    CONFIDENCE_LEVEL
)
from data_loader import load_train_data, load_validation_data, load_locked_test_data
from model_loader import load_frozen_artifacts
from prediction_runner import run_predictions, save_predictions_csv
from classification_metrics import compute_classification_metrics, confusion_matrix_to_dataframe
from extraction_metrics import evaluate_dataset_extractions
from negation_metrics import evaluate_diagnostic_negation_suite, evaluate_dataset_polarity_distribution
from feature_metrics import evaluate_feature_matrix, compute_feature_drift
from calibration import compute_calibration_metrics
from robustness import evaluate_perturbation_stability, evaluate_controlled_negation_probes
from error_analysis import analyze_classification_errors, analyze_extraction_errors
from leakage_checks import (
    verify_upstream_file_hashes,
    verify_split_isolation,
    verify_text_duplicates_across_splits,
    verify_temporal_consistency,
    verify_target_leakage_terms
)
from confidence_intervals import compute_patient_bootstrap_ci
from utils import (
    save_json,
    plot_confusion_matrix,
    plot_per_class_f1_bars,
    plot_calibration_curve,
    plot_generalization_comparison
)


def run_full_evaluation() -> Dict[str, Any]:
    print("=" * 70)
    print("STAGE 3 CLINICAL NLP — INDEPENDENT EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/9] Loading upstream datasets in read-only mode...")
    df_train = load_train_data()
    df_val = load_validation_data()
    df_test = load_locked_test_data()
    print(f"Loaded: TRAIN={len(df_train)}, VAL={len(df_val)}, LOCKED_TEST={len(df_test)}")

    # 2. Load Frozen Artifacts
    print("\n[2/9] Deserializing frozen NLP artifacts...")
    artifacts = load_frozen_artifacts()
    urg_classes = list(artifacts.urgency_encoder.classes_)
    haz_classes = list(artifacts.hazard_encoder.classes_)
    print(f"Urgency Classes: {urg_classes}")
    print(f"Hazard Classes: {haz_classes}")

    # 3. Leakage & Split Invariance Checks
    print("\n[3/9] Performing cryptographic hash and split isolation verification...")
    hash_verification = verify_upstream_file_hashes()
    split_isolation = verify_split_isolation(df_train, df_val, df_test)
    text_dups = verify_text_duplicates_across_splits(df_train, df_val, df_test)
    temporal_val = verify_temporal_consistency(df_val)
    temporal_test = verify_temporal_consistency(df_test)
    target_leak_val = verify_target_leakage_terms(df_val)
    target_leak_test = verify_target_leakage_terms(df_test)

    leakage_summary = {
        "upstream_file_hashes": hash_verification,
        "split_isolation": split_isolation,
        "cross_split_text_duplicates": text_dups,
        "temporal_consistency": {
            "validation": temporal_val,
            "locked_test": temporal_test
        },
        "target_leakage_scan": {
            "validation": target_leak_val,
            "locked_test": target_leak_test
        }
    }
    save_json(leakage_summary, RESULTS_METRICS_DIR / "leakage_verification_results.json")
    print(f"Leakage Checks Completed. Completely Isolated: {split_isolation['is_completely_isolated']}")

    # 4. Predictions & Inference
    print("\n[4/9] Running batch inference on Validation and Locked Test partitions...")
    val_pred_df, val_urg_probs, val_haz_probs, X_val, struct_val = run_predictions(df_val, artifacts)
    test_pred_df, test_urg_probs, test_haz_probs, X_test, struct_test = run_predictions(df_test, artifacts)

    save_predictions_csv(val_pred_df, "validation_predictions.csv")
    save_predictions_csv(test_pred_df, "locked_test_predictions.csv")
    print("Inference complete. Predictions saved to CSV.")

    # 5. Validation Evaluation
    print("\n[5/9] Evaluating Validation partition metrics...")
    val_urg_metrics = compute_classification_metrics(
        val_pred_df["ground_truth_urgency"].values,
        val_pred_df["predicted_urgency"].values,
        urg_classes,
        task_name="urgency_validation"
    )
    val_haz_metrics = compute_classification_metrics(
        val_pred_df["ground_truth_hazard"].values,
        val_pred_df["predicted_hazard"].values,
        haz_classes,
        task_name="hazard_validation"
    )

    # Entity Extraction
    val_ner_exact = evaluate_dataset_extractions(df_val, match_type="exact")
    val_ner_relaxed = evaluate_dataset_extractions(df_val, match_type="relaxed")

    # Calibration
    val_true_urg_idx = artifacts.urgency_encoder.transform(val_pred_df["ground_truth_urgency"])
    val_true_haz_idx = artifacts.hazard_encoder.transform(val_pred_df["ground_truth_hazard"])
    val_urg_calib = compute_calibration_metrics(val_true_urg_idx, val_urg_probs, task_name="urgency_val")
    val_haz_calib = compute_calibration_metrics(val_true_haz_idx, val_haz_probs, task_name="hazard_val")

    # Feature Diagnostics
    val_feat_diag = evaluate_feature_matrix(X_val, struct_val, artifacts.numeric_feature_cols, split_name="validation")

    val_summary = {
        "split": "validation",
        "sample_count": len(df_val),
        "urgency_classification": val_urg_metrics,
        "hazard_classification": val_haz_metrics,
        "entity_extraction_exact": val_ner_exact,
        "entity_extraction_relaxed": val_ner_relaxed,
        "calibration": {
            "urgency": val_urg_calib,
            "hazard": val_haz_calib
        },
        "feature_diagnostics": val_feat_diag
    }
    save_json(val_summary, RESULTS_VAL_DIR / "validation_metrics.json")

    # 6. Locked Test Evaluation
    print("\n[6/9] Evaluating Locked Test partition metrics & Bootstrap Confidence Intervals...")
    test_urg_metrics = compute_classification_metrics(
        test_pred_df["ground_truth_urgency"].values,
        test_pred_df["predicted_urgency"].values,
        urg_classes,
        task_name="urgency_locked_test"
    )
    test_haz_metrics = compute_classification_metrics(
        test_pred_df["ground_truth_hazard"].values,
        test_pred_df["predicted_hazard"].values,
        haz_classes,
        task_name="hazard_locked_test"
    )

    # Test Bootstrap 95% CIs
    print("Computing 1,000-iteration patient-clustered bootstrap CIs for Locked Test...")
    test_urg_ci = compute_patient_bootstrap_ci(
        df_test,
        test_pred_df["ground_truth_urgency"].values,
        test_pred_df["predicted_urgency"].values,
        n_iterations=BOOTSTRAP_ITERATIONS,
        random_seed=RANDOM_SEED,
        critical_class="CRITICAL"
    )
    test_haz_ci = compute_patient_bootstrap_ci(
        df_test,
        test_pred_df["ground_truth_hazard"].values,
        test_pred_df["predicted_hazard"].values,
        n_iterations=BOOTSTRAP_ITERATIONS,
        random_seed=RANDOM_SEED,
        critical_class="HEPATIC"
    )

    # Test NER
    test_ner_exact = evaluate_dataset_extractions(df_test, match_type="exact")
    test_ner_relaxed = evaluate_dataset_extractions(df_test, match_type="relaxed")

    # Test Calibration
    test_true_urg_idx = artifacts.urgency_encoder.transform(test_pred_df["ground_truth_urgency"])
    test_true_haz_idx = artifacts.hazard_encoder.transform(test_pred_df["ground_truth_hazard"])
    test_urg_calib = compute_calibration_metrics(test_true_urg_idx, test_urg_probs, task_name="urgency_test")
    test_haz_calib = compute_calibration_metrics(test_true_haz_idx, test_haz_probs, task_name="hazard_test")

    # Test Feature Diagnostics & Drift
    test_feat_diag = evaluate_feature_matrix(X_test, struct_test, artifacts.numeric_feature_cols, split_name="locked_test")
    _, struct_train = extract_train_struct(df_train, artifacts)
    feature_drift = compute_feature_drift(struct_train, struct_test, artifacts.numeric_feature_cols)

    test_summary = {
        "split": "locked_test",
        "sample_count": len(df_test),
        "unique_patients": int(df_test["patient_id"].nunique()),
        "unique_encounters": int(df_test["encounter_id"].nunique()),
        "urgency_classification": test_urg_metrics,
        "hazard_classification": test_haz_metrics,
        "urgency_confidence_intervals": test_urg_ci,
        "hazard_confidence_intervals": test_haz_ci,
        "entity_extraction_exact": test_ner_exact,
        "entity_extraction_relaxed": test_ner_relaxed,
        "calibration": {
            "urgency": test_urg_calib,
            "hazard": test_haz_calib
        },
        "feature_diagnostics": test_feat_diag,
        "feature_drift": feature_drift
    }
    save_json(test_summary, RESULTS_TEST_DIR / "locked_test_metrics.json")

    # Save Confusion Matrix DataFrames
    val_cm_urg_df = confusion_matrix_to_dataframe(val_urg_metrics["confusion_matrix"], urg_classes)
    val_cm_urg_df.to_csv(RESULTS_CM_DIR / "confusion_matrix_urgency_val.csv")
    test_cm_urg_df = confusion_matrix_to_dataframe(test_urg_metrics["confusion_matrix"], urg_classes)
    test_cm_urg_df.to_csv(RESULTS_CM_DIR / "confusion_matrix_urgency_test.csv")

    val_cm_haz_df = confusion_matrix_to_dataframe(val_haz_metrics["confusion_matrix"], haz_classes)
    val_cm_haz_df.to_csv(RESULTS_CM_DIR / "confusion_matrix_hazard_val.csv")
    test_cm_haz_df = confusion_matrix_to_dataframe(test_haz_metrics["confusion_matrix"], haz_classes)
    test_cm_haz_df.to_csv(RESULTS_CM_DIR / "confusion_matrix_hazard_test.csv")

    # 7. Diagnostic Suites (Negation, Robustness, Error Analysis)
    print("\n[7/9] Running Negation Diagnostic Suite, Robustness Batteries, and Error Analysis...")
    neg_suite = evaluate_diagnostic_negation_suite()
    val_pol_dist = evaluate_dataset_polarity_distribution(df_val)
    test_pol_dist = evaluate_dataset_polarity_distribution(df_test)
    save_json({
        "diagnostic_suite": neg_suite,
        "polarity_distribution_val": val_pol_dist,
        "polarity_distribution_test": test_pol_dist
    }, RESULTS_METRICS_DIR / "negation_diagnostic_results.json")

    # Robustness
    robust_results = evaluate_perturbation_stability(df_val, artifacts, sample_size=300)
    neg_probes = evaluate_controlled_negation_probes(artifacts)
    save_json({
        "text_perturbations": robust_results,
        "negation_probes": neg_probes
    }, RESULTS_ROBUST_DIR / "robustness_evaluation_results.json")

    # Error Analysis
    val_urg_errors = analyze_classification_errors(val_pred_df, task_prefix="urgency")
    test_urg_errors = analyze_classification_errors(test_pred_df, task_prefix="urgency")
    val_haz_errors = analyze_classification_errors(val_pred_df, task_prefix="hazard")
    test_haz_errors = analyze_classification_errors(test_pred_df, task_prefix="hazard")
    ner_errors = analyze_extraction_errors(df_test, sample_limit=200)

    save_json({
        "urgency_errors_validation": val_urg_errors,
        "urgency_errors_locked_test": test_urg_errors,
        "hazard_errors_validation": val_haz_errors,
        "hazard_errors_locked_test": test_haz_errors,
        "entity_extraction_errors": ner_errors
    }, RESULTS_ERROR_DIR / "detailed_error_analysis.json")

    # Calibration Bin Exports
    save_json(val_urg_calib["bin_details"], RESULTS_CALIB_DIR / "calibration_bins_urgency_val.json")
    save_json(test_urg_calib["bin_details"], RESULTS_CALIB_DIR / "calibration_bins_urgency_test.json")

    # 8. Reproducibility Verification
    print("\n[8/9] Verifying deterministic reproducibility across twin passes...")
    test_pred_df2, _, _, _, _ = run_predictions(df_test, artifacts)
    urg_pred_match = bool((test_pred_df["predicted_urgency"] == test_pred_df2["predicted_urgency"]).all())
    haz_pred_match = bool((test_pred_df["predicted_hazard"] == test_pred_df2["predicted_hazard"]).all())
    urg_conf_diff = float(np.max(np.abs(test_pred_df["urgency_confidence"] - test_pred_df2["urgency_confidence"])))

    repro_results = {
        "evaluation_seed": RANDOM_SEED,
        "urgency_predictions_identical": urg_pred_match,
        "hazard_predictions_identical": haz_pred_match,
        "max_confidence_difference": urg_conf_diff,
        "is_perfectly_reproducible": (urg_pred_match and haz_pred_match and urg_conf_diff < 1e-6)
    }
    save_json(repro_results, RESULTS_METRICS_DIR / "reproducibility_results.json")

    # 9. Publication-Grade Visualizations
    print("\n[9/9] Generating publication figures...")
    plot_confusion_matrix(
        val_urg_metrics["confusion_matrix"], urg_classes,
        "Validation Urgency Confusion Matrix (Normalized)",
        FIGURES_DIR / "classification" / "cm_urgency_validation.png", normalize=True
    )
    plot_confusion_matrix(
        test_urg_metrics["confusion_matrix"], urg_classes,
        "Locked Test Urgency Confusion Matrix (Normalized)",
        FIGURES_DIR / "classification" / "cm_urgency_locked_test.png", normalize=True
    )
    plot_confusion_matrix(
        val_haz_metrics["confusion_matrix"], haz_classes,
        "Validation Hazard Confusion Matrix (Normalized)",
        FIGURES_DIR / "classification" / "cm_hazard_validation.png", normalize=True
    )
    plot_confusion_matrix(
        test_haz_metrics["confusion_matrix"], haz_classes,
        "Locked Test Hazard Confusion Matrix (Normalized)",
        FIGURES_DIR / "classification" / "cm_hazard_locked_test.png", normalize=True
    )
    plot_per_class_f1_bars(
        test_urg_metrics["per_class"],
        "Locked Test Urgency Classification Performance by Class",
        FIGURES_DIR / "classification" / "per_class_f1_urgency_test.png"
    )
    plot_per_class_f1_bars(
        test_haz_metrics["per_class"],
        "Locked Test Hazard Classification Performance by Class",
        FIGURES_DIR / "classification" / "per_class_f1_hazard_test.png"
    )
    plot_calibration_curve(
        test_urg_calib["bin_details"], test_urg_calib["expected_calibration_error"],
        "Locked Test Urgency Reliability Diagram",
        FIGURES_DIR / "calibration" / "calibration_urgency_test.png"
    )
    plot_generalization_comparison(
        val_urg_metrics, test_urg_metrics,
        ["accuracy", "macro_f1", "weighted_f1", "critical_recall"],
        FIGURES_DIR / "generalization" / "generalization_urgency_val_vs_test.png",
        title="Urgency Triage: Validation vs. Locked Test Generalization"
    )
    plot_generalization_comparison(
        val_haz_metrics, test_haz_metrics,
        ["accuracy", "macro_f1", "weighted_f1"],
        FIGURES_DIR / "generalization" / "generalization_hazard_val_vs_test.png",
        title="Hazard Classification: Validation vs. Locked Test Generalization"
    )

    # Comprehensive Consolidated Summary
    consolidated_summary = {
        "metadata": {
            "evaluator": "Stage 3 Evaluation Engineer",
            "date": "2026-09-09",
            "eval_status": "OFFICIAL_LOCKED_TEST_VERIFIED"
        },
        "sample_counts": {
            "train_documents": len(df_train),
            "val_documents": len(df_val),
            "test_documents": len(df_test),
            "total_documents": len(df_train) + len(df_val) + len(df_test)
        },
        "validation_key_metrics": {
            "urgency_macro_f1": val_urg_metrics["macro_f1"],
            "urgency_accuracy": val_urg_metrics["accuracy"],
            "urgency_critical_recall": val_urg_metrics["critical_recall"],
            "hazard_macro_f1": val_haz_metrics["macro_f1"],
            "hazard_accuracy": val_haz_metrics["accuracy"],
            "ner_relaxed_f1": val_ner_relaxed["macro_mean_f1"],
            "ner_exact_f1": val_ner_exact["macro_mean_f1"]
        },
        "locked_test_key_metrics": {
            "urgency_macro_f1": test_urg_metrics["macro_f1"],
            "urgency_accuracy": test_urg_metrics["accuracy"],
            "urgency_critical_recall": test_urg_metrics["critical_recall"],
            "hazard_macro_f1": test_haz_metrics["macro_f1"],
            "hazard_accuracy": test_haz_metrics["accuracy"],
            "ner_relaxed_f1": test_ner_relaxed["macro_mean_f1"],
            "ner_exact_f1": test_ner_exact["macro_mean_f1"]
        },
        "generalization_gaps": {
            "urgency_macro_f1_delta": round(test_urg_metrics["macro_f1"] - val_urg_metrics["macro_f1"], 4),
            "urgency_critical_recall_delta": round(test_urg_metrics["critical_recall"] - val_urg_metrics["critical_recall"], 4),
            "hazard_macro_f1_delta": round(test_haz_metrics["macro_f1"] - val_haz_metrics["macro_f1"], 4),
            "ner_relaxed_f1_delta": round(test_ner_relaxed["macro_mean_f1"] - val_ner_relaxed["macro_mean_f1"], 4)
        },
        "reproducibility": repro_results,
        "leakage": {
            "is_completely_isolated": split_isolation["is_completely_isolated"],
            "all_upstream_hashes_match": hash_verification["all_files_invariant"]
        }
    }
    save_json(consolidated_summary, RESULTS_METRICS_DIR / "evaluation_summary.json")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Validation Urgency Macro F1: {val_urg_metrics['macro_f1']} | Locked Test: {test_urg_metrics['macro_f1']}")
    print(f"Validation Critical Recall: {val_urg_metrics['critical_recall']} | Locked Test: {test_urg_metrics['critical_recall']}")
    print(f"Validation Hazard Macro F1: {val_haz_metrics['macro_f1']} | Locked Test: {test_haz_metrics['macro_f1']}")
    print("=" * 70)
    return consolidated_summary


def extract_train_struct(df_train, artifacts):
    from prediction_runner import extract_features
    return extract_features(df_train, artifacts)


if __name__ == "__main__":
    run_full_evaluation()
