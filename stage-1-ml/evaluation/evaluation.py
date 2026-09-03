import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Ensure stage-1-ml/ml/src is accessible for FeatureEngineer and utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "ml", "src"))
if ML_SRC_DIR not in sys.path:
    sys.path.insert(0, ML_SRC_DIR)

from data_loader import load_master_dataset, prepare_features_and_target, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from feature_engineering import FeatureEngineer
from preprocessing import PreprocessingArtifactManager
from utils import patient_level_split

# Constants & Paths
MASTER_DATASET_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data-engineering", "data", "processed", "master_patient_dataset.csv"))
MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "ml", "models", "best_model", "model.joblib"))
PREPROCESSOR_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "ml", "artifacts", "preprocessor", "preprocessor.joblib"))
MAPPING_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "ml", "artifacts", "encoders", "target_mapping.json"))

RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "metrics.json")
CLASSIFICATION_REPORT_CSV_PATH = os.path.join(RESULTS_DIR, "classification_report.csv")
CONFUSION_MATRIX_PNG_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")

TARGET_MAPPING = {"Low": 0, "Moderate": 1, "High": 2}
REVERSE_TARGET_MAPPING = {0: "Low", 1: "Moderate", 2: "High"}


def validate_probabilities(y_prob, y_pred, tolerance=1e-4):
    """
    Validates class prediction probabilities.
    Checks: 3 classes, bounded [0,1], sum to ~1.0, no NaNs/Infs, highest prob matches predicted class.
    """
    num_samples, num_classes = y_prob.shape
    if num_classes != 3:
        raise ValueError(f"Expected 3 class probabilities, found {num_classes}")
    
    if np.isnan(y_prob).any():
        raise ValueError("NaN values detected in prediction probabilities!")
    
    if np.isinf(y_prob).any():
        raise ValueError("Infinite values detected in prediction probabilities!")
    
    if (y_prob < -tolerance).any() or (y_prob > 1.0 + tolerance).any():
        raise ValueError("Probabilities outside range [0, 1] detected!")
    
    prob_sums = np.sum(y_prob, axis=1)
    if not np.allclose(prob_sums, 1.0, atol=tolerance):
        raise ValueError("Prediction probabilities do not sum to 1.0 within tolerance!")
    
    argmax_classes = np.argmax(y_prob, axis=1)
    mismatches = np.sum(argmax_classes != y_pred)
    if mismatches > 0:
        raise ValueError(f"Found {mismatches} predictions where argmax probability does not match predicted class!")
    
    return True


def run_evaluation():
    print("=" * 70)
    print("STAGE 1 ML EVALUATION ENGINEER PIPELINE")
    print("=" * 70)
    
    # 1. Load master dataset
    print(f"\n[Step 1] Loading master dataset from:\n  {MASTER_DATASET_PATH}")
    if not os.path.exists(MASTER_DATASET_PATH):
        raise FileNotFoundError(f"Master dataset missing at {MASTER_DATASET_PATH}")
    
    raw_df = pd.read_csv(MASTER_DATASET_PATH)
    print(f"Master dataset loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns.")
    
    # 2. Schema and Target Validation
    print("\n[Step 2] Validating columns, target existence, and target mapping...")
    required_cols = ["patient_id", "encounter_id", "toxicity_risk"]
    for col in required_cols:
        if col not in raw_df.columns:
            raise KeyError(f"Required column '{col}' missing from master dataset!")
            
    unique_targets = set(raw_df["toxicity_risk"].unique())
    expected_targets = {"Low", "Moderate", "High"}
    if unique_targets != expected_targets:
        raise ValueError(f"Unexpected target values: {unique_targets}. Expected: {expected_targets}")
    print("Column & Target validation PASSED.")
    
    # 3. Patient-Level Split Verification
    print("\n[Step 3] Performing patient-level split (80% train / 20% test)...")
    train_df, test_df = patient_level_split(raw_df, group_col="patient_id", target_col="toxicity_risk", test_size=0.20, random_state=42)
    
    train_pids = set(train_df["patient_id"])
    test_pids = set(test_df["patient_id"])
    overlap = train_pids.intersection(test_pids)
    
    print(f"  Train split: {len(train_df)} rows ({len(train_pids)} unique patients)")
    print(f"  Locked Test split: {len(test_df)} rows ({len(test_pids)} unique patients)")
    print(f"  Patient Overlap Count: {len(overlap)} (MUST BE 0)")
    
    if len(overlap) > 0:
        raise RuntimeError(f"CRITICAL ERROR: Patient overlap detected between train and test sets! Overlap size: {len(overlap)}")
    
    # 4. Feature Extraction & Feature Engineering
    print("\n[Step 4] Extracting features and running feature engineering on test set...")
    X_test_raw, y_test_series, meta_test = prepare_features_and_target(test_df)
    y_test = y_test_series.values
    
    fe = FeatureEngineer(include_engineered=True)
    X_test_fe = fe.transform(X_test_raw)
    
    # 5. Load Preprocessor & Transform Test Data
    print(f"\n[Step 5] Loading saved preprocessor from:\n  {PREPROCESSOR_PATH}")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor artifact missing at {PREPROCESSOR_PATH}")
    
    pm = PreprocessingArtifactManager.load(PREPROCESSOR_PATH)
    X_test_proc = pm.transform(X_test_fe)
    print(f"Preprocessed test matrix shape: {X_test_proc.shape}")
    
    # 6. Load Best Model (Tuned LightGBM) & Predict
    print(f"\n[Step 6] Loading saved Tuned LightGBM model from:\n  {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact missing at {MODEL_PATH}")
    
    model = joblib.load(MODEL_PATH)
    y_test_pred = model.predict(X_test_proc)
    y_test_prob = model.predict_proba(X_test_proc)
    
    # 7. Probability Validation
    print("\n[Step 7] Validating prediction probabilities...")
    validate_probabilities(y_test_prob, y_test_pred)
    print("Probability validation PASSED.")
    
    # 8. Metric Computation
    print("\n[Step 8] Calculating evaluation metrics...")
    acc = float(accuracy_score(y_test, y_test_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_test_pred, average="macro")
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_test, y_test_pred, average="weighted")
    
    p_class, r_class, f1_class, s_class = precision_recall_fscore_support(y_test, y_test_pred, average=None, labels=[0, 1, 2])
    
    high_risk_recall = float(r_class[2]) # Class High = 2
    
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2])
    
    print("\n" + "=" * 50)
    print("INDEPENDENTLY REPRODUCED TEST METRICS (Tuned LightGBM)")
    print("=" * 50)
    print(f"Accuracy:           {acc:.4f}")
    print(f"Macro Precision:    {p_macro:.4f}")
    print(f"Macro Recall:       {r_macro:.4f}")
    print(f"Macro F1 Score:     {f1_macro:.4f}")
    print(f"Weighted F1 Score:  {f1_weighted:.4f}")
    print(f"High-Risk Recall:   {high_risk_recall:.4f}")
    print("Per-class performance:")
    class_names = ["Low", "Moderate", "High"]
    for idx, cname in enumerate(class_names):
        print(f"  Class {cname:8s} -> Precision: {p_class[idx]:.4f}, Recall: {r_class[idx]:.4f}, F1: {f1_class[idx]:.4f}, Support: {s_class[idx]}")
    print("=" * 50)
    
    # 9. Save Results
    print("\n[Step 9] Saving metrics and classification report...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    metrics_dict = {
        "model_name": "Tuned LightGBM",
        "evaluation_role": "Evaluation Engineer Independent Verification",
        "test_rows": int(len(test_df)),
        "test_unique_patients": int(len(test_pids)),
        "train_rows": int(len(train_df)),
        "train_unique_patients": int(len(train_pids)),
        "patient_overlap_count": int(len(overlap)),
        "accuracy": float(round(acc, 4)),
        "macro_precision": float(round(p_macro, 4)),
        "macro_recall": float(round(r_macro, 4)),
        "macro_f1": float(round(f1_macro, 4)),
        "weighted_f1": float(round(f1_weighted, 4)),
        "high_risk_recall": float(round(high_risk_recall, 4)),
        "target_mapping": TARGET_MAPPING,
        "per_class": {
            "Low": {
                "precision": float(round(p_class[0], 4)),
                "recall": float(round(r_class[0], 4)),
                "f1": float(round(f1_class[0], 4)),
                "support": int(s_class[0])
            },
            "Moderate": {
                "precision": float(round(p_class[1], 4)),
                "recall": float(round(r_class[1], 4)),
                "f1": float(round(f1_class[1], 4)),
                "support": int(s_class[1])
            },
            "High": {
                "precision": float(round(p_class[2], 4)),
                "recall": float(round(r_class[2], 4)),
                "f1": float(round(f1_class[2], 4)),
                "support": int(s_class[2])
            }
        }
    }
    
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved metrics JSON to: {METRICS_JSON_PATH}")
    
    # Save classification report CSV
    report_data = []
    for idx, cname in enumerate(class_names):
        report_data.append({
            "class": cname,
            "encoded_label": idx,
            "precision": float(round(p_class[idx], 4)),
            "recall": float(round(r_class[idx], 4)),
            "f1_score": float(round(f1_class[idx], 4)),
            "support": int(s_class[idx])
        })
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(CLASSIFICATION_REPORT_CSV_PATH, index=False)
    print(f"Saved classification report CSV to: {CLASSIFICATION_REPORT_CSV_PATH}")
    
    # 10. Generate and Save Confusion Matrix Plot
    print("\n[Step 10] Generating confusion matrix plot...")
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Locked Test Set Confusion Matrix (Tuned LightGBM)")
    plt.xlabel("Predicted Risk Label")
    plt.ylabel("Actual Risk Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG_PATH, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {CONFUSION_MATRIX_PNG_PATH}")
    
    print("\nSTAGE 1 ML EVALUATION COMPLETED SUCCESSFULLY!")
    return metrics_dict, cm, df_preds if 'df_preds' in locals() else None


if __name__ == "__main__":
    run_evaluation()
