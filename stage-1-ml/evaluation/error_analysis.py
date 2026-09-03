import os
import sys
import numpy as np
import pandas as pd
import joblib

# Ensure stage-1-ml/ml/src is accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "ml", "src"))
if ML_SRC_DIR not in sys.path:
    sys.path.insert(0, ML_SRC_DIR)

from data_loader import load_master_dataset, prepare_features_and_target
from feature_engineering import FeatureEngineer
from preprocessing import PreprocessingArtifactManager
from utils import patient_level_split

MASTER_DATASET_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data-engineering", "data", "processed", "master_patient_dataset.csv"))
MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "ml", "models", "best_model", "model.joblib"))
PREPROCESSOR_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "ml", "artifacts", "preprocessor", "preprocessor.joblib"))

RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
ERROR_ANALYSIS_CSV_PATH = os.path.join(RESULTS_DIR, "error_analysis.csv")

TARGET_MAPPING = {"Low": 0, "Moderate": 1, "High": 2}
REVERSE_TARGET_MAPPING = {0: "Low", 1: "Moderate", 2: "High"}


def run_error_analysis():
    print("=" * 70)
    print("STAGE 1 EVALUATION ENGINEER — ERROR & CONFIDENCE ANALYSIS")
    print("=" * 70)
    
    # 1. Load dataset & test split
    print("\n[Step 1] Loading master dataset and test split...")
    raw_df = pd.read_csv(MASTER_DATASET_PATH)
    train_df, test_df = patient_level_split(raw_df, group_col="patient_id", target_col="toxicity_risk", test_size=0.20, random_state=42)
    
    X_test_raw, y_test_series, meta_test = prepare_features_and_target(test_df)
    y_test_encoded = y_test_series.values
    y_test_labels = test_df["toxicity_risk"].values
    
    # 2. Preprocess & Predict
    print("\n[Step 2] Transforming test data and generating model predictions...")
    fe = FeatureEngineer(include_engineered=True)
    X_test_fe = fe.transform(X_test_raw)
    
    pm = PreprocessingArtifactManager.load(PREPROCESSOR_PATH)
    X_test_proc = pm.transform(X_test_fe)
    
    model = joblib.load(MODEL_PATH)
    y_test_pred_encoded = model.predict(X_test_proc)
    y_test_prob = model.predict_proba(X_test_proc)
    
    y_test_pred_labels = [REVERSE_TARGET_MAPPING[idx] for idx in y_test_pred_encoded]
    
    # 3. Construct Error Analysis DataFrame
    print("\n[Step 3] Constructing detailed error analysis table...")
    records = []
    for i in range(len(test_df)):
        true_label = y_test_labels[i]
        pred_label = y_test_pred_labels[i]
        p_low = float(round(y_test_prob[i, 0], 4))
        p_mod = float(round(y_test_prob[i, 1], 4))
        p_high = float(round(y_test_prob[i, 2], 4))
        confidence = float(round(np.max(y_test_prob[i]), 4))
        
        is_correct = (true_label == pred_label)
        if is_correct:
            error_type = "Correct"
        else:
            error_type = f"{true_label} -> {pred_label}"
            
        records.append({
            "patient_id": meta_test["patient_id"].iloc[i],
            "encounter_id": meta_test["encounter_id"].iloc[i],
            "true_toxicity_risk": true_label,
            "predicted_toxicity_risk": pred_label,
            "probability_low": p_low,
            "probability_moderate": p_mod,
            "probability_high": p_high,
            "confidence": confidence,
            "correct": is_correct,
            "error_type": error_type
        })
        
    df_err = pd.DataFrame(records)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_err.to_csv(ERROR_ANALYSIS_CSV_PATH, index=False)
    print(f"Saved error analysis CSV to: {ERROR_ANALYSIS_CSV_PATH}")
    
    # 4. Error Category Frequency Analysis
    print("\n[Step 4] Analyzing error transition frequencies...")
    error_transitions = [
        "Moderate -> Low",
        "Moderate -> High",
        "High -> Low",
        "High -> Moderate",
        "Low -> Moderate",
        "Low -> High"
    ]
    
    print("-" * 50)
    print("ERROR TRANSITION BREAKDOWN:")
    print("-" * 50)
    total_errors = len(df_err[~df_err["correct"]])
    total_samples = len(df_err)
    print(f"Total Test Samples: {total_samples}")
    print(f"Total Errors:       {total_errors} (Error Rate: {total_errors/total_samples:.2%})")
    print(f"Total Correct:      {total_samples - total_errors} (Accuracy: {(total_samples - total_errors)/total_samples:.2%})\n")
    
    transition_counts = df_err["error_type"].value_counts()
    for trans in error_transitions:
        count = transition_counts.get(trans, 0)
        pct = (count / total_errors) * 100 if total_errors > 0 else 0
        print(f"  {trans:18s} : {count:4d} occurrences ({pct:5.1f}% of errors)")
    print("-" * 50)
    
    # 5. Dedicated High-Risk Analysis
    print("\n[Step 5] Performing High-Risk class analysis...")
    high_df = df_err[df_err["true_toxicity_risk"] == "High"]
    total_high = len(high_df)
    high_tp = len(high_df[high_df["predicted_toxicity_risk"] == "High"])
    high_fn_mod = len(high_df[high_df["predicted_toxicity_risk"] == "Moderate"])
    high_fn_low = len(high_df[high_df["predicted_toxicity_risk"] == "Low"])
    high_fn_total = high_fn_mod + high_fn_low
    
    pred_high_df = df_err[df_err["predicted_toxicity_risk"] == "High"]
    total_pred_high = len(pred_high_df)
    high_fp = total_pred_high - high_tp
    
    high_precision = high_tp / total_pred_high if total_pred_high > 0 else 0
    high_recall = high_tp / total_high if total_high > 0 else 0
    high_f1 = (2 * high_precision * high_recall / (high_precision + high_recall)) if (high_precision + high_recall) > 0 else 0
    
    print("-" * 50)
    print("HIGH RISK CLASS METRICS & ERROR BREAKDOWN:")
    print("-" * 50)
    print(f"  Total Actual High Risk Encounters:  {total_high}")
    print(f"  True Positives (Predicted High):    {high_tp}")
    print(f"  False Positives (Incorrect High):   {high_fp}")
    print(f"  False Negatives (Missed High):      {high_fn_total}")
    print(f"    - High predicted as Moderate:     {high_fn_mod}")
    print(f"    - High predicted as Low:          {high_fn_low}")
    print(f"  High Risk Precision:                {high_precision:.4f}")
    print(f"  High Risk Recall:                   {high_recall:.4f}")
    print(f"  High Risk F1-Score:                 {high_f1:.4f}")
    print("-" * 50)
    
    # 6. Model Confidence Analysis
    print("\n[Step 6] Analyzing prediction confidence vs accuracy...")
    bins = [0.33, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    labels = ["0.33-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
    
    df_err["conf_bin"] = pd.cut(df_err["confidence"], bins=bins, labels=labels, right=True, include_lowest=True)
    
    print("-" * 65)
    print(f"{'Confidence Bin':15s} | {'Count':8s} | {'Pct Total':10s} | {'Accuracy':10s}")
    print("-" * 65)
    for bin_name in labels:
        bin_df = df_err[df_err["conf_bin"] == bin_name]
        bin_count = len(bin_df)
        bin_pct = (bin_count / total_samples) * 100
        bin_acc = bin_df["correct"].mean() if bin_count > 0 else 0.0
        print(f"{bin_name:15s} | {bin_count:8d} | {bin_pct:9.1f}% | {bin_acc:9.2%}")
    print("-" * 65)
    
    print("\nERROR ANALYSIS COMPLETED SUCCESSFULLY!")
    return df_err


if __name__ == "__main__":
    run_error_analysis()
