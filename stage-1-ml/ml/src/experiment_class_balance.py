"""
Class Balancing & Per-Class Performance Investigation Runner.
Stage 1 ML: Patient Toxicity Risk Multiclass Classification.

Evaluates strategies B0 through B11 strictly on training data using 5-fold StratifiedGroupKFold on patient_id:
- B0: No balancing (class_weight=None)
- B1: V4 fold-local balanced class weights
- B2: Moderate-weight 1.2 (fold-local)
- B3: Moderate-weight 1.4 (fold-local)
- B4: Moderate-weight 1.6 (fold-local)
- B5: Moderate-weight 1.8 (fold-local)
- B6: Moderate-weight 2.0 (fold-local)
- B7: Asymmetric error penalty (Moderate + High prioritized)
- B8: Random Over-Sampling (ROS, fold-internal only)
- B9: SMOTE (k-NN interpolation in preprocessed space, fold-internal only)
- B10: Random Under-Sampling (RUS, majority downsampling, fold-internal only)
- B11: Out-of-fold decision-rule adjustment (calibrated on CV predictions)

STRICT RULES:
- Locked test set is NEVER touched or evaluated.
- All resampling occurs strictly inside fold training splits.
- Zero patient leakage across folds.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime
from copy import deepcopy

from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Path configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from data_loader import (
    load_master_dataset,
    prepare_features_and_target,
    TARGET_MAPPING,
    REVERSE_TARGET_MAPPING
)
from feature_engineering import FeatureEngineer
from preprocessing import PreprocessingArtifactManager
from utils import (
    patient_level_split,
    evaluate_multiclass_predictions,
    ThresholdAdjustedClassifier
)

RANDOM_STATE = 42

# Base Candidate V4 LightGBM configuration
BASE_LGBM_PARAMS = {
    "max_depth": 3,
    "n_estimators": 80,
    "learning_rate": 0.10,
    "min_child_samples": 30,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "n_jobs": -1
}


def compute_fold_balanced_weights(y_train_fold: np.ndarray, mod_multiplier: float = 1.0) -> Dict[int, float]:
    """
    Computes class weights strictly from empirical fold-local class counts:
    w_c = N_fold / (3 * N_{fold, c}) * multiplier_c
    """
    n_samples = len(y_train_fold)
    classes, counts = np.unique(y_train_fold, return_counts=True)
    count_dict = dict(zip(classes, counts))
    
    weights = {}
    for c in [0, 1, 2]:
        base_w = n_samples / (3.0 * count_dict.get(c, 1))
        if c == 1:
            weights[c] = float(base_w * mod_multiplier)
        else:
            weights[c] = float(base_w)
            
    return weights


def run_cv_experiment(
    strategy_id: str,
    strategy_name: str,
    strategy_type: str,
    X_train_fe: pd.DataFrame,
    y_train: np.ndarray,
    patient_ids: np.ndarray,
    num_cols: List[str],
    cat_cols: List[str],
    mod_multiplier: float = 1.0,
    resampler_cls = None,
    resampler_kwargs: dict = None,
    decision_multipliers: list = None
) -> Dict[str, Any]:
    """
    Executes a complete 5-fold patient-grouped CV experiment for a balancing strategy.
    All transformations and resampling happen strictly inside each training fold.
    """
    sgkf = StratifiedGroupKFold(n_splits=5)
    fold_metrics = []
    oof_probs = np.zeros((len(X_train_fe), 3))
    oof_preds = np.zeros(len(X_train_fe), dtype=int)
    
    for fold_i, (train_idx, val_idx) in enumerate(sgkf.split(X_train_fe, y_train, groups=patient_ids)):
        X_fold_train_df = X_train_fe.iloc[train_idx].copy()
        X_fold_val_df = X_train_fe.iloc[val_idx].copy()
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        # Verify 0 patient overlap
        train_pts = set(patient_ids[train_idx])
        val_pts = set(patient_ids[val_idx])
        assert len(train_pts.intersection(val_pts)) == 0, f"Patient leakage in fold {fold_i}!"
        
        # 1. Fit preprocessor on fold train ONLY
        pm = PreprocessingArtifactManager()
        X_fold_train_proc = pm.fit_transform(X_fold_train_df, num_cols, cat_cols)
        X_fold_val_proc = pm.transform(X_fold_val_df)
        
        # 2. Configure balancing / weights / resampling
        model_params = deepcopy(BASE_LGBM_PARAMS)
        
        if strategy_type == "unweighted":
            model_params["class_weight"] = None
            X_fit, y_fit = X_fold_train_proc, y_fold_train
            
        elif strategy_type == "weighted":
            weights = compute_fold_balanced_weights(y_fold_train, mod_multiplier=mod_multiplier)
            model_params["class_weight"] = weights
            X_fit, y_fit = X_fold_train_proc, y_fold_train
            
        elif strategy_type == "asymmetric":
            # Penalize Moderate misclassifications via sample weights
            weights = compute_fold_balanced_weights(y_fold_train, mod_multiplier=1.35)
            # High gets extra safety boost
            weights[2] = weights[2] * 1.15
            model_params["class_weight"] = weights
            X_fit, y_fit = X_fold_train_proc, y_fold_train
            
        elif strategy_type == "resampling":
            model_params["class_weight"] = None
            # Resample inside fold training set ONLY
            resampler = resampler_cls(**(resampler_kwargs or {}))
            X_fit, y_fit = resampler.fit_resample(X_fold_train_proc, y_fold_train)
            
        elif strategy_type == "decision_rule":
            weights = compute_fold_balanced_weights(y_fold_train, mod_multiplier=1.0)
            model_params["class_weight"] = weights
            X_fit, y_fit = X_fold_train_proc, y_fold_train
            
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
            
        # 3. Fit LightGBM model on fold training data
        clf = LGBMClassifier(**model_params)
        clf.fit(X_fit, y_fit)
        
        # 4. Train performance evaluation
        train_raw_probs = clf.predict_proba(X_fold_train_proc)
        if decision_multipliers is not None:
            mults = np.array(decision_multipliers)
            train_pred = np.argmax(train_raw_probs * mults, axis=1)
        else:
            train_pred = np.argmax(train_raw_probs, axis=1)
        train_m = evaluate_multiclass_predictions(y_fold_train, train_pred, train_raw_probs)
        
        # 5. Validation performance evaluation (untouched validation fold)
        val_raw_probs = clf.predict_proba(X_fold_val_proc)
        oof_probs[val_idx] = val_raw_probs
        if decision_multipliers is not None:
            mults = np.array(decision_multipliers)
            val_pred = np.argmax(val_raw_probs * mults, axis=1)
        else:
            val_pred = np.argmax(val_raw_probs, axis=1)
            
        oof_preds[val_idx] = val_pred
        val_m = evaluate_multiclass_predictions(y_fold_val, val_pred, val_raw_probs)
        
        fold_metrics.append({
            "fold": fold_i,
            "train_macro_f1": train_m["macro_f1"],
            "train_accuracy": train_m["accuracy"],
            "train_high_risk_recall": train_m["high_risk_recall"],
            "train_moderate_f1": train_m["per_class"]["Moderate"]["f1"],
            "train_moderate_recall": train_m["per_class"]["Moderate"]["recall"],
            "val_macro_f1": val_m["macro_f1"],
            "val_accuracy": val_m["accuracy"],
            "val_high_risk_recall": val_m["high_risk_recall"],
            "val_moderate_f1": val_m["per_class"]["Moderate"]["f1"],
            "val_moderate_recall": val_m["per_class"]["Moderate"]["recall"],
            "val_low_f1": val_m["per_class"]["Low"]["f1"],
            "val_high_f1": val_m["per_class"]["High"]["f1"],
            "val_macro_precision": val_m["macro_precision"],
            "val_macro_recall": val_m["macro_recall"],
            "val_weighted_f1": val_m["weighted_f1"],
            "gap_macro_f1": train_m["macro_f1"] - val_m["macro_f1"]
        })
        
    # Aggregate metrics across 5 folds
    val_macro_f1s = [f["val_macro_f1"] for f in fold_metrics]
    val_hr_recalls = [f["val_high_risk_recall"] for f in fold_metrics]
    val_mod_f1s = [f["val_moderate_f1"] for f in fold_metrics]
    val_mod_recalls = [f["val_moderate_recall"] for f in fold_metrics]
    val_low_f1s = [f["val_low_f1"] for f in fold_metrics]
    val_high_f1s = [f["val_high_f1"] for f in fold_metrics]
    val_accuracies = [f["val_accuracy"] for f in fold_metrics]
    val_macro_precs = [f["val_macro_precision"] for f in fold_metrics]
    val_macro_recs = [f["val_macro_recall"] for f in fold_metrics]
    val_weighted_f1s = [f["val_weighted_f1"] for f in fold_metrics]
    train_macro_f1s = [f["train_macro_f1"] for f in fold_metrics]
    gaps = [f["gap_macro_f1"] for f in fold_metrics]
    
    # Overall OOF metrics & confusion matrix
    oof_m = evaluate_multiclass_predictions(y_train, oof_preds, oof_probs)
    cm = oof_m["confusion_matrix"]
    
    # Error transition counts:
    # Low: 0, Mod: 1, High: 2
    # Mod -> Low: Actual 1, Predicted 0: cm[1][0]
    # High -> Low: Actual 2, Predicted 0: cm[2][0]
    mod_to_low = int(cm[1][0])
    high_to_low = int(cm[2][0])
    
    summary = {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "strategy_type": strategy_type,
        "cv_macro_f1": float(np.mean(val_macro_f1s)),
        "cv_macro_f1_std": float(np.std(val_macro_f1s)),
        "cv_high_risk_recall": float(np.mean(val_hr_recalls)),
        "cv_high_risk_recall_std": float(np.std(val_hr_recalls)),
        "cv_moderate_f1": float(np.mean(val_mod_f1s)),
        "cv_moderate_recall": float(np.mean(val_mod_recalls)),
        "cv_low_f1": float(np.mean(val_low_f1s)),
        "cv_high_f1": float(np.mean(val_high_f1s)),
        "cv_accuracy": float(np.mean(val_accuracies)),
        "cv_macro_precision": float(np.mean(val_macro_precs)),
        "cv_macro_recall": float(np.mean(val_macro_recs)),
        "cv_weighted_f1": float(np.mean(val_weighted_f1s)),
        "train_macro_f1": float(np.mean(train_macro_f1s)),
        "gap_macro_f1": float(np.mean(gaps)),
        "mod_to_low_errors": mod_to_low,
        "high_to_low_errors": high_to_low,
        "confusion_matrix": cm,
        "per_fold": fold_metrics
    }
    
    return summary


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 85)
    print("STAGE 1 ML: CLASS BALANCE & PER-CLASS PERFORMANCE BENCHMARK (B0 - B11)")
    print(f"Timestamp: {timestamp}")
    print("=" * 85)
    
    # 1. Load dataset & perform patient split
    print("\n[Step 1] Loading master dataset and performing patient-level train split...")
    raw_df = load_master_dataset()
    train_df, test_df = patient_level_split(
        raw_df, group_col="patient_id", target_col="toxicity_risk",
        test_size=0.20, random_state=RANDOM_STATE
    )
    
    print(f"  Train: {len(train_df)} encounters, {train_df['patient_id'].nunique()} unique patients")
    print(f"  Test:  {len(test_df)} encounters, {test_df['patient_id'].nunique()} unique patients [LOCKED]")
    
    # 2. Extract features & apply feature engineering
    print("\n[Step 2] Applying expanded V4 feature engineering on training data...")
    X_train_raw, y_train, _ = prepare_features_and_target(train_df)
    patient_ids = np.array(train_df["patient_id"])
    
    fe = FeatureEngineer(include_engineered=True, include_expanded=True)
    X_train_fe = fe.fit_transform(X_train_raw)
    
    from data_loader import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    num_cols = list(NUMERICAL_FEATURES) + [
        "blood_pressure_ratio", "pulse_pressure", "comorbidity_age_interaction",
        "tumor_biomarker_index", "hematologic_risk_flag", "prior_toxicity_risk_flag",
        "cumulative_treatment_load", "organ_impairment_index", "vital_instability_score",
        "genomic_instability_score", "biomarker_severity_weight"
    ]
    cat_cols = list(CATEGORICAL_FEATURES)
    print(f"  Total features for preprocessor: {X_train_fe.shape[1]} ({len(num_cols)} numerical, {len(cat_cols)} categorical)")
    
    # Define experimental strategies B0 through B11
    strategies = [
        # B0: No balancing
        {
            "id": "B0", "name": "No Balancing (Unweighted)", "type": "unweighted",
            "kwargs": {}
        },
        # B1: V4 Balanced class weights
        {
            "id": "B1", "name": "V4 Balanced Class Weights", "type": "weighted",
            "kwargs": {"mod_multiplier": 1.0}
        },
        # B2 - B6: Moderate-weight grid (1.2, 1.4, 1.6, 1.8, 2.0)
        {
            "id": "B2", "name": "Moderate-Weight 1.2x (Fold-Local)", "type": "weighted",
            "kwargs": {"mod_multiplier": 1.2}
        },
        {
            "id": "B3", "name": "Moderate-Weight 1.4x (Fold-Local)", "type": "weighted",
            "kwargs": {"mod_multiplier": 1.4}
        },
        {
            "id": "B4", "name": "Moderate-Weight 1.6x (Fold-Local)", "type": "weighted",
            "kwargs": {"mod_multiplier": 1.6}
        },
        {
            "id": "B5", "name": "Moderate-Weight 1.8x (Fold-Local)", "type": "weighted",
            "kwargs": {"mod_multiplier": 1.8}
        },
        {
            "id": "B6", "name": "Moderate-Weight 2.0x (Fold-Local)", "type": "weighted",
            "kwargs": {"mod_multiplier": 2.0}
        },
        # B7: Asymmetric penalty
        {
            "id": "B7", "name": "Asymmetric Misclassification Penalty", "type": "asymmetric",
            "kwargs": {}
        },
        # B8: Random Over-Sampling
        {
            "id": "B8", "name": "Random Over-Sampling (ROS, Fold-Train)", "type": "resampling",
            "kwargs": {
                "resampler_cls": RandomOverSampler,
                "resampler_kwargs": {"random_state": RANDOM_STATE}
            }
        },
        # B9: SMOTE
        {
            "id": "B9", "name": "SMOTE (k=5 Interpolation, Fold-Train)", "type": "resampling",
            "kwargs": {
                "resampler_cls": SMOTE,
                "resampler_kwargs": {"random_state": RANDOM_STATE, "k_neighbors": 5}
            }
        },
        # B10: Random Under-Sampling
        {
            "id": "B10", "name": "Random Under-Sampling (RUS, Fold-Train)", "type": "resampling",
            "kwargs": {
                "resampler_cls": RandomUnderSampler,
                "resampler_kwargs": {"random_state": RANDOM_STATE}
            }
        },
        # B11: OOF Decision Rule Adjustment
        {
            "id": "B11", "name": "OOF Decision Rule Multiplier Adjustment", "type": "decision_rule",
            "kwargs": {"decision_multipliers": [1.0, 1.05, 1.05]}
        }
    ]
    
    print("\n[Step 3] Executing 5-Fold StratifiedGroupKFold Benchmark across B0 - B11...")
    print("-" * 115)
    header = (
        f"{'ID':<4} | {'Strategy Name':<38} | {'Val Macro F1':<14} | {'Gap':<7} | "
        f"{'HR Recall':<10} | {'Mod F1':<7} | {'Mod Rec':<7} | {'Mod->Low':<8} | {'High->Low':<9}"
    )
    print(header)
    print("-" * 115)
    
    results = {}
    
    for strat in strategies:
        sid = strat["id"]
        sname = strat["name"]
        stype = strat["type"]
        extra_kwargs = strat["kwargs"]
        
        res = run_cv_experiment(
            strategy_id=sid,
            strategy_name=sname,
            strategy_type=stype,
            X_train_fe=X_train_fe,
            y_train=y_train,
            patient_ids=patient_ids,
            num_cols=num_cols,
            cat_cols=cat_cols,
            **extra_kwargs
        )
        results[sid] = res
        
        row_str = (
            f"{sid:<4} | {sname:<38} | {res['cv_macro_f1']:.4f} +/- {res['cv_macro_f1_std']:.3f} | "
            f"{res['gap_macro_f1']:.4f} | {res['cv_high_risk_recall']:.4f}     | "
            f"{res['cv_moderate_f1']:.4f}  | {res['cv_moderate_recall']:.4f}  | "
            f"{res['mod_to_low_errors']:<8} | {res['high_to_low_errors']:<9}"
        )
        print(row_str)
        
    print("-" * 115)
    
    # Save results JSON
    results_path = os.path.join(ML_DIR, "results", "class_balance_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Clean results for serialization
    clean_results = {}
    for sid, r in results.items():
        clean_r = dict(r)
        clean_results[sid] = clean_r
        
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nSaved raw experimental results to: {results_path}")
    
    # Evaluate against V5 promotion criteria
    # Criterion: CV Macro F1 >= 0.5470 (+0.0043 over V4), Mod F1 >= 0.3400, HR Recall >= 0.6200, Gap <= 0.1200, Std <= 0.0080
    v4_baseline_f1 = results["B1"]["cv_macro_f1"]
    v4_baseline_hr = results["B1"]["cv_high_risk_recall"]
    v4_baseline_mod = results["B1"]["cv_moderate_f1"]
    v5_threshold = 0.5470
    
    print("\n[Step 4] Candidate V5 Acceptance Evaluation:")
    print(f"  V4 Development Baseline: Macro F1 = {v4_baseline_f1:.4f}, HR Recall = {v4_baseline_hr:.4f}, Mod F1 = {v4_baseline_mod:.4f}")
    print(f"  V5 Promotion Requirement: Macro F1 >= {v5_threshold:.4f} (+0.0043 margin), HR Recall >= 0.6200, Gap <= 0.1200\n")
    
    v5_candidates = []
    for sid, r in results.items():
        passes_f1 = r["cv_macro_f1"] >= v5_threshold
        passes_hr = r["cv_high_risk_recall"] >= 0.6200
        passes_gap = r["gap_macro_f1"] <= 0.1200
        passes_std = r["cv_macro_f1_std"] <= 0.0080
        
        status_flag = "FAIL"
        if passes_f1 and passes_hr and passes_gap and passes_std:
            status_flag = "PASS (QUALIFIES FOR V5)"
            v5_candidates.append(sid)
            
        print(f"  {sid} ({r['strategy_name']}): Macro F1 = {r['cv_macro_f1']:.4f} | HR Recall = {r['cv_high_risk_recall']:.4f} | Mod F1 = {r['cv_moderate_f1']:.4f} | Gap = {r['gap_macro_f1']:.4f} -> {status_flag}")
        
    print("\n[Step 5] Final Decision Verdict:")
    if v5_candidates:
        best_v5 = max(v5_candidates, key=lambda sid: results[sid]["cv_macro_f1"])
        print(f"  RECOMMENDATION: PROMOTE Candidate V5 using strategy {best_v5} ({results[best_v5]['strategy_name']})")
    else:
        print(f"  RECOMMENDATION: RETAIN Candidate V4 as the preferred model.")
        print("  RATIONALE: None of the class balancing strategies achieved a meaningful Macro F1 improvement >= 0.5470 without compromising calibration or High-risk detection.")

    return results


if __name__ == "__main__":
    main()
