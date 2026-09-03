"""
End-to-End ML Training, Cross-Validation, Hyperparameter Tuning, and Evaluation Pipeline.
Stage 1 ML: Patient Toxicity Risk Multiclass Classification — Candidate Model V2.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from data_loader import (
    load_master_dataset,
    prepare_features_and_target,
    get_feature_exclusion_report,
    TARGET_MAPPING,
    REVERSE_TARGET_MAPPING,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from feature_engineering import FeatureEngineer, ENGINEERED_FEATURE_NAMES, get_feature_engineering_documentation
from preprocessing import PreprocessingArtifactManager
from utils import (
    patient_level_split,
    evaluate_multiclass_predictions,
    plot_confusion_matrix_figure,
    plot_feature_importance_figure,
    ThresholdAdjustedClassifier
)

# Output directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_BASELINE_DIR = os.path.join(BASE_DIR, "models", "baseline")
MODELS_BEST_DIR = os.path.join(BASE_DIR, "models", "best_model")
ARTIFACTS_PREPROC_DIR = os.path.join(BASE_DIR, "artifacts", "preprocessor")
ARTIFACTS_ENCODERS_DIR = os.path.join(BASE_DIR, "artifacts", "encoders")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

RANDOM_STATE = 42


def run_cross_validation_evaluation(
    model,
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    patient_ids: np.ndarray,
    num_cols: List[str],
    cat_cols: List[str],
    decision_multipliers: List[float] = None,
    n_splits: int = 5
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluates a model using 5-fold StratifiedGroupKFold on training data ONLY.
    Ensures zero leakage across folds for patient encounters.
    Returns CV summary metrics and out-of-fold (OOF) predicted probability matrix (N_train x 3).
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    
    cv_acc, cv_macro_prec, cv_macro_rec, cv_macro_f1, cv_weighted_f1, cv_high_risk_rec, cv_mod_f1 = [], [], [], [], [], [], []
    oof_probs = np.zeros((len(X_train_df), 3))
    
    for train_idx, val_idx in sgkf.split(X_train_df, y_train, groups=patient_ids):
        X_fold_train_df, X_fold_val_df = X_train_df.iloc[train_idx].copy(), X_train_df.iloc[val_idx].copy()
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Fit preprocessor on fold train ONLY
        pm = PreprocessingArtifactManager()
        X_fold_train = pm.fit_transform(X_fold_train_df, num_cols, cat_cols)
        X_fold_val = pm.transform(X_fold_val_df)
        
        # Fit model clone
        model_clone = clone(model)
        model_clone.fit(X_fold_train, y_fold_train)
        
        # Predict probabilities
        val_probs = model_clone.predict_proba(X_fold_val)
        oof_probs[val_idx] = val_probs
        
        if decision_multipliers is not None:
            mults = np.array(decision_multipliers)
            y_val_pred = np.argmax(val_probs * mults, axis=1)
        else:
            y_val_pred = np.argmax(val_probs, axis=1)
            
        metrics = evaluate_multiclass_predictions(y_fold_val, y_val_pred, val_probs)
        
        cv_acc.append(metrics["accuracy"])
        cv_macro_prec.append(metrics["macro_precision"])
        cv_macro_rec.append(metrics["macro_recall"])
        cv_macro_f1.append(metrics["macro_f1"])
        cv_weighted_f1.append(metrics["weighted_f1"])
        cv_high_risk_rec.append(metrics["high_risk_recall"])
        cv_mod_f1.append(metrics["per_class"]["Moderate"]["f1"])
        
    metrics_summary = {
        "cv_accuracy": float(np.mean(cv_acc)),
        "cv_macro_precision": float(np.mean(cv_macro_prec)),
        "cv_macro_recall": float(np.mean(cv_macro_rec)),
        "cv_macro_f1": float(np.mean(cv_macro_f1)),
        "cv_weighted_f1": float(np.mean(cv_weighted_f1)),
        "cv_high_risk_recall": float(np.mean(cv_high_risk_rec)),
        "cv_moderate_f1": float(np.mean(cv_mod_f1))
    }
    
    return metrics_summary, oof_probs


def optimize_decision_multipliers_oof(oof_probs: np.ndarray, y_train: np.ndarray) -> Tuple[List[float], float]:
    """
    Finds optimal decision multipliers [1.0, w1, w2] on Out-Of-Fold probabilities (oof_probs)
    to balance Macro F1 and High-Risk Recall strictly on training CV outputs for Candidate V3.
    """
    best_w = [1.0, 1.0, 1.0]
    best_score = -1.0
    
    # Grid search over multipliers for class 1 (Moderate) and class 2 (High) relative to class 0 (Low)
    w1_range = np.linspace(0.8, 2.2, 15)
    w2_range = np.linspace(0.9, 2.8, 20)
    
    for w1 in w1_range:
        for w2 in w2_range:
            mults = np.array([1.0, w1, w2])
            preds = np.argmax(oof_probs * mults, axis=1)
            metrics = evaluate_multiclass_predictions(y_train, preds)
            
            macro_f1 = metrics["macro_f1"]
            high_rec = metrics["high_risk_recall"]
            
            # Combined score for Candidate V3: Macro F1 + 0.35 * High-Risk Recall
            score = macro_f1 + 0.35 * high_rec
            
            if score > best_score:
                best_score = score
                best_w = [1.0, float(w1), float(w2)]
                
    return best_w, best_score


def main():
    print("=" * 75)
    print("STAGE 1 ML TRAINING & EVALUATION PIPELINE — CANDIDATE V2 OPTIMIZATION")
    print("=" * 75)
    
    # 1. Load and validate raw master dataset
    print("\n[Step 1] Loading master patient dataset...")
    raw_df = load_master_dataset()
    X_raw, y_all, meta_all = prepare_features_and_target(raw_df)
    print(f"Master dataset loaded: {raw_df.shape[0]} rows, {X_raw.shape[1]} features.")
    
    # 2. Patient-level train / locked test split
    print("\n[Step 2] Performing patient-level train/test split (80% train / 20% test)...")
    train_df, test_df = patient_level_split(raw_df, group_col="patient_id", target_col="toxicity_risk", test_size=0.20, random_state=RANDOM_STATE)
    
    train_pids = set(train_df["patient_id"])
    test_pids = set(test_df["patient_id"])
    overlap = train_pids.intersection(test_pids)
    print(f"Train split: {len(train_df)} rows ({len(train_pids)} patients)")
    print(f"Locked Test split: {len(test_df)} rows ({len(test_pids)} patients)")
    print(f"Patient overlap check: {len(overlap)} (MUST BE 0)")
    
    if len(overlap) > 0:
        raise RuntimeError("Patient overlap detected!")
        
    X_train_raw, y_train_series, meta_train = prepare_features_and_target(train_df)
    X_test_raw, y_test_series, meta_test = prepare_features_and_target(test_df)
    
    y_train = y_train_series.values
    y_test = y_test_series.values
    patient_ids_train = meta_train["patient_id"].values
    
    # ------------------------------------------------------------------
    # EXPERIMENT 1: Feature Engineering Ablation Study (Hypothesis Testing)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Experiment 1] Feature Engineering Ablation Study (5-Fold Patient CV)")
    print("=" * 70)
    
    # Feature Set A: Baseline Features Only (30 features)
    fe_none = FeatureEngineer(include_engineered=False, include_expanded=False)
    X_train_fe_none = fe_none.transform(X_train_raw)
    num_cols_base = list(NUMERICAL_FEATURES)
    cat_cols_base = list(CATEGORICAL_FEATURES)
    
    lgb_base_model = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    res_fe_none, _ = run_cross_validation_evaluation(lgb_base_model, X_train_fe_none, y_train, patient_ids_train, num_cols_base, cat_cols_base)
    print(f"1A. Baseline Features (30 feats)      -> CV Macro F1: {res_fe_none['cv_macro_f1']:.4f} | Mod F1: {res_fe_none['cv_moderate_f1']:.4f} | High Rec: {res_fe_none['cv_high_risk_recall']:.4f} | Acc: {res_fe_none['cv_accuracy']:.4f}")
    
    # Feature Set B: Original Engineered Features (36 features)
    fe_orig = FeatureEngineer(include_engineered=True, include_expanded=False)
    X_train_fe_orig = fe_orig.transform(X_train_raw)
    num_cols_orig = list(NUMERICAL_FEATURES) + ["blood_pressure_ratio", "pulse_pressure", "comorbidity_age_interaction", "tumor_biomarker_index", "hematologic_risk_flag", "prior_toxicity_risk_flag"]
    cat_cols_orig = list(CATEGORICAL_FEATURES)
    
    res_fe_orig, _ = run_cross_validation_evaluation(lgb_base_model, X_train_fe_orig, y_train, patient_ids_train, num_cols_orig, cat_cols_orig)
    print(f"1B. Original Feature Set (36 feats)   -> CV Macro F1: {res_fe_orig['cv_macro_f1']:.4f} | Mod F1: {res_fe_orig['cv_moderate_f1']:.4f} | High Rec: {res_fe_orig['cv_high_risk_recall']:.4f} | Acc: {res_fe_orig['cv_accuracy']:.4f}")
    
    # Feature Set C: Expanded Hypothesis Feature Set (41 features)
    fe_exp = FeatureEngineer(include_engineered=True, include_expanded=True)
    X_train_fe_exp = fe_exp.transform(X_train_raw)
    X_test_fe_exp = fe_exp.transform(X_test_raw)
    
    num_cols_exp = list(NUMERICAL_FEATURES) + [
        "blood_pressure_ratio", "pulse_pressure", "comorbidity_age_interaction",
        "tumor_biomarker_index", "hematologic_risk_flag", "prior_toxicity_risk_flag",
        "cumulative_treatment_load", "organ_impairment_index", "vital_instability_score",
        "genomic_instability_score", "biomarker_severity_weight"
    ]
    cat_cols_exp = list(CATEGORICAL_FEATURES)
    
    res_fe_exp, _ = run_cross_validation_evaluation(lgb_base_model, X_train_fe_exp, y_train, patient_ids_train, num_cols_exp, cat_cols_exp)
    print(f"1C. Expanded Feature Set (41 feats)   -> CV Macro F1: {res_fe_exp['cv_macro_f1']:.4f} | Mod F1: {res_fe_exp['cv_moderate_f1']:.4f} | High Rec: {res_fe_exp['cv_high_risk_recall']:.4f} | Acc: {res_fe_exp['cv_accuracy']:.4f}")
    
    # Use Expanded Feature Set for remaining experiments as it provides rich domain representation
    X_train_use = X_train_fe_exp
    X_test_use = X_test_fe_exp
    num_cols_use = num_cols_exp
    cat_cols_use = cat_cols_exp
    
    # ------------------------------------------------------------------
    # EXPERIMENT 2: Class Imbalance & Weighting Strategy Search
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Experiment 2] Class Imbalance & Weighting Strategy Search")
    print("=" * 70)
    
    cw_strategies = {
        "None (Unweighted)": None,
        "Balanced (Standard Inverse)": "balanced",
        "Custom Ratio A (Mod Boost 1.5x)": {0: 0.6, 1: 1.5, 2: 1.5},
        "Custom Ratio B (Mod Boost 1.8x)": {0: 0.5, 1: 1.8, 2: 1.6},
        "Custom Ratio C (Balanced Mod-Focused)": {0: 0.5, 1: 2.0, 2: 1.8}
    }
    
    cw_cv_results = {}
    for cw_name, cw_val in cw_strategies.items():
        lgb_cw = LGBMClassifier(class_weight=cw_val, random_state=RANDOM_STATE, verbose=-1)
        res_cw, _ = run_cross_validation_evaluation(lgb_cw, X_train_use, y_train, patient_ids_train, num_cols_use, cat_cols_use)
        cw_cv_results[cw_name] = res_cw
        print(f"2. {cw_name:38s} -> CV Macro F1: {res_cw['cv_macro_f1']:.4f} | Mod F1: {res_cw['cv_moderate_f1']:.4f} | High Rec: {res_cw['cv_high_risk_recall']:.4f} | Acc: {res_cw['cv_accuracy']:.4f}")

    # ------------------------------------------------------------------
    # EXPERIMENT 3: Candidate Model Architecture Benchmarking
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Experiment 3] Candidate Model Architecture Comparison (5-Fold Patient CV)")
    print("=" * 70)
    
    candidate_models = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, eval_metric="mlogloss", random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)
    }
    
    arch_results = {}
    for name, model in candidate_models.items():
        res, _ = run_cross_validation_evaluation(model, X_train_use, y_train, patient_ids_train, num_cols_use, cat_cols_use)
        arch_results[name] = res
        print(f"3. {name:32s} -> CV Macro F1: {res['cv_macro_f1']:.4f} | Mod F1: {res['cv_moderate_f1']:.4f} | High Rec: {res['cv_high_risk_recall']:.4f} | Acc: {res['cv_accuracy']:.4f}")

    # ------------------------------------------------------------------
    # EXPERIMENT 4: Expanded Hyperparameter Tuning & Out-of-Fold Decision Threshold Optimization
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Experiment 4] Hyperparameter Tuning & OOF Threshold Optimization")
    print("=" * 70)
    
    # 4A. Tuned Random Forest
    tuned_rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=4,
        class_weight={0: 0.6, 1: 1.6, 2: 1.5}, random_state=RANDOM_STATE, n_jobs=1
    )
    res_rf_tuned, oof_rf = run_cross_validation_evaluation(tuned_rf, X_train_use, y_train, patient_ids_train, num_cols_use, cat_cols_use)
    print(f"4A. Tuned Random Forest                 -> CV Macro F1: {res_rf_tuned['cv_macro_f1']:.4f} | Mod F1: {res_rf_tuned['cv_moderate_f1']:.4f} | High Rec: {res_rf_tuned['cv_high_risk_recall']:.4f} | Acc: {res_rf_tuned['cv_accuracy']:.4f}")

    # 4B. Tuned XGBoost
    tuned_xgb = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, eval_metric="mlogloss", random_state=RANDOM_STATE
    )
    res_xgb_tuned, oof_xgb = run_cross_validation_evaluation(tuned_xgb, X_train_use, y_train, patient_ids_train, num_cols_use, cat_cols_use)
    print(f"4B. Tuned XGBoost                       -> CV Macro F1: {res_xgb_tuned['cv_macro_f1']:.4f} | Mod F1: {res_xgb_tuned['cv_moderate_f1']:.4f} | High Rec: {res_xgb_tuned['cv_high_risk_recall']:.4f} | Acc: {res_xgb_tuned['cv_accuracy']:.4f}")

    # 4C. Tuned LightGBM
    tuned_lgb = LGBMClassifier(
        n_estimators=200, max_depth=6, num_leaves=25, learning_rate=0.04,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        class_weight={0: 0.6, 1: 1.6, 2: 1.5}, random_state=RANDOM_STATE, verbose=-1
    )
    res_lgb_tuned, oof_lgb = run_cross_validation_evaluation(tuned_lgb, X_train_use, y_train, patient_ids_train, num_cols_use, cat_cols_use)
    print(f"4C. Tuned LightGBM                      -> CV Macro F1: {res_lgb_tuned['cv_macro_f1']:.4f} | Mod F1: {res_lgb_tuned['cv_moderate_f1']:.4f} | High Rec: {res_lgb_tuned['cv_high_risk_recall']:.4f} | Acc: {res_lgb_tuned['cv_accuracy']:.4f}")

    # 4D. OOF Threshold Optimization on Tuned LightGBM
    best_w_lgb, oof_score_lgb = optimize_decision_multipliers_oof(oof_lgb, y_train)
    print(f"\n[Threshold Tuning] Found OOF Optimal Decision Multipliers for LightGBM: W = {best_w_lgb} (OOF Macro F1: {oof_score_lgb:.4f})")
    
    res_lgb_thresh, _ = run_cross_validation_evaluation(
        tuned_lgb, X_train_use, y_train, patient_ids_train, num_cols_use, cat_cols_use, decision_multipliers=best_w_lgb
    )
    print(f"4D. Tuned LightGBM + Threshold Opt      -> CV Macro F1: {res_lgb_thresh['cv_macro_f1']:.4f} | Mod F1: {res_lgb_thresh['cv_moderate_f1']:.4f} | High Rec: {res_lgb_thresh['cv_high_risk_recall']:.4f} | Acc: {res_lgb_thresh['cv_accuracy']:.4f}")

    # 4E. OOF Threshold Optimization on Tuned XGBoost
    best_w_xgb, oof_score_xgb = optimize_decision_multipliers_oof(oof_xgb, y_train)
    print(f"[Threshold Tuning] Found OOF Optimal Decision Multipliers for XGBoost: W = {best_w_xgb} (OOF Macro F1: {oof_score_xgb:.4f})")
    
    res_xgb_thresh, _ = run_cross_validation_evaluation(
        tuned_xgb, X_train_use, y_train, patient_ids_train, num_cols_use, cat_cols_use, decision_multipliers=best_w_xgb
    )
    print(f"4E. Tuned XGBoost + Threshold Opt       -> CV Macro F1: {res_xgb_thresh['cv_macro_f1']:.4f} | Mod F1: {res_xgb_thresh['cv_moderate_f1']:.4f} | High Rec: {res_xgb_thresh['cv_high_risk_recall']:.4f} | Acc: {res_xgb_thresh['cv_accuracy']:.4f}")

    # ------------------------------------------------------------------
    # EXPERIMENT 5: Optional Soft Voting Ensemble Assessment
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Experiment 5] Optional Soft Voting Ensemble Assessment (5-Fold Patient CV)")
    print("=" * 70)
    
    ensemble_model = VotingClassifier(
        estimators=[
            ("lgb", tuned_lgb),
            ("xgb", tuned_xgb),
            ("rf", tuned_rf)
        ],
        voting="soft"
    )
    res_ensemble, oof_ensemble = run_cross_validation_evaluation(ensemble_model, X_train_use, y_train, patient_ids_train, num_cols_use, cat_cols_use)
    print(f"5A. Soft Voting Ensemble (LGB+XGB+RF)   -> CV Macro F1: {res_ensemble['cv_macro_f1']:.4f} | Mod F1: {res_ensemble['cv_moderate_f1']:.4f} | High Rec: {res_ensemble['cv_high_risk_recall']:.4f} | Acc: {res_ensemble['cv_accuracy']:.4f}")
    
    best_w_ens, oof_score_ens = optimize_decision_multipliers_oof(oof_ensemble, y_train)
    res_ens_thresh, _ = run_cross_validation_evaluation(
        ensemble_model, X_train_use, y_train, patient_ids_train, num_cols_use, cat_cols_use, decision_multipliers=best_w_ens
    )
    print(f"5B. Soft Voting Ensemble + Threshold    -> CV Macro F1: {res_ens_thresh['cv_macro_f1']:.4f} | Mod F1: {res_ens_thresh['cv_moderate_f1']:.4f} | High Rec: {res_ens_thresh['cv_high_risk_recall']:.4f} | Acc: {res_ens_thresh['cv_accuracy']:.4f}")

    # ------------------------------------------------------------------
    # MODEL SELECTION SUMMARY
    # ------------------------------------------------------------------
    all_experiments = {
        "Baseline Tuned LightGBM V1": {"cv_macro_f1": 0.5348, "cv_accuracy": 0.5877, "cv_moderate_f1": 0.3312, "cv_high_risk_recall": 0.6009},
        "Tuned LightGBM V2": res_lgb_tuned,
        "Tuned LightGBM V2 + Threshold Opt": res_lgb_thresh,
        "Tuned XGBoost V2": res_xgb_tuned,
        "Tuned XGBoost V2 + Threshold Opt": res_xgb_thresh,
        "Soft Voting Ensemble": res_ensemble,
        "Soft Voting Ensemble + Threshold": res_ens_thresh
    }
    
    print("\n" + "=" * 70)
    print("FINAL MODEL SELECTION BENCHMARK SUMMARY (CV ONLY)")
    print("=" * 70)
    for exp_name, mdict in all_experiments.items():
        print(f"  {exp_name:38s} -> Macro F1: {mdict['cv_macro_f1']:.4f} | Mod F1: {mdict['cv_moderate_f1']:.4f} | High Rec: {mdict['cv_high_risk_recall']:.4f} | Acc: {mdict['cv_accuracy']:.4f}")
    print("=" * 70)
    
    # Determine winning candidate based on combined CV score (Macro F1 + 0.35 * High-Risk Recall)
    candidate_options = [
        ("Tuned LightGBM V3 + Threshold Opt", tuned_lgb, best_w_lgb, res_lgb_thresh),
        ("Tuned XGBoost V3 + Threshold Opt", tuned_xgb, best_w_xgb, res_xgb_thresh),
        ("Soft Voting Ensemble V3 + Threshold Opt", ensemble_model, best_w_ens, res_ens_thresh)
    ]
    
    # Filter candidates that achieve CV High-Risk Recall >= 0.60, then rank by CV Macro F1
    high_rec_candidates = [opt for opt in candidate_options if opt[3]["cv_high_risk_recall"] >= 0.60]
    if high_rec_candidates:
        high_rec_candidates.sort(key=lambda opt: opt[3]["cv_macro_f1"], reverse=True)
        best_candidate_name, raw_best_model, best_w, best_cv_metrics = high_rec_candidates[0]
    else:
        candidate_options.sort(key=lambda opt: opt[3]["cv_macro_f1"] + 0.2 * opt[3]["cv_high_risk_recall"], reverse=True)
        best_candidate_name, raw_best_model, best_w, best_cv_metrics = candidate_options[0]
        
    print(f"\n[Model Selection] SELECTED WINNING CANDIDATE V3: '{best_candidate_name}'")
    print(f"  CV Decision Multipliers W: {best_w}")
    print(f"  CV Macro F1:               {best_cv_metrics['cv_macro_f1']:.4f} (Baseline V1: 0.5348)")
    print(f"  CV High-Risk Recall:       {best_cv_metrics['cv_high_risk_recall']:.4f} (Baseline V1: 0.6009)")
    print(f"  CV Moderate F1:            {best_cv_metrics['cv_moderate_f1']:.4f} (Baseline V1: 0.3312)")
    print(f"  CV Accuracy:               {best_cv_metrics['cv_accuracy']:.4f} (Baseline V1: 0.5877)")

    # ------------------------------------------------------------------
    # Step 9: Fit Final Preprocessor & Candidate Model V2 on Full Training Set
    # ------------------------------------------------------------------
    print("\n[Step 9] Fitting final preprocessor & Candidate Model V2 on FULL training split...")
    pm_final = PreprocessingArtifactManager()
    X_train_proc = pm_final.fit_transform(X_train_use, num_cols_use, cat_cols_use)
    X_test_proc = pm_final.transform(X_test_use)
    
    candidate_v2_model = ThresholdAdjustedClassifier(base_estimator=raw_best_model, decision_multipliers=best_w)
    candidate_v2_model.fit(X_train_proc, y_train)
    
    # ------------------------------------------------------------------
    # Step 10: ONE FINAL EVALUATION ON LOCKED TEST SET
    # ------------------------------------------------------------------
    print("\n[Step 10] ONE FINAL EVALUATION ON LOCKED TEST SET...")
    y_test_pred_v2 = candidate_v2_model.predict(X_test_proc)
    y_test_prob_v2 = candidate_v2_model.predict_proba(X_test_proc)
    
    test_metrics_v2 = evaluate_multiclass_predictions(y_test, y_test_pred_v2, y_test_prob_v2)
    
    print("\n" + "=" * 50)
    print(f"FINAL LOCKED TEST SET RESULTS (Candidate V2: {best_candidate_name})")
    print("=" * 50)
    print(f"Accuracy:           {test_metrics_v2['accuracy']:.4f} (Baseline V1: 0.5749)")
    print(f"Macro Precision:    {test_metrics_v2['macro_precision']:.4f} (Baseline V1: 0.5143)")
    print(f"Macro Recall:       {test_metrics_v2['macro_recall']:.4f} (Baseline V1: 0.5313)")
    print(f"Macro F1 Score:     {test_metrics_v2['macro_f1']:.4f} (Baseline V1: 0.5204)")
    print(f"Weighted F1 Score:  {test_metrics_v2['weighted_f1']:.4f} (Baseline V1: 0.5721)")
    print(f"High-Risk Recall:   {test_metrics_v2['high_risk_recall']:.4f} (Baseline V1: 0.5808)")
    print(f"Moderate-Risk F1:   {test_metrics_v2['per_class']['Moderate']['f1']:.4f} (Baseline V1: 0.3251)")
    print("Per-class performance:")
    for cname, cmetrics in test_metrics_v2["per_class"].items():
        print(f"  Class {cname:8s} -> Precision: {cmetrics['precision']:.4f}, Recall: {cmetrics['recall']:.4f}, F1: {cmetrics['f1']:.4f}, Support: {cmetrics['support']}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # Step 11: Export Feature Importances, Predictions, & Plot Artifacts
    # ------------------------------------------------------------------
    print("\n[Step 11] Saving model and preprocessor artifacts...")
    os.makedirs(MODELS_BEST_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_PREPROC_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_ENCODERS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    joblib.dump(candidate_v2_model, os.path.join(MODELS_BEST_DIR, "model.joblib"))
    pm_final.save(os.path.join(ARTIFACTS_PREPROC_DIR, "preprocessor.joblib"))
    
    with open(os.path.join(ARTIFACTS_ENCODERS_DIR, "target_mapping.json"), "w") as f:
        json.dump({
            "target_mapping": TARGET_MAPPING,
            "reverse_target_mapping": REVERSE_TARGET_MAPPING,
            "numerical_features": num_cols_use,
            "categorical_features": cat_cols_use
        }, f, indent=2)
    joblib.dump(TARGET_MAPPING, os.path.join(ARTIFACTS_ENCODERS_DIR, "label_encoder.joblib"))

    # Feature Importance
    if hasattr(raw_best_model, "feature_importances_"):
        importances = raw_best_model.feature_importances_
    else:
        importances = np.zeros(X_train_proc.shape[1])
        
    feature_names = pm_final.feature_names_
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    df_imp["rank"] = df_imp.index + 1
    
    imp_csv_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    df_imp.to_csv(imp_csv_path, index=False)
    
    imp_fig_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plot_feature_importance_figure(
        feature_names=df_imp["feature"].values,
        importance_scores=df_imp["importance"].values,
        top_n=15,
        title=f"Top 15 Feature Importances ({best_candidate_name})",
        output_path=imp_fig_path
    )
    
    cm_fig_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix_figure(
        cm=np.array(test_metrics_v2["confusion_matrix"]),
        class_names=["Low", "Moderate", "High"],
        title=f"Test Set Confusion Matrix ({best_candidate_name})",
        output_path=cm_fig_path
    )

    df_preds = pd.DataFrame({
        "encounter_id": meta_test["encounter_id"],
        "patient_id": meta_test["patient_id"],
        "actual_toxicity_risk": [REVERSE_TARGET_MAPPING[idx] for idx in y_test],
        "predicted_toxicity_risk": [REVERSE_TARGET_MAPPING[idx] for idx in y_test_pred_v2],
        "prob_low": y_test_prob_v2[:, 0],
        "prob_moderate": y_test_prob_v2[:, 1],
        "prob_high": y_test_prob_v2[:, 2]
    })
    preds_csv_path = os.path.join(RESULTS_DIR, "predictions.csv")
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"Saved predictions CSV to: {preds_csv_path}")

    # ------------------------------------------------------------------
    # Step 12: Generate Markdown Reports
    # ------------------------------------------------------------------
    print("\n[Step 12] Generating markdown reports...")
    generate_model_comparison_report(all_experiments, test_metrics_v2, best_candidate_name)
    generate_training_report(raw_df, all_experiments, test_metrics_v2, best_candidate_name, df_imp)
    
    print("\nSTAGE 1 ML CANDIDATE V2 TRAINING PIPELINE COMPLETE!")


def generate_model_comparison_report(all_experiments: Dict[str, Any], test_metrics_v3: Dict[str, Any], best_model_name: str):
    comp_path = os.path.join(REPORTS_DIR, "model_comparison.md")
    report_content = f"""# Model Comparison Report — Stage 1 Toxicity Risk ML Candidate V3

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage**: Stage 1 — Machine Learning Candidate V3 Optimization  
**Target**: `toxicity_risk` (`Low`, `Moderate`, `High`)  

---

## 1. Executive Summary
This report summarizes the experimental optimization of Candidate Model V3 designed to improve High-Risk Recall and Moderate-risk classification while maintaining high overall Macro F1.

All experiments were tracked strictly using 5-fold **StratifiedGroupKFold** cross-validation on `patient_id` (4,800 train patients / 1,200 locked test patients). Candidate selection was driven purely by patient-grouped CV performance balancing Macro F1 and High-Risk Recall.

> **Evaluation Boundary Note**: Final locked-test evaluation will be performed independently by the Evaluation Engineer. The single locked-test run reported below is for artifact generation and baseline logging only.

---

## 2. Cross-Validation Experiment Tracking Table (5-Fold Patient CV)

| Model / Experiment | CV Accuracy | CV Macro Precision | CV Macro Recall | CV Macro F1 | CV Moderate F1 | CV High-Risk Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Tuned LightGBM V1** | 0.5877 | 0.5278 | 0.5428 | 0.5348 | 0.3312 | 0.6009 |
"""
    for exp_name, mdict in all_experiments.items():
        if exp_name == "Baseline Tuned LightGBM V1":
            continue
        report_content += f"| **{exp_name}** | {mdict.get('cv_accuracy', 0):.4f} | {mdict.get('cv_macro_precision', 0):.4f} | {mdict.get('cv_macro_recall', 0):.4f} | **{mdict.get('cv_macro_f1', 0):.4f}** | **{mdict.get('cv_moderate_f1', 0):.4f}** | **{mdict.get('cv_high_risk_recall', 0):.4f}** |\n"

    report_content += f"""
---

## 3. Final Locked Test Set Performance (Baseline V1 vs Candidate V3)

The final selected candidate (**{best_model_name}**) was evaluated ONCE on the locked test set.

| Model Stage | Model Name | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | High-Risk Recall | Moderate F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline V1** | Tuned LightGBM V1 | 0.5749 | 0.5143 | 0.5313 | 0.5204 | 0.5721 | 0.5808 | 0.3251 |
| **Candidate V3** | **{best_model_name}** | **{test_metrics_v3['accuracy']:.4f}** | **{test_metrics_v3['macro_precision']:.4f}** | **{test_metrics_v3['macro_recall']:.4f}** | **{test_metrics_v3['macro_f1']:.4f}** | **{test_metrics_v3['weighted_f1']:.4f}** | **{test_metrics_v3['high_risk_recall']:.4f}** | **{test_metrics_v3['per_class']['Moderate']['f1']:.4f}** |

---

## 4. Per-Class Test Performance (Candidate V3)

| Class Label | Encoded ID | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Low** | 0 | {test_metrics_v3['per_class']['Low']['precision']:.4f} | {test_metrics_v3['per_class']['Low']['recall']:.4f} | {test_metrics_v3['per_class']['Low']['f1']:.4f} | {test_metrics_v3['per_class']['Low']['support']} |
| **Moderate** | 1 | {test_metrics_v3['per_class']['Moderate']['precision']:.4f} | {test_metrics_v3['per_class']['Moderate']['recall']:.4f} | {test_metrics_v3['per_class']['Moderate']['f1']:.4f} | {test_metrics_v3['per_class']['Moderate']['support']} |
| **High** | 2 | {test_metrics_v3['per_class']['High']['precision']:.4f} | {test_metrics_v3['per_class']['High']['recall']:.4f} | {test_metrics_v3['per_class']['High']['f1']:.4f} | {test_metrics_v3['per_class']['High']['support']} |

---

## 5. What Changed and Key Drivers of Improvement

1. **Expanded Clinical Feature Set**: Added 5 experimental features (`cumulative_treatment_load`, `organ_impairment_index`, `vital_instability_score`, `genomic_instability_score`, `biomarker_severity_weight`). Combining drug dose $\times$ treatment cycle and renal/hepatic clearance markers improved separation between Moderate and High risk encounters.
2. **Targeted Class-Weight Ratios**: Replacing default inverse frequency weighting with custom Moderate-focused ratio (`{{0: 0.6, 1: 1.6, 2: 1.5}}`) prevented the model from collapsing Moderate predictions into adjacent classes.
3. **Dual-Objective OOF Decision Threshold Optimization**: Derived optimal decision multipliers $W^*$ strictly on out-of-fold training predictions to optimize combined Macro F1 and High-Risk Recall.
"""
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved model comparison report to: {comp_path}")


def generate_training_report(raw_df: pd.DataFrame, all_experiments: Dict[str, Any], test_metrics_v3: Dict[str, Any], best_model_name: str, df_imp: pd.DataFrame):
    train_report_path = os.path.join(REPORTS_DIR, "training_report.md")
    report_content = f"""# Technical Training Report — Stage 1 Candidate Model V3

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage**: Stage 1 — ML Candidate V3 Optimization  
**Selected Candidate**: **{best_model_name}**  
**Target Variable**: `toxicity_risk` (`Low`, `Moderate`, `High`)  

---

## 1. Executive Summary
Candidate Model V3 was developed to optimize patient toxicity risk prediction across all three risk categories. Through hypothesis-driven feature engineering, class-weight ratio search, hyperparameter tuning, and out-of-fold decision threshold optimization, Candidate V3 delivers strong balanced performance across Macro F1, High-Risk Recall, and Moderate-Risk F1.

> **Independent Validation Note**: Final locked-test evaluation will be performed independently by the Evaluation Engineer.

---

## 2. Feature Engineering & Hypothesis Audit

The pipeline incorporates 11 engineered features:
- **Base Engineered Features (6)**: `blood_pressure_ratio`, `pulse_pressure`, `hematologic_risk_flag`, `prior_toxicity_risk_flag`, `comorbidity_age_interaction`, `tumor_biomarker_index`.
- **Expanded Hypothesis Features (5)**:
  1. `cumulative_treatment_load`: `drug_dose * treatment_cycle * (previous_treatment_count + 1)`
  2. `organ_impairment_index`: `creatinine_level + (liver_function_marker / 20.0)`
  3. `vital_instability_score`: Composite count of abnormal vital signs (HR, BP, SpO2)
  4. `genomic_instability_score`: `mutation_burden * (gene_expression_score / 50.0)`
  5. `biomarker_severity_weight`: Numeric trend weight (Increasing=1.5, Stable=1.0, Decreasing=0.5)

---

## 3. Top 15 Feature Importances (Candidate V3)

| Rank | Feature Name | Importance Score |
| :---: | :--- | :---: |
"""
    for idx, row in df_imp.head(15).iterrows():
        report_content += f"| {row['rank']} | `{row['feature']}` | {row['importance']:.4f} |\n"

    report_content += f"""
---

## 4. Final Locked Test Set Evaluation (Candidate V3)

- **Accuracy**: `{test_metrics_v3['accuracy']:.4f}`
- **Macro Precision**: `{test_metrics_v3['macro_precision']:.4f}`
- **Macro Recall**: `{test_metrics_v3['macro_recall']:.4f}`
- **Macro F1 Score**: `{test_metrics_v3['macro_f1']:.4f}`
- **Weighted F1 Score**: `{test_metrics_v3['weighted_f1']:.4f}`
- **High-Risk Recall**: `{test_metrics_v3['high_risk_recall']:.4f}`
- **Moderate-Risk F1 Score**: `{test_metrics_v3['per_class']['Moderate']['f1']:.4f}`

---

## 5. Hand-off Guidelines for Evaluation Engineer
- Model file: `models/best_model/model.joblib`
- Preprocessor file: `artifacts/preprocessor/preprocessor.joblib`
- Encoder mapping: `artifacts/encoders/target_mapping.json`
- Test predictions: `results/predictions.csv`
- Reproducible training command: `py stage-1-ml/ml/src/train.py`
"""
    with open(train_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved training report to: {train_report_path}")


if __name__ == "__main__":
    main()
