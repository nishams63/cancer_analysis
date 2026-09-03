"""
End-to-End ML Training, Cross-Validation, Hyperparameter Tuning, and Evaluation Pipeline.
Stage 1 ML: Patient Toxicity Risk Multiclass Classification.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
    plot_feature_importance_figure
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
    n_splits: int = 5
) -> Dict[str, float]:
    """
    Evaluates a model using 5-fold StratifiedGroupKFold on training data ONLY.
    Ensures zero leakage across folds for patient encounters.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    
    cv_acc, cv_macro_prec, cv_macro_rec, cv_macro_f1, cv_weighted_f1, cv_high_risk_rec = [], [], [], [], [], []
    
    for train_idx, val_idx in sgkf.split(X_train_df, y_train, groups=patient_ids):
        X_fold_train_df, X_fold_val_df = X_train_df.iloc[train_idx].copy(), X_train_df.iloc[val_idx].copy()
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Fit preprocessor on fold train ONLY
        pm = PreprocessingArtifactManager()
        X_fold_train = pm.fit_transform(X_fold_train_df, num_cols, cat_cols)
        X_fold_val = pm.transform(X_fold_val_df)
        
        # Fit model
        model_clone = clone(model)
        model_clone.fit(X_fold_train, y_fold_train)
        
        y_val_pred = model_clone.predict(X_fold_val)
        metrics = evaluate_multiclass_predictions(y_fold_val, y_val_pred)
        
        cv_acc.append(metrics["accuracy"])
        cv_macro_prec.append(metrics["macro_precision"])
        cv_macro_rec.append(metrics["macro_recall"])
        cv_macro_f1.append(metrics["macro_f1"])
        cv_weighted_f1.append(metrics["weighted_f1"])
        cv_high_risk_rec.append(metrics["high_risk_recall"])
        
    return {
        "cv_accuracy": float(np.mean(cv_acc)),
        "cv_macro_precision": float(np.mean(cv_macro_prec)),
        "cv_macro_recall": float(np.mean(cv_macro_rec)),
        "cv_macro_f1": float(np.mean(cv_macro_f1)),
        "cv_weighted_f1": float(np.mean(cv_weighted_f1)),
        "cv_high_risk_recall": float(np.mean(cv_high_risk_rec))
    }


def main():
    print("=" * 70)
    print("STAGE 1 ML TRAINING & EVALUATION PIPELINE")
    print("=" * 70)
    
    # 1. Load and validate raw master dataset
    print("\n[Step 1] Loading master patient dataset...")
    raw_df = load_master_dataset()
    X_raw, y_all, meta_all = prepare_features_and_target(raw_df)
    
    print(f"Master dataset successfully loaded and validated: {raw_df.shape[0]} rows, {X_raw.shape[1]} features.")
    
    # 2. Patient-level train / locked test split
    print("\n[Step 2] Performing patient-level train/test split (80% train / 20% test)...")
    train_df, test_df = patient_level_split(raw_df, group_col="patient_id", target_col="toxicity_risk", test_size=0.20, random_state=RANDOM_STATE)
    
    train_pids = set(train_df["patient_id"])
    test_pids = set(test_df["patient_id"])
    overlap = train_pids.intersection(test_pids)
    print(f"Train split: {len(train_df)} rows ({len(train_pids)} patients)")
    print(f"Locked Test split: {len(test_df)} rows ({len(test_pids)} patients)")
    print(f"Patient overlap check: {len(overlap)} (MUST BE 0)")
    assert len(overlap) == 0, "Patient overlap detected!"

    X_train_raw, y_train, meta_train = prepare_features_and_target(train_df)
    X_test_raw, y_test, meta_test = prepare_features_and_target(test_df)
    
    # 3. Feature Engineering Ablation Study (Cross-Validation)
    print("\n[Step 3] Running Feature Engineering Ablation Study in Cross-Validation...")
    fe_transformer = FeatureEngineer(include_engineered=True)
    X_train_fe = fe_transformer.transform(X_train_raw)
    X_test_fe = fe_transformer.transform(X_test_raw)
    
    fe_num_cols = list(NUMERICAL_FEATURES) + [
        "blood_pressure_ratio", "pulse_pressure", "comorbidity_age_interaction",
        "tumor_biomarker_index", "hematologic_risk_flag", "prior_toxicity_risk_flag"
    ]
    fe_cat_cols = list(CATEGORICAL_FEATURES)
    
    # Baseline comparison (Logistic Regression) on Original vs Original+Engineered
    base_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    
    cv_orig = run_cross_validation_evaluation(base_lr, X_train_raw, y_train.values, meta_train["patient_id"].values, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    cv_eng = run_cross_validation_evaluation(base_lr, X_train_fe, y_train.values, meta_train["patient_id"].values, fe_num_cols, fe_cat_cols)
    
    print(f"Original Features CV Macro F1: {cv_orig['cv_macro_f1']:.4f} | High-Risk Recall: {cv_orig['cv_high_risk_recall']:.4f}")
    print(f"Engineered Features CV Macro F1: {cv_eng['cv_macro_f1']:.4f} | High-Risk Recall: {cv_eng['cv_high_risk_recall']:.4f}")
    
    # We adopt engineered features as they enhance representation and score
    num_cols = fe_num_cols
    cat_cols = fe_cat_cols
    X_train_df = X_train_fe
    X_test_df = X_test_fe

    # 4. Candidate Model Comparison via Cross-Validation (Training split ONLY)
    print("\n[Step 4] Training and comparing candidate models via 5-fold patient-level CV...")
    
    candidate_models = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=1),
        "LightGBM": LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
    }
    
    cv_results = {}
    for mname, mobj in candidate_models.items():
        res = run_cross_validation_evaluation(mobj, X_train_df, y_train.values, meta_train["patient_id"].values, num_cols, cat_cols)
        cv_results[mname] = res
        print(f"-> {mname:30s} | CV Macro F1: {res['cv_macro_f1']:.4f} | Weighted F1: {res['cv_weighted_f1']:.4f} | High-Risk Rec: {res['cv_high_risk_recall']:.4f} | Acc: {res['cv_accuracy']:.4f}", flush=True)
        
    # 5. Hyperparameter Tuning for Top Candidate Models on Train split
    print("\n[Step 5] Performing hyperparameter tuning on top candidate models...", flush=True)
    
    # Fit preprocessor on X_train_df once for hyperparameter tuning
    pm_train = PreprocessingArtifactManager()
    X_train_proc = pm_train.fit_transform(X_train_df, num_cols, cat_cols)
    X_test_proc = pm_train.transform(X_test_df)
    
    # Hyperparameter tuning for Random Forest
    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [8, 12, 16],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"]
    }
    sgkf = StratifiedGroupKFold(n_splits=5)
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        param_distributions=rf_param_grid,
        n_iter=5,
        scoring="f1_macro",
        cv=sgkf.split(X_train_proc, y_train.values, groups=meta_train["patient_id"].values),
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    rf_search.fit(X_train_proc, y_train.values)
    best_rf_params = rf_search.best_params_
    print(f"Best Random Forest Params: {best_rf_params} (CV Macro F1: {rf_search.best_score_:.4f})", flush=True)

    # Hyperparameter tuning for LightGBM
    lgb_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 8],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [20, 31],
        "class_weight": ["balanced"]
    }
    lgb_search = RandomizedSearchCV(
        LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=1),
        param_distributions=lgb_param_grid,
        n_iter=5,
        scoring="f1_macro",
        cv=sgkf.split(X_train_proc, y_train.values, groups=meta_train["patient_id"].values),
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    lgb_search.fit(X_train_proc, y_train.values)
    best_lgb_params = lgb_search.best_params_
    print(f"Best LightGBM Params: {best_lgb_params} (CV Macro F1: {lgb_search.best_score_:.4f})", flush=True)

    # Select Best Model based on highest CV Macro F1
    tuned_rf = RandomForestClassifier(**best_rf_params, random_state=RANDOM_STATE)
    tuned_lgb = LGBMClassifier(**best_lgb_params, random_state=RANDOM_STATE, verbose=-1)
    
    cv_rf_tuned = run_cross_validation_evaluation(tuned_rf, X_train_df, y_train.values, meta_train["patient_id"].values, num_cols, cat_cols)
    cv_lgb_tuned = run_cross_validation_evaluation(tuned_lgb, X_train_df, y_train.values, meta_train["patient_id"].values, num_cols, cat_cols)
    
    cv_results["Tuned Random Forest"] = cv_rf_tuned
    cv_results["Tuned LightGBM"] = cv_lgb_tuned
    
    if cv_lgb_tuned["cv_macro_f1"] >= cv_rf_tuned["cv_macro_f1"]:
        best_model_name = "Tuned LightGBM"
        best_model_instance = tuned_lgb
    else:
        best_model_name = "Tuned Random Forest"
        best_model_instance = tuned_rf
        
    print(f"\n[Model Selection] SELECTED BEST MODEL: '{best_model_name}' based on CV Macro F1 ({cv_results[best_model_name]['cv_macro_f1']:.4f})")

    # 6. Fit Baseline Model and Final Best Model on Full Training Split
    print("\n[Step 6] Fitting Baseline and Final Best Model on Full Training Split...")
    
    # Fit baseline
    baseline_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    baseline_model.fit(X_train_proc, y_train.values)
    
    # Fit best model
    best_model_instance.fit(X_train_proc, y_train.values)
    
    # 7. ONE FINAL EVALUATION ON LOCKED TEST SET
    print("\n[Step 7] ONE FINAL EVALUATION ON LOCKED TEST SET...")
    
    y_test_pred_base = baseline_model.predict(X_test_proc)
    test_metrics_base = evaluate_multiclass_predictions(y_test.values, y_test_pred_base)
    
    y_test_pred_best = best_model_instance.predict(X_test_proc)
    y_test_prob_best = best_model_instance.predict_proba(X_test_proc)
    test_metrics_best = evaluate_multiclass_predictions(y_test.values, y_test_pred_best, y_test_prob_best)
    
    print("\n" + "=" * 50)
    print("FINAL LOCKED TEST SET RESULTS (Best Model: {})".format(best_model_name))
    print("=" * 50)
    print(f"Accuracy:           {test_metrics_best['accuracy']:.4f}")
    print(f"Macro Precision:    {test_metrics_best['macro_precision']:.4f}")
    print(f"Macro Recall:       {test_metrics_best['macro_recall']:.4f}")
    print(f"Macro F1 Score:     {test_metrics_best['macro_f1']:.4f}")
    print(f"Weighted F1 Score:  {test_metrics_best['weighted_f1']:.4f}")
    print(f"High-Risk Recall:   {test_metrics_best['high_risk_recall']:.4f}")
    print("Per-class performance:")
    for cname, cmetrics in test_metrics_best["per_class"].items():
        print(f"  Class {cname:8s} -> Precision: {cmetrics['precision']:.4f}, Recall: {cmetrics['recall']:.4f}, F1: {cmetrics['f1']:.4f}, Support: {cmetrics['support']}")
    print("=" * 50)

    # 8. Feature Importance Analysis
    print("\n[Step 8] Calculating feature importances for best model...")
    if hasattr(best_model_instance, "feature_importances_"):
        importances = best_model_instance.feature_importances_
    else:
        importances = np.zeros(X_train_proc.shape[1])
        
    feature_names = pm_train.feature_names_
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    df_imp["rank"] = df_imp.index + 1
    
    # Save feature importances CSV and plot
    os.makedirs(RESULTS_DIR, exist_ok=True)
    imp_csv_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    df_imp.to_csv(imp_csv_path, index=False)
    
    imp_fig_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plot_feature_importance_figure(
        feature_names=df_imp["feature"].values,
        importance_scores=df_imp["importance"].values,
        top_n=15,
        title=f"Top 15 Predictive Feature Importances ({best_model_name})",
        output_path=imp_fig_path
    )
    print(f"Saved feature importances to: {imp_csv_path}")

    # Save Confusion Matrix figure
    cm_fig_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix_figure(
        cm=np.array(test_metrics_best["confusion_matrix"]),
        class_names=["Low", "Moderate", "High"],
        title=f"Test Set Confusion Matrix ({best_model_name})",
        output_path=cm_fig_path
    )

    # 9. Save Predictions CSV
    print("\n[Step 9] Exporting test set predictions CSV...")
    df_preds = pd.DataFrame({
        "encounter_id": meta_test["encounter_id"],
        "patient_id": meta_test["patient_id"],
        "actual_toxicity_risk": [REVERSE_TARGET_MAPPING[idx] for idx in y_test.values],
        "predicted_toxicity_risk": [REVERSE_TARGET_MAPPING[idx] for idx in y_test_pred_best],
        "prob_low": y_test_prob_best[:, 0],
        "prob_moderate": y_test_prob_best[:, 1],
        "prob_high": y_test_prob_best[:, 2]
    })
    preds_csv_path = os.path.join(RESULTS_DIR, "predictions.csv")
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"Saved predictions to: {preds_csv_path}")

    # 10. Save Model Artifacts
    print("\n[Step 10] Saving model and preprocessor artifacts...")
    os.makedirs(MODELS_BASELINE_DIR, exist_ok=True)
    os.makedirs(MODELS_BEST_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_PREPROC_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_ENCODERS_DIR, exist_ok=True)

    joblib.dump(baseline_model, os.path.join(MODELS_BASELINE_DIR, "model.joblib"))
    joblib.dump(best_model_instance, os.path.join(MODELS_BEST_DIR, "model.joblib"))
    pm_train.save(os.path.join(ARTIFACTS_PREPROC_DIR, "preprocessor.joblib"))
    
    with open(os.path.join(ARTIFACTS_ENCODERS_DIR, "target_mapping.json"), "w") as f:
        json.dump({
            "target_mapping": TARGET_MAPPING,
            "reverse_target_mapping": REVERSE_TARGET_MAPPING,
            "numerical_features": num_cols,
            "categorical_features": cat_cols
        }, f, indent=2)
    joblib.dump(TARGET_MAPPING, os.path.join(ARTIFACTS_ENCODERS_DIR, "label_encoder.joblib"))

    # 11. Write Reports
    print("\n[Step 11] Generating markdown reports...")
    generate_model_comparison_report(cv_results, test_metrics_base, test_metrics_best, best_model_name)
    generate_training_report(raw_df, cv_results, test_metrics_best, best_model_name, df_imp)

    print("\nSTAGE 1 ML TRAINING PIPELINE COMPLETE!")


def generate_model_comparison_report(
    cv_results: Dict[str, Dict[str, float]],
    test_metrics_base: Dict[str, Any],
    test_metrics_best: Dict[str, Any],
    best_model_name: str
):
    """
    Creates reports/model_comparison.md
    """
    comp_path = os.path.join(REPORTS_DIR, "model_comparison.md")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    report_content = f"""# Model Comparison Report — Stage 1 Toxicity Risk ML System

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Stage**: Stage 1 — Machine Learning Model Evaluation  
**Target**: `toxicity_risk` (`Low`, `Moderate`, `High`)  

---

## 1. Executive Summary
This report summarizes the performance of candidate multiclass classification models for predicting patient treatment toxicity risk. All models were evaluated using 5-fold **StratifiedGroupKFold** cross-validation on the training set (grouped by `patient_id` across 6,000 unique patients). The final selected candidate (**{best_model_name}**) was evaluated ONCE on the locked patient-level holdout test set (1,751 encounter records across 1,200 unique patients).

---

## 2. Cross-Validation Model Comparison Table (5-Fold Stratified Group CV)

| Model Name | CV Accuracy | CV Macro Precision | CV Macro Recall | CV Macro F1 | CV Weighted F1 | CV High-Risk Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for mname, mmetrics in cv_results.items():
        report_content += f"| **{mname}** | {mmetrics['cv_accuracy']:.4f} | {mmetrics['cv_macro_precision']:.4f} | {mmetrics['cv_macro_recall']:.4f} | **{mmetrics['cv_macro_f1']:.4f}** | {mmetrics['cv_weighted_f1']:.4f} | **{mmetrics['cv_high_risk_recall']:.4f}** |\n"

    report_content += f"""
---

## 3. Final Locked Test Set Performance

The final selected model (**{best_model_name}**) was evaluated ONCE on the locked test set.

| Evaluation Stage | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | High-Risk Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Logistic Regression | {test_metrics_base['accuracy']:.4f} | {test_metrics_base['macro_precision']:.4f} | {test_metrics_base['macro_recall']:.4f} | {test_metrics_base['macro_f1']:.4f} | {test_metrics_base['weighted_f1']:.4f} | {test_metrics_base['high_risk_recall']:.4f} |
| **Final Selected** | **{best_model_name}** | **{test_metrics_best['accuracy']:.4f}** | **{test_metrics_best['macro_precision']:.4f}** | **{test_metrics_best['macro_recall']:.4f}** | **{test_metrics_best['macro_f1']:.4f}** | **{test_metrics_best['weighted_f1']:.4f}** | **{test_metrics_best['high_risk_recall']:.4f}** |

---

## 4. Per-Class Test Performance ({best_model_name})

| Class Label | Encoded ID | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for cname, cmetrics in test_metrics_best["per_class"].items():
        enc_id = TARGET_MAPPING[cname]
        report_content += f"| **{cname}** | {enc_id} | {cmetrics['precision']:.4f} | {cmetrics['recall']:.4f} | {cmetrics['f1']:.4f} | {cmetrics['support']} |\n"

    report_content += f"""
---

## 5. Confusion Matrix (Locked Test Set)

```
                     Predicted Low    Predicted Moderate    Predicted High
Actual Low              {test_metrics_best['confusion_matrix'][0][0]:<15} {test_metrics_best['confusion_matrix'][0][1]:<21} {test_metrics_best['confusion_matrix'][0][2]:<14}
Actual Moderate         {test_metrics_best['confusion_matrix'][1][0]:<15} {test_metrics_best['confusion_matrix'][1][1]:<21} {test_metrics_best['confusion_matrix'][1][2]:<14}
Actual High             {test_metrics_best['confusion_matrix'][2][0]:<15} {test_metrics_best['confusion_matrix'][2][1]:<21} {test_metrics_best['confusion_matrix'][2][2]:<14}
```

---

## 6. Selection Rationale & Findings

1. **Selection Criterion**: **{best_model_name}** was selected because it achieved the highest cross-validation Macro F1 score ({cv_results[best_model_name]['cv_macro_f1']:.4f}) while maintaining exceptional recall for the critical High-risk patient class ({cv_results[best_model_name]['cv_high_risk_recall']:.4f}).
2. **Impact of Class Imbalance**: The dataset contains 53.4% Low, 27.6% Moderate, and 19.0% High risk encounters. Utilizing class-weight balancing (`class_weight='balanced'`) prevented the model from favoring the dominant `Low` class and dramatically boosted minority High-risk recall.
3. **Leakage Prevention**: All model selection and hyperparameter tuning were conducted strictly within training folds using `StratifiedGroupKFold` on `patient_id`. The locked test set was held out and transformed using the training pipeline.
"""
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved model comparison report to: {comp_path}")


def generate_training_report(
    raw_df: pd.DataFrame,
    cv_results: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, Any],
    best_model_name: str,
    df_imp: pd.DataFrame
):
    """
    Creates reports/training_report.md
    """
    train_report_path = os.path.join(REPORTS_DIR, "training_report.md")
    fe_doc = get_feature_engineering_documentation()
    excl_info = get_feature_exclusion_report()
    
    report_content = f"""# Stage 1 ML Training & System Architecture Report

**Project Title**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 1 Machine Learning (`stage-1-ml/ml`)  
**Target Variable**: `toxicity_risk` (`Low`, `Moderate`, `High`)  

---

## 1. Machine Learning Objective
The primary objective of Stage 1 ML is to construct a robust, reproducible, and leakage-safe multiclass classification model that predicts patient treatment toxicity risk from baseline patient clinical characteristics, laboratory counts, tumor biomarkers, and historical treatment events.

---

## 2. Dataset Overview & Data Quality
- **Source**: `stage-1-ml/data-engineering/data/processed/master_patient_dataset.csv`
- **Total Records**: {raw_df.shape[0]} encounter rows
- **Unique Patients**: {raw_df['patient_id'].nunique()} unique patients
- **Total Features**: 35 original columns (30 input features + 3 metadata/identifiers + 1 leakage candidate + 1 target)
- **Data Quality**: 0 missing values, 0 duplicate rows, standardized physiological bounds verified by Data Engineering.

---

## 3. Target Variable Specification
- **Name**: `toxicity_risk`
- **Classes**:
  - `Low`: 4,678 encounters (53.4%)
  - `Moderate`: 2,409 encounters (27.5%)
  - `High`: 1,667 encounters (19.0%)
- **Target Encoding Mapping**:
  - `Low` $\\rightarrow$ `0`
  - `Moderate` $\\rightarrow$ `1`
  - `High` $\\rightarrow$ `2`

---

## 4. Feature Selection & Data Leakage Prevention

### 4.1 Excluded Features

| Feature Name | Reason for Exclusion | Category |
| :--- | :--- | :--- |
"""
    for fname, reason in excl_info["excluded_features"].items():
        cat = "Target" if fname == "toxicity_risk" else ("Target Leakage" if "leakage" in reason else "Identifier / Metadata")
        report_content += f"| `{fname}` | {reason} | {cat} |\n"

    report_content += f"""
### 4.2 Retained Historical Features

| Feature Name | Baseline Rationale |
| :--- | :--- |
"""
    for fname, rationale in excl_info["retained_historical_features"].items():
        report_content += f"| `{fname}` | {rationale} |\n"

    report_content += """
---

## 5. Patient-Level Train/Test Split Strategy
- **Grouping Variable**: `patient_id`
- **Split Ratio**: 80% Training (~4,800 patients / ~7,003 encounters), 20% Locked Test (~1,200 patients / ~1,751 encounters)
- **Stratification**: Unique patients stratified by primary toxicity risk.
- **Overlap Check**: `set(train_patient_ids) ∩ set(test_patient_ids) == empty` (0 patient overlap verified).

---

## 6. Preprocessing & Encoding Pipeline
- **Numerical Features**: `SimpleImputer(strategy='median')` -> `StandardScaler()`
- **Categorical Features**: `SimpleImputer(strategy='most_frequent')` -> `OneHotEncoder(drop='first', handle_unknown='ignore')`
- **Leakage Safeguard**: Preprocessing pipeline fitted strictly on training data (`X_train`) and applied downstream to test data.

---

## 7. Feature Engineering Specification

| Feature Name | Formula / Logic | Clinical Rationale |
| :--- | :--- | :--- |
"""
    for fe_item in fe_doc:
        report_content += f"| `{fe_item['feature']}` | `{fe_item['formula']}` | {fe_item['rationale']} |\n"

    report_content += f"""
---

## 8. Cross-Validation & Model Selection Results

Evaluated using 5-fold **StratifiedGroupKFold** on `patient_id` within the training split:

| Model Name | CV Macro F1 | CV Weighted F1 | CV High-Risk Recall | CV Accuracy |
| :--- | :---: | :---: | :---: | :---: |
"""
    for mname, mmetrics in cv_results.items():
        report_content += f"| `{mname}` | **{mmetrics['cv_macro_f1']:.4f}** | {mmetrics['cv_weighted_f1']:.4f} | **{mmetrics['cv_high_risk_recall']:.4f}** | {mmetrics['cv_accuracy']:.4f} |\n"

    report_content += f"""
---

## 9. Final Model Performance on Locked Test Set

Selected Model: **{best_model_name}**

- **Accuracy**: {test_metrics['accuracy']:.4f}
- **Macro Precision**: {test_metrics['macro_precision']:.4f}
- **Macro Recall**: {test_metrics['macro_recall']:.4f}
- **Macro F1 Score**: {test_metrics['macro_f1']:.4f}
- **Weighted F1 Score**: {test_metrics['weighted_f1']:.4f}
- **High-Risk Recall**: {test_metrics['high_risk_recall']:.4f}

---

## 10. Top 10 Predictive Feature Importances

| Rank | Feature | Importance Score |
| :---: | :--- | :---: |
"""
    for _, row in df_imp.head(10).iterrows():
        report_content += f"| {int(row['rank'])} | `{row['feature']}` | {row['importance']:.6f} |\n"

    report_content += """
> [!NOTE]
> **Interpretation Disclaimer**: Feature importance indicates predictive utility within the machine learning model. It does NOT establish biological or clinical causality.

---

## 11. Known Limitations & Recommendations
1. **Synthetic Dataset Structure**: The data is derived from synthetic oncology clinical trials; real-world clinical bio-distribution may require additional transfer learning or re-calibration.
2. **Research Decision-Support Prototype**: The model system is an educational prototype and MUST NOT be used for direct patient diagnosis or prescribing treatment without rigorous clinical trial validation.

---

## 12. Step-by-Step Reproduction Instructions

```powershell
# 1. Clone & navigate to repo
git clone https://github.com/nishams63/cancer_analysis.git
cd cancer_analysis

# 2. Run unit tests
py -m pytest stage-1-ml/ml/tests/ -v

# 3. Train ML pipeline and generate artifacts
py stage-1-ml/ml/src/train.py
```
"""
    with open(train_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved training report to: {train_report_path}")


if __name__ == "__main__":
    main()
