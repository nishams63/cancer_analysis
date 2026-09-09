"""
Baseline NLP Modeling Module for Stage 3 NLP.
Implements class-weighted linear classifiers for Urgency Level and Hazard Type,
and evaluates entity span extraction on the official VALIDATION partition.
"""

from typing import Dict, Any, Tuple
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
import joblib

from config import (
    ARTIFACTS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED
)
from data_loader import load_train_data, load_validation_data
from label_preparation import TargetLabelManager
from feature_extraction import ClinicalFeaturePipeline
from clinical_concepts import extract_clinical_concepts, evaluate_entity_spans


class ClinicalNLPBaselines:
    """Manages baseline model training, evaluation, and artifact persistence."""

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.urgency_model = LogisticRegression(
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state
        )
        self.hazard_model = LogisticRegression(
            class_weight="balanced",
            C=0.5,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state
        )
        self.feature_pipeline = ClinicalFeaturePipeline()
        self.label_manager = TargetLabelManager()
        self.is_trained = False

    def fit(self, df_train: pd.DataFrame) -> "ClinicalNLPBaselines":
        """Train models strictly on the TRAIN partition."""
        # 1. Fit Label Manager
        self.label_manager.fit(df_train)
        y_urg_train = self.label_manager.transform_urgency(df_train["urgency_level"])
        y_haz_train = self.label_manager.transform_hazard(df_train["hazard_type"])

        # 2. Fit Feature Pipeline
        self.feature_pipeline.fit(df_train)
        X_train, struct_train_df = self.feature_pipeline.transform(df_train)

        # 3. Train Primary Urgency Model
        self.urgency_model.fit(X_train, y_urg_train)

        # 4. Train Secondary Hazard Model
        self.hazard_model.fit(X_train, y_haz_train)

        self.is_trained = True

        # Save intermediate feature outputs
        struct_train_df.to_parquet(OUTPUTS_DIR / "nlp_features_train.parquet")
        return self

    def evaluate_validation(self, df_val: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate baseline models strictly on the official VALIDATION partition."""
        if not self.is_trained:
            raise RuntimeError("Models must be fitted on TRAIN before evaluation!")

        # 1. Transform Validation Data
        X_val, struct_val_df = self.feature_pipeline.transform(df_val)
        struct_val_df.to_parquet(OUTPUTS_DIR / "nlp_features_val.parquet")

        y_urg_val = self.label_manager.transform_urgency(df_val["urgency_level"])
        y_haz_val = self.label_manager.transform_hazard(df_val["hazard_type"])

        # 2. Urgency Predictions & Metrics
        urg_preds = self.urgency_model.predict(X_val)
        urg_probs = self.urgency_model.predict_proba(X_val)
        urg_names = list(self.label_manager.urgency_encoder.classes_)

        urg_metrics = {
            "macro_f1": round(float(f1_score(y_urg_val, urg_preds, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y_urg_val, urg_preds, average="weighted")), 4),
            "accuracy": round(float(accuracy_score(y_urg_val, urg_preds)), 4),
            "classification_report": classification_report(y_urg_val, urg_preds, target_names=urg_names, output_dict=True),
            "confusion_matrix": confusion_matrix(y_urg_val, urg_preds).tolist(),
            "class_names": urg_names
        }

        # 3. Hazard Predictions & Metrics
        haz_preds = self.hazard_model.predict(X_val)
        haz_probs = self.hazard_model.predict_proba(X_val)
        haz_names = list(self.label_manager.hazard_encoder.classes_)

        haz_metrics = {
            "macro_f1": round(float(f1_score(y_haz_val, haz_preds, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_haz_val, haz_preds, average="weighted")), 4),
            "accuracy": round(float(accuracy_score(y_haz_val, haz_preds)), 4),
            "classification_report": classification_report(y_haz_val, haz_preds, target_names=haz_names, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_haz_val, haz_preds).tolist(),
            "class_names": haz_names
        }

        # 4. Clinical Concept Extraction (NER) Evaluation on Validation
        ner_eval_results = []
        for _, row in df_val.iterrows():
            pred_entities = extract_clinical_concepts(row["text"], assign_polarity=True)
            gt_entities = json.loads(row["ner_entities"])
            eval_score = evaluate_entity_spans(pred_entities, gt_entities, match_type="relaxed")
            ner_eval_results.append(eval_score)

        mean_ner_p = float(np.mean([r["precision"] for r in ner_eval_results]))
        mean_ner_r = float(np.mean([r["recall"] for r in ner_eval_results]))
        mean_ner_f1 = float(np.mean([r["f1"] for r in ner_eval_results]))

        ner_metrics = {
            "mean_precision": round(mean_ner_p, 4),
            "mean_recall": round(mean_ner_r, 4),
            "mean_span_f1": round(mean_ner_f1, 4),
            "evaluated_documents": len(ner_eval_results)
        }

        # 5. Save Validation Predictions
        val_preds_df = pd.DataFrame({
            "document_id": df_val["document_id"],
            "patient_id": df_val["patient_id"],
            "ground_truth_urgency": df_val["urgency_level"],
            "predicted_urgency": self.label_manager.inverse_transform_urgency(urg_preds),
            "urgency_confidence": np.max(urg_probs, axis=1).round(4),
            "ground_truth_hazard": df_val["hazard_type"],
            "predicted_hazard": self.label_manager.inverse_transform_hazard(haz_preds),
            "hazard_confidence": np.max(haz_probs, axis=1).round(4)
        })
        val_preds_df.to_csv(PREDICTIONS_DIR / "validation_predictions.csv", index=False)

        metrics_summary = {
            "urgency_classification": urg_metrics,
            "hazard_classification": haz_metrics,
            "entity_extraction_ner": ner_metrics
        }

        # Save Metrics JSON
        with open(METRICS_DIR / "baseline_validation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

        return metrics_summary

    def save(self, artifacts_dir=ARTIFACTS_DIR) -> None:
        """Serialize baseline models and constituent components."""
        joblib.dump(self.urgency_model, artifacts_dir / "urgency_baseline_model.joblib")
        joblib.dump(self.hazard_model, artifacts_dir / "hazard_baseline_model.joblib")
        self.feature_pipeline.save()
        self.label_manager.save()

    @classmethod
    def load(cls, artifacts_dir=ARTIFACTS_DIR) -> "ClinicalNLPBaselines":
        """Load fitted baseline models."""
        baselines = cls()
        baselines.urgency_model = joblib.load(artifacts_dir / "urgency_baseline_model.joblib")
        baselines.hazard_model = joblib.load(artifacts_dir / "hazard_baseline_model.joblib")
        baselines.feature_pipeline = ClinicalFeaturePipeline.load()
        baselines.label_manager = TargetLabelManager.load()
        baselines.is_trained = True
        return baselines


def run_baseline_pipeline() -> Dict[str, Any]:
    """Execute complete end-to-end baseline training and validation."""
    df_train = load_train_data()
    df_val = load_validation_data()

    baselines = ClinicalNLPBaselines()
    baselines.fit(df_train)
    metrics = baselines.evaluate_validation(df_val)
    baselines.save()
    return metrics
