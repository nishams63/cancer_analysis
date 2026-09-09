"""
Configuration and Path Management for Stage 3 Clinical NLP Evaluation.
Enforces strict read-only access to upstream artifacts and provides centralized paths.
"""

from pathlib import Path
from typing import Dict, List, Set
import yaml

# Base directory for the evaluation module
EVAL_MODULE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = EVAL_MODULE_DIR / "configs" / "evaluation_config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    EVAL_CONFIG = yaml.safe_load(f)

# Upstream Data Paths (Read-Only)
UPSTREAM_DATA_DIR = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["upstream_data_dir"]).resolve()
RAW_PARQUET_PATH = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["raw_dataset_path"]).resolve()
TRAIN_PARQUET_PATH = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["train_split_path"]).resolve()
VAL_PARQUET_PATH = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["validation_split_path"]).resolve()
LOCKED_TEST_PARQUET_PATH = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["locked_test_split_path"]).resolve()

# Upstream NLP Paths & Frozen Artifacts (Read-Only)
NLP_ROOT_DIR = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["nlp_root_dir"]).resolve()
NLP_ARTIFACTS_DIR = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["nlp_artifacts_dir"]).resolve()
NLP_TOKENIZERS_DIR = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["nlp_tokenizers_dir"]).resolve()
NLP_ENCODERS_DIR = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["nlp_encoders_dir"]).resolve()

URGENCY_MODEL_PATH = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["nlp_urgency_model"]).resolve()
HAZARD_MODEL_PATH = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["nlp_hazard_model"]).resolve()
TFIDF_VECTORIZER_PATH = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["nlp_tfidf_vectorizer"]).resolve()
FEATURE_SCALER_PATH = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["nlp_feature_scaler"]).resolve()
NUMERIC_COLS_PATH = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["nlp_numeric_cols"]).resolve()
URGENCY_ENCODER_PATH = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["nlp_urgency_encoder"]).resolve()
HAZARD_ENCODER_PATH = (EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["nlp_hazard_encoder"]).resolve()

# Compatibility Aliases for Upstream NLP modules that import from config
TOKENIZERS_DIR = NLP_TOKENIZERS_DIR
OUTPUTS_DIR = (NLP_ROOT_DIR / "data" / "outputs").resolve()
ARTIFACTS_DIR = NLP_ARTIFACTS_DIR

# Evaluation Output Directories
RESULTS_DIR = EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["results_dir"]
RESULTS_VAL_DIR = EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["results_validation_dir"]
RESULTS_TEST_DIR = EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["results_locked_test_dir"]
RESULTS_METRICS_DIR = EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["results_metrics_dir"]
RESULTS_PRED_DIR = EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["results_predictions_dir"]
RESULTS_CM_DIR = EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["results_confusion_matrices_dir"]
RESULTS_ERROR_DIR = EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["results_error_analysis_dir"]
RESULTS_ROBUST_DIR = EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["results_robustness_dir"]
RESULTS_CALIB_DIR = EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["results_calibration_dir"]
FIGURES_DIR = EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["figures_dir"]
REPORTS_DIR = EVAL_MODULE_DIR / EVAL_CONFIG["paths"]["reports_dir"]

# Ensure output directories exist
for d in [
    RESULTS_DIR, RESULTS_VAL_DIR, RESULTS_TEST_DIR, RESULTS_METRICS_DIR,
    RESULTS_PRED_DIR, RESULTS_CM_DIR, RESULTS_ERROR_DIR, RESULTS_ROBUST_DIR,
    RESULTS_CALIB_DIR, FIGURES_DIR, REPORTS_DIR,
    FIGURES_DIR / "classification", FIGURES_DIR / "extraction",
    FIGURES_DIR / "negation", FIGURES_DIR / "calibration",
    FIGURES_DIR / "robustness", FIGURES_DIR / "generalization"
]:
    d.mkdir(parents=True, exist_ok=True)

# Vocabulary and Schema Constraints
VALID_DOC_TYPES: Set[str] = {
    "oncology_consultation", "nurse_intake_note",
    "patient_symptom_log", "pathology_report"
}

VALID_URGENCY_LEVELS: List[str] = EVAL_CONFIG["tasks"]["primary_classification"]["classes"]
VALID_HAZARD_TYPES: List[str] = EVAL_CONFIG["tasks"]["secondary_classification"]["classes"]
VALID_NER_LABELS: List[str] = EVAL_CONFIG["tasks"]["entity_extraction"]["entity_types"]

RANDOM_SEED: int = EVAL_CONFIG["metadata"]["random_seed"]
BOOTSTRAP_ITERATIONS: int = EVAL_CONFIG["bootstrap"]["n_iterations"]
CONFIDENCE_LEVEL: float = EVAL_CONFIG["bootstrap"]["confidence_level"]

# Frozen Upstream Checksums (SHA-256)
EXPECTED_SHA256: Dict[str, str] = {
    "clinical_nlp_dataset_v1.parquet": "426ea0c51f354a5fa9e1177ec0d8fabc0ed26ed708f748ee8f8f4b0f18554230",
    "train.parquet": "5862db7ec4e42fb71bf207906c9acf21f9d43280a32336ebf43dff77e6f55133",
    "validation.parquet": "2b6787a394d262c8f8b94b92914777f495e30a7be4eaf925ce862f1302899a4c",
    "locked_test.parquet": "491562c892a6af2f08695eefefe782daeae12d52f1fef5a6ad11060d96af4509"
}
