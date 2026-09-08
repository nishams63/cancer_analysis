"""
Configuration loader and constants for Stage 3 NLP Data Engineering.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import yaml

def find_repo_root() -> Path:
    """Find repository root by looking for known root markers."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "stage-1-ml").exists() and (current / "stage-2-dl").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent.parent

REPO_ROOT = find_repo_root()
CONFIG_PATH = REPO_ROOT / "stage-3-nlp-slm" / "data-engineering" / "configs" / "data_config.yaml"

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

CFG = load_config()

# Derived absolute paths
MASTER_PATIENT_PATH = REPO_ROOT / CFG["paths"]["master_patient_dataset"]
RAW_DATA_DIR = REPO_ROOT / CFG["paths"]["raw_data_dir"]
PROCESSED_DATA_DIR = REPO_ROOT / CFG["paths"]["processed_data_dir"]
REPORTS_DIR = REPO_ROOT / CFG["paths"]["reports_dir"]

RAW_JSONL_PATH = RAW_DATA_DIR / "raw_clinical_notes_v1.jsonl"
RAW_METADATA_PATH = RAW_DATA_DIR / "raw_clinical_notes_metadata.csv"

PROCESSED_PARQUET_PATH = PROCESSED_DATA_DIR / "clinical_nlp_dataset_v1.parquet"
PROCESSED_JSONL_PATH = PROCESSED_DATA_DIR / "clinical_nlp_dataset_v1.jsonl"
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "clinical_nlp_dataset_v1.csv"

TRAIN_PARQUET_PATH = PROCESSED_DATA_DIR / "train.parquet"
VAL_PARQUET_PATH = PROCESSED_DATA_DIR / "validation.parquet"
TEST_PARQUET_PATH = PROCESSED_DATA_DIR / "locked_test.parquet"

TRAIN_JSONL_PATH = PROCESSED_DATA_DIR / "train.jsonl"
VAL_JSONL_PATH = PROCESSED_DATA_DIR / "validation.jsonl"
TEST_JSONL_PATH = PROCESSED_DATA_DIR / "locked_test.jsonl"

RANDOM_SEED = CFG["project"]["random_seed"]
TOTAL_PATIENTS = CFG["cohort"]["total_patients"]
TRAIN_PATIENTS = CFG["cohort"]["split_ratio"]["train_patients"]
VAL_PATIENTS = CFG["cohort"]["split_ratio"]["val_patients"]
TEST_PATIENTS = CFG["cohort"]["split_ratio"]["test_patients"]

VALID_DOC_TYPES = set(CFG["document_types"])
VALID_URGENCY_LEVELS = set(CFG["targets"]["urgency_levels"])
VALID_HAZARD_TYPES = set(CFG["targets"]["hazard_types"])
VALID_NER_ENTITIES = set(CFG["targets"]["ner_entities"])

CLINICAL_DISCLAIMER = (
    "RESEARCH PROTOTYPE DATASET: Synthetically generated clinical text modeled after "
    "oncology clinical trial workflows and pharmacogenomic records. Not real patient PHI. "
    "Not clinically approved for autonomous diagnosis or medical decision making."
)
