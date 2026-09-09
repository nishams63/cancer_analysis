"""
Configuration and Path Resolver for Stage 3 Clinical NLP.
Dynamically resolves local module directories and upstream read-only data paths.
"""

from pathlib import Path
from typing import Dict, Any
import yaml

SRC_DIR = Path(__file__).resolve().parent
NLP_DIR = SRC_DIR.parent
CONFIG_PATH = NLP_DIR / "configs" / "nlp_config.yaml"


def load_yaml_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file missing at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_yaml_config()

# Upstream Data Paths (Read-Only)
UPSTREAM_DATA_DIR = (NLP_DIR / CONFIG["paths"]["upstream_data_dir"]).resolve()
RAW_PARQUET_PATH = (NLP_DIR / CONFIG["paths"]["raw_dataset_path"]).resolve()
TRAIN_PARQUET_PATH = (NLP_DIR / CONFIG["paths"]["train_split_path"]).resolve()
VAL_PARQUET_PATH = (NLP_DIR / CONFIG["paths"]["validation_split_path"]).resolve()
LOCKED_TEST_PARQUET_PATH = (NLP_DIR / CONFIG["paths"]["locked_test_split_path"]).resolve()

# NLP Local Directories
INTERMEDIATE_DIR = NLP_DIR / CONFIG["paths"]["intermediate_dir"]
OUTPUTS_DIR = NLP_DIR / CONFIG["paths"]["outputs_dir"]
ARTIFACTS_DIR = NLP_DIR / CONFIG["paths"]["artifacts_dir"]
VOCAB_DIR = NLP_DIR / CONFIG["paths"]["vocab_dir"]
TOKENIZERS_DIR = NLP_DIR / CONFIG["paths"]["tokenizers_dir"]
ENCODERS_DIR = NLP_DIR / CONFIG["paths"]["encoders_dir"]
RESULTS_DIR = NLP_DIR / CONFIG["paths"]["results_dir"]
PREDICTIONS_DIR = NLP_DIR / CONFIG["paths"]["predictions_dir"]
METRICS_DIR = NLP_DIR / CONFIG["paths"]["metrics_dir"]
FEATURE_OUTPUTS_DIR = NLP_DIR / CONFIG["paths"]["feature_outputs_dir"]
REPORTS_DIR = NLP_DIR / CONFIG["paths"]["reports_dir"]

# Ensure local directories exist
for p in [INTERMEDIATE_DIR, OUTPUTS_DIR, ARTIFACTS_DIR, VOCAB_DIR, TOKENIZERS_DIR,
          ENCODERS_DIR, RESULTS_DIR, PREDICTIONS_DIR, METRICS_DIR, FEATURE_OUTPUTS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = CONFIG["metadata"]["random_seed"]
VALID_DOC_TYPES = set(CONFIG["vocabularies"]["document_types"])
VALID_URGENCY_LEVELS = set(CONFIG["vocabularies"]["urgency_levels"])
VALID_HAZARD_TYPES = set(CONFIG["vocabularies"]["hazard_types"])
VALID_NER_LABELS = set(CONFIG["vocabularies"]["ner_labels"])
