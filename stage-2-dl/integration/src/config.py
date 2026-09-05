"""
Stage 2 Deep Learning - Multimodal Integration Configuration Module

IMPORTANT FUSION VALIDATION NOTICE:
The default fusion weights and alert thresholds in this configuration are prototype
engineering parameters only. They are NOT clinically validated.
They were NOT optimized or tuned using the locked test set.
LOW, MODERATE, and HIGH represent prototype engineering alert categories, NOT clinically
established risk strata.
"""
from pathlib import Path

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
INTEGRATION_DIR = SRC_DIR.parent
STAGE_2_DIR = INTEGRATION_DIR.parent
PROJECT_ROOT = STAGE_2_DIR.parent

# Frozen DL Module & Checkpoints (Strictly Read-Only)
DL_DIR = STAGE_2_DIR / 'dl'
CHECKPOINTS_DIR = DL_DIR / 'checkpoints'
FROZEN_IMAGE_CHECKPOINT = CHECKPOINTS_DIR / 'best_pathology_cnn.pt'
FROZEN_TEMPORAL_CHECKPOINT = CHECKPOINTS_DIR / 'best_temporal_lstm.pt'

# Data Engineering Source Data (Strictly Read-Only)
DATA_ENG_V2 = STAGE_2_DIR / 'data-engineering' / 'data' / 'v2'
IMAGE_METADATA_PATH = DATA_ENG_V2 / 'processed' / 'image_metadata.csv'
BIOMARKERS_PATH = DATA_ENG_V2 / 'processed' / 'biomarkers_processed.csv'
PATHOLOGY_TILES_DIR = DATA_ENG_V2 / 'processed' / 'pathology_tiles'
TEST_SPLIT_PATH = DATA_ENG_V2 / 'splits' / 'test_patients.csv'
TRAIN_SPLIT_PATH = DATA_ENG_V2 / 'splits' / 'train_patients.csv'
VAL_SPLIT_PATH = DATA_ENG_V2 / 'splits' / 'validation_patients.csv'

# Output Reports & Artifacts
REPORTS_DIR = INTEGRATION_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
INTEGRATION_REPORT_PATH = REPORTS_DIR / 'integration_report.md'

# Mandatory Regulatory Disclaimers
MANDATORY_DISCLAIMER = (
    "This model was developed using synthetic data and has not been clinically validated. "
    "Performance on this dataset does not establish clinical effectiveness, safety, or real-world patient benefit."
)
EQUIVALENCE_DISCLAIMER = "Synthetic data != Real patient evidence != Clinical validation"
CLINICAL_VALIDATION_STATUS = "NOT_CLINICALLY_VALIDATED"
DATA_SOURCE = "synthetic"

# Temporal Sequence Constraints
FORECAST_SPLIT_DAY = 90
FORBIDDEN_COLUMNS = [
    'future_ctDNA_30d_target',
    'future_progression_trend',
    'trajectory_pattern'
]

# Patient Pathology Aggregation
DEFAULT_TILE_AGGREGATION = 'mean'
ALLOWED_TILE_AGGREGATIONS = ['mean', 'median', 'max']

# Multimodal Fusion Configuration (PROTOTYPE ENGINEERING CHOICES)
# These weights are transparent heuristic baseline parameters.
# They are NOT clinically calibrated or optimized on test data.
DEFAULT_FUSION_WEIGHTS = {
    'pathology_malignant': 0.35,
    'temporal_progression': 0.40,
    'ctdna_vaf_risk': 0.25
}

# Fallback weights when only temporal modality is available
TEMPORAL_ONLY_WEIGHTS = {
    'temporal_progression': 0.60,
    'ctdna_vaf_risk': 0.40
}

# ctDNA Normalization Scaling Parameters
# Scales continuous VAF (%) into [0, 1] index for fusion
CTDNA_VAF_MIN = 0.0
CTDNA_VAF_MAX = 5.0  # 99th percentile across synthetic distribution is ~4.7%

# Prototype Alert Thresholds (PROTOTYPE ENGINEERING CATEGORIES ONLY)
# NOT clinically validated risk categories or treatment thresholds.
PROTOTYPE_ALERT_THRESHOLDS = {
    'low_upper': 0.35,      # [0.00, 0.35) -> LOW (Prototype)
    'moderate_upper': 0.70  # [0.35, 0.70) -> MODERATE (Prototype), [0.70, 1.00] -> HIGH (Prototype)
}

PROTOTYPE_ALERT_LABELS = {
    'LOW': 'LOW (Prototype)',
    'MODERATE': 'MODERATE (Prototype)',
    'HIGH': 'HIGH (Prototype)',
    'INSUFFICIENT_DATA': 'INSUFFICIENT_DATA'
}

# API Configuration
API_HOST = '0.0.0.0'
API_PORT = 8000
