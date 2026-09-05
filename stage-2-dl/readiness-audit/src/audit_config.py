"""
Stage 2 Deep Learning - Real-World Readiness Audit Configuration Module

Defines base paths, evaluation ratings, readiness tiers, and mandatory disclaimers.
"""
from pathlib import Path

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
AUDIT_DIR = SRC_DIR.parent
STAGE_2_DIR = AUDIT_DIR.parent
PROJECT_ROOT = STAGE_2_DIR.parent

# Module Directories
DATA_ENG_DIR = STAGE_2_DIR / 'data-engineering'
EDA_DIR = STAGE_2_DIR / 'eda'
DL_DIR = STAGE_2_DIR / 'dl'
EVAL_DIR = STAGE_2_DIR / 'evaluation'
INTEGRATION_DIR = STAGE_2_DIR / 'integration'

# Output Paths
REPORTS_DIR = AUDIT_DIR / 'reports'
FIGURES_DIR = AUDIT_DIR / 'figures'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Key Artifacts to Audit
PATHOLOGY_CHECKPOINT = DL_DIR / 'checkpoints' / 'best_pathology_cnn.pt'
TEMPORAL_CHECKPOINT = DL_DIR / 'checkpoints' / 'best_temporal_lstm.pt'
IMAGE_METADATA_PATH = DATA_ENG_DIR / 'data' / 'v2' / 'processed' / 'image_metadata.csv'
BIOMARKERS_PATH = DATA_ENG_DIR / 'data' / 'v2' / 'processed' / 'biomarkers_processed.csv'
TEST_SPLIT_PATH = DATA_ENG_DIR / 'data' / 'v2' / 'splits' / 'test_patients.csv'
DATA_DICTIONARY_PATH = DATA_ENG_DIR / 'docs' / 'data_dictionary.md'
DL_REPORT_PATH = DL_DIR / 'reports' / 'dl_report.md'
EVAL_REPORT_PATH = EVAL_DIR / 'reports' / 'evaluation_report.md'
INTEGRATION_REPORT_PATH = INTEGRATION_DIR / 'reports' / 'integration_report.md'

# Output Report Files
FINAL_AUDIT_REPORT_PATH = REPORTS_DIR / 'stage2_real_world_readiness_report.md'
SCORECARD_CSV_PATH = REPORTS_DIR / 'readiness_scorecard.csv'
GAP_ANALYSIS_CSV_PATH = REPORTS_DIR / 'validation_gap_analysis.csv'
RISK_REGISTER_CSV_PATH = REPORTS_DIR / 'risk_register.csv'

# Audit Rating Definitions
RATINGS = {
    'GREEN': 'GREEN (Strong evidence for technical prototype purpose)',
    'YELLOW': 'YELLOW (Limitation / additional simulation or empirical validation required)',
    'RED': 'RED (Major blocker for real-world or clinical patient use)'
}

# Readiness Tiers
TIER_TECHNICAL_PROTOTYPE = 'READY'
TIER_RESEARCH_PROTOTYPE = 'READY WITH LIMITATIONS'
TIER_CLINICAL_DEPLOYMENT = 'NOT READY'

# Mandatory Regulatory Disclaimers
MANDATORY_DISCLAIMER = (
    "This model was developed using synthetic data and has not been clinically validated. "
    "Performance on this dataset does not establish clinical effectiveness, safety, or real-world patient benefit."
)
EQUIVALENCE_DISCLAIMER = "Synthetic data != Real patient evidence != Clinical validation"
