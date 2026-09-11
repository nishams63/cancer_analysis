"""Pytest fixtures and configuration for Stage 5 Data Engineering."""
import sys
from pathlib import Path
import pytest
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

@pytest.fixture
def sample_stage1_df():
    return pd.DataFrame({
        "patient_id": ["PT-000001", "PT-000002", "PT-000003"],
        "age": [62.0, 115.0, 45.0],
        "sex": ["Male", "female", None],
        "cancer_type": ["Breast Cancer", "Non-Small Cell Lung Cancer", "Unknown"],
        "cancer_stage": ["Stage IV", "Stage II", "Unknown"],
        "mutation_primary": ["KRAS", "EGFR-mut", "None/Unknown"],
        "mutation_secondary": ["TP53", "None", "None/Unknown"],
        "ctdna_level": [0.758, 120.0, 1.2],
        "tumor_marker_level": [50.33, -5.0, 30.0],
        "drug_name": ["Docetaxel", "Osimertinib", "Cisplatin"],
        "drug_dose": [265.4, 80.0, -10.0],
        "treatment_cycle": [4, -1, 2],
        "previous_toxicity_grade": [0.0, 6.0, 2.0],
        "toxicity_risk": ["High", "invalid_risk", "Low"],
        "observation_date": ["2024-03-13", "invalid-date", "2024-05-20"]
    })

@pytest.fixture
def sample_stage2_df():
    return pd.DataFrame({
        "patient_id": ["PAT-001", "PAT-001"],
        "timestamp": ["2026-01-13", "2026-01-25"],
        "delta_days": [0, -5],
        "ctDNA_vaf_percent": [0.5142, 0.4119]
    })
