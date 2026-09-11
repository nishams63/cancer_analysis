"""Tests for data cleaning, quarantine tracking, and bound checks."""
import pandas as pd
from src.cleaning.clean_demographics import DemographicsCleaner
from src.cleaning.clean_mutations import MutationCleaner
from src.cleaning.clean_biomarkers import BiomarkersCleaner
from src.cleaning.clean_treatments import TreatmentCleaner
from src.cleaning.clean_adverse_events import AdverseEventCleaner
from src.cleaning.clean_timestamps import TimestampCleaner

def test_clean_demographics(sample_stage1_df):
    cleaner = DemographicsCleaner(age_min=18, age_max=105)
    df_clean, issues = cleaner.clean(sample_stage1_df)
    # Age 115 should be clamped to 105
    assert df_clean.loc[1, "age"] == 105.0
    # Missing sex should be set to Unknown
    assert df_clean.loc[2, "sex"] == "Unknown"
    assert len(issues) >= 2

def test_clean_mutations(sample_stage1_df):
    cleaner = MutationCleaner()
    df_clean, issues = cleaner.clean(sample_stage1_df)
    # EGFR-mut should be recognized and cleaned to EGFR
    assert df_clean.loc[1, "mutation_primary"] == "EGFR"
    assert df_clean.loc[1, "mutation_secondary"] == "None/Unknown"

def test_clean_biomarkers(sample_stage1_df):
    cleaner = BiomarkersCleaner()
    df_clean, issues = cleaner.clean(sample_stage1_df)
    # ctDNA 120 should be clamped to max 50
    assert df_clean.loc[1, "ctdna_level"] <= 50.0
    # Negative tumor marker should be clamped to 0
    assert df_clean.loc[1, "tumor_marker_level"] >= 0.0

def test_clean_treatments(sample_stage1_df):
    cleaner = TreatmentCleaner()
    df_clean, issues = cleaner.clean(sample_stage1_df)
    # Negative dosage should be clamped to 0
    assert df_clean.loc[2, "drug_dose"] == 0.0
    # Negative cycle count should be clamped to 0
    assert df_clean.loc[1, "treatment_cycle"] == 0

def test_clean_adverse_events(sample_stage1_df):
    cleaner = AdverseEventCleaner()
    df_clean, issues = cleaner.clean(sample_stage1_df)
    # Toxicity grade 6 should be clamped to 5
    assert df_clean.loc[1, "previous_toxicity_grade"] <= 5.0
    # Invalid risk tier normalized
    assert df_clean.loc[1, "toxicity_risk"] in {"Low", "Moderate", "High", "Critical"}

def test_clean_timestamps(sample_stage1_df, sample_stage2_df):
    cleaner = TimestampCleaner()
    df_clean, issues = cleaner.clean(sample_stage1_df)
    # Unparseable date should be null/NaN and logged
    assert pd.isna(df_clean.loc[1, "observation_date"])
    # Negative delta days in stage2 should be clamped
    s2_clean, s2_issues = cleaner.clean(sample_stage2_df)
    assert (s2_clean["delta_days"] >= 0).all()
