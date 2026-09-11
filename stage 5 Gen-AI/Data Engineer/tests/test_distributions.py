"""Tests for empirical reference distributions."""
from pathlib import Path
from src.utils.io import read_parquet

def test_all_distribution_files_exist():
    processed = Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/processed")
    expected = [
        "reference_distributions.parquet",
        "mutation_frequencies.parquet",
        "mutation_cooccurrence.parquet",
        "biomarker_distributions.parquet",
        "treatment_distributions.parquet",
        "dosage_ranges.parquet",
        "adverse_event_distributions.parquet",
        "missingness_patterns.parquet",
        "temporal_patterns.parquet"
    ]
    for fname in expected:
        assert (processed / fname).exists(), f"Missing expected distribution file: {fname}"

def test_mutation_frequencies_sum_and_bounds():
    df = read_parquet("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/processed/mutation_frequencies.parquet")
    assert len(df) > 0
    assert (df["count"] >= 0).all()
    assert (df["frequency"] >= 0.0).all()
    assert (df["frequency"] <= 1.0).all()

def test_mutation_cooccurrence_validity():
    df = read_parquet("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/processed/mutation_cooccurrence.parquet")
    assert len(df) > 0
    assert "mutation_A" in df.columns
    assert "mutation_B" in df.columns
    assert (df["cooccurrence_count"] >= 0).all()
    assert (df["cooccurrence_frequency"] >= 0.0).all()
    assert set(df["joint_rarity"].unique()).issubset({"common", "uncommon", "rare", "very_rare"})

def test_biomarker_quantiles_ordered():
    df = read_parquet("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/processed/biomarker_distributions.parquet")
    for _, row in df.iterrows():
        assert row["min"] <= row["p05"] <= row["p25"] <= row["p50"] <= row["p75"] <= row["p95"] <= row["max"], (
            f"Quantile ordering violated for {row['biomarker_name']}"
        )

def test_dosage_ranges_non_negative():
    df = read_parquet("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/processed/dosage_ranges.parquet")
    assert (df["minimum_observed"] >= 0.0).all()
    assert (df["maximum_observed"] >= df["minimum_observed"]).all()
    assert (df["median"] >= df["minimum_observed"]).all()
