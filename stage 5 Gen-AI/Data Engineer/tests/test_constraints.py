"""Tests for constraint specifications."""
from pathlib import Path
from src.utils.io import load_yaml

def test_constraint_spec_file_valid():
    p = Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/constraint_spec.yaml")
    assert p.exists()
    spec = load_yaml(p)
    assert "schema_constraints" in spec
    assert "biomarker_ranges" in spec
    assert "temporal_rules" in spec
    assert "mutation_rules" in spec
    assert "treatment_rules" in spec

def test_biomarker_range_rules():
    spec = load_yaml("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/constraint_spec.yaml")
    bio_ranges = spec["biomarker_ranges"]
    for bio, r in bio_ranges.items():
        assert r["min"] <= r["max"]
        assert "unit" in r

def test_temporal_rules_causality():
    spec = load_yaml("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/constraint_spec.yaml")
    rules = spec["temporal_rules"]
    assert len(rules) >= 4
    rule_ids = [r["rule_id"] for r in rules]
    assert "T001" in rule_ids
    assert "T002" in rule_ids

def test_mutation_cooccurrence_constraints():
    spec = load_yaml("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/constraint_spec.yaml")
    mut_rules = spec["mutation_rules"]
    assert "allowed_driver_mutations" in mut_rules
    assert "forbidden_contradictions" in mut_rules
    assert "EGFR" in mut_rules["allowed_driver_mutations"]
