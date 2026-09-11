"""Tests for rare combination space catalog."""
from pathlib import Path
from src.utils.io import load_yaml

def test_rare_space_catalog_valid():
    p = Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/rare_combination_space.yaml")
    assert p.exists()
    catalog = load_yaml(p)
    assert "scenarios" in catalog
    assert len(catalog["scenarios"]) >= 6

def test_rare_scenarios_schema():
    catalog = load_yaml("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/rare_combination_space.yaml")
    for sc in catalog["scenarios"]:
        assert "scenario_id" in sc
        assert "scenario_name" in sc
        assert "required_features" in sc
        assert "joint_frequency" in sc
        assert "rarity_category" in sc
        assert sc["allowed"] is True
        assert sc["rarity_category"] in {"common", "uncommon", "rare", "very_rare"}

def test_rare_thresholds_honored():
    catalog = load_yaml("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/rare_combination_space.yaml")
    thresh = catalog["rarity_thresholds"]
    assert thresh["common"] > thresh["uncommon"] > thresh["rare"] > thresh["very_rare"]
