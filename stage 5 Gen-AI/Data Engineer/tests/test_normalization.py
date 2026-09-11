"""Tests for ontology normalization and category harmonization."""
import pandas as pd
from src.normalization.normalize_mutations import MutationNormalizer
from src.normalization.normalize_biomarkers import BiomarkerNormalizer
from src.normalization.normalize_treatments import TreatmentNormalizer
from src.normalization.normalize_categories import CategoryNormalizer
from src.normalization.normalize_units import UnitNormalizer

def test_normalize_mutations():
    df = pd.DataFrame({
        "mutation_primary": ["EGFR-MUT", "KRAS+", "wildtype", "None"],
        "mutation_secondary": ["TP53-MUT", "MET AMP", "None", None]
    })
    normalizer = MutationNormalizer()
    df_norm = normalizer.normalize(df)
    assert df_norm.loc[0, "mutation_primary"] == "EGFR"
    assert df_norm.loc[1, "mutation_primary"] == "KRAS"
    assert df_norm.loc[2, "mutation_primary"] == "None/Unknown"
    assert df_norm.loc[1, "mutation_secondary"] == "MET"

def test_normalize_biomarkers():
    df = pd.DataFrame({
        "ctdna_level": [1.5],
        "ctDNA_vaf_percent": [2.4]
    })
    normalizer = BiomarkerNormalizer()
    df_norm = normalizer.normalize(df)
    assert "ctdna_ng_ml" in df_norm.columns
    assert "ctdna_vaf_percent" in df_norm.columns

def test_normalize_treatments():
    df = pd.DataFrame({"drug_name": ["docetaxel", "TAGRISSO", "radiation"]})
    normalizer = TreatmentNormalizer()
    df_norm = normalizer.normalize(df)
    assert df_norm.loc[0, "drug_name"] == "Docetaxel"
    assert df_norm.loc[1, "drug_name"] == "Osimertinib"
    assert df_norm.loc[2, "drug_name"] == "radiotherapy-standard"

def test_normalize_categories():
    df = pd.DataFrame({
        "cancer_stage": ["stage iv", "Stage 2", "ii", "invalid"],
        "hazard_type": ["hepatic", "renal", "None", "NONE"]
    })
    normalizer = CategoryNormalizer()
    df_norm = normalizer.normalize(df)
    assert df_norm.loc[0, "cancer_stage"] == "Stage IV"
    assert df_norm.loc[1, "cancer_stage"] == "Stage II"
    assert df_norm.loc[0, "hazard_type"] == "HEPATIC"
    assert df_norm.loc[1, "hazard_type"] == "RENAL"

def test_unit_normalizer():
    un = UnitNormalizer()
    assert un.get_standard_unit("drug_dose") == "mg"
    assert un.get_standard_unit("creatinine_level") == "mg/dL"
