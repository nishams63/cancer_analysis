"""
Tests for Entity Density and Target Retention Module.
"""

import pandas as pd
import pytest
from entity_analysis import EntityAnalyzer


@pytest.fixture
def entity_test_df():
    return pd.DataFrame({
        "target_risk": [
            "Increased pneumonitis hazard with Pembrolizumab (200 mg).",
            "Low toxicity risk."
        ],
        "target_key_finding": [
            "EGFR mutation confirmed; Pembrolizumab administered at 200 mg.",
            "Stable disease."
        ],
        "target_action": [
            "Hold Pembrolizumab and treat pneumonitis.",
            "Continue standard monitoring."
        ],
        "ner_genes": [["EGFR"], ["None/Unknown"]],
        "ner_drugs": [["Pembrolizumab"], []],
        "ner_dosages": [["200 mg"], []],
        "ner_adverse_events": [["pneumonitis"], []],
    })


def test_entity_density_analysis(entity_test_df):
    analyzer = EntityAnalyzer()
    res = analyzer.analyze_entity_density(entity_test_df)
    assert "density_metrics" in res
    assert res["density_metrics"]["overall"]["max"] == 4
    assert res["density_metrics"]["gene"]["mean"] == 0.5


def test_entity_retention_analysis(entity_test_df):
    analyzer = EntityAnalyzer()
    ret = analyzer.analyze_entity_retention(entity_test_df)
    assert ret["overall_retention_rate"] == 1.0
    assert ret["total_source_entities"] == 4
    assert ret["total_preserved_entities"] == 4
    assert ret["categories"]["drug"]["retention_rate"] == 1.0
