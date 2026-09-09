"""
Tests for Token Analyzer and Critical Entity Truncation Risk Module.
"""

import pandas as pd
import pytest
from token_analysis import TokenAnalyzer


@pytest.fixture
def token_test_df():
    return pd.DataFrame({
        "clinical_note": [
            "Patient receiving Pembrolizumab 200 mg every 3 weeks.",
            "Short note.",
            "A " * 300 + "Patient experienced neutropenia."
        ],
        "instruction": ["Analyze the note."] * 3,
        "target_risk": ["Increased neutropenia hazard.", "Low risk.", "High toxicity."],
        "target_key_finding": ["Pneumonitis monitored.", "Stable.", "Neutropenia present."],
        "target_action": ["Hold Pembrolizumab.", "Continue.", "Administer G-CSF."],
        "ner_genes": [[], [], []],
        "ner_drugs": [["Pembrolizumab"], [], []],
        "ner_dosages": [["200 mg"], [], []],
        "ner_adverse_events": [[], [], ["neutropenia"]],
    })


def test_token_counting_and_distributions(token_test_df):
    analyzer = TokenAnalyzer(tokenizer_model="cl100k_base", context_limits=[50, 100, 500])
    results = analyzer.analyze_token_distributions(token_test_df)

    metrics = results["metrics"]
    assert "source_tokens" in metrics
    assert metrics["source_tokens"]["mean"] > 0
    assert "context_limits_evaluation" in metrics
    assert "limit_50" in metrics["context_limits_evaluation"]


def test_critical_entity_truncation_detection(token_test_df):
    analyzer = TokenAnalyzer(tokenizer_model="cl100k_base")
    # Low limit so third record triggers tail truncation check
    risk = analyzer.analyze_critical_entity_truncation_risk(
        token_test_df, target_context_limit=300, tail_fraction=0.30
    )
    assert "potential_entity_truncation_count" in risk
    assert risk["potential_entity_truncation_count"] >= 1
