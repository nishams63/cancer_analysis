"""
Tests for Ablation Study, Model Comparison, and Disqualification Logic.
"""

from comparison import ModelComparator


def test_composite_score_calculation():
    comparator = ModelComparator("stage-4-slm/slm-engineer/results")
    metrics = {
        "risk_macro_f1": 0.85,
        "entity_retention_rate": 0.95,
        "negation_preservation_rate": 1.0,
        "format_compliance_rate": 0.98,
        "unsupported_entity_rate": 0.02
    }
    score = comparator.calculate_composite_score(metrics)
    assert score > 0.80
    assert score <= 1.0


def test_disqualification_on_negation_flip():
    comparator = ModelComparator("stage-4-slm/slm-engineer/results")
    experiments = [
        {
            "experiment_id": "model_unsafe",
            "configuration": {"model_name": "unsafe", "dataset_variant": "raw"},
            "metrics": {
                "risk_macro_f1": 0.90,
                "entity_retention_rate": 0.95,
                "negation_preservation_rate": 0.90,
                "negation_flip_rate": 0.10,  # > 5% limit -> DISQUALIFIED
                "format_compliance_rate": 0.95,
                "unsupported_entity_rate": 0.0
            }
        },
        {
            "experiment_id": "model_safe",
            "configuration": {"model_name": "safe", "dataset_variant": "filtered"},
            "metrics": {
                "risk_macro_f1": 0.85,
                "entity_retention_rate": 0.95,
                "negation_preservation_rate": 1.0,
                "negation_flip_rate": 0.0,
                "format_compliance_rate": 0.95,
                "unsupported_entity_rate": 0.0
            }
        }
    ]

    df_comp = comparator.build_comparison_table(experiments)
    assert len(df_comp) == 2
    assert "DISQUALIFIED" in df_comp.loc[df_comp["experiment_id"] == "model_unsafe", "status"].iloc[0]
    assert "QUALIFIED" in df_comp.loc[df_comp["experiment_id"] == "model_safe", "status"].iloc[0]
