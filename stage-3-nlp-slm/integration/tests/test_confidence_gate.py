"""
Unit tests for Confidence-Gated Handoff Logic (confidence_gate.py).
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))

from contract_validation import (
    Stage3OutputPayload,
    TriageUrgencyPayload,
    ToxicityHazardPayload,
    ClinicalEntityPayload,
)
from confidence_gate import (
    ConfidenceGate,
    ConfidenceGateConfig,
    RoutingResult
)


def create_mock_payload(
    urg_class="HIGH",
    urg_conf=0.88,
    haz_class="HEMATOLOGIC",
    haz_conf=0.85,
    p_crit=0.05
):
    return Stage3OutputPayload(
        schema_version="1.0.0",
        document_id="DOC-CONF-TEST",
        patient_id="PT-CONF-001",
        inference_timestamp=datetime.now(timezone.utc).isoformat(),
        model_version="stage3-minilm-hybrid-v3.3-config-c",
        ner_model_version="stage3-trainable-ner-v1.0",
        triage_urgency=TriageUrgencyPayload(
            predicted_class=urg_class,
            confidence=urg_conf,
            class_probabilities={"LOW": 0.05, "MEDIUM": 0.10, "HIGH": urg_conf, "CRITICAL": p_crit}
        ),
        toxicity_hazard=ToxicityHazardPayload(
            predicted_class=haz_class,
            confidence=haz_conf,
            class_probabilities={"NONE": 0.05, haz_class: haz_conf}
        ),
        clinical_entities=[
            ClinicalEntityPayload(
                start=0,
                end=10,
                label="DRUG_NAME",
                text="cisplatin",
                polarity="AFFIRMED"
            )
        ]
    )


def test_high_confidence_passthrough():
    gate = ConfidenceGate()
    payload = create_mock_payload(urg_class="HIGH", urg_conf=0.92, haz_class="HEMATOLOGIC", haz_conf=0.88)
    res = gate.evaluate(payload)
    assert res.destination == "STAGE_4_AUTOMATED"
    assert res.is_automated is True
    assert "satisfies all clinical safety thresholds" in res.routing_reasons[0]


def test_low_urgency_confidence_routing():
    gate = ConfidenceGate()
    payload = create_mock_payload(urg_class="MEDIUM", urg_conf=0.55, haz_class="HEMATOLOGIC", haz_conf=0.88)
    res = gate.evaluate(payload)
    assert res.destination == "HUMAN_REVIEW"
    assert res.is_automated is False
    assert any("Low urgency confidence" in r for r in res.routing_reasons)


def test_low_standard_hazard_confidence_routing():
    gate = ConfidenceGate()
    # HEMATOLOGIC is a standard hazard (threshold 0.65)
    payload = create_mock_payload(urg_class="HIGH", urg_conf=0.85, haz_class="HEMATOLOGIC", haz_conf=0.60)
    res = gate.evaluate(payload)
    assert res.destination == "HUMAN_REVIEW"
    assert res.is_automated is False
    assert any("Low hazard confidence" in r for r in res.routing_reasons)


def test_rare_hazard_elevated_threshold():
    gate = ConfidenceGate()
    # CARDIAC is a rare hazard class (elevated threshold = 0.70)
    # A confidence of 0.68 would pass standard (0.65) but fails rare hazard (0.70)
    payload_marginal = create_mock_payload(urg_class="HIGH", urg_conf=0.85, haz_class="CARDIAC", haz_conf=0.68)
    res_marginal = gate.evaluate(payload_marginal)
    assert res_marginal.destination == "HUMAN_REVIEW"
    assert any("Elevated rare hazard threshold" in r for r in res_marginal.routing_reasons)

    # A confidence of 0.72 passes rare hazard threshold
    payload_strong = create_mock_payload(urg_class="HIGH", urg_conf=0.85, haz_class="CARDIAC", haz_conf=0.72)
    res_strong = gate.evaluate(payload_strong)
    assert res_strong.destination == "STAGE_4_AUTOMATED"


def test_critical_safety_override():
    gate = ConfidenceGate(ConfidenceGateConfig(critical_override_threshold=0.30))
    # Note where argmax was MEDIUM (prob 0.55), but P(CRITICAL) = 0.38 >= 0.30
    payload = create_mock_payload(urg_class="MEDIUM", urg_conf=0.55, haz_class="NONE", haz_conf=0.90, p_crit=0.38)
    res = gate.evaluate(payload)
    assert res.effective_urgency_class == "CRITICAL"
    assert any("CRITICAL safety override triggered" in r for r in res.routing_reasons)
    # Since rescued CRITICAL has P(CRITICAL) = 0.38 < 0.65, routes to human review for safety confirmation
    assert res.destination == "HUMAN_REVIEW"


def test_boundary_edge_values():
    gate = ConfidenceGate(ConfidenceGateConfig(
        urgency_min_confidence=0.65,
        hazard_min_confidence=0.65,
        rare_hazard_min_confidence=0.70
    ))

    # Exact 0.6500 urgency -> pass
    p1 = create_mock_payload(urg_class="HIGH", urg_conf=0.6500, haz_class="NONE", haz_conf=0.80)
    assert gate.evaluate(p1).destination == "STAGE_4_AUTOMATED"

    # 0.6499 urgency -> fail to human review
    p2 = create_mock_payload(urg_class="HIGH", urg_conf=0.6499, haz_class="NONE", haz_conf=0.80)
    assert gate.evaluate(p2).destination == "HUMAN_REVIEW"

    # Exact 0.7000 rare hazard -> pass
    p3 = create_mock_payload(urg_class="HIGH", urg_conf=0.80, haz_class="DERMATOLOGIC", haz_conf=0.7000)
    assert gate.evaluate(p3).destination == "STAGE_4_AUTOMATED"

    # 0.6999 rare hazard -> fail
    p4 = create_mock_payload(urg_class="HIGH", urg_conf=0.80, haz_class="DERMATOLOGIC", haz_conf=0.6999)
    assert gate.evaluate(p4).destination == "HUMAN_REVIEW"


def test_invalid_config():
    with pytest.raises(ValueError):
        ConfidenceGateConfig(urgency_min_confidence=-0.1)
    with pytest.raises(ValueError):
        ConfidenceGateConfig(hazard_min_confidence=1.2)
    with pytest.raises(ValueError):
        ConfidenceGateConfig(critical_override_threshold=2.0)
