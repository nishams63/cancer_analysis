"""Tests for Pydantic Schema Validation."""
import pytest
from pydantic import ValidationError
from src.prompts.prompt_schema import (
    ScenarioDefinition, ScenarioCategory, DifficultyLevel, ClinicalSeverity,
    StatisticalRarity, TargetStage, RequiredEntity, EntityPresence,
    ForbiddenChange, RAGRetrievalIntent, PatientSkeleton
)


def test_valid_scenario_definition():
    sc = ScenarioDefinition(
        scenario_id="PROMPT-TEST",
        title="Test Scenario",
        category=ScenarioCategory.DUAL_MUTATION_RESISTANCE,
        target_blind_spots=["BS01"],
        target_stages=[TargetStage.STAGE_4_SLM],
        difficulty=DifficultyLevel.HARD,
        clinical_severity=ClinicalSeverity.HIGH,
        statistical_rarity=StatisticalRarity.RARE,
        clinical_premise="Clinical premise test",
        target_stage_vulnerability="Vulnerability test",
        patient_skeleton=PatientSkeleton(
            age=60, sex="F", cancer_type="NSCLC", histology="Adeno",
            primary_site="Lung", prior_lines_therapy=1, ecog_ps=1
        ),
        required_entities=[
            RequiredEntity(name="mut1", entity_type="mutation", description="mut test")
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-01", target_domain="guidelines",
            primary_query="test query"
        ),
        prompt_template="Generate test case"
    )
    assert sc.scenario_id == "PROMPT-TEST"
    assert sc.category == ScenarioCategory.DUAL_MUTATION_RESISTANCE


def test_invalid_schema_raises():
    with pytest.raises(ValidationError):
        # Missing required entities and RAG intent
        ScenarioDefinition(
            scenario_id="PROMPT-FAIL",
            title="Broken Scenario",
            category=ScenarioCategory.BORDERLINE_BIOMARKER
        )