"""Pydantic Schema for Stage 5 Stress-Test Scenarios and Prompts."""
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ScenarioCategory(str, Enum):
    DUAL_MUTATION_RESISTANCE = "DUAL_MUTATION_RESISTANCE"
    COMPLEX_NEGATION = "COMPLEX_NEGATION"
    CROSS_STAGE_DISCORDANCE = "CROSS_STAGE_DISCORDANCE"
    BORDERLINE_BIOMARKER = "BORDERLINE_BIOMARKER"
    ORGAN_TOXICITY_OVERRIDE = "ORGAN_TOXICITY_OVERRIDE"
    RAPID_SEPSIS_TEMPORAL = "RAPID_SEPSIS_TEMPORAL"
    SPARSE_MATRIX_EXTREME = "SPARSE_MATRIX_EXTREME"
    QUADRUPLE_MUTATION_RARE = "QUADRUPLE_MUTATION_RARE"
    ELDERLY_DOSE_TOXICITY = "ELDERLY_DOSE_TOXICITY"
    RARE_HISTOLOGY_MUTATION = "RARE_HISTOLOGY_MUTATION"
    IO_RECHALLENGE_MYOCARDITIS = "IO_RECHALLENGE_MYOCARDITIS"
    INVERSE_WEIGHT_HAZARD = "INVERSE_WEIGHT_HAZARD"
    OOD_DRIFT_RESISTANCE = "OOD_DRIFT_RESISTANCE"
    TIMELINE_DISCORDANCE = "TIMELINE_DISCORDANCE"
    ASYMPTOMATIC_HIGH_HAZARD = "ASYMPTOMATIC_HIGH_HAZARD"


class DifficultyLevel(str, Enum):
    MODERATE = "MODERATE"
    HARD = "HARD"
    EXTREME = "EXTREME"


class ClinicalSeverity(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class StatisticalRarity(str, Enum):
    COMMON = "COMMON"
    UNCOMMON = "UNCOMMON"
    RARE = "RARE"
    EXTREMELY_RARE = "EXTREMELY_RARE"


class TargetStage(str, Enum):
    STAGE_1_ML = "STAGE_1_ML"
    STAGE_2_DL = "STAGE_2_DL"
    STAGE_3_NLP = "STAGE_3_NLP"
    STAGE_4_SLM = "STAGE_4_SLM"


class EntityPresence(str, Enum):
    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"
    FORBIDDEN = "FORBIDDEN"


class RequiredEntity(BaseModel):
    name: str = Field(..., description="Entity identifier or name")
    entity_type: str = Field(..., description="Entity type (mutation, biomarker, lab, vital, treatment, clause)")
    allowed_values: Optional[List[str]] = Field(default=None, description="Allowed discrete values or formats")
    min_value: Optional[float] = Field(default=None, description="Minimum numeric value if applicable")
    max_value: Optional[float] = Field(default=None, description="Maximum numeric value if applicable")
    unit: Optional[str] = Field(default=None, description="Unit of measurement")
    presence: EntityPresence = Field(default=EntityPresence.REQUIRED, description="Required, optional, or forbidden")
    description: str = Field(..., description="Clinical explanation of the entity constraint")


class ForbiddenChange(BaseModel):
    rule_id: str = Field(..., description="Unique forbidden rule identifier")
    description: str = Field(..., description="Plain-language description of prohibited modification")
    forbidden_pattern: str = Field(..., description="Pattern, regex, or value that must not appear")
    rationale: str = Field(..., description="Clinical or evaluation rationale")


class RAGRetrievalIntent(BaseModel):
    intent_id: str = Field(..., description="Unique retrieval intent ID")
    target_domain: str = Field(..., description="Target knowledge domain (guidelines, drug_labels, trials, interactions)")
    primary_query: str = Field(..., description="Primary search query for RAG retriever")
    search_terms: List[str] = Field(default_factory=list, description="Key search tokens")
    required_keywords: List[str] = Field(default_factory=list, description="Keywords that MUST be matched in top passages")
    forbidden_keywords: List[str] = Field(default_factory=list, description="Keywords indicating irrelevant/misleading evidence")
    min_documents: int = Field(default=1, description="Minimum relevant documents required")
    guideline_reference: Optional[str] = Field(default=None, description="Reference guideline ID, e.g., NCCN-NSCLC-v4.2024")


class ValidationRule(BaseModel):
    rule_id: str = Field(..., description="Rule ID")
    rule_type: str = Field(..., description="Validation type: range, entity_presence, negation, contradiction, ood")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for validator")
    error_message: str = Field(..., description="Message if scenario violates this rule")


class PatientSkeleton(BaseModel):
    age: int = Field(..., description="Patient age in years")
    sex: str = Field(..., description="Biological sex (M/F)")
    cancer_type: str = Field(..., description="Primary cancer type (e.g. NSCLC, Breast, Colorectal)")
    histology: str = Field(..., description="Histological classification")
    primary_site: str = Field(..., description="Anatomical primary site")
    prior_lines_therapy: int = Field(default=1, description="Number of prior systemic therapy lines")
    ecog_ps: int = Field(default=1, description="ECOG Performance Status (0-4)")


class ScenarioDefinition(BaseModel):
    scenario_id: str = Field(..., description="Unique scenario ID, e.g. PROMPT-R01")
    title: str = Field(..., description="Short descriptive title")
    category: ScenarioCategory = Field(..., description="Scenario taxonomy category")
    target_blind_spots: List[str] = Field(..., description="List of targeted blind spots, e.g. [BS01]")
    target_stages: List[TargetStage] = Field(..., description="Stages specifically targeted for stress testing")
    difficulty: DifficultyLevel = Field(..., description="AI difficulty rating")
    clinical_severity: ClinicalSeverity = Field(..., description="True patient clinical severity")
    statistical_rarity: StatisticalRarity = Field(..., description="Dataset prevalence rarity")
    clinical_premise: str = Field(..., description="Clinical mechanism and why this scenario challenges AI")
    target_stage_vulnerability: str = Field(..., description="Observed vulnerability in upstream models being probed")
    patient_skeleton: PatientSkeleton = Field(..., description="Demographic and tumor baseline")
    required_entities: List[RequiredEntity] = Field(..., description="Required clinical facts and lab values")
    forbidden_modifications: List[ForbiddenChange] = Field(default_factory=list, description="Prohibited changes")
    rag_intent: RAGRetrievalIntent = Field(..., description="Knowledge retrieval intent")
    validation_rules: List[ValidationRule] = Field(default_factory=list, description="Automated verification checks")
    prompt_template: str = Field(..., description="Prompt instruction template for synthetic scenario generation")
    version: str = Field(default="1.0.0", description="Semantic scenario version")


class ScenarioCatalog(BaseModel):
    version: str = Field(default="1.0.0", description="Catalog version")
    catalog_id: str = Field(default="STAGE5-STRESS-CATALOG-V1", description="Catalog identifier")
    updated_at: str = Field(..., description="ISO 8601 timestamp")
    total_scenarios: int = Field(..., description="Number of registered scenarios")
    scenarios: List[ScenarioDefinition] = Field(..., description="List of scenario definitions")
