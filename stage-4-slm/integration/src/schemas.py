"""
Pydantic Schemas for Stage 4 SLM Clinical Decision Support Service.
Enforces strict input validation, clinical bounds, and Stage 3 contract compatibility.
"""

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["OK", "DEGRADED", "ERROR"] = Field(..., description="Service operational status.")
    service: str = Field(default="stage4-slm-decision-support", description="Service identifier.")
    model_version: str = Field(..., description="Active SLM model version.")
    device: str = Field(..., description="Hardware compute device (cuda / cpu).")
    uptime_seconds: float = Field(..., ge=0.0, description="Service uptime in seconds.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp.")


class ClinicalNoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: Optional[str] = Field(default="DOC-UNKNOWN", description="Unique document ID.")
    patient_id: Optional[str] = Field(default="PT-UNKNOWN", description="Patient identifier.")
    clinical_note: str = Field(..., min_length=10, description="Narrative clinical oncology progress note.")


class RiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clinical_note: str = Field(..., min_length=10, description="Clinical progress note.")
    patient_id: Optional[str] = Field(default="PT-UNKNOWN", description="Patient ID.")


class RiskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    target_risk: str
    hazard_category: str
    urgency_tier: str
    latency_ms: float


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clinical_note: str = Field(..., min_length=10, description="Clinical progress note.")
    patient_id: Optional[str] = Field(default="PT-UNKNOWN", description="Patient ID.")


class ActionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    target_action: str
    dose_modification: str
    urgency_tier: str
    latency_ms: float


class DecisionSupportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    patient_id: str
    target_risk: str
    target_key_finding: str
    target_action: str
    preserved_entities: Dict[str, List[str]]
    safety_gate_passed: bool
    routing: Literal["STAGE_4_AUTO", "FALLBACK_STAGE3_BASELINE", "HUMAN_REVIEW"]
    guardrail_warnings: List[str] = Field(default_factory=list)
    latency_ms: float


class BatchDecisionSupportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    notes: List[ClinicalNoteRequest] = Field(..., min_length=1, description="List of clinical notes for batch inference.")


class BatchDecisionSupportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_processed: int
    successful_count: int
    results: List[DecisionSupportResponse]
    batch_latency_ms: float
