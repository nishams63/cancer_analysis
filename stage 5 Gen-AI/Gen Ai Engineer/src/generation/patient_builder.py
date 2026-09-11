"""Structured patient builder and schema representation."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Demographics(BaseModel):
    age: int = Field(..., ge=18, le=105)
    sex: str = Field(..., description="Male, Female, Unknown")
    cancer_type: str = Field(...)
    cancer_stage: str = Field(default="Stage IV")


class Resistance(BaseModel):
    status: bool = Field(default=False)
    mechanism: Optional[str] = Field(default=None)


class TimelineEvent(BaseModel):
    event_id: str
    event_type: str
    timestamp: str
    delta_days: int
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)


class StructuredSyntheticPatient(BaseModel):
    scenario_id: str
    patient_id: str
    synthetic: bool = Field(default=True)
    demographics: Demographics
    mutations: List[str] = Field(default_factory=list)
    biomarkers: Dict[str, Any] = Field(default_factory=dict)
    treatments: List[Dict[str, Any]] = Field(default_factory=list)
    dosages: Dict[str, float] = Field(default_factory=dict)
    adverse_events: List[Dict[str, Any]] = Field(default_factory=list)
    resistance: Resistance = Field(default_factory=Resistance)
    timeline: List[TimelineEvent] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    scenario_requirements: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()