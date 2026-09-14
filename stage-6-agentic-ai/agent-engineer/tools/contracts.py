"""Tool metadata and contract specifications."""
from __future__ import annotations
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ToolRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolMetadata(BaseModel):
    """Authoritative descriptor for registered analytical tools."""
    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Functional purpose of tool")
    risk_level: ToolRiskLevel = Field(ToolRiskLevel.LOW, description="Operational risk profile")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="Expected JSON Schema parameter specs")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Expected output schema")
    timeout_sec: float = Field(30.0, description="Max execution duration before timeout")
