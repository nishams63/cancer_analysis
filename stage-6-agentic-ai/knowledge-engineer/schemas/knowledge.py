"""Authoritative Pydantic Schemas for Knowledge Items and Retrieval Contracts."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class KnowledgeCategory(str, Enum):
    """The 8 core domains supported by the Autonomous AI Data Analyst."""
    DATA_QUALITY = "data_quality"
    DATA_CLEANING = "data_cleaning"
    EDA = "eda"
    STATISTICS = "statistics"
    MACHINE_LEARNING = "machine_learning"
    ANOMALY_DETECTION = "anomaly_detection"
    VISUALIZATION = "visualization"
    ROOT_CAUSE = "root_cause"
    BUSINESS_ANALYSIS = "business_analysis"


class KnowledgeItem(BaseModel):
    """Atomic, structured, versioned knowledge unit."""
    knowledge_id: str = Field(..., description="Unique deterministic identifier, e.g. DQ-001, STAT-004")
    title: str = Field(..., min_length=3, description="Concise, descriptive title")
    category: KnowledgeCategory = Field(..., description="High-level domain category")
    subcategory: str = Field(..., min_length=2, description="Specific sub-domain, e.g. missing_values, imputation")
    description: str = Field(..., min_length=15, description="Comprehensive semantic explanation")
    conditions: List[str] = Field(default_factory=list, description="Prerequisites or scenarios where this applies")
    recommended_methods: List[str] = Field(default_factory=list, description="Valid techniques or algorithms")
    selection_rules: List[str] = Field(default_factory=list, description="Decision logic for choosing between methods")
    limitations: List[str] = Field(default_factory=list, description="Caveats, edge cases, risks, or trade-offs")
    source: str = Field("AADA Internal Methodology", description="Authoritative reference or standard")
    version: str = Field("1.0", description="Semantic version string, e.g. 1.0, 1.1")
    effective_date: str = Field("2026-01-01", description="Effective date in YYYY-MM-DD")
    status: str = Field("active", description="Status: active, deprecated, or draft")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Completeness score")

    @field_validator("knowledge_id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        v = v.strip().upper()
        if not v or "-" not in v:
            raise ValueError(f"Invalid knowledge_id '{v}'. Must follow format PREFIX-NUM (e.g. DQ-001)")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid_statuses = {"active", "deprecated", "draft"}
        v = v.strip().lower()
        if v not in valid_statuses:
            raise ValueError(f"Invalid status '{v}'. Must be one of {valid_statuses}")
        return v

    def compute_quality_score(self) -> float:
        """Compute an automated quality/completeness score (0.0 to 1.0)."""
        score = 0.0
        if len(self.title) >= 5:
            score += 0.15
        if len(self.description) >= 50:
            score += 0.25
        elif len(self.description) >= 20:
            score += 0.15
        if len(self.conditions) >= 2:
            score += 0.15
        elif len(self.conditions) >= 1:
            score += 0.08
        if len(self.recommended_methods) >= 2:
            score += 0.15
        elif len(self.recommended_methods) >= 1:
            score += 0.08
        if len(self.selection_rules) >= 1:
            score += 0.15
        if len(self.limitations) >= 1:
            score += 0.15
        return round(min(score, 1.0), 2)

    def to_searchable_text(self) -> str:
        """Format complete knowledge item into a unified semantic search document."""
        parts = [
            f"Title: {self.title}",
            f"Category: {self.category.value}",
            f"Subcategory: {self.subcategory}",
            f"Description: {self.description}",
        ]
        if self.conditions:
            parts.append("Conditions: " + " | ".join(self.conditions))
        if self.recommended_methods:
            parts.append("Recommended Methods: " + " | ".join(self.recommended_methods))
        if self.selection_rules:
            parts.append("Selection Rules: " + " | ".join(self.selection_rules))
        if self.limitations:
            parts.append("Limitations: " + " | ".join(self.limitations))
        return "\n".join(parts)


class RetrievalQuery(BaseModel):
    """Query contract submitted by an agent or user to the Knowledge Base."""
    query: str = Field(..., min_length=1, description="Natural language analytics question or intent")
    category: Optional[KnowledgeCategory] = Field(None, description="Optional domain filter")
    subcategory: Optional[str] = Field(None, description="Optional subcategory filter")
    source: Optional[str] = Field(None, description="Optional source filter")
    version: Optional[str] = Field(None, description="Optional version filter")
    status: Optional[str] = Field("active", description="Filter by status (default: active)")
    top_k: int = Field(5, ge=1, le=50, description="Max number of results to return")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum relevance threshold")


class SearchResult(BaseModel):
    """Individual retrieved knowledge item with matching scores and ranking."""
    item: KnowledgeItem
    score: float = Field(..., description="Hybrid combined score (0.0 - 1.0)")
    semantic_score: float = Field(..., description="Cosine similarity score (0.0 - 1.0)")
    keyword_score: float = Field(..., description="BM25/TF-IDF keyword score (0.0 - 1.0)")
    rank: int = Field(..., ge=1, description="1-based rank position")


class RetrievalResponse(BaseModel):
    """Standardized response object returned by knowledge retrieval tools and endpoints."""
    query: str
    total_found: int
    results: List[SearchResult]
    latency_ms: float
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
