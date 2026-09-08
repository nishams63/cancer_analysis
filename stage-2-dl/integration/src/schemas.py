"""
Stage 2 Deep Learning - Multimodal Integration Data Contracts & Schemas

Typed Pydantic schemas defining contracts for:
  - Input validation (Patient, Tiles, Longitudinal Biomarkers)
  - Modality outputs (Pathology, Attention-MIL, Temporal, Fusion)
  - Reliability layers (Uncertainty, Calibration, OOD)
  - Patient-level response and Batch response
"""
import sys
from pathlib import Path
_STAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_STAGE_ROOT / '.runtime') not in sys.path:
    sys.path.insert(0, str(_STAGE_ROOT / '.runtime'))

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class BiomarkerObservation(BaseModel):
    """Single chronological biomarker observation visit (must be <= Day 90)."""
    days_from_baseline: int = Field(..., ge=0, le=90, description="Days elapsed since baseline (must be <= 90)")
    ctDNA_vaf_percent: Optional[float] = Field(None, ge=0.0, description="ctDNA variant allele frequency (%)")
    cea_ng_ml: Optional[float] = Field(None, ge=0.0, description="Carcinoembryonic antigen (ng/mL)")
    ca125_u_ml: Optional[float] = Field(None, ge=0.0, description="Cancer antigen 125 (U/mL)")
    ldh_u_l: Optional[float] = Field(None, ge=0.0, description="Lactate dehydrogenase (U/L)")
    crp_mg_l: Optional[float] = Field(None, ge=0.0, description="C-reactive protein (mg/L)")
    delta_days: Optional[float] = Field(None, ge=0.0, description="Days since previous visit (>= 0)")
    ctDNA_velocity_30d: Optional[float] = Field(None, description="Backward-looking 30-day velocity")
    ctDNA_missing: Optional[int] = Field(0, description="ctDNA missingness mask (1 if imputed)")
    cea_missing: Optional[int] = Field(0, description="CEA missingness mask")
    ca125_missing: Optional[int] = Field(0, description="CA125 missingness mask")
    ldh_missing: Optional[int] = Field(0, description="LDH missingness mask")
    crp_missing: Optional[int] = Field(0, description="CRP missingness mask")


class PatientInferenceRequest(BaseModel):
    """Unified request for patient-level multimodal inference."""
    patient_id: str = Field(..., min_length=1, description="Unique patient identifier")
    tile_paths: Optional[List[str]] = Field(None, description="Paths to biopsy pathology tiles")
    temporal_observations: Optional[List[BiomarkerObservation]] = Field(
        None, description="Chronological biomarker history strictly within Day 0-90"
    )
    model_version: Optional[str] = Field(
        'auto',
        pattern='^(auto|upgraded|baseline|validated|pathology_only|temporal_only)$',
        description="Model version selector ('auto', 'upgraded', 'baseline', 'validated')"
    )
    model_configuration: Optional[str] = Field(
        'auto',
        pattern='^(auto|upgraded|baseline|validated|pathology_only|temporal_only)$',
        description="Target model configuration ('auto', 'upgraded', 'baseline', etc.)"
    )
    fusion_mode: Optional[str] = Field(
        'gated',
        pattern='^(gated|concat_mlp|cross_attention|fixed_weighted)$',
        description="Multimodal fusion strategy"
    )
    aggregation_method: Optional[str] = Field(
        'attention',
        pattern='^(attention|mean|median|max)$',
        description="Tile aggregation method ('attention' for Gated Attention-MIL)"
    )
    include_explanations: bool = Field(
        False,
        description="Whether to include expensive explainability outputs (Grad-CAM, feature sensitivities)"
    )


class BatchInferenceRequest(BaseModel):
    """Batch request for multiple patients."""
    patients: List[PatientInferenceRequest] = Field(..., min_length=1, max_length=100)


# --- Intermediate & Output Schemas ---

class TileAttentionInfo(BaseModel):
    """Individual tile attention weight from Attention-MIL."""
    tile_index: int
    tile_path: Optional[str] = None
    attention_weight: float
    rank: int


class PathologyOutput(BaseModel):
    """Pathology modality inference summary."""
    available: bool = True
    model_name: str
    malignant_probability: Optional[float] = None
    benign_probability: Optional[float] = None
    inflammation_probability: Optional[float] = None
    aggregation_method: str = "attention"
    num_tiles_analyzed: int = 0
    attention_tiles: Optional[List[TileAttentionInfo]] = None
    embedding_dimension: int = 128


class TemporalOutput(BaseModel):
    """Temporal modality inference summary."""
    available: bool = True
    model_name: str
    progression_probability: Optional[float] = None
    ctdna_30d_forecast: Optional[float] = None
    sequence_length: int = 0
    max_historical_day: int = 0
    embedding_dimension: int = 64


class FusionOutput(BaseModel):
    """Multimodal fusion output."""
    available: bool = True
    model_name: str
    progression_probability: Optional[float] = None
    ctdna_forecast: Optional[float] = None
    multimodal_risk_score: Optional[float] = None
    fusion_dimension: int = 32


class UncertaintyOutput(BaseModel):
    """Epistemic uncertainty and calibration parameters."""
    confidence: Optional[float] = None
    uncertainty_score: Optional[float] = None
    predictive_variance: Optional[float] = None
    method: str = "mc_dropout_15_passes"
    calibration_status: str = "NOT_CALIBRATED"
    confidence_definition: str = "Maximum predicted class probability; not a clinical guarantee"


class OODOutput(BaseModel):
    """Out-of-distribution detection summary."""
    status: str = "NOT_EVALUATED"  # "IN_DISTRIBUTION", "POTENTIAL_DISTRIBUTION_SHIFT", "NOT_EVALUATED"
    mahalanobis_distance: Optional[float] = None
    threshold: Optional[float] = None


class RiskOutput(BaseModel):
    """Model-estimated research risk stratification."""
    level: str = "INSUFFICIENT_DATA"  # "LOW", "MODERATE", "HIGH", "INSUFFICIENT_DATA"
    joint_confidence_status: str = "INTERMEDIATE"  # "HIGH_RISK_HIGH_CONFIDENCE", etc.
    description: str = "Model-estimated research risk score. Not for clinical diagnosis."


class LatencySummary(BaseModel):
    """Component execution latency in milliseconds."""
    validation_ms: float = 0.0
    pathology_ms: float = 0.0
    temporal_ms: float = 0.0
    fusion_ms: float = 0.0
    uncertainty_ms: float = 0.0
    ood_ms: float = 0.0
    total_ms: float = 0.0


class PatientInferenceResponse(BaseModel):
    """Standardized response schema for patient-level multimodal inference."""
    patient_id: str
    inference_mode: str  # FULL_MULTIMODAL, PATHOLOGY_ONLY, TEMPORAL_ONLY, INSUFFICIENT_DATA
    available_modalities: List[str]
    model_configuration: str
    
    # Core predictions
    progression_probability: Optional[float] = None
    ctdna_forecast_30d: Optional[float] = None
    risk: RiskOutput
    
    # Detailed sub-modality blocks
    pathology: Optional[PathologyOutput] = None
    temporal: Optional[TemporalOutput] = None
    fusion: Optional[FusionOutput] = None
    
    # Reliability & Safety
    uncertainty: UncertaintyOutput
    ood: OODOutput
    
    # Observability & Latency
    latency: LatencySummary
    system: Dict[str, Any]
    
    # Mandatory disclaimers
    mandatory_disclaimer: str
    equivalence_disclaimer: str


class PatientBatchResultItem(BaseModel):
    """Single patient result within a batch response."""
    patient_id: str
    status: str  # "success" or "error"
    status_code: int = 200
    detail: Optional[str] = None
    prediction: Optional[PatientInferenceResponse] = None


class BatchInferenceResponse(BaseModel):
    """Response schema for batch inference."""
    total: int
    successful: int
    failed: int
    results: List[PatientBatchResultItem]
