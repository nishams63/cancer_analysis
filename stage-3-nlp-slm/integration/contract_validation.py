"""
Contract Definition and Schema Validation Module for Stage 3 to Stage 4 Integration.
Enforces strict schema validation at Stage 3 output boundary and Stage 4 input boundary.
Zero malformed payloads are silently passed downstream.
"""

from typing import Dict, Any, List, Optional, Literal, Union
from datetime import datetime, timezone
import re
import json
import logging
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ValidationError,
    ConfigDict
)

logger = logging.getLogger("stage3_stage4.contract_validation")

SUPPORTED_SCHEMA_VERSION = "1.0.0"
SEMVER_REGEX = re.compile(r"^\d+\.\d+\.\d+$")
HEX_SHA256_REGEX = re.compile(r"^[a-f0-9]{64}$")

URGENCY_CLASSES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
HAZARD_CLASSES = (
    "NONE",
    "HEMATOLOGIC",
    "HEPATIC",
    "RENAL",
    "CARDIAC",
    "PULMONARY",
    "NEUROPATHIC",
    "DERMATOLOGIC"
)
ENTITY_LABELS = ("GENE_MUTATION", "DRUG_NAME", "DOSAGE", "ADVERSE_EVENT")
POLARITY_VALUES = ("AFFIRMED", "NEGATED", "HISTORICAL", "RESOLVED")


class ContractValidationError(ValueError):
    """Raised when an integration payload fails schema validation."""
    def __init__(self, message: str, errors: Optional[List[Dict[str, Any]]] = None, raw_payload: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or []
        self.raw_payload = raw_payload


class TriageUrgencyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_class: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        ..., description="Categorical triage urgency class."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score in [0.0, 1.0]."
    )
    class_probabilities: Optional[Dict[str, float]] = Field(
        default=None, description="Optional class-wise posterior probability distribution."
    )

    @field_validator("class_probabilities")
    @classmethod
    def validate_probabilities(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if v is not None:
            for c, p in v.items():
                if c not in URGENCY_CLASSES:
                    raise ValueError(f"Unknown urgency class in probabilities: {c}")
                if not (0.0 <= p <= 1.0):
                    raise ValueError(f"Probability for {c} must be in [0.0, 1.0], got {p}")
        return v


class ToxicityHazardPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_class: Literal[
        "NONE",
        "HEMATOLOGIC",
        "HEPATIC",
        "RENAL",
        "CARDIAC",
        "PULMONARY",
        "NEUROPATHIC",
        "DERMATOLOGIC"
    ] = Field(..., description="Categorical toxicity hazard class.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score in [0.0, 1.0]."
    )
    class_probabilities: Optional[Dict[str, float]] = Field(
        default=None, description="Optional class-wise posterior probability distribution."
    )

    @field_validator("class_probabilities")
    @classmethod
    def validate_probabilities(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if v is not None:
            for c, p in v.items():
                if c not in HAZARD_CLASSES:
                    raise ValueError(f"Unknown hazard class in probabilities: {c}")
                if not (0.0 <= p <= 1.0):
                    raise ValueError(f"Probability for {c} must be in [0.0, 1.0], got {p}")
        return v


class ClinicalEntityPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int = Field(..., ge=0, description="0-indexed start character offset in clinical text.")
    end: int = Field(..., gt=0, description="0-indexed end character offset (exclusive).")
    label: Literal["GENE_MUTATION", "DRUG_NAME", "DOSAGE", "ADVERSE_EVENT"] = Field(
        ..., description="Standard 4-class taxonomy entity label."
    )
    text: str = Field(..., min_length=1, description="Entity text verbatim from clinical note.")
    polarity: Optional[Literal["AFFIRMED", "NEGATED", "HISTORICAL", "RESOLVED"]] = Field(
        default=None, description="Clinical polarity attribution."
    )
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Optional entity detection confidence."
    )

    @model_validator(mode="after")
    def check_span_consistency(self) -> "ClinicalEntityPayload":
        if self.end <= self.start:
            raise ValueError(f"Entity character offset end ({self.end}) must be strictly greater than start ({self.start})")
        return self


class Stage3OutputPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=SUPPORTED_SCHEMA_VERSION,
        description="Semver schema version string."
    )
    document_id: str = Field(..., min_length=1, description="Unique document identifier.")
    patient_id: str = Field(..., min_length=1, description="De-identified patient construct ID.")
    inference_timestamp: str = Field(
        ..., description="ISO 8601 UTC timestamp of NLP extraction."
    )
    model_version: str = Field(..., min_length=1, description="Triage/hazard classifier model version.")
    ner_model_version: str = Field(..., min_length=1, description="Clinical NER tagger model version.")

    triage_urgency: TriageUrgencyPayload
    toxicity_hazard: ToxicityHazardPayload
    clinical_entities: List[ClinicalEntityPayload] = Field(
        default_factory=list, description="List of extracted clinical entities."
    )
    raw_text_hash: Optional[str] = Field(
        default=None, description="SHA-256 hex digest of raw normalized document text."
    )
    integration_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Runtime telemetry metadata."
    )

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if not SEMVER_REGEX.match(v):
            raise ValueError(f"schema_version '{v}' must be a valid semver string (e.g. '1.0.0')")
        major = v.split(".")[0]
        if major != SUPPORTED_SCHEMA_VERSION.split(".")[0]:
            raise ValueError(
                f"Incompatible schema major version '{major}'. Expected major version '{SUPPORTED_SCHEMA_VERSION.split('.')[0]}'."
            )
        return v

    @field_validator("inference_timestamp")
    @classmethod
    def validate_iso_timestamp(cls, v: str) -> str:
        try:
            # Validate ISO 8601 string parseability
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception as e:
            raise ValueError(f"inference_timestamp '{v}' is not a valid ISO 8601 string: {e}")
        return v

    @field_validator("raw_text_hash")
    @classmethod
    def validate_sha256_hash(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not HEX_SHA256_REGEX.match(v):
            raise ValueError(f"raw_text_hash must be a 64-character lowercase hex SHA-256 digest, got '{v}'")
        return v


def validate_stage3_output(payload: Union[Dict[str, Any], Stage3OutputPayload]) -> Stage3OutputPayload:
    """
    Validates payload at Stage 3 output boundary.
    Raises ContractValidationError and logs rejection on malformed structure.
    """
    if isinstance(payload, Stage3OutputPayload):
        return payload
    if not isinstance(payload, dict):
        msg = f"Stage 3 output validation failed: expected dict or Stage3OutputPayload, got {type(payload).__name__}"
        logger.error(msg)
        raise ContractValidationError(msg, raw_payload=payload)
    try:
        validated = Stage3OutputPayload.model_validate(payload)
        return validated
    except ValidationError as e:
        logger.error(f"Stage 3 output boundary validation failed with {len(e.errors())} errors for doc {payload.get('document_id', 'UNKNOWN')}")
        raise ContractValidationError(
            f"Stage 3 output contract validation rejected: {e.errors()}",
            errors=e.errors(),
            raw_payload=payload
        ) from e


def validate_stage4_input(payload: Union[Dict[str, Any], Stage3OutputPayload]) -> Stage3OutputPayload:
    """
    Validates payload at Stage 4 input boundary.
    Guarantees Stage 4 treatment optimization engine never consumes corrupted or unverified data.
    """
    if isinstance(payload, Stage3OutputPayload):
        return payload
    if not isinstance(payload, dict):
        msg = f"Stage 4 input validation failed: expected dict or Stage3OutputPayload, got {type(payload).__name__}"
        logger.error(msg)
        raise ContractValidationError(msg, raw_payload=payload)
    try:
        validated = Stage3OutputPayload.model_validate(payload)
        return validated
    except ValidationError as e:
        logger.error(f"Stage 4 input boundary validation failed with {len(e.errors())} errors for doc {payload.get('document_id', 'UNKNOWN')}")
        raise ContractValidationError(
            f"Stage 4 input contract validation rejected: {e.errors()}",
            errors=e.errors(),
            raw_payload=payload
        ) from e


def export_canonical_json(payload: Stage3OutputPayload) -> str:
    """Serializes a Stage3OutputPayload to deterministic, canonical JSON for hashing and transmission."""
    return json.dumps(payload.model_dump(), sort_keys=True, separators=(",", ":"))
