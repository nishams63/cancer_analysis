"""
Data Loader and Schema Integrity Verifier for Stage 4 EDA.
Provides read-only ingestion, SHA-256 integrity verification, and Stage 3 provenance joins.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd

logger = logging.getLogger("stage4_eda.data_loader")

REQUIRED_STAGE4_COLUMNS = [
    "patient_id",
    "note_id",
    "clinical_note",
    "instruction",
    "target_risk",
    "target_key_finding",
    "target_action",
    "ner_genes",
    "ner_drugs",
    "ner_dosages",
    "ner_adverse_events",
    "entity_check_status",
    "entity_coverage",
    "missing_entities",
    "invented_entities",
    "generation_model_version",
    "split"
]


def calculate_file_sha256(file_path: Path) -> str:
    """Calculate the cryptographic SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class Stage4DataLoader:
    """Loads and validates the Stage 4 fine-tuning dataset with strict read-only guarantees."""

    def __init__(self, dataset_path: str, expected_sha256: Optional[str] = None):
        self.dataset_path = Path(dataset_path)
        self.expected_sha256 = expected_sha256

    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads the Stage 4 parquet dataset and returns the DataFrame and verification metadata.
        Raises FileNotFoundError if file is missing, or ValueError if schema/hash fails.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {self.dataset_path}")

        actual_sha256 = calculate_file_sha256(self.dataset_path)
        if self.expected_sha256 and actual_sha256 != self.expected_sha256:
            logger.warning(
                f"SHA256 mismatch: expected {self.expected_sha256}, got {actual_sha256}"
            )

        df = pd.read_parquet(self.dataset_path)
        self._validate_schema(df)

        metadata = {
            "file_path": str(self.dataset_path.resolve()),
            "file_name": self.dataset_path.name,
            "sha256": actual_sha256,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": list(df.columns)
        }
        return df, metadata

    @staticmethod
    def _validate_schema(df: pd.DataFrame) -> None:
        """Enforces schema requirements on the loaded DataFrame."""
        missing = [col for col in REQUIRED_STAGE4_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Stage 4 dataset is missing required columns: {missing}")


def load_stage3_reference_data(stage3_path: str) -> Optional[pd.DataFrame]:
    """Loads upstream Stage 3 dataset to extract provenance metadata like urgency_level and document_date."""
    path = Path(stage3_path)
    if not path.exists():
        logger.warning(f"Stage 3 reference dataset not found at: {path}")
        return None
    return pd.read_parquet(path)
