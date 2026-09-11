"""Loader for upstream project datasets (Stages 1–4) ensuring raw immutability."""
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from .source_registry import SourceRegistry
from ..utils.io import ensure_parent_dir
from ..utils.hashing import compute_file_hash
from ..utils.logging import get_logger

logger = get_logger("load_project_data")

class ProjectDataLoader:
    def __init__(self, registry: SourceRegistry, raw_project_dir: str | Path = "C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/raw/project"):
        self.registry = registry
        self.raw_project_dir = Path(raw_project_dir)
        self.raw_project_dir.mkdir(parents=True, exist_ok=True)

    def ingest_stage1_data(self) -> Dict[str, Any]:
        src = self.registry.get_source("PROJECT_STAGE1")
        self.registry.validate_source_approval("PROJECT_STAGE1")
        src_path = Path(src["original_path"])
        dest_path = self.raw_project_dir / "master_patient_dataset.csv"
        
        # Copy to preserve raw original immutability
        shutil.copy2(src_path, dest_path)
        file_hash = compute_file_hash(dest_path)
        df = pd.read_csv(dest_path)
        logger.info(f"Ingested Stage 1 ML data: {df.shape[0]} rows, {df.shape[1]} cols. Hash: {file_hash[:12]}")
        return {
            "source_id": "PROJECT_STAGE1",
            "dest_path": str(dest_path),
            "file_hash": file_hash,
            "row_count": len(df),
            "col_count": len(df.columns),
            "dataframe": df
        }

    def ingest_stage2_data(self) -> Dict[str, Any]:
        src = self.registry.get_source("PROJECT_STAGE2")
        self.registry.validate_source_approval("PROJECT_STAGE2")
        src_path = Path(src["original_path"])
        dest_path = self.raw_project_dir / "biomarkers_processed.csv"
        
        shutil.copy2(src_path, dest_path)
        file_hash = compute_file_hash(dest_path)
        df = pd.read_csv(dest_path)
        logger.info(f"Ingested Stage 2 DL data: {df.shape[0]} rows, {df.shape[1]} cols. Hash: {file_hash[:12]}")
        return {
            "source_id": "PROJECT_STAGE2",
            "dest_path": str(dest_path),
            "file_hash": file_hash,
            "row_count": len(df),
            "col_count": len(df.columns),
            "dataframe": df
        }

    def ingest_stage3_data(self) -> Dict[str, Any]:
        src = self.registry.get_source("PROJECT_STAGE3")
        self.registry.validate_source_approval("PROJECT_STAGE3")
        src_path = Path(src["original_path"])
        dest_path = self.raw_project_dir / "clinical_nlp_dataset_v1.parquet"
        
        shutil.copy2(src_path, dest_path)
        file_hash = compute_file_hash(dest_path)
        df = pd.read_parquet(dest_path)
        logger.info(f"Ingested Stage 3 NLP data: {df.shape[0]} rows, {df.shape[1]} cols. Hash: {file_hash[:12]}")
        return {
            "source_id": "PROJECT_STAGE3",
            "dest_path": str(dest_path),
            "file_hash": file_hash,
            "row_count": len(df),
            "col_count": len(df.columns),
            "dataframe": df
        }

    def ingest_stage4_data(self) -> Dict[str, Any]:
        src = self.registry.get_source("PROJECT_STAGE4")
        self.registry.validate_source_approval("PROJECT_STAGE4")
        src_path = Path(src["original_path"])
        dest_path = self.raw_project_dir / "slm_finetune_dataset_v1.parquet"
        
        shutil.copy2(src_path, dest_path)
        file_hash = compute_file_hash(dest_path)
        df = pd.read_parquet(dest_path)
        logger.info(f"Ingested Stage 4 SLM data: {df.shape[0]} rows, {df.shape[1]} cols. Hash: {file_hash[:12]}")
        return {
            "source_id": "PROJECT_STAGE4",
            "dest_path": str(dest_path),
            "file_hash": file_hash,
            "row_count": len(df),
            "col_count": len(df.columns),
            "dataframe": df
        }

    def ingest_all_project_data(self) -> Dict[str, Dict[str, Any]]:
        return {
            "stage1": self.ingest_stage1_data(),
            "stage2": self.ingest_stage2_data(),
            "stage3": self.ingest_stage3_data(),
            "stage4": self.ingest_stage4_data()
        }
