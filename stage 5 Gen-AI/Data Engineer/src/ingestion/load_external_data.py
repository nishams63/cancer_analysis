"""Loader for approved external oncology reference datasets."""
import json
from pathlib import Path
from typing import Dict, Any
from .source_registry import SourceRegistry
from ..utils.hashing import compute_file_hash
from ..utils.logging import get_logger

logger = get_logger("load_external_data")

class ExternalDataLoader:
    def __init__(self, registry: SourceRegistry, raw_external_dir: str | Path = "C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/raw/external"):
        self.registry = registry
        self.raw_external_dir = Path(raw_external_dir)

    def load_nccn_biomarkers(self) -> Dict[str, Any]:
        self.registry.validate_source_approval("NCCN_NSCLC_REF_002")
        p = self.raw_external_dir / "nccn_nslc_biomarkers.json"
        file_hash = compute_file_hash(p)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded NCCN reference: {len(data.get('biomarkers', []))} biomarkers. Hash: {file_hash[:12]}")
        return {"source_id": "NCCN_NSCLC_REF_002", "path": str(p), "file_hash": file_hash, "data": data}

    def load_fda_drugs(self) -> Dict[str, Any]:
        self.registry.validate_source_approval("FDA_DRUG_LABEL_003")
        p = self.raw_external_dir / "fda_oncology_drugs.json"
        file_hash = compute_file_hash(p)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded FDA drug labeling: {len(data.get('drugs', []))} drugs. Hash: {file_hash[:12]}")
        return {"source_id": "FDA_DRUG_LABEL_003", "path": str(p), "file_hash": file_hash, "data": data}

    def load_ctcae_categories(self) -> Dict[str, Any]:
        self.registry.validate_source_approval("CTCAE_TOXICITY_004")
        p = self.raw_external_dir / "ctcae_v5_categories.json"
        file_hash = compute_file_hash(p)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded CTCAE organ toxicity: {len(data.get('organ_systems', []))} systems. Hash: {file_hash[:12]}")
        return {"source_id": "CTCAE_TOXICITY_004", "path": str(p), "file_hash": file_hash, "data": data}

    def load_all_external(self) -> Dict[str, Any]:
        return {
            "nccn": self.load_nccn_biomarkers(),
            "fda": self.load_fda_drugs(),
            "ctcae": self.load_ctcae_categories()
        }
