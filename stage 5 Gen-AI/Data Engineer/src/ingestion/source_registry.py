"""Source registry validator and manager."""
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..utils.io import load_yaml
from ..utils.hashing import compute_file_hash
from ..utils.logging import get_logger

logger = get_logger("source_registry")

class SourceRegistry:
    def __init__(self, registry_path: str | Path = "C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/source_registry.yaml"):
        self.registry_path = Path(registry_path)
        self.config = load_yaml(self.registry_path)
        self.sources: Dict[str, Dict[str, Any]] = {
            s["source_id"]: s for s in self.config.get("sources", [])
        }

    def get_source(self, source_id: str) -> Dict[str, Any]:
        if source_id not in self.sources:
            raise KeyError(f"Source ID '{source_id}' is not registered in {self.registry_path}")
        return self.sources[source_id]

    def is_approved(self, source_id: str) -> bool:
        src = self.get_source(source_id)
        return src.get("status") == "approved"

    def validate_source_approval(self, source_id: str) -> None:
        if not self.is_approved(source_id):
            raise PermissionError(f"Source '{source_id}' is NOT approved for Stage 5 ingestion.")

    def list_approved_sources(self) -> List[str]:
        return [sid for sid, s in self.sources.items() if s.get("status") == "approved"]

    def verify_source_file(self, source_id: str, filepath: str | Path) -> Dict[str, Any]:
        self.validate_source_approval(source_id)
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Registered file for '{source_id}' does not exist: {p}")
        file_hash = compute_file_hash(p)
        logger.info(f"Verified source '{source_id}' [{p}] - SHA-256: {file_hash[:12]}...")
        return {
            "source_id": source_id,
            "path": str(p),
            "file_hash": file_hash,
            "version": self.sources[source_id].get("version", "unknown"),
            "status": "approved"
        }
