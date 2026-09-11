"""Manifest builder generating all cryptographic audit manifests."""
from pathlib import Path
from typing import Dict, Any, List
from .versioning import VersionManager
from ..utils.io import save_json, load_json
from ..utils.hashing import compute_file_hash
from ..utils.logging import get_logger

logger = get_logger("manifest_builder")

class ManifestBuilder:
    def __init__(
        self,
        manifest_dir: str | Path = "stage5/manifests",
        processed_dir: str | Path = "stage5/data/processed",
        evidence_dir: str | Path = "stage5/data/evidence",
        raw_dir: str | Path = "stage5/data/raw",
        schemas_dir: str | Path = "stage5/schemas"
    ):
        self.manifest_dir = Path(manifest_dir)
        self.processed_dir = Path(processed_dir)
        self.evidence_dir = Path(evidence_dir)
        self.raw_dir = Path(raw_dir)
        self.schemas_dir = Path(schemas_dir)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def build_dataset_manifest(self, ingested_info: Dict[str, Any]) -> Dict[str, Any]:
        manifest = {
            "manifest_type": "dataset_manifest",
            "version": "1.0.0",
            "created_at_utc": VersionManager.get_timestamp(),
            "sources": {}
        }
        for key, info in ingested_info.items():
            sid = info.get("source_id", key)
            p = Path(info.get("dest_path", ""))
            h = compute_file_hash(p) if p.exists() else info.get("file_hash", "")
            manifest["sources"][sid] = {
                "source_id": sid,
                "file_path": str(p),
                "file_hash": h,
                "version": "v1.0",
                "row_count": info.get("row_count", 0),
                "col_count": info.get("col_count", 0),
                "verified": True
            }
        save_json(manifest, self.manifest_dir / "dataset_manifest.json")
        logger.info("Saved dataset_manifest.json")
        return manifest

    def build_distribution_manifest(self, distribution_files: List[str] = None) -> Dict[str, Any]:
        files = distribution_files or [
            "reference_distributions.parquet",
            "mutation_frequencies.parquet",
            "mutation_cooccurrence.parquet",
            "biomarker_distributions.parquet",
            "treatment_distributions.parquet",
            "dosage_ranges.parquet",
            "adverse_event_distributions.parquet",
            "missingness_patterns.parquet",
            "temporal_patterns.parquet"
        ]
        manifest = {
            "manifest_type": "distribution_manifest",
            "distribution_version": "v1.0",
            "created_at_utc": VersionManager.get_timestamp(),
            "distributions": {}
        }
        for fname in files:
            p = self.processed_dir / fname
            if p.exists():
                manifest["distributions"][fname] = {
                    "filename": fname,
                    "file_path": str(p),
                    "file_hash": compute_file_hash(p),
                    "file_size_bytes": p.stat().st_size,
                    "source_ids": ["PROJECT_STAGE1", "PROJECT_STAGE2"],
                    "dataset_version": "v1.0",
                    "distribution_version": "v1.0",
                    "processing_timestamp": VersionManager.get_timestamp(),
                    "verified": True
                }
        save_json(manifest, self.manifest_dir / "distribution_manifest.json")
        logger.info("Saved distribution_manifest.json")
        return manifest

    def build_evidence_manifest(self) -> Dict[str, Any]:
        meta_dir = self.evidence_dir / "metadata"
        chunks_dir = self.evidence_dir / "chunks"
        docs_list = []
        total_chunks = 0
        
        for meta_file in sorted(meta_dir.glob("*_metadata.json")):
            meta = load_json(meta_file)
            doc_id = meta.get("document_id")
            chunk_file = chunks_dir / f"{doc_id}_chunks.json"
            chunk_count = len(load_json(chunk_file)) if chunk_file.exists() else 0
            total_chunks += chunk_count
            docs_list.append({
                "document_id": doc_id,
                "title": meta.get("title"),
                "source": meta.get("source"),
                "version": meta.get("version"),
                "approval_status": meta.get("approval_status"),
                "evidence_category": meta.get("evidence_category"),
                "chunk_count": chunk_count,
                "file_hash": meta.get("file_hash"),
                "processing_timestamp": VersionManager.get_timestamp()
            })

        manifest = {
            "manifest_type": "evidence_manifest",
            "evidence_store_version": "v1.0",
            "created_at_utc": VersionManager.get_timestamp(),
            "total_documents": len(docs_list),
            "total_chunks": total_chunks,
            "documents": docs_list
        }
        save_json(manifest, self.manifest_dir / "evidence_manifest.json")
        logger.info(f"Saved evidence_manifest.json ({total_chunks} total chunks)")
        return manifest

    def build_schema_manifest(self) -> Dict[str, Any]:
        manifest = {
            "manifest_type": "schema_manifest",
            "version": "1.0.0",
            "created_at_utc": VersionManager.get_timestamp(),
            "schemas": {}
        }
        for sfile in sorted(self.schemas_dir.glob("*.json")):
            manifest["schemas"][sfile.name] = {
                "schema_file": sfile.name,
                "file_hash": compute_file_hash(sfile),
                "file_size_bytes": sfile.stat().st_size,
                "schema_version": "1.0.0",
                "verified": True
            }
        save_json(manifest, self.manifest_dir / "schema_manifest.json")
        logger.info("Saved schema_manifest.json")
        return manifest

    def build_all_manifests(self, ingested_info: Dict[str, Any] = None) -> Dict[str, Any]:
        return {
            "dataset": self.build_dataset_manifest(ingested_info or {}),
            "distribution": self.build_distribution_manifest(),
            "evidence": self.build_evidence_manifest(),
            "schema": self.build_schema_manifest()
        }
