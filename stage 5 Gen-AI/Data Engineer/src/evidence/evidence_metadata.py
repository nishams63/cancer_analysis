"""Metadata management for approved evidence documents."""
from pathlib import Path
from typing import Dict, Any
from ..utils.hashing import compute_file_hash

class EvidenceMetadataManager:
    @staticmethod
    def create_doc_metadata(
        document_id: str,
        title: str,
        source: str,
        source_type: str,
        version: str,
        evidence_category: str,
        filepath: Path,
        approval_status: str = "approved"
    ) -> Dict[str, Any]:
        return {
            "document_id": document_id,
            "title": title,
            "source": source,
            "source_type": source_type,
            "version": version,
            "approval_status": approval_status,
            "evidence_category": evidence_category,
            "file_hash": compute_file_hash(filepath),
            "file_size_bytes": filepath.stat().st_size
        }
