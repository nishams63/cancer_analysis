"""Evidence validation against approval and integrity constraints."""
from typing import Dict, Any, Tuple
from ..utils.hashing import compute_file_hash, compute_string_hash

class EvidenceValidator:
    def validate_document_metadata(self, meta: Dict[str, Any]) -> Tuple[bool, str]:
        required = ["document_id", "title", "source", "version", "approval_status", "evidence_category"]
        for field in required:
            if field not in meta or not meta[field]:
                return False, f"Missing required metadata field: '{field}'"
        if meta.get("approval_status") != "approved":
            return False, f"Document '{meta.get('document_id')}' status is not 'approved' (got '{meta.get('approval_status')}')"
        return True, "Valid"

    def validate_chunk(self, chunk: Dict[str, Any]) -> Tuple[bool, str]:
        required = ["chunk_id", "document_id", "text", "source", "version", "approval_status", "evidence_category"]
        for field in required:
            if field not in chunk or not chunk[field]:
                return False, f"Missing required chunk field: '{field}'"
        if len(str(chunk["text"]).strip()) < 10:
            return False, "Chunk text is too short or empty"
        if chunk.get("approval_status") != "approved":
            return False, "Chunk approval status is not 'approved'"
        return True, "Valid"
