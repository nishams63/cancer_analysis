"""Evidence store loader and orchestrator."""
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
from .evidence_cleaner import EvidenceCleaner
from .evidence_chunker import EvidenceChunker
from .evidence_metadata import EvidenceMetadataManager
from .evidence_validator import EvidenceValidator
from ..utils.io import save_json
from ..utils.logging import get_logger

logger = get_logger("evidence_loader")

class EvidenceLoader:
    def __init__(self, evidence_dir: str | Path = "C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/evidence"):
        self.evidence_dir = Path(evidence_dir)
        self.docs_dir = self.evidence_dir / "documents"
        self.chunks_dir = self.evidence_dir / "chunks"
        self.metadata_dir = self.evidence_dir / "metadata"
        self.approved_dir = self.evidence_dir / "approved_sources"
        self.rejected_dir = self.evidence_dir / "rejected"

        for d in [self.docs_dir, self.chunks_dir, self.metadata_dir, self.approved_dir, self.rejected_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.validator = EvidenceValidator()
        self.chunker = EvidenceChunker()

    def process_document(
        self,
        filepath: Path,
        document_id: str,
        title: str,
        source: str,
        source_type: str,
        version: str,
        evidence_category: str,
        section: str = "Clinical Guidelines"
    ) -> Dict[str, Any]:
        dest_doc = self.docs_dir / filepath.name
        if filepath.resolve() != dest_doc.resolve():
            shutil.copy2(filepath, dest_doc)

        meta = EvidenceMetadataManager.create_doc_metadata(
            document_id=document_id,
            title=title,
            source=source,
            source_type=source_type,
            version=version,
            evidence_category=evidence_category,
            filepath=dest_doc
        )

        is_valid, msg = self.validator.validate_document_metadata(meta)
        if not is_valid:
            logger.warning(f"Rejected document {document_id}: {msg}")
            save_json({"metadata": meta, "rejection_reason": msg}, self.rejected_dir / f"{document_id}_rejected.json")
            return {"status": "rejected", "reason": msg}

        # Read text content
        if dest_doc.suffix.lower() in [".txt", ".md"]:
            doc_text = dest_doc.read_text(encoding="utf-8")
        elif dest_doc.suffix.lower() == ".json":
            data = json.loads(dest_doc.read_text(encoding="utf-8"))
            doc_text = json.dumps(data, indent=2)
        else:
            doc_text = f"Binary clinical reference document: {title}. See original document at {dest_doc.name}"

        chunks = self.chunker.chunk_document(doc_text, meta, default_section=section)

        # Validate all chunks
        valid_chunks = []
        for chk in chunks:
            c_valid, c_msg = self.validator.validate_chunk(chk)
            if c_valid:
                valid_chunks.append(chk)

        # Save metadata and chunks
        save_json(meta, self.metadata_dir / f"{document_id}_metadata.json")
        save_json(valid_chunks, self.chunks_dir / f"{document_id}_chunks.json")

        logger.info(f"Processed evidence {document_id}: {len(valid_chunks)} valid chunks. Category: {evidence_category}")
        return {
            "status": "approved",
            "metadata": meta,
            "chunk_count": len(valid_chunks),
            "chunks": valid_chunks
        }
