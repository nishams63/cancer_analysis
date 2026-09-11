"""Traceable chunker splitting evidence documents into schema-compliant chunks."""
from typing import List, Dict, Any
from .evidence_cleaner import EvidenceCleaner

class EvidenceChunker:
    def __init__(self, chunk_size_words: int = 150, chunk_overlap_words: int = 30):
        self.chunk_size = chunk_size_words
        self.chunk_overlap = chunk_overlap_words
        self.cleaner = EvidenceCleaner()

    def chunk_document(self, doc_text: str, doc_metadata: Dict[str, Any], default_section: str = "General") -> List[Dict[str, Any]]:
        cleaned = self.cleaner.clean_text(doc_text)
        words = cleaned.split()
        chunks: List[Dict[str, Any]] = []
        if not words:
            return chunks

        step = max(1, self.chunk_size - self.chunk_overlap)
        chunk_idx = 1

        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            # Estimate page number (approx 350 words per page)
            page_estimate = (i // 350) + 1

            chunk_id = f"CHUNK-{doc_metadata['document_id']}-{chunk_idx:04d}"
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc_metadata["document_id"],
                "section": default_section,
                "page": page_estimate,
                "text": chunk_text,
                "word_count": len(chunk_words),
                "source": doc_metadata["source"],
                "version": doc_metadata["version"],
                "approval_status": doc_metadata["approval_status"],
                "evidence_category": doc_metadata["evidence_category"]
            })
            chunk_idx += 1

        return chunks
