"""Tests for approved evidence store and traceable chunking."""
from pathlib import Path
from src.utils.io import load_json
from src.evidence.evidence_validator import EvidenceValidator

def test_evidence_chunks_exist_and_valid():
    chunks_dir = Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/evidence/chunks")
    chunk_files = list(chunks_dir.glob("*_chunks.json"))
    assert len(chunk_files) >= 4
    validator = EvidenceValidator()
    
    total_checked = 0
    for cf in chunk_files:
        chunks = load_json(cf)
        for chk in chunks:
            is_valid, msg = validator.validate_chunk(chk)
            assert is_valid, f"Chunk {chk.get('chunk_id')} failed validation: {msg}"
            total_checked += 1
    assert total_checked >= 10

def test_chunk_traceability_metadata():
    chunks_dir = Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/evidence/chunks")
    chunk_files = list(chunks_dir.glob("*_chunks.json"))
    sample_chunk = load_json(chunk_files[0])[0]
    assert "chunk_id" in sample_chunk
    assert "document_id" in sample_chunk
    assert "section" in sample_chunk
    assert "page" in sample_chunk
    assert "source" in sample_chunk
    assert "version" in sample_chunk
    assert sample_chunk["approval_status"] == "approved"
    assert len(sample_chunk["text"]) >= 10
