"""Tests for cryptographic manifests and lineage traversal."""
from pathlib import Path
from src.utils.io import load_json
from src.provenance.lineage import LineageTracker

def test_manifest_files_exist():
    mdir = Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/manifests")
    for m in ["dataset_manifest.json", "distribution_manifest.json", "evidence_manifest.json", "schema_manifest.json"]:
        p = mdir / m
        assert p.exists(), f"Missing manifest: {m}"
        data = load_json(p)
        assert "created_at_utc" in data

def test_manifest_hashes_non_empty():
    dist_manifest = load_json("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/manifests/distribution_manifest.json")
    for fname, info in dist_manifest["distributions"].items():
        assert len(info["file_hash"]) == 64
        assert info["verified"] is True

def test_programmatic_question_answers():
    lineage = LineageTracker()
    
    # Q1: Where did mutation probability come from?
    q1 = lineage.trace_mutation_probability("KRAS")
    assert q1["traceable"] is True
    assert q1["source_id"] == "PROJECT_STAGE1"
    assert q1["distribution_version"] == "v1.0"

    # Q4/Q5: Trace evidence chunk
    q4 = lineage.trace_evidence_chunk("CHUNK-DOC-SPEC-001-0001")
    assert q4["traceable"] is True
    assert q4["approval_status"] == "approved"

    # Q5: All sources traceable
    assert lineage.verify_all_sources_traceable() is True
