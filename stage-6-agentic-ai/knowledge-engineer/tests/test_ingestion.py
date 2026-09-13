"""Unit tests for file loading, parsing, validation, chunking, and deduplication."""
import json
from pathlib import Path
from schemas.knowledge import KnowledgeItem, KnowledgeCategory
from ingestion.loader import KnowledgeLoader
from ingestion.parser import KnowledgeParser
from ingestion.validator import KnowledgeValidator
from ingestion.chunker import KnowledgeChunker
from ingestion.deduplicator import KnowledgeDeduplicator
from ingestion.ingest import run_ingestion_pipeline


def test_loader_finds_files(tmp_path):
    d = tmp_path / "data_quality"
    d.mkdir()
    f = d / "dq-001.json"
    f.write_text('{"knowledge_id": "DQ-001", "title": "Test Title"}', encoding="utf-8")

    loader = KnowledgeLoader(tmp_path)
    files = loader.load_all_files()
    assert len(files) == 1
    assert files[0][1]["knowledge_id"] == "DQ-001"


def test_parser_normalizes_fields():
    parser = KnowledgeParser()
    raw = {
        "KNOWLEDGE_ID": "DQ-001 ",
        "Title": " Normalization Test ",
        "Conditions": [" Cond 1 ", "Cond 2 "],
    }
    normalized = parser.normalize_raw_dict(raw)
    assert normalized["knowledge_id"] == "DQ-001"
    assert normalized["title"] == "Normalization Test"
    assert normalized["conditions"] == ["Cond 1", "Cond 2"]
    assert normalized["source"] == "AADA Internal Methodology"
    assert normalized["version"] == "1.0"


def test_validator_detects_malformed():
    validator = KnowledgeValidator()
    # Missing required description and category
    item, err = validator.validate_item({"knowledge_id": "DQ-001", "title": "No Category"})
    assert item is None
    assert err is not None


def test_chunker_handles_arrays_and_objects():
    chunker = KnowledgeChunker()
    # Array input
    items_array = [{"id": "1"}, {"id": "2"}]
    assert len(chunker.chunk(items_array)) == 2

    # Single object input
    single_obj = {"id": "3"}
    assert len(chunker.chunk(single_obj)) == 1

    # Wrapped object input
    wrapped_obj = {"knowledge_items": [{"id": "4"}, {"id": "5"}]}
    assert len(chunker.chunk(wrapped_obj)) == 2


def test_deduplicator_prevents_duplicates(sample_item):
    dedup = KnowledgeDeduplicator()
    is_dup, reason = dedup.check_duplicate(sample_item)
    assert not is_dup
    assert reason == "ok"

    # Same ID check
    is_dup2, reason2 = dedup.check_duplicate(sample_item)
    assert is_dup2
    assert "Duplicate knowledge_id" in reason2


def test_end_to_end_ingestion(tmp_path):
    kb_dir = tmp_path / "kb"
    (kb_dir / "eda").mkdir(parents=True)
    item_json = {
        "knowledge_id": "EDA-999",
        "title": "Temporary Ingestion Test",
        "category": "eda",
        "subcategory": "testing",
        "description": "Exploratory testing item for end-to-end ingestion pipeline.",
        "conditions": ["Condition 1", "Condition 2"],
        "recommended_methods": ["Method A", "Method B"],
        "selection_rules": ["Rule X"],
        "limitations": ["Limitation Y"],
        "source": "AADA Internal Methodology",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "status": "active",
    }
    (kb_dir / "eda" / "eda-999.json").write_text(json.dumps(item_json), encoding="utf-8")

    db_file = tmp_path / "ingest_test.db"
    index_dir = tmp_path / "index_test"

    report = run_ingestion_pipeline(kb_dir=kb_dir, db_path=db_file, index_dir=index_dir)
    assert report["total_files"] == 1
    assert report["valid_items"] == 1
    assert report["inserted_items"] == 1
    assert report["rejected_items"] == 0
    assert report["indexed_documents"] == 1
