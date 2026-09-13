"""Master Ingestion Script and CLI for Knowledge Engineer."""
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure parent directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.knowledge import KnowledgeItem
from database.db_manager import DatabaseManager
from retrieval.vector_store import LocalVectorStore
from ingestion.loader import KnowledgeLoader
from ingestion.parser import KnowledgeParser
from ingestion.validator import KnowledgeValidator
from ingestion.chunker import KnowledgeChunker
from ingestion.deduplicator import KnowledgeDeduplicator


def run_ingestion_pipeline(
    kb_dir: Path | str,
    db_path: Path | str | None = None,
    index_dir: Path | str | None = None,
) -> Dict[str, Any]:
    """Execute end-to-end ingestion workflow."""
    start_time = time.perf_counter()
    kb_path = Path(kb_dir)
    print(f"============================================================")
    print(f"AADA STAGE 06: KNOWLEDGE BASE INGESTION PIPELINE")
    print(f"Target Knowledge Directory: {kb_path}")
    print(f"============================================================")

    db = DatabaseManager(db_path=db_path)
    vstore = LocalVectorStore(index_dir=index_dir)
    loader = KnowledgeLoader(kb_path)
    chunker = KnowledgeChunker()
    parser = KnowledgeParser()
    validator = KnowledgeValidator()
    dedup = KnowledgeDeduplicator()

    files = loader.load_all_files()
    print(f"[*] Discovered {len(files)} JSON knowledge files.")

    total_chunks = 0
    valid_items: List[KnowledgeItem] = []
    rejected_items: List[Dict[str, Any]] = []
    duplicate_count = 0
    inserted_count = 0
    updated_count = 0

    for file_path, raw_content in files:
        chunks = chunker.chunk(raw_content)
        total_chunks += len(chunks)

        for raw_chunk in chunks:
            normalized = parser.normalize_raw_dict(raw_chunk)
            item, error = validator.validate_item(normalized)

            if item is None:
                rejected_items.append({
                    "file": str(file_path.relative_to(kb_path.parent)),
                    "error": error,
                    "item": normalized.get("knowledge_id", "UNKNOWN"),
                })
                continue

            is_dup, reason = dedup.check_duplicate(item)
            if is_dup:
                duplicate_count += 1
                print(f"  [-] Skipped duplicate item: {reason}")
                continue

            valid_items.append(item)

    print(f"[*] Validated {len(valid_items)} discrete knowledge items ({len(rejected_items)} rejected).")

    # Upsert valid items into SQLite
    for item in valid_items:
        is_inserted, action = db.upsert_item(item)
        if is_inserted:
            inserted_count += 1
        else:
            if action == "updated":
                updated_count += 1

    # Build local vector store
    print(f"[*] Building dense TF-IDF vector index for {len(valid_items)} items...")
    indexed_docs = vstore.build_index(valid_items)

    duration = round(time.perf_counter() - start_time, 3)

    report = {
        "total_files": len(files),
        "total_chunks": total_chunks,
        "valid_items": len(valid_items),
        "inserted_items": inserted_count,
        "updated_items": updated_count,
        "rejected_items": len(rejected_items),
        "duplicate_items": duplicate_count,
        "indexed_documents": indexed_docs,
        "duration_seconds": duration,
        "categories": db.get_categories(),
        "sources": db.get_sources(),
        "versions": db.get_versions(),
    }

    db.record_ingestion(
        total_files=len(files),
        total_items=len(valid_items),
        inserted_items=inserted_count,
        updated_items=updated_count,
        rejected_items=len(rejected_items),
        duplicate_items=duplicate_count,
        duration_seconds=duration,
        report=report,
    )

    print(f"------------------------------------------------------------")
    print(f"Ingestion Finished in {duration}s")
    print(f"  - Total Files:        {len(files)}")
    print(f"  - Valid Items:        {len(valid_items)}")
    print(f"  - Inserted Items:     {inserted_count}")
    print(f"  - Updated Items:      {updated_count}")
    print(f"  - Duplicates Skipped: {duplicate_count}")
    print(f"  - Rejected Items:     {len(rejected_items)}")
    print(f"  - Vector Index Size:  {indexed_docs}")
    print(f"============================================================")

    return report


if __name__ == "__main__":
    default_kb_dir = PROJECT_ROOT / "knowledge_base"
    run_ingestion_pipeline(default_kb_dir)
