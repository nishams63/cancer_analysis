"""Database Manager for SQLite persistent storage and retrieval."""
import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from schemas.knowledge import KnowledgeItem, KnowledgeCategory
from database.models import CREATE_TABLES_SQL


class DatabaseManager:
    """Thread-safe SQLite manager for Knowledge Base items and ingestion logs."""

    def __init__(self, db_path: Optional[Path | str] = None):
        if db_path is None:
            # Default location inside stage-6-agentic-ai/knowledge-engineer/database/
            base_dir = Path(__file__).resolve().parent
            self.db_path = base_dir / "knowledge_base.db"
        else:
            self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initialize schema and indexes."""
        with self._get_connection() as conn:
            conn.executescript(CREATE_TABLES_SQL)
            conn.commit()

    @staticmethod
    def compute_content_hash(item: KnowledgeItem) -> str:
        """Compute SHA-256 hash of core content to detect changes or exact duplicates."""
        normalized = (
            f"{item.title.strip()}|{item.category.value}|{item.subcategory.strip()}|"
            f"{item.description.strip()}|{'|'.join(sorted(item.conditions))}|"
            f"{'|'.join(sorted(item.recommended_methods))}|{'|'.join(sorted(item.selection_rules))}|"
            f"{'|'.join(sorted(item.limitations))}"
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def upsert_item(self, item: KnowledgeItem) -> Tuple[bool, str]:
        """Upsert a knowledge item. Returns (is_inserted, action_reason)."""
        content_hash = self.compute_content_hash(item)
        now_iso = datetime.utcnow().isoformat() + "Z"
        created_at = item.created_at or now_iso
        updated_at = item.updated_at or now_iso
        quality_score = item.quality_score if item.quality_score is not None else item.compute_quality_score()

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT content_hash FROM knowledge_items WHERE knowledge_id = ?", (item.knowledge_id,))
            existing = cur.fetchone()

            if existing is None:
                cur.execute(
                    """
                    INSERT INTO knowledge_items (
                        knowledge_id, title, category, subcategory, description,
                        conditions_json, recommended_methods_json, selection_rules_json, limitations_json,
                        source, version, effective_date, status, created_at, updated_at, quality_score, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.knowledge_id,
                        item.title,
                        item.category.value,
                        item.subcategory,
                        item.description,
                        json.dumps(item.conditions),
                        json.dumps(item.recommended_methods),
                        json.dumps(item.selection_rules),
                        json.dumps(item.limitations),
                        item.source,
                        item.version,
                        item.effective_date,
                        item.status,
                        created_at,
                        updated_at,
                        quality_score,
                        content_hash,
                    ),
                )
                conn.commit()
                return True, "inserted"
            else:
                if existing["content_hash"] == content_hash:
                    # Content unchanged, update timestamps/metadata only
                    cur.execute(
                        """
                        UPDATE knowledge_items
                        SET source = ?, version = ?, effective_date = ?, status = ?, updated_at = ?, quality_score = ?
                        WHERE knowledge_id = ?
                        """,
                        (item.source, item.version, item.effective_date, item.status, updated_at, quality_score, item.knowledge_id),
                    )
                    conn.commit()
                    return False, "unchanged_duplicate"
                else:
                    cur.execute(
                        """
                        UPDATE knowledge_items
                        SET title = ?, category = ?, subcategory = ?, description = ?,
                            conditions_json = ?, recommended_methods_json = ?, selection_rules_json = ?, limitations_json = ?,
                            source = ?, version = ?, effective_date = ?, status = ?, updated_at = ?, quality_score = ?, content_hash = ?
                        WHERE knowledge_id = ?
                        """,
                        (
                            item.title,
                            item.category.value,
                            item.subcategory,
                            item.description,
                            json.dumps(item.conditions),
                            json.dumps(item.recommended_methods),
                            json.dumps(item.selection_rules),
                            json.dumps(item.limitations),
                            item.source,
                            item.version,
                            item.effective_date,
                            item.status,
                            updated_at,
                            quality_score,
                            content_hash,
                            item.knowledge_id,
                        ),
                    )
                    conn.commit()
                    return False, "updated"

    def get_item(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """Fetch a single knowledge item by its unique ID."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM knowledge_items WHERE knowledge_id = ?", (knowledge_id.upper().strip(),))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_item(row)

    def list_items(
        self,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        source: Optional[str] = None,
        version: Optional[str] = None,
        status: Optional[str] = "active",
        limit: int = 100,
        offset: int = 0,
    ) -> List[KnowledgeItem]:
        """Filter knowledge items with criteria."""
        clauses = []
        params: List[Any] = []

        if category:
            clauses.append("category = ?")
            params.append(category)
        if subcategory:
            clauses.append("subcategory = ?")
            params.append(subcategory)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if version:
            clauses.append("version = ?")
            params.append(version)
        if status:
            clauses.append("status = ?")
            params.append(status)

        where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM knowledge_items {where_str} ORDER BY knowledge_id LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            return [self._row_to_item(row) for row in cur.fetchall()]

    def count_items(self, status: Optional[str] = None) -> int:
        with self._get_connection() as conn:
            cur = conn.cursor()
            if status:
                cur.execute("SELECT COUNT(*) FROM knowledge_items WHERE status = ?", (status,))
            else:
                cur.execute("SELECT COUNT(*) FROM knowledge_items")
            return cur.fetchone()[0]

    def get_categories(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT category, COUNT(*) as count 
                FROM knowledge_items 
                WHERE status = 'active' 
                GROUP BY category 
                ORDER BY category
            """)
            return [{"category": row["category"], "count": row["count"]} for row in cur.fetchall()]

    def get_sources(self) -> List[str]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT source FROM knowledge_items ORDER BY source")
            return [row["source"] for row in cur.fetchall()]

    def get_versions(self) -> List[str]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT version FROM knowledge_items ORDER BY version")
            return [row["version"] for row in cur.fetchall()]

    def record_ingestion(
        self,
        total_files: int,
        total_items: int,
        inserted_items: int,
        updated_items: int,
        rejected_items: int,
        duplicate_items: int,
        duration_seconds: float,
        report: Dict[str, Any],
    ) -> int:
        now_iso = datetime.utcnow().isoformat() + "Z"
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ingestion_history (
                    ingestion_timestamp, total_files, total_items, inserted_items,
                    updated_items, rejected_items, duplicate_items, duration_seconds, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso,
                    total_files,
                    total_items,
                    inserted_items,
                    updated_items,
                    rejected_items,
                    duplicate_items,
                    duration_seconds,
                    json.dumps(report),
                ),
            )
            conn.commit()
            return cur.lastrowid

    def _row_to_item(self, row: sqlite3.Row) -> KnowledgeItem:
        return KnowledgeItem(
            knowledge_id=row["knowledge_id"],
            title=row["title"],
            category=KnowledgeCategory(row["category"]),
            subcategory=row["subcategory"],
            description=row["description"],
            conditions=json.loads(row["conditions_json"]),
            recommended_methods=json.loads(row["recommended_methods_json"]),
            selection_rules=json.loads(row["selection_rules_json"]),
            limitations=json.loads(row["limitations_json"]),
            source=row["source"],
            version=row["version"],
            effective_date=row["effective_date"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            quality_score=row["quality_score"],
        )
