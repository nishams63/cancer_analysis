"""SQLite schema definitions and row mapping utilities."""
CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_items (
    knowledge_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    description TEXT NOT NULL,
    conditions_json TEXT NOT NULL,
    recommended_methods_json TEXT NOT NULL,
    selection_rules_json TEXT NOT NULL,
    limitations_json TEXT NOT NULL,
    source TEXT NOT NULL,
    version TEXT NOT NULL,
    effective_date TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    quality_score REAL NOT NULL,
    content_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ki_category ON knowledge_items(category);
CREATE INDEX IF NOT EXISTS idx_ki_subcategory ON knowledge_items(subcategory);
CREATE INDEX IF NOT EXISTS idx_ki_status ON knowledge_items(status);
CREATE INDEX IF NOT EXISTS idx_ki_version ON knowledge_items(version);
CREATE INDEX IF NOT EXISTS idx_ki_source ON knowledge_items(source);

CREATE TABLE IF NOT EXISTS ingestion_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingestion_timestamp TEXT NOT NULL,
    total_files INTEGER NOT NULL,
    total_items INTEGER NOT NULL,
    inserted_items INTEGER NOT NULL,
    updated_items INTEGER NOT NULL,
    rejected_items INTEGER NOT NULL,
    duplicate_items INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    report_json TEXT NOT NULL
);
"""
