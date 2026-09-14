"""SQLite relational tables for execution traces and run logs."""
CREATE_TRACE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    total_tasks INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    state_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trace_events (
    trace_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    task_id TEXT,
    event_type TEXT NOT NULL,
    tool_name TEXT,
    tool_input_json TEXT,
    tool_output_json TEXT,
    decision TEXT,
    decision_reason TEXT,
    confidence REAL,
    status TEXT NOT NULL,
    error_json TEXT,
    next_task_id TEXT,
    metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_te_run_id ON trace_events(run_id);
CREATE INDEX IF NOT EXISTS idx_te_task_id ON trace_events(task_id);
CREATE INDEX IF NOT EXISTS idx_te_event_type ON trace_events(event_type);

CREATE TABLE IF NOT EXISTS human_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    task_id TEXT,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ho_run_id ON human_overrides(run_id);
"""
