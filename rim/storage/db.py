from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(os.getenv("RIM_DB_PATH", "rim.db"))

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    input_idea TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    confidence_score REAL,
    error_summary TEXT
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    parent_node_id TEXT,
    depth INTEGER NOT NULL,
    component_text TEXT NOT NULL,
    node_type TEXT NOT NULL,
    confidence REAL,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS critic_findings (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    critic_type TEXT NOT NULL,
    issue TEXT NOT NULL,
    severity TEXT NOT NULL,
    confidence REAL NOT NULL,
    suggested_fix TEXT NOT NULL,
    provider TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id),
    FOREIGN KEY(node_id) REFERENCES nodes(id)
);

CREATE TABLE IF NOT EXISTS synthesis_outputs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    synthesized_idea TEXT NOT NULL,
    changes_summary_json TEXT NOT NULL,
    residual_risks_json TEXT NOT NULL,
    next_experiments_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS memory_entries (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    entry_text TEXT NOT NULL,
    score REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS stage_logs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    provider TEXT,
    latency_ms INTEGER,
    status TEXT NOT NULL,
    meta_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
"""


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()
