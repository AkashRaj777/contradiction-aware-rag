"""
SQLite logging for queries, decisions, and eval runs.
"""
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

import config


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.RUNS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they do not exist."""
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                mode TEXT NOT NULL,
                decision TEXT,
                answer TEXT,
                confidence TEXT,
                metadata TEXT
            );
            """
        )
        conn.commit()


def log_run(
    question: str,
    mode: str,
    decision: str,
    answer: str,
    confidence: str,
    metadata: dict[str, Any],
) -> None:
    """Save one pipeline execution for later analysis."""
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO runs (created_at, question, mode, decision, answer, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                question,
                mode,
                decision,
                answer,
                confidence,
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
        conn.commit()
