"""
tool_connectors/audit.py — a durable record of every connector action, for
the requirement that all executions (attempted or completed) be auditable.

Kept as its own small SQLite store rather than folded into any other
module's database, so this package stays a self-contained deliverable that
doesn't reach into unrelated parts of the app to log something.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
DB_PATH = BASE_DIR / "memory" / "tool_connector_audit.db"

_lock = Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_connector_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    connector   TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    safety      TEXT    NOT NULL,
    params      TEXT,
    output      TEXT,
    success     INTEGER NOT NULL,
    stage       TEXT    NOT NULL DEFAULT 'executed'
);
CREATE INDEX IF NOT EXISTS idx_tce_timestamp ON tool_connector_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_tce_connector ON tool_connector_events(connector, action);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def record(
    connector: str,
    action: str,
    safety: str,
    params: str,
    output: str,
    success: bool,
    stage: str = "executed",
    db_path: Path | None = None,
) -> None:
    """stage is 'executed' for a completed run, or 'confirmation_requested'
    for a REVERSIBLE/DESTRUCTIVE action parked behind the confirmation gate —
    so the audit trail shows an action was proposed even if it's later
    cancelled or times out."""
    with _lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                "INSERT INTO tool_connector_events "
                "(timestamp, connector, action, safety, params, output, success, stage) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    connector,
                    action,
                    safety,
                    params[:2000],
                    output[:2000],
                    1 if success else 0,
                    stage,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def recent(limit: int = 100, db_path: Path | None = None) -> list[sqlite3.Row]:
    conn = _connect(db_path)
    try:
        return conn.execute(
            "SELECT * FROM tool_connector_events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
