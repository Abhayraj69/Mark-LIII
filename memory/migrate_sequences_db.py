"""
memory/migrate_sequences_db.py — one-shot / idempotent migration for the
sequence ("macro") store (memory/sequences.db).

Adds the "confirm" column to sequence_steps (record mode re-gates a replayed
step that originally required a confirm token — see core/sequence_memory.py's
RECORD MODE docstring) for any database created before that column existed.
ensure_schema() itself is idempotent, so running this against an already
up-to-date or brand-new database is a no-op either way.

Usage:
    python memory/migrate_sequences_db.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.sequence_memory import DB_PATH, ensure_schema  # noqa: E402


def migrate(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        print(f"[migrate_sequences_db] schema up to date at {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
