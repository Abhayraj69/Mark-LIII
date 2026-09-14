"""
memory/migrate_causal_graph_db.py — one-shot / idempotent migration for the
cause-and-effect reasoning store (memory/causal_graph.db).

Usage:
    python memory/migrate_causal_graph_db.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.causal_reasoning import DB_PATH, ensure_schema  # noqa: E402


def migrate(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        print(f"[migrate_causal_graph_db] schema up to date at {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
