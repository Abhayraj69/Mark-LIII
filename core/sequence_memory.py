"""
core/sequence_memory.py — permanent storage for named multi-step action
sequences ("macros"): an ordered list of {tool, args} pairs, saved once and
replayed by name forever.

WHY THIS EXISTS
    core/predictive_assistant.py already mines workflow_events for patterns
    and *suggests* repeats, but a suggestion is never stored as a runnable
    thing — it is re-derived from history every time and can decay below the
    confidence threshold and disappear. memory/memory_manager.py stores facts
    about the user, not procedures. Neither gives the assistant a durable,
    explicitly-named "do these N steps" it can recall on request ("run my
    morning routine"). This module is that missing piece: a small SQLite
    store, never trimmed, that only forgets a sequence when the user deletes
    it.

SCHEMA
    sequences       — one row per named sequence (name is the primary key).
    sequence_steps  — one row per step, ordered by step_index, each holding
                       the tool name and its JSON-encoded arguments.

    Two tables instead of one JSON blob column so a single step can be
    inspected or counted with plain SQL, matching the pattern already used by
    core/predictive_assistant.py and core/context_manager.py (DB_PATH +
    ensure_schema(conn) in the owning module, migrated via a thin
    memory/migrate_*.py runner).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

# Sequences may run tools that themselves already ran through a full
# planning agent (dev_agent) — bound generously so a legitimate long recipe
# isn't rejected, while still catching a runaway "record everything" bug.
MAX_STEPS = 50
NAME_MAX_LEN = 64


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
DB_PATH = BASE_DIR / "memory" / "sequences.db"

_lock = Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS sequences (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    created     TEXT NOT NULL,
    updated     TEXT NOT NULL,
    run_count   INTEGER NOT NULL DEFAULT 0,
    last_run    TEXT
);

CREATE TABLE IF NOT EXISTS sequence_steps (
    sequence_name TEXT    NOT NULL,
    step_index    INTEGER NOT NULL,
    tool_name     TEXT    NOT NULL,
    args_json     TEXT    NOT NULL DEFAULT '{}',
    note          TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (sequence_name, step_index),
    FOREIGN KEY (sequence_name) REFERENCES sequences(name) ON DELETE CASCADE
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_schema(conn)
    return conn


@dataclass
class Step:
    tool: str
    args: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class Sequence:
    name: str
    description: str
    created: str
    updated: str
    run_count: int
    last_run: Optional[str]
    steps: list[Step]


def _row_to_sequence(row: sqlite3.Row, step_rows: list[sqlite3.Row]) -> Sequence:
    steps = []
    for s in step_rows:
        try:
            args = json.loads(s["args_json"])
        except (json.JSONDecodeError, TypeError):
            args = {}
        steps.append(Step(tool=s["tool_name"], args=args, note=s["note"] or ""))
    return Sequence(
        name=row["name"],
        description=row["description"] or "",
        created=row["created"],
        updated=row["updated"],
        run_count=row["run_count"],
        last_run=row["last_run"],
        steps=steps,
    )


def save_sequence(
    name: str,
    steps: list[dict],
    description: str = "",
    db_path: Optional[Path] = None,
) -> str:
    """Create or overwrite a named sequence. `steps` is a list of
    {"tool": str, "args": dict, "note": str} — "args"/"note" optional.
    Returns a short human-readable confirmation, or raises ValueError on bad
    input (empty name/steps, too many steps, missing tool name)."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Sequence name cannot be empty.")
    if len(name) > NAME_MAX_LEN:
        raise ValueError(f"Sequence name too long (max {NAME_MAX_LEN} chars).")
    if not steps:
        raise ValueError("A sequence needs at least one step.")
    if len(steps) > MAX_STEPS:
        raise ValueError(f"Too many steps (max {MAX_STEPS}).")

    normalized: list[tuple[str, dict, str]] = []
    for i, step in enumerate(steps):
        tool = (step.get("tool") or "").strip() if isinstance(step, dict) else ""
        if not tool:
            raise ValueError(f"Step {i + 1} is missing a tool name.")
        args = step.get("args") if isinstance(step, dict) else None
        args = args if isinstance(args, dict) else {}
        note = str(step.get("note", "") or "") if isinstance(step, dict) else ""
        normalized.append((tool, args, note))

    now = datetime.now().isoformat(timespec="seconds")
    with _lock:
        conn = _connect(db_path)
        try:
            existing = conn.execute(
                "SELECT created FROM sequences WHERE name = ?", (name,)
            ).fetchone()
            created = existing["created"] if existing else now
            conn.execute(
                "INSERT INTO sequences (name, description, created, updated, run_count, last_run) "
                "VALUES (?, ?, ?, ?, COALESCE((SELECT run_count FROM sequences WHERE name = ?), 0), "
                "(SELECT last_run FROM sequences WHERE name = ?)) "
                "ON CONFLICT(name) DO UPDATE SET description = excluded.description, updated = excluded.updated",
                (name, description.strip(), created, now, name, name),
            )
            conn.execute("DELETE FROM sequence_steps WHERE sequence_name = ?", (name,))
            conn.executemany(
                "INSERT INTO sequence_steps (sequence_name, step_index, tool_name, args_json, note) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (name, i, tool, json.dumps(args, ensure_ascii=False), note)
                    for i, (tool, args, note) in enumerate(normalized)
                ],
            )
            conn.commit()
        finally:
            conn.close()

    verb = "Updated" if existing else "Saved"
    return f"{verb} sequence '{name}' with {len(normalized)} step(s)."


def get_sequence(name: str, db_path: Optional[Path] = None) -> Optional[Sequence]:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM sequences WHERE name = ?", (name.strip(),)).fetchone()
        if row is None:
            return None
        step_rows = conn.execute(
            "SELECT * FROM sequence_steps WHERE sequence_name = ? ORDER BY step_index ASC",
            (row["name"],),
        ).fetchall()
        return _row_to_sequence(row, step_rows)
    finally:
        conn.close()


def list_sequences(db_path: Optional[Path] = None) -> list[Sequence]:
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM sequences ORDER BY updated DESC").fetchall()
        out = []
        for row in rows:
            step_rows = conn.execute(
                "SELECT * FROM sequence_steps WHERE sequence_name = ? ORDER BY step_index ASC",
                (row["name"],),
            ).fetchall()
            out.append(_row_to_sequence(row, step_rows))
        return out
    finally:
        conn.close()


def delete_sequence(name: str, db_path: Optional[Path] = None) -> bool:
    with _lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute("DELETE FROM sequences WHERE name = ?", (name.strip(),))
            conn.execute("DELETE FROM sequence_steps WHERE sequence_name = ?", (name.strip(),))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def record_run(name: str, db_path: Optional[Path] = None) -> None:
    """Bump run_count/last_run after a sequence finishes executing. Best-effort
    bookkeeping only — never raises, since a failed count bump must not undo a
    sequence that already ran."""
    try:
        with _lock:
            conn = _connect(db_path)
            try:
                conn.execute(
                    "UPDATE sequences SET run_count = run_count + 1, last_run = ? WHERE name = ?",
                    (datetime.now().isoformat(timespec="seconds"), name.strip()),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        print(f"[SequenceMemory] ⚠️ record_run failed for '{name}': {e}")
