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
                       the tool name, its JSON-encoded arguments, and whether
                       it originally required a confirm token (see RECORD
                       MODE below).

    Two tables instead of one JSON blob column so a single step can be
    inspected or counted with plain SQL, matching the pattern already used by
    core/predictive_assistant.py and core/context_manager.py (DB_PATH +
    ensure_schema(conn) in the owning module, migrated via a thin
    memory/migrate_*.py runner).

RECORD MODE
    Originally a sequence had to be dictated up front as a full steps list.
    start_recording(name) / stop_recording() / discard_recording() add the
    natural alternative: do the steps once while JARVIS watches. The actual
    "watching" happens outside this module — main.py's tool dispatcher calls
    record_step() after every tool it runs, and this module just buffers
    whatever arrives in memory (_recording) until stop_recording() persists
    it via save_sequence(). record_step() is a deliberate no-op, never a
    raise, whenever nothing is being recorded, so the dispatcher can call it
    unconditionally on every turn without an is_recording() check first.
    manage_sequence itself is refused as a step (see
    _SELF_RECORDING_FORBIDDEN) so a macro can never record itself.

PARAMETERS
    A step's string arg values may contain {name} placeholders. render_steps()
    substitutes them from a replay's params and reports which names are still
    missing rather than running a partially-substituted step. parametrize()
    is the inverse: given a sequence and a literal value that appears in it,
    replace every occurrence with {placeholder} so future edits don't require
    re-recording.
"""

from __future__ import annotations

import json
import re
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

# A sequence cannot contain a call to itself — recording or replaying
# manage_sequence as a step would allow infinite/exponential recursion.
_SELF_RECORDING_FORBIDDEN = {"manage_sequence"}

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
    confirm       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (sequence_name, step_index),
    FOREIGN KEY (sequence_name) REFERENCES sequences(name) ON DELETE CASCADE
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # CREATE TABLE IF NOT EXISTS doesn't add columns to a table that already
    # existed before "confirm" was introduced — cover that with an explicit
    # ALTER, the same idempotent-migration shape memory/migrate_sequences_db.py
    # runs standalone.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(sequence_steps)")}
    if "confirm" not in cols:
        conn.execute("ALTER TABLE sequence_steps ADD COLUMN confirm INTEGER NOT NULL DEFAULT 0")
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
    confirm: bool = False   # required a confirm token when recorded — re-gated on replay


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
        steps.append(Step(tool=s["tool_name"], args=args, note=s["note"] or "",
                          confirm=bool(s["confirm"]) if "confirm" in s.keys() else False))
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

    normalized: list[tuple[str, dict, str, bool]] = []
    for i, step in enumerate(steps):
        tool = (step.get("tool") or "").strip() if isinstance(step, dict) else ""
        if not tool:
            raise ValueError(f"Step {i + 1} is missing a tool name.")
        args = step.get("args") if isinstance(step, dict) else None
        args = args if isinstance(args, dict) else {}
        note = str(step.get("note", "") or "") if isinstance(step, dict) else ""
        confirm = bool(step.get("confirm", False)) if isinstance(step, dict) else False
        normalized.append((tool, args, note, confirm))

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
                "INSERT INTO sequence_steps (sequence_name, step_index, tool_name, args_json, note, confirm) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (name, i, tool, json.dumps(args, ensure_ascii=False), note, int(confirm))
                    for i, (tool, args, note, confirm) in enumerate(normalized)
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


def edit_step(
    name: str,
    index: int,
    tool: Optional[str] = None,
    args: Optional[dict] = None,
    note: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> str:
    """Replaces one step (1-based `index`, matching the numbering
    _format_sequence shows the user) in place. Any of tool/args/note left as
    None keeps that field unchanged. Raises ValueError if the sequence or
    index doesn't exist."""
    seq = get_sequence(name, db_path)
    if seq is None:
        raise ValueError(f"No saved sequence named '{name}'.")
    i = index - 1
    if i < 0 or i >= len(seq.steps):
        raise ValueError(f"Step {index} out of range — '{name}' has {len(seq.steps)} step(s).")

    current = seq.steps[i]
    new_tool = tool.strip() if tool else current.tool
    if new_tool in _SELF_RECORDING_FORBIDDEN:
        raise ValueError(f"A sequence cannot contain '{new_tool}' as a step.")
    seq.steps[i] = Step(
        tool=new_tool,
        args=args if args is not None else current.args,
        note=note if note is not None else current.note,
        confirm=current.confirm,
    )
    steps_payload = [
        {"tool": s.tool, "args": s.args, "note": s.note, "confirm": s.confirm} for s in seq.steps
    ]
    save_sequence(name, steps_payload, seq.description, db_path)
    return f"Updated step {index} of '{name}'."


_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _walk_args(value, fn):
    """Applies fn to every string found anywhere inside value (which may be a
    str, a dict, a list, or any JSON-safe scalar), rebuilding dicts/lists and
    leaving non-strings untouched."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: _walk_args(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk_args(v, fn) for v in value]
    return value


def required_params(seq: Sequence) -> set[str]:
    """Every {placeholder} name referenced anywhere in the sequence's steps."""
    needed: set[str] = set()

    def collect(s: str) -> str:
        needed.update(_PLACEHOLDER_RE.findall(s))
        return s

    for step in seq.steps:
        _walk_args(step.args, collect)
    return needed


def render_steps(seq: Sequence, params: dict) -> tuple[list[Step], set[str]]:
    """Substitutes {placeholder} values in every step's args from `params`.
    Returns (rendered_steps, missing_params) — if any placeholder has no
    matching param, rendered_steps is [] and missing_params names what's
    needed, so the caller can stop before running anything rather than
    replay a half-substituted sequence."""
    missing = required_params(seq) - set(params.keys())
    if missing:
        return [], missing

    def substitute(s: str) -> str:
        return _PLACEHOLDER_RE.sub(lambda m: str(params[m.group(1)]), s)

    rendered = [
        Step(tool=s.tool, args=_walk_args(s.args, substitute), note=s.note, confirm=s.confirm)
        for s in seq.steps
    ]
    return rendered, set()


def parametrize(name: str, literal_value: str, placeholder: str, db_path: Optional[Path] = None) -> str:
    """Replaces every exact occurrence of `literal_value` inside any step's
    string args with {placeholder}, across the whole sequence, and persists
    the result — turning "open report_Q3.pdf" into "open {file}" without
    re-recording."""
    seq = get_sequence(name, db_path)
    if seq is None:
        raise ValueError(f"No saved sequence named '{name}'.")

    ph = placeholder.strip().strip("{}").strip()
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", ph):
        raise ValueError("Placeholder must be a valid identifier, e.g. 'file' or 'city'.")
    token = "{" + ph + "}"

    count = 0

    def replace(s: str) -> str:
        nonlocal count
        if s == literal_value:
            count += 1
            return token
        return s

    steps_payload = []
    for step in seq.steps:
        steps_payload.append({
            "tool": step.tool, "args": _walk_args(step.args, replace),
            "note": step.note, "confirm": step.confirm,
        })
    if count == 0:
        return f"'{literal_value}' does not appear in any step of '{name}' — nothing changed."
    save_sequence(name, steps_payload, seq.description, db_path)
    return f"Replaced {count} occurrence(s) of '{literal_value}' with {token} in '{name}'."


# ── Record mode ──────────────────────────────────────────────────────────
# One recording active at a time, process-wide — matches the rest of this
# single-user assistant's state model (one Live session, one undo stack).
_rec_lock = Lock()
_recording: Optional[dict] = None   # {"name": str, "description": str, "steps": list[dict]}


def is_recording() -> bool:
    return _recording is not None


def recording_name() -> Optional[str]:
    return _recording["name"] if _recording else None


def start_recording(name: str, description: str = "") -> str:
    global _recording
    name = (name or "").strip()
    if not name:
        raise ValueError("Sequence name cannot be empty.")
    if len(name) > NAME_MAX_LEN:
        raise ValueError(f"Sequence name too long (max {NAME_MAX_LEN} chars).")
    with _rec_lock:
        if _recording is not None:
            raise ValueError(f"Already recording '{_recording['name']}' — stop or discard it first.")
        _recording = {"name": name, "description": description, "steps": []}
    return f"Recording started as '{name}'. Every action taken now becomes a step — say 'stop recording' when done."


def record_step(tool: str, args: dict, confirm: bool = False) -> None:
    """Appends one step to the in-progress recording, if any — called
    unconditionally by main.py's tool dispatcher after every tool call, so
    it must be silent and cheap when nothing is being recorded, and must
    never raise (a bookkeeping failure must never break the tool call that
    actually ran)."""
    if _recording is None or tool in _SELF_RECORDING_FORBIDDEN:
        return
    try:
        with _rec_lock:
            if _recording is None:
                return
            if len(_recording["steps"]) >= MAX_STEPS:
                return
            _recording["steps"].append(
                {"tool": tool, "args": dict(args or {}), "note": "", "confirm": bool(confirm)}
            )
    except Exception as e:
        print(f"[SequenceMemory] ⚠️ record_step failed: {e}")


def stop_recording(db_path: Optional[Path] = None) -> str:
    global _recording
    with _rec_lock:
        if _recording is None:
            raise ValueError("Not currently recording.")
        rec, _recording = _recording, None
    if not rec["steps"]:
        return f"Stopped recording '{rec['name']}' — no steps were captured, nothing saved."
    return save_sequence(rec["name"], rec["steps"], rec["description"], db_path)


def discard_recording() -> str:
    global _recording
    with _rec_lock:
        if _recording is None:
            raise ValueError("Not currently recording.")
        name, _recording = _recording["name"], None
    return f"Discarded recording '{name}'."


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
