"""
core/predictive_assistant.py — learns from workflow history and proactively
suggests next actions.

WHY THIS EXISTS
    Every command JARVIS runs is a data point about what you tend to do next.
    Most of that signal was thrown away. This module keeps it (workflow_events),
    mines it for three cheap patterns — "A is usually followed by B", "you run
    X around the same time every day", and "you keep doing this by hand" — and
    turns matches above a confidence threshold into dismissible suggestions.
    Nothing here auto-executes: a suggestion is always one explicit confirm
    away from actually running.

HOW SCORING WORKS
    Confidence starts as plain association-rule support/confidence (frequency
    counting — no model, no training step). Accept/dismiss feedback then
    nudges a per-pattern weight multiplier up or down, so patterns you keep
    dismissing quietly fade below the threshold over time.
"""

from __future__ import annotations

import ast
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
DB_PATH = BASE_DIR / "memory" / "workflow.db"

_lock = Lock()

# Confidence-weight bounds for the feedback loop — wide enough that repeated
# dismissals can genuinely silence a pattern, narrow enough that one dismissal
# can't zero it out or one accept send it over any real threshold on its own.
_MIN_WEIGHT = 0.2
_MAX_WEIGHT = 2.0
_WEIGHT_STEP = 0.15

DEFAULT_CONFIDENCE_THRESHOLD = 0.6

SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    action_type TEXT    NOT NULL,
    context     TEXT,
    input       TEXT,
    output      TEXT,
    success     INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_workflow_events_timestamp ON workflow_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_workflow_events_action_type ON workflow_events(action_type);

CREATE TABLE IF NOT EXISTS suggestion_feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_key TEXT    NOT NULL,
    action_type TEXT    NOT NULL,
    context     TEXT,
    accepted    INTEGER NOT NULL,
    timestamp   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS suggestion_weights (
    pattern_key TEXT PRIMARY KEY,
    weight      REAL NOT NULL DEFAULT 1.0
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
    ensure_schema(conn)
    return conn


# ── Data capture ─────────────────────────────────────────────────────────────

def log_event(
    action_type: str,
    context: str = "",
    input_data: str = "",
    output: str = "",
    success: bool = True,
    db_path: Optional[Path] = None,
) -> int:
    """Record one user action. Returns the new row's id."""
    with _lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute(
                "INSERT INTO workflow_events "
                "(timestamp, action_type, context, input, output, success) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    action_type,
                    context,
                    input_data,
                    output,
                    1 if success else 0,
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_recent_events(
    days: Optional[int] = None,
    limit: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> list[sqlite3.Row]:
    """Rolling-window fetch: last `days` days and/or last `limit` events,
    oldest first (the order every detector below assumes)."""
    conn = _connect(db_path)
    try:
        clauses = []
        params: list = []
        if days is not None:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
            clauses.append("timestamp >= ?")
            params.append(cutoff)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        if limit is not None:
            query = (
                f"SELECT * FROM workflow_events {where} "
                "ORDER BY timestamp DESC LIMIT ?"
            )
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            rows.reverse()
            return rows
        query = f"SELECT * FROM workflow_events {where} ORDER BY timestamp ASC"
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


# ── Pattern detection ────────────────────────────────────────────────────────

def detect_sequential_patterns(
    events: list[sqlite3.Row], min_count: int = 2
) -> list[dict]:
    """n-gram (bigram) sequence counting: how often action A is immediately
    followed by action B, restricted to successful events. Confidence is the
    standard association-rule value: count(A→B) / count(A)."""
    successful = [e for e in events if e["success"]]
    pair_counts: Counter = Counter()
    first_counts: Counter = Counter()
    for a, b in zip(successful, successful[1:]):
        first_counts[a["action_type"]] += 1
        pair_counts[(a["action_type"], b["action_type"])] += 1

    patterns = []
    for (a, b), count in pair_counts.items():
        if count < min_count:
            continue
        confidence = count / first_counts[a]
        patterns.append(
            {
                "type": "sequential",
                "pattern_key": f"seq:{a}->{b}",
                "trigger": a,
                "suggested_action": b,
                "support": count,
                "confidence": confidence,
            }
        )
    return patterns


def detect_time_based_patterns(
    events: list[sqlite3.Row], min_count: int = 3, hour_window: int = 1
) -> list[dict]:
    """Groups events by (action_type, hour-of-day bucket) to find things run
    at roughly the same time of day repeatedly (e.g. "every morning" or
    "before deploys" if deploys cluster at a time). Confidence is how much of
    the action's total occurrences fall in that one time bucket — a habit, not
    a coincidence, needs most of its occurrences clustered there."""
    buckets: defaultdict = defaultdict(set)
    action_days: defaultdict = defaultdict(set)
    for e in events:
        try:
            ts = datetime.fromisoformat(e["timestamp"])
        except ValueError:
            continue
        bucket_hour = (ts.hour // hour_window) * hour_window
        buckets[(e["action_type"], bucket_hour)].add(ts.date())
        action_days[e["action_type"]].add(ts.date())

    patterns = []
    for (action_type, hour), days in buckets.items():
        distinct_days = len(days)
        if distinct_days < min_count:
            continue
        confidence = distinct_days / len(action_days[action_type])
        patterns.append(
            {
                "type": "time_based",
                "pattern_key": f"time:{action_type}@{hour}",
                "trigger": f"~{hour:02d}:00 daily",
                "suggested_action": action_type,
                "support": distinct_days,
                "confidence": confidence,
            }
        )
    return patterns


def detect_repeated_manual_steps(
    events: list[sqlite3.Row], min_count: int = 3, ngram_size: int = 2
) -> list[dict]:
    """Finds short sequences of manual steps (n-grams over raw action_type,
    ignoring already-automated actions) that repeat verbatim often enough to
    be worth turning into a single one-click action."""
    manual = [
        e["action_type"]
        for e in events
        if e["success"] and not e["action_type"].startswith("automation:")
    ]
    if len(manual) < ngram_size:
        return []

    ngram_counts: Counter = Counter()
    for i in range(len(manual) - ngram_size + 1):
        ngram = tuple(manual[i : i + ngram_size])
        ngram_counts[ngram] += 1

    patterns = []
    for ngram, count in ngram_counts.items():
        if count < min_count:
            continue
        confidence = min(1.0, count / max(1, len(manual) // ngram_size))
        patterns.append(
            {
                "type": "repeated_manual",
                "pattern_key": f"manual:{'>'.join(ngram)}",
                "trigger": ngram[0],
                "suggested_action": " then ".join(ngram),
                "support": count,
                "confidence": confidence,
            }
        )
    return patterns


def get_manual_sequence_steps(
    pattern_key: str,
    days: int = 14,
    db_path: Optional[Path] = None,
) -> Optional[list[dict]]:
    """
    Reconstruct the actual {tool, args} steps behind a 'manual:a>b>...'
    pattern_key (as produced by detect_repeated_manual_steps) from the
    workflow log, so accepting a "you keep doing this by hand" suggestion
    can be handed straight to manage_sequence(action='save') instead of
    dead-ending — the suggestion only carries a human-readable "a then b"
    description, not runnable arguments.

    Walks the log backwards to find the MOST RECENT occurrence of the exact
    tool n-gram, on the theory that the freshest arguments (e.g. which file
    was opened) are the ones most likely to still be what the user wants a
    macro of. Returns None if the pattern can't be found (log trimmed,
    wrong key, etc.) — the caller should treat that as "nothing to save".
    """
    if not pattern_key.startswith("manual:"):
        return None
    ngram = tuple(pattern_key[len("manual:"):].split(">"))
    if not ngram or not all(ngram):
        return None

    events = get_recent_events(days=days, db_path=db_path)
    manual = [
        e for e in events
        if e["success"] and not e["action_type"].startswith("automation:")
    ]

    n = len(ngram)
    for i in range(len(manual) - n, -1, -1):
        window = manual[i:i + n]
        if tuple(e["action_type"] for e in window) != ngram:
            continue
        steps = []
        for e in window:
            args: dict = {}
            raw = e["input"]
            if raw:
                try:
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, dict):
                        args = parsed
                except (ValueError, SyntaxError):
                    pass
            steps.append({"tool": e["action_type"], "args": args})
        return steps
    return None


# ── Suggestion engine ────────────────────────────────────────────────────────

@dataclass
class Suggestion:
    action: str
    confidence_score: float
    reasoning: str
    one_click_command: str
    # Not part of the original 4-field contract — carried along so a caller
    # (e.g. the UI's accept/dismiss handler) can attribute feedback to the
    # exact pattern that produced this suggestion via record_suggestion_feedback,
    # without having to re-derive it from action/one_click_command.
    pattern_key: str = ""


def _get_weight(conn: sqlite3.Connection, pattern_key: str) -> float:
    row = conn.execute(
        "SELECT weight FROM suggestion_weights WHERE pattern_key = ?",
        (pattern_key,),
    ).fetchone()
    return row["weight"] if row else 1.0


def _to_command(action_type: str) -> str:
    """Best-effort slug for the one-click command a suggestion would run."""
    return re.sub(r"[^a-z0-9]+", "_", action_type.lower()).strip("_")


def _reasoning_for(pattern: dict) -> str:
    if pattern["type"] == "sequential":
        return (
            f"You ran '{pattern['suggested_action']}' after "
            f"'{pattern['trigger']}' {pattern['support']} times."
        )
    if pattern["type"] == "time_based":
        return (
            f"You've run '{pattern['suggested_action']}' around "
            f"{pattern['trigger']} on {pattern['support']} occasions."
        )
    return (
        f"You've manually done '{pattern['suggested_action']}' "
        f"{pattern['support']} times — this could be one action."
    )


def get_proactive_suggestions(
    current_context: str = "",
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    days: int = 14,
    db_path: Optional[Path] = None,
) -> list[Suggestion]:
    """Runs all three detectors over the rolling window, applies any learned
    feedback weight, and returns suggestions whose adjusted confidence clears
    `threshold` — highest confidence first. current_context, when supplied,
    restricts sequential-pattern triggers to events logged in that context."""
    events = get_recent_events(days=days, db_path=db_path)
    if current_context:
        contextual = [e for e in events if e["context"] == current_context]
        seq_source = contextual or events
    else:
        seq_source = events

    raw_patterns = (
        detect_sequential_patterns(seq_source)
        + detect_time_based_patterns(events)
        + detect_repeated_manual_steps(events)
    )

    conn = _connect(db_path)
    try:
        suggestions = []
        for pattern in raw_patterns:
            weight = _get_weight(conn, pattern["pattern_key"])
            adjusted = max(0.0, min(1.0, pattern["confidence"] * weight))
            if adjusted < threshold:
                continue
            suggestions.append(
                Suggestion(
                    action=pattern["suggested_action"],
                    confidence_score=round(adjusted, 3),
                    reasoning=_reasoning_for(pattern),
                    one_click_command=_to_command(pattern["suggested_action"]),
                    pattern_key=pattern["pattern_key"],
                )
            )
    finally:
        conn.close()

    suggestions.sort(key=lambda s: s.confidence_score, reverse=True)
    return suggestions


# ── Feedback loop ────────────────────────────────────────────────────────────

def record_suggestion_feedback(
    pattern_key: str,
    action_type: str,
    accepted: bool,
    context: str = "",
    db_path: Optional[Path] = None,
) -> None:
    """Logs the accept/dismiss decision and nudges that pattern's confidence
    weight — a simple bounded weighted update, not a full RL loop."""
    with _lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                "INSERT INTO suggestion_feedback "
                "(pattern_key, action_type, context, accepted, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    pattern_key,
                    action_type,
                    context,
                    1 if accepted else 0,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            current = _get_weight(conn, pattern_key)
            step = _WEIGHT_STEP if accepted else -_WEIGHT_STEP
            new_weight = max(_MIN_WEIGHT, min(_MAX_WEIGHT, current + step))
            conn.execute(
                "INSERT INTO suggestion_weights (pattern_key, weight) VALUES (?, ?) "
                "ON CONFLICT(pattern_key) DO UPDATE SET weight = excluded.weight",
                (pattern_key, new_weight),
            )
            conn.commit()
        finally:
            conn.close()
