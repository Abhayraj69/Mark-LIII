"""
memory/study_history.py — durable, cross-session record of quiz answers.

Study Q&A live in core context (session_memory) only for the length of one
conversation — the live model grades each answer turn-by-turn per
core/prompt.txt's routing, and that grading never touches Python. This
module is the durable twin: a small SQLite table, matching the pattern
already used by core/context_manager.py (context_store.db) and
core/predictive_assistant.py (workflow.db), that the model asks JARVIS to
write to via actions/study_progress.py after each graded answer, so
"what am I weak on" and "quiz me on what I got wrong" can be answered in a
brand-new session that has no memory of the last one.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
DB_PATH  = BASE_DIR / "memory" / "study_history.db"

_lock = Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS quiz_answers (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT    NOT NULL,
    topic          TEXT    NOT NULL DEFAULT '',
    question       TEXT    NOT NULL,
    correct_answer TEXT    NOT NULL,
    user_answer    TEXT    NOT NULL DEFAULT '',
    correct        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quiz_answers_topic ON quiz_answers(topic);
CREATE INDEX IF NOT EXISTS idx_quiz_answers_correct ON quiz_answers(correct);
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


def log_answer(
    question: str,
    correct_answer: str,
    correct: bool,
    topic: str = "",
    user_answer: str = "",
    db_path: Optional[Path] = None,
) -> int:
    """Persists one graded quiz answer. The grading itself already happened
    in the live conversation (by meaning, not string-match) — this only
    records the outcome."""
    question = (question or "").strip()
    if not question:
        return -1
    with _lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute(
                "INSERT INTO quiz_answers "
                "(timestamp, topic, question, correct_answer, user_answer, correct) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    (topic or "").strip(),
                    question,
                    (correct_answer or "").strip(),
                    (user_answer or "").strip(),
                    1 if correct else 0,
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_weak_topics(limit: int = 5, min_attempts: int = 2, db_path: Optional[Path] = None) -> list[dict]:
    """Topics with the lowest accuracy, among those attempted at least
    `min_attempts` times (so one unlucky guess on a brand-new topic doesn't
    outrank a topic genuinely struggled with over many questions)."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT topic,
                   COUNT(*)                       AS attempts,
                   SUM(correct)                    AS correct_count,
                   1.0 * SUM(correct) / COUNT(*)   AS accuracy
            FROM quiz_answers
            WHERE topic != ''
            GROUP BY topic
            HAVING attempts >= ?
            ORDER BY accuracy ASC, attempts DESC
            LIMIT ?
            """,
            (min_attempts, limit),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "topic": r["topic"],
            "attempts": r["attempts"],
            "correct": r["correct_count"],
            "accuracy": round(r["accuracy"], 2),
        }
        for r in rows
    ]


def get_missed_questions(topic: str = "", limit: int = 10, db_path: Optional[Path] = None) -> list[dict]:
    """Most recent questions the user got wrong, optionally filtered to one
    topic — the source list for "quiz me again on what I missed"."""
    conn = _connect(db_path)
    try:
        if topic:
            rows = conn.execute(
                "SELECT * FROM quiz_answers WHERE correct = 0 AND topic = ? "
                "ORDER BY id DESC LIMIT ?",
                (topic, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM quiz_answers WHERE correct = 0 "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    finally:
        conn.close()
    return [
        {
            "topic": r["topic"],
            "question": r["question"],
            "correct_answer": r["correct_answer"],
            "user_answer": r["user_answer"],
            "timestamp": r["timestamp"],
        }
        for r in rows
    ]


def get_stats(db_path: Optional[Path] = None) -> dict:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS total, SUM(correct) AS correct_count FROM quiz_answers"
        ).fetchone()
    finally:
        conn.close()
    total   = row["total"] or 0
    correct = row["correct_count"] or 0
    return {
        "total_answered": total,
        "total_correct": correct,
        "accuracy": round(correct / total, 2) if total else None,
    }


def clear_history(db_path: Optional[Path] = None) -> int:
    """Deletes every stored quiz answer. Returns how many were removed —
    the privacy control, matching context_manager.clear_turns()."""
    with _lock:
        conn = _connect(db_path)
        try:
            n = conn.execute("SELECT COUNT(*) AS n FROM quiz_answers").fetchone()["n"]
            conn.execute("DELETE FROM quiz_answers")
            conn.commit()
            return n
        finally:
            conn.close()
