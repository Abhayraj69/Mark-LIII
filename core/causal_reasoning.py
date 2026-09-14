"""
core/causal_reasoning.py — a causal-link graph that augments Jarvis's
autonomous decision-making with cause-and-effect analysis of screen events
and the tool calls that follow them.

WHAT THIS IS, PRECISELY
    core/predictive_assistant.py already mines "A is usually followed by B"
    bigrams from workflow_events, but only over tool calls, and it only ever
    reports raw confidence (count(A->B) / count(A)). That number cannot tell
    "B follows A because A causes it" apart from "B just happens to be common
    on its own, and would follow almost anything" — a screen alert that
    fires constantly will look "caused" by whatever else is also constant.

    This module adds the missing half: LIFT. lift(A->B) = confidence(A->B) /
    base_rate(B), i.e. how much more often B follows A than B happens anyway.
    A link with lift near 1.0 is coincidence; a link with high confidence AND
    high lift is the closest a frequency-counting system can get to "A
    actually drives B" without running a controlled experiment. This is
    still a heuristic, not verified causal inference (no do-calculus, no
    randomized intervention) — the API is named and documented honestly so
    nothing downstream mistakes correlation-with-lift for proof.

    It is fed by two event streams that predictive_assistant does not see
    together: `tool:<name>` events (mirrored from the same dispatch main.py
    already logs to predictive_assistant) and `screen:<watch_for>` events
    (from actions/screen_monitor.py's alerts). Putting both in one timeline
    is what lets Jarvis learn things like "after screen:download_finished,
    tool:open_app usually follows" — a genuinely cause-shaped fact that
    neither existing event log could produce alone, because each only ever
    saw one half of the story.

HOW IT'S USED
    - actions/screen_monitor.py calls record_event() when an alert fires.
    - main.py's _dispatch_tool calls record_event() alongside its existing
      predictive_assistant.log_event() call (same best-effort try/except).
    - actions/causal_insight.py exposes explain()/predict_effects() as a tool
      so the model can consult learned cause-effect structure before
      deciding what to do next, instead of guessing — that consultation is
      the "augmented autonomous decision-making" this module provides.
      Nothing here executes anything on its own; it only informs.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional

# A cause only "explains" an effect if the effect showed up within this many
# seconds of it — long enough to cover a slow app launch or page load, short
# enough that unrelated events hours apart don't get linked by accident.
MAX_LAG_SECONDS = 90

# Below this many co-occurrences a link is noise, not a pattern — matches
# predictive_assistant's own min_count=2/3 bar for the same reason: two
# coincidences is not yet a rule.
DEFAULT_MIN_SUPPORT = 2

# Rolling window: only the most recent events are kept, so the DB stays a
# few KB forever and old, no-longer-true habits age out on their own instead
# of permanently outvoting new behavior.
MAX_EVENTS_KEPT = 1000


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
DB_PATH = BASE_DIR / "memory" / "causal_graph.db"

_lock = Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS causal_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_causal_events_timestamp ON causal_events(timestamp);

CREATE TABLE IF NOT EXISTS causal_links (
    cause         TEXT NOT NULL,
    effect        TEXT NOT NULL,
    support       INTEGER NOT NULL DEFAULT 0,
    cause_total   INTEGER NOT NULL DEFAULT 0,
    avg_lag_secs  REAL NOT NULL DEFAULT 0,
    updated       TEXT NOT NULL,
    PRIMARY KEY (cause, effect)
);

CREATE TABLE IF NOT EXISTS causal_event_totals (
    event_type TEXT PRIMARY KEY,
    total      INTEGER NOT NULL DEFAULT 0
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


@dataclass
class CausalLink:
    cause: str
    effect: str
    support: int          # how many times cause -> effect was observed
    confidence: float     # P(effect within MAX_LAG_SECONDS | cause) = support / cause_total
    lift: float           # confidence / base_rate(effect) — >1 means "more than coincidence"
    avg_lag_seconds: float


def _bump_total(conn: sqlite3.Connection, event_type: str) -> None:
    conn.execute(
        "INSERT INTO causal_event_totals (event_type, total) VALUES (?, 1) "
        "ON CONFLICT(event_type) DO UPDATE SET total = total + 1",
        (event_type,),
    )


def record_event(event_type: str, detail: str = "", timestamp: Optional[datetime] = None,
                  db_path: Optional[Path] = None) -> None:
    """Log one event and update every causal_links row it completes as an
    effect of a recent prior event. Best-effort: never raises, since a
    reasoning hiccup must not be allowed to break the screen monitor or the
    tool dispatcher that call this as a side effect of their real job."""
    event_type = (event_type or "").strip()
    if not event_type:
        return
    now = timestamp or datetime.now()
    cutoff = now - timedelta(seconds=MAX_LAG_SECONDS)

    try:
        with _lock:
            conn = _connect(db_path)
            try:
                recent = conn.execute(
                    "SELECT event_type, timestamp FROM causal_events WHERE timestamp >= ? "
                    "ORDER BY timestamp DESC",
                    (cutoff.isoformat(timespec="seconds"),),
                ).fetchall()

                # Collapse to the single most recent occurrence per cause type
                # within the window: if "tool:web_search" fired three times in
                # the last 90s before this effect, that is still only one
                # opportunity for it to have "caused" this effect, not three —
                # counting all three would inflate support far past how often
                # the effect actually happened.
                latest_by_cause: dict[str, str] = {}
                for row in recent:
                    cause = row["event_type"]
                    if cause == event_type or cause in latest_by_cause:
                        continue  # skip self-links and older duplicates of the same cause
                    latest_by_cause[cause] = row["timestamp"]

                for cause, cause_ts in latest_by_cause.items():
                    try:
                        lag = (now - datetime.fromisoformat(cause_ts)).total_seconds()
                    except ValueError:
                        continue
                    existing = conn.execute(
                        "SELECT support, cause_total, avg_lag_secs FROM causal_links "
                        "WHERE cause = ? AND effect = ?",
                        (cause, event_type),
                    ).fetchone()
                    if existing:
                        new_support = existing["support"] + 1
                        new_avg = (existing["avg_lag_secs"] * existing["support"] + lag) / new_support
                        conn.execute(
                            "UPDATE causal_links SET support = ?, avg_lag_secs = ?, updated = ? "
                            "WHERE cause = ? AND effect = ?",
                            (new_support, new_avg, now.isoformat(timespec="seconds"), cause, event_type),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO causal_links "
                            "(cause, effect, support, cause_total, avg_lag_secs, updated) "
                            "VALUES (?, ?, 1, 0, ?, ?)",
                            (cause, event_type, lag, now.isoformat(timespec="seconds")),
                        )

                conn.execute(
                    "INSERT INTO causal_events (timestamp, event_type, detail) VALUES (?, ?, ?)",
                    (now.isoformat(timespec="seconds"), event_type, detail[:300]),
                )
                _bump_total(conn, event_type)

                # cause_total for every link whose cause is this event's type must
                # count "how many times this event happened at all", not just the
                # times it happened to be followed by something — recompute it from
                # causal_event_totals rather than incrementing per-link so a cause
                # with many different effects still gets one true denominator.
                total_row = conn.execute(
                    "SELECT total FROM causal_event_totals WHERE event_type = ?",
                    (event_type,),
                ).fetchone()

                conn.execute(
                    "DELETE FROM causal_events WHERE id NOT IN "
                    "(SELECT id FROM causal_events ORDER BY id DESC LIMIT ?)",
                    (MAX_EVENTS_KEPT,),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        print(f"[CausalReasoning] ⚠️ record_event failed for '{event_type}': {e}")


def _refresh_cause_totals(conn: sqlite3.Connection) -> None:
    """cause_total has to reflect the CURRENT total occurrence count of the
    cause event type (causal_event_totals grows after some links were first
    created), so it's synced here rather than trusted from insert-time."""
    conn.execute(
        "UPDATE causal_links SET cause_total = ("
        "  SELECT total FROM causal_event_totals WHERE event_type = causal_links.cause"
        ")"
    )
    conn.commit()


def get_links(min_support: int = DEFAULT_MIN_SUPPORT, db_path: Optional[Path] = None,
              since: Optional[datetime] = None) -> list[CausalLink]:
    """All causal_links clearing `min_support`, with confidence and lift
    computed fresh, sorted strongest-first (confidence * lift).

    `since`, when given, restricts this to links whose most recent
    occurrence (`updated`) falls on or after that time — e.g. the weekly
    causal-graph digest (actions/causal_insight.py) uses this to report
    patterns actively seen in the last 7 days rather than the all-time
    leaderboard. `support`/`confidence` on a `since`-filtered link are still
    the all-time figures (this table doesn't keep a per-period count) — only
    which links are included is time-restricted, not their stats."""
    conn = _connect(db_path)
    try:
        _refresh_cause_totals(conn)
        totals = {
            r["event_type"]: r["total"]
            for r in conn.execute("SELECT event_type, total FROM causal_event_totals").fetchall()
        }
        grand_total = sum(totals.values()) or 1

        if since is not None:
            rows = conn.execute(
                "SELECT * FROM causal_links WHERE support >= ? AND updated >= ? ORDER BY support DESC",
                (min_support, since.isoformat(timespec="seconds")),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM causal_links WHERE support >= ? ORDER BY support DESC",
                (min_support,),
            ).fetchall()

        links = []
        for row in rows:
            cause_total = row["cause_total"] or totals.get(row["cause"], 0) or 1
            confidence = min(1.0, row["support"] / cause_total)
            base_rate = max(1, totals.get(row["effect"], 0)) / grand_total
            lift = confidence / base_rate if base_rate > 0 else 0.0
            links.append(CausalLink(
                cause=row["cause"], effect=row["effect"], support=row["support"],
                confidence=round(confidence, 3), lift=round(lift, 2),
                avg_lag_seconds=round(row["avg_lag_secs"], 1),
            ))
        links.sort(key=lambda l: l.confidence * l.lift, reverse=True)
        return links
    finally:
        conn.close()


def explain(event_type: str, min_support: int = DEFAULT_MIN_SUPPORT,
            db_path: Optional[Path] = None) -> dict:
    """What tends to cause `event_type`, and what `event_type` tends to
    cause — the two directions of "why" a decision-making step would want."""
    event_type = (event_type or "").strip()
    links = get_links(min_support=min_support, db_path=db_path)
    return {
        "causes_of":  [l for l in links if l.effect == event_type],
        "effects_of": [l for l in links if l.cause == event_type],
    }


def predict_effects(event_type: str, top_k: int = 3, min_support: int = DEFAULT_MIN_SUPPORT,
                     db_path: Optional[Path] = None) -> list[CausalLink]:
    """The top-k things most likely to follow `event_type`, for a decision
    step to weigh before acting (e.g. "a download-finished alert has
    historically been followed by open_app 80% of the time — worth
    offering that instead of just reading the alert aloud")."""
    effects = explain(event_type, min_support=min_support, db_path=db_path)["effects_of"]
    return effects[:max(1, top_k)]
