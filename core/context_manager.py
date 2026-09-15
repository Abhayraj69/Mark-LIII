"""
core/context_manager.py — merges session history, project facts, and stored
user preferences into one structured ContextBundle before a response is
generated, so JARVIS's answers can be tailored to "what project is this,"
"what does this person prefer," and "what did we just talk about" instead of
starting from a blank slate every turn.

WHERE EACH LAYER ACTUALLY LIVES
    This module does not reinvent storage that already exists elsewhere —
    it assembles from it:

      - User preferences  → memory/memory_manager.py's long_term.json
                             ("preferences" category). Already has its own
                             viewer/editor (ui.py's MemoryOverlay) and its
                             own delete path (memory_manager.forget()).
      - Open tasks         → the same store's "wishes" category.
      - Recently touched
        files/projects     → core/predictive_assistant.py's workflow_events
                             table, if present (soft dependency — falls back
                             to "" if that module hasn't logged anything yet).
      - Git branch          → tool_connectors' GitConnector, called directly
                             (branch_info is READ_ONLY, so no confirmation
                             gate applies) — soft dependency, same reasoning.
      - Session turns       → passed in by the caller (main.py's
                             self._session_log is the live source of truth
                             for "the current conversation"); this module
                             only decides how many of them fit the budget.

    What THIS module owns is the one thing nothing else in the app has: a
    durable, searchable log of past turns (context_turns, below) for
    semantic recall of "we talked about this before," plus the assembly and
    truncation logic that turns all of the above into one prompt-ready
    ContextBundle.

WHY A HAND-ROLLED EMBEDDING INSTEAD OF Chroma/FAISS
    Same call as core/predictive_assistant.py's frequency counting instead of
    a trained model: this app has no ML/vector-search dependency today, and
    a single user's turn history is a few thousand rows at most — nowhere
    near where an approximate-nearest-neighbor index would pay for its
    dependency weight. embed()/cosine() below are a deterministic
    hashing-trick bag-of-words with brute-force cosine ranking, good enough
    to surface "this sounds like that other conversation" and completely
    replaceable: if this ever needs real semantic recall over tens of
    thousands of turns, swap embed() for a real sentence-embedding call and
    semantic_search()'s linear scan for a Chroma/FAISS index — the
    ContextBundle/build_context() contract on top does not need to change.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
DB_PATH = BASE_DIR / "memory" / "context_store.db"

# Run directly (`python core/context_manager.py show`), Python only puts
# core/ itself on sys.path, not the project root — so the sibling-package
# imports below (memory.memory_manager, tool_connectors...) would silently
# fail and, worse, be swallowed by their own `except Exception: return {}`
# fallbacks, making real stored preferences look like "(none stored)".
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

_lock = Lock()

EMBED_DIM = 256
DEFAULT_MAX_CHARS = 2400          # char budget for the assembled prompt (~600 tokens at ~4 chars/token)
DEFAULT_SESSION_TURNS = 10
DEFAULT_TOP_K = 5
_SCAN_CAP = 3000                  # most-recent rows considered per semantic_search call

# Directories skipped when inferring the project's language from file extensions.
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
_EXT_TO_LANGUAGE = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".jsx": "JavaScript", ".java": "Java", ".go": "Go", ".rs": "Rust", ".rb": "Ruby",
    ".cpp": "C++", ".cc": "C++", ".c": "C", ".h": "C/C++", ".cs": "C#", ".html": "HTML",
    ".css": "CSS", ".swift": "Swift", ".kt": "Kotlin", ".php": "PHP",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS context_turns (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    role      TEXT    NOT NULL,
    content   TEXT    NOT NULL,
    project   TEXT    NOT NULL DEFAULT '',
    embedding BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_context_turns_timestamp ON context_turns(timestamp);
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


# ── Lightweight local "vector store" ─────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def embed(text: str) -> np.ndarray:
    """Deterministic hashing-trick bag-of-words, L2-normalized so cosine
    similarity is a plain dot product. No model download, no network call,
    no GPU — see the module docstring for when this stops being enough."""
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    for tok in _tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % EMBED_DIM] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _row_to_vec(row: sqlite3.Row) -> np.ndarray:
    return np.frombuffer(row["embedding"], dtype=np.float32)


def log_turn(role: str, content: str, project: str = "", db_path: Optional[Path] = None) -> int:
    """Persists one conversation turn with its embedding, for later semantic
    recall. Call this alongside wherever a turn already gets appended to the
    live session log (main.py's self._session_log) — this is the durable,
    cross-session twin of that in-memory list."""
    content = (content or "").strip()
    if not content:
        return -1
    vec = embed(content)
    with _lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute(
                "INSERT INTO context_turns (timestamp, role, content, project, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    role,
                    content,
                    project,
                    vec.tobytes(),
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def semantic_search(query: str, top_k: int = DEFAULT_TOP_K, db_path: Optional[Path] = None) -> list[dict]:
    """Cosine-ranked past turns for `query`. Linear scan over the most recent
    _SCAN_CAP rows — see the module docstring for the Chroma/FAISS upgrade
    path if that scan ever becomes the bottleneck."""
    if not query.strip():
        return []
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM context_turns ORDER BY id DESC LIMIT ?", (_SCAN_CAP,)
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return []

    qvec = embed(query)
    scored = [(float(np.dot(qvec, _row_to_vec(r))), r) for r in rows]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [
        {"content": r["content"], "role": r["role"], "score": round(score, 3), "timestamp": r["timestamp"]}
        for score, r in scored[:top_k]
        if score > 0.0
    ]


def count_turns(db_path: Optional[Path] = None) -> int:
    conn = _connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) AS n FROM context_turns").fetchone()["n"]
    finally:
        conn.close()


def clear_turns(db_path: Optional[Path] = None) -> int:
    """Deletes every stored turn. Returns how many were removed."""
    with _lock:
        conn = _connect(db_path)
        try:
            n = conn.execute("SELECT COUNT(*) AS n FROM context_turns").fetchone()["n"]
            conn.execute("DELETE FROM context_turns")
            conn.commit()
            return n
        finally:
            conn.close()


def forget_turns(substring: str, db_path: Optional[Path] = None) -> int:
    """Deletes stored turns whose content contains `substring` (case
    insensitive) — the privacy control for "forget the time I mentioned X"
    without wiping the whole history."""
    with _lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute(
                "DELETE FROM context_turns WHERE content LIKE ? COLLATE NOCASE",
                (f"%{substring}%",),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()


# ── Preferences (delegates to memory_manager — not reimplemented here) ──────

def get_preferences() -> dict:
    try:
        from memory.memory_manager import load_memory
        return load_memory().get("preferences", {})
    except Exception:
        return {}


def get_open_tasks() -> dict:
    try:
        from memory.memory_manager import load_memory
        return load_memory().get("wishes", {})
    except Exception:
        return {}


# ── Project context ─────────────────────────────────────────────────────────

def _detect_languages(project_root: Path, sample_limit: int = 4000) -> list[str]:
    counts: Counter = Counter()
    scanned = 0
    for path in project_root.rglob("*"):
        if scanned >= sample_limit:
            break
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        scanned += 1
        lang = _EXT_TO_LANGUAGE.get(path.suffix.lower())
        if lang:
            counts[lang] += 1
    return [lang for lang, _ in counts.most_common(3)]


def _recent_files(days: int = 7, limit: int = 5) -> list[str]:
    try:
        from core.predictive_assistant import get_recent_events
    except Exception:
        return []
    try:
        events = get_recent_events(days=days)
    except Exception:
        return []
    seen: list[str] = []
    for e in reversed(events):  # most recent first
        ctx = (e["context"] or "").strip()
        if ctx and ctx not in seen:
            seen.append(ctx)
        if len(seen) >= limit:
            break
    return seen


def _git_branch(project_root: Path) -> str:
    try:
        from tool_connectors.connectors.git_connector import GitConnector
        connector = GitConnector(repo_path=project_root)
        if not connector.health_check():
            return ""
        result = connector.execute("branch_info", {})
        if not result.success or not isinstance(result.output, dict):
            return ""
        # Just the branch name — result.message/upstream can run to a full
        # `git status -sb` dump (one line per changed file), which has no
        # business bloating a "project context" fact meant to stay one line.
        return str(result.output.get("branch", "")).strip()
    except Exception:
        return ""


def get_project_context(project_root: Optional[Path] = None) -> dict:
    root = Path(project_root or BASE_DIR)
    return {
        "root": str(root),
        "languages": _detect_languages(root),
        "recent_files": _recent_files(),
        "open_tasks": [f"{k}: {v.get('value', '')}" for k, v in get_open_tasks().items()],
        "git_branch": _git_branch(root),
    }


# ── Assembly ─────────────────────────────────────────────────────────────────

@dataclass
class ContextBundle:
    session: list[str] = field(default_factory=list)
    project: dict = field(default_factory=dict)
    preferences: dict = field(default_factory=dict)
    retrieved: list[dict] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Structured, labeled sections — never a raw dump — so the model
        can tell 'this is a stored preference' apart from 'this is something
        that happened five minutes ago' apart from 'this is an old,
        semantically-related conversation'."""
        sections: list[str] = []

        if self.preferences:
            lines = [f"- {k}: {v.get('value', v) if isinstance(v, dict) else v}" for k, v in self.preferences.items()]
            sections.append("[USER PREFERENCES]\n" + "\n".join(lines))

        if self.project:
            p = self.project
            lines = []
            if p.get("languages"):
                lines.append(f"- Languages in use: {', '.join(p['languages'])}")
            if p.get("git_branch"):
                lines.append(f"- Git: {p['git_branch']}")
            if p.get("recent_files"):
                lines.append(f"- Recently touched: {', '.join(p['recent_files'])}")
            if p.get("open_tasks"):
                lines.append("- Open tasks: " + "; ".join(p["open_tasks"]))
            if lines:
                sections.append("[PROJECT CONTEXT]\n" + "\n".join(lines))

        if self.session:
            sections.append("[RECENT CONVERSATION]\n" + "\n".join(self.session))

        if self.retrieved:
            lines = [f"- ({r['timestamp']}) {r['content']}" for r in self.retrieved]
            sections.append("[RELEVANT PAST CONTEXT]\n" + "\n".join(lines))

        return "\n\n".join(sections)

    def __len__(self) -> int:
        return len(self.to_prompt())


def _truncate_to_budget(bundle: ContextBundle, max_chars: int) -> ContextBundle:
    """Greedy trim, lowest-priority first: semantically-retrieved history is
    the least essential (it's a bonus recall, not current state), then
    project facts, then the oldest session turns — preferences are dropped
    last since they're both small and the most decision-relevant single
    fact ("always answer in Spanish") a truncated bundle could still carry."""
    while len(bundle) > max_chars:
        if bundle.retrieved:
            bundle.retrieved.pop()
        elif bundle.project.get("open_tasks"):
            bundle.project["open_tasks"].pop()
        elif bundle.project.get("recent_files"):
            bundle.project["recent_files"].pop()
        elif len(bundle.session) > 1:
            bundle.session.pop(0)  # drop the oldest turn first
        else:
            break
    return bundle


def build_context(
    query: str,
    session_log: Optional[list[str]] = None,
    project_root: Optional[Path] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    session_turns: int = DEFAULT_SESSION_TURNS,
    top_k: int = DEFAULT_TOP_K,
    db_path: Optional[Path] = None,
) -> ContextBundle:
    """Ranks and truncates the three context layers to fit max_chars,
    prioritizing recency (session) and relevance (semantic retrieval) the
    same way the docstring above describes. `query` is normally the user's
    latest utterance — it drives which past turns semantic_search surfaces."""
    session = list(session_log or [])[-session_turns:]
    project = get_project_context(project_root)
    preferences = get_preferences()

    # Session lines are role-prefixed ("User: ...", "JARVIS: ..."), but
    # stored turn content isn't — compare on the bare text after the first
    # ": " so an already-visible turn doesn't also show up as "relevant past
    # context" a few lines down.
    session_bare = {line.split(": ", 1)[-1] for line in session}
    retrieved = [
        hit for hit in semantic_search(query, top_k=top_k * 2, db_path=db_path)
        if hit["content"] not in session_bare
    ][:top_k]

    bundle = ContextBundle(session=session, project=project, preferences=preferences, retrieved=retrieved)
    return _truncate_to_budget(bundle, max_chars)


# ── CLI: inspect / clear stored context ──────────────────────────────────────

def _print_show(query: str) -> None:
    prefs = get_preferences()
    project = get_project_context()
    n_turns = count_turns()

    print("=== USER PREFERENCES (memory/long_term.json - edit via the JARVIS settings memory panel) ===")
    if prefs:
        for k, v in prefs.items():
            print(f"  {k}: {v.get('value', v) if isinstance(v, dict) else v}")
    else:
        print("  (none stored)")

    print("\n=== PROJECT CONTEXT (derived, not stored - nothing here to delete directly) ===")
    print(f"  root: {project['root']}")
    print(f"  languages: {', '.join(project['languages']) or '(none detected)'}")
    print(f"  git branch: {project['git_branch'] or '(not a git repo / git unavailable)'}")
    print(f"  recently touched: {', '.join(project['recent_files']) or '(none logged yet)'}")
    print(f"  open tasks: {'; '.join(project['open_tasks']) or '(none)'}")

    print(f"\n=== STORED CONVERSATION TURNS: {n_turns} (core/context_manager.py's own store) ===")
    if query:
        hits = semantic_search(query, top_k=10)
        print(f"  Top matches for query {query!r}:")
        for h in hits:
            print(f"  [{h['score']:.3f}] ({h['timestamp']}) {h['role']}: {h['content'][:120]}")
        if not hits:
            print("  (no matches)")


def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="context_manager",
        description="Inspect or clear what JARVIS's contextual memory has stored — "
                     "\"what do you know about me/this project\".",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="Show preferences, project context, and stored turns.")
    show.add_argument("--query", default="", help="Also show semantically-relevant stored turns for this text.")

    sub.add_parser("clear-turns", help="Delete ALL stored conversation turns. Preferences/project notes are untouched.")

    forget = sub.add_parser("forget", help="Delete stored turns containing a substring (case-insensitive).")
    forget.add_argument("text")

    args = parser.parse_args()

    if args.command == "show":
        _print_show(args.query)
    elif args.command == "clear-turns":
        n = clear_turns()
        print(f"Deleted {n} stored turn(s).")
    elif args.command == "forget":
        n = forget_turns(args.text)
        print(f"Deleted {n} turn(s) containing {args.text!r}.")


if __name__ == "__main__":
    _cli()
