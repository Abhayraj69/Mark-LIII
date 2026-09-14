"""
actions/causal_insight.py — exposes core/causal_reasoning.py's learned
cause-and-effect graph as a tool, so the model can consult it before
deciding what to do next instead of guessing. This is the "advanced
reasoning module" surface: it only reads and reports; it never executes
anything by itself (running an action on a prediction is a separate,
explicit call to that action, exactly like every other tool in this app).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from core.causal_reasoning import DEFAULT_MIN_SUPPORT, explain, get_links, predict_effects

_TOOL_PREFIX = "tool:"
_SCREEN_PREFIX = "screen:"

# ── Weekly causal-graph digest ────────────────────────────────────────────────
# "This week I noticed X usually leads to Y" — a plain-language summary of
# whatever causal_reasoning has learned recently, pushed into the morning
# briefing (actions/proactive.py's build_morning_brief) instead of sitting
# behind the on-demand causal_insight tool where nobody thinks to ask for it.
_DIGEST_INTERVAL_DAYS = 7


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower().strip())[:40].strip("_")


def _resolve_event_type(hint: str, known: set[str]) -> str:
    """Free-text like 'the download finished' or 'web search' needs to map
    onto an actual namespaced event_type ('screen:download_finished' or
    'tool:web_search'). Tries an exact slug match against both namespaces
    first, then falls back to substring search over whatever event types
    have actually been observed."""
    hint = hint.strip()
    if hint in known:
        return hint
    slug = _slug(hint)
    for candidate in (f"{_SCREEN_PREFIX}{slug}", f"{_TOOL_PREFIX}{slug}", slug):
        if candidate in known:
            return candidate
    for candidate in known:
        if slug and slug in candidate:
            return candidate
    return hint or slug


def _format_link(link, other_field: str) -> str:
    other = getattr(link, other_field)
    return (
        f"{other} (confidence {link.confidence:.0%}, lift {link.lift}x, "
        f"seen {link.support}x, ~{link.avg_lag_seconds:.0f}s later)"
    )


def _load_digest_state() -> dict:
    from memory.memory_manager import load_memory
    data = load_memory().get("causal_digest", {})
    return data if isinstance(data, dict) else {}


def _save_digest_state(state: dict) -> None:
    from memory.memory_manager import MEMORY_PATH, _lock, load_memory
    memory = load_memory()
    memory["causal_digest"] = state
    with _lock:
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_PATH.write_text(
            json.dumps(memory, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _format_digest_link(link) -> str:
    return (
        f"{link.cause} usually leads to {link.effect} "
        f"({link.support}x seen, {link.confidence:.0%} confidence)"
    )


def weekly_digest_text(
    min_support: int = DEFAULT_MIN_SUPPORT,
    max_links: int = 3,
    db_path: Optional[Path] = None,
) -> str:
    """
    Plain-language summary of the strongest causal links actively observed
    in the last 7 days (a link's most recent occurrence falls in that
    window) — NOT the all-time leaderboard causal_insight's own 'stats'
    action returns. Empty string when nothing cleared min_support this
    week; never fabricates a pattern to fill the digest.
    """
    since = datetime.now() - timedelta(days=_DIGEST_INTERVAL_DAYS)
    links = get_links(min_support=min_support, db_path=db_path, since=since)
    if not links:
        return ""

    top = links[:max(1, max_links)]
    if len(top) == 1:
        return f"This week I noticed: {_format_digest_link(top[0])}."
    return "This week I noticed a few patterns:\n" + "\n".join(
        f"  - {_format_digest_link(l)}" for l in top
    )


def get_scheduled_weekly_digest(
    min_support: int = DEFAULT_MIN_SUPPORT,
    max_links: int = 3,
    db_path: Optional[Path] = None,
) -> str:
    """
    The 'scheduled task' entry point: due at most once every 7 days,
    self-throttled via a persisted last-delivered timestamp (mirrors
    actions/background_monitor.py's own 'once per day per topic' pattern).
    Meant to be called from anything that runs periodically — currently
    actions/proactive.py's build_morning_brief() — so it is safe to call on
    every run without re-delivering the same digest daily.

    Returns "" both when it isn't due yet and when it's due but nothing
    cleared the bar this week; the caller can't tell those apart and
    shouldn't need to — either way there's nothing to say right now.
    """
    state = _load_digest_state()
    last_sent = state.get("last_sent", "")
    if last_sent:
        try:
            if datetime.now() - datetime.fromisoformat(last_sent) < timedelta(days=_DIGEST_INTERVAL_DAYS):
                return ""
        except ValueError:
            pass  # corrupt state — treat as never sent, fall through and regenerate

    text = weekly_digest_text(min_support=min_support, max_links=max_links, db_path=db_path)
    # Mark delivered regardless of whether there was anything to report —
    # otherwise a quiet week would retry (and pay for a get_links() scan) on
    # every single call until content finally appears, defeating the whole
    # point of a weekly cadence.
    _save_digest_state({"last_sent": datetime.now().isoformat(timespec="seconds")})
    return text


def causal_insight(parameters: dict) -> str:
    action = (parameters.get("action") or "explain").strip().lower()
    event_hint = (parameters.get("event") or "").strip()

    all_links = get_links(min_support=DEFAULT_MIN_SUPPORT)
    known_types = {l.cause for l in all_links} | {l.effect for l in all_links}

    if action == "stats":
        if not all_links:
            return "No causal patterns learned yet — this builds up as screen alerts and tool calls occur."
        top = all_links[:5]
        return "Strongest known cause -> effect links:\n" + "\n".join(
            f"- {l.cause} -> {l.effect} (confidence {l.confidence:.0%}, lift {l.lift}x, seen {l.support}x)"
            for l in top
        )

    if action == "weekly_digest":
        # On-demand: always fresh, and unlike get_scheduled_weekly_digest()
        # (used by the morning briefing) this never touches the delivery
        # state — asking for it out loud doesn't consume this week's slot.
        return weekly_digest_text() or "No new causal patterns stood out this week."

    if not event_hint:
        return "Specify 'event' — what happened, e.g. 'download finished' or 'web_search'."

    resolved = _resolve_event_type(event_hint, known_types)

    if action == "predict":
        effects = predict_effects(resolved, top_k=3)
        if not effects:
            return f"No strong pattern yet for what follows '{event_hint}'."
        return f"After '{event_hint}', this has usually followed:\n" + "\n".join(
            f"- {_format_link(l, 'effect')}" for l in effects
        )

    # action == "explain" (default): both directions
    result = explain(resolved)
    causes, effects = result["causes_of"], result["effects_of"]
    if not causes and not effects:
        return f"No causal pattern learned yet involving '{event_hint}'."
    lines = [f"What I've learned about '{event_hint}':"]
    if causes:
        lines.append("Tends to happen after:")
        lines.extend(f"  - {_format_link(l, 'cause')}" for l in causes[:5])
    if effects:
        lines.append("Tends to be followed by:")
        lines.extend(f"  - {_format_link(l, 'effect')}" for l in effects[:5])
    return "\n".join(lines)


TOOL = {
    "name": "causal_insight",
    "description": (
        "Consults Jarvis's learned cause-and-effect graph over past screen alerts "
        "(from screen_monitor) and tool actions, to reason about WHY something "
        "happens or WHAT is likely to happen next before deciding what to do. "
        "Use action='predict' with 'event' before reacting to a screen_monitor "
        "alert or a just-completed action, to check what has historically followed "
        "it and weigh that in your next step. Use action='explain' to answer a "
        "user's 'why does X happen' / 'what usually causes X' question. Use "
        "action='stats' to summarize the strongest patterns learned so far, and "
        "action='weekly_digest' to summarize only what's been actively observed "
        "in the last 7 days (the same content proactively surfaced in the "
        "morning briefing). This tool only reports patterns — it never runs "
        "anything itself; treat its output as evidence to reason with, not as "
        "instructions to blindly follow."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "description": "explain | predict | stats | weekly_digest"},
            "event": {
                "type": "STRING",
                "description": "Plain description of the event to look up (e.g. 'download finished', 'web_search'). Required for explain/predict.",
            },
        },
        "required": ["action"],
    },
    "handler": causal_insight,
}
