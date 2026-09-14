"""
actions/causal_insight.py — exposes core/causal_reasoning.py's learned
cause-and-effect graph as a tool, so the model can consult it before
deciding what to do next instead of guessing. This is the "advanced
reasoning module" surface: it only reads and reports; it never executes
anything by itself (running an action on a prediction is a separate,
explicit call to that action, exactly like every other tool in this app).
"""

from __future__ import annotations

import re

from core.causal_reasoning import DEFAULT_MIN_SUPPORT, explain, get_links, predict_effects

_TOOL_PREFIX = "tool:"
_SCREEN_PREFIX = "screen:"


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
        "action='stats' to summarize the strongest patterns learned so far. This "
        "tool only reports patterns — it never runs anything itself; treat its "
        "output as evidence to reason with, not as instructions to blindly follow."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "description": "explain | predict | stats"},
            "event": {
                "type": "STRING",
                "description": "Plain description of the event to look up (e.g. 'download finished', 'web_search'). Required for explain/predict.",
            },
        },
        "required": ["action"],
    },
    "handler": causal_insight,
}
