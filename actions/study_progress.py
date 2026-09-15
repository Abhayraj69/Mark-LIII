"""
Study progress: cross-session quiz history — log a graded answer, or ask
for weak topics / previously missed questions / overall stats.

Grading happens entirely in the live conversation (core/prompt.txt has the
model grade each spoken answer by meaning, not string-match) — this tool
only persists the outcome the model already decided, via
memory/study_history.py's SQLite store, and reads it back later.
"""
from __future__ import annotations

from memory.study_history import (
    log_answer,
    get_weak_topics,
    get_missed_questions,
    get_stats,
    clear_history,
)

_VALID_ACTIONS = {"log_result", "weak_topics", "missed_questions", "stats", "clear"}


def study_progress(
    parameters: dict = None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    action = str(params.get("action", "")).strip().lower()

    if action not in _VALID_ACTIONS:
        return f"Unknown action: '{action}'. Use one of: {', '.join(sorted(_VALID_ACTIONS))}."

    if player:
        player.write_log(f"[study_progress] {action}")

    if action == "log_result":
        question = params.get("question", "").strip()
        if not question:
            return "No question provided to log."
        row_id = log_answer(
            question=question,
            correct_answer=params.get("correct_answer", ""),
            correct=bool(params.get("correct", False)),
            topic=params.get("topic", ""),
            user_answer=params.get("user_answer", ""),
        )
        return "Logged." if row_id >= 0 else "Nothing to log."

    if action == "weak_topics":
        limit = _safe_int(params.get("limit"), default=5)
        topics = get_weak_topics(limit=limit)
        if not topics:
            return "Not enough quiz history yet to identify weak topics."
        lines = [
            f"- {t['topic']}: {t['correct']}/{t['attempts']} correct ({t['accuracy']:.0%})"
            for t in topics
        ]
        return "Weakest topics so far:\n" + "\n".join(lines)

    if action == "missed_questions":
        limit = _safe_int(params.get("limit"), default=10)
        topic = params.get("topic", "").strip()
        missed = get_missed_questions(topic=topic, limit=limit)
        if not missed:
            return "No missed questions on record" + (f" for '{topic}'." if topic else ".")
        lines = [
            f"- [{m['topic'] or 'general'}] {m['question']} "
            f"(correct answer: {m['correct_answer']})"
            for m in missed
        ]
        return "Previously missed questions:\n" + "\n".join(lines)

    if action == "stats":
        stats = get_stats()
        if not stats["total_answered"]:
            return "No quiz history recorded yet."
        return (
            f"{stats['total_correct']}/{stats['total_answered']} correct overall "
            f"({stats['accuracy']:.0%})."
        )

    # action == "clear"
    n = clear_history()
    return f"Cleared {n} stored quiz answer(s)."


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ── Tool declaration (auto-discovered by core/action_loader.py) ─────────────
TOOL = {
    "name": "study_progress",
    "description": (
        "Tracks quiz performance across sessions, since the live quiz "
        "conversation itself does not persist. Actions: 'log_result' (call "
        "once right after YOU grade each quiz answer, so it's remembered "
        "for next time — pass topic, question, correct_answer, user_answer, "
        "and correct); 'weak_topics' (topics the user gets wrong most, for "
        "'what should I review'); 'missed_questions' (previously missed "
        "questions, to requiz on them); 'stats' (overall accuracy); 'clear' "
        "(erase all stored quiz history, if the user asks to reset/forget "
        "their progress)."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": "log_result | weak_topics | missed_questions | stats | clear",
            },
            "topic": {
                "type": "STRING",
                "description": "Subject of the question, e.g. the notes' topic_hint. Optional.",
            },
            "question": {
                "type": "STRING",
                "description": "The quiz question text (required for log_result).",
            },
            "correct_answer": {
                "type": "STRING",
                "description": "The correct answer (for log_result).",
            },
            "user_answer": {
                "type": "STRING",
                "description": "What the user actually said (for log_result).",
            },
            "correct": {
                "type": "BOOLEAN",
                "description": "Whether the user's answer was graded correct (for log_result).",
            },
            "limit": {
                "type": "INTEGER",
                "description": "Max results for weak_topics/missed_questions. Default: 5/10.",
            },
        },
        "required": ["action"],
    },
    "handler": study_progress,
}
