"""
Study quiz: turn the last study notes (or a fresh screen capture) into a set
of quiz questions with answers, for the live model to ask one at a time.

Deliberately does NOT run the quiz itself — grading a spoken, free-form
answer ("um, I think it's mitochondria") against a stored string is an NLU
problem the live conversational model already handles well, and a Python
string-match would mark correct answers wrong constantly. This tool's job
ends at producing a well-formed Q&A set; core/prompt.txt tells the model how
to run the quiz turn-by-turn from there.
"""
from __future__ import annotations

import json
import re

from actions.screen_processor import _text_query
from actions.study_notes import get_last_notes, _capture_and_extract_notes

_QUIZ_PROMPT = (
    "You will be given a set of study notes. Generate exactly {count} quiz "
    "questions at {difficulty} difficulty that test understanding of these "
    "notes. Base every question, answer, and explanation strictly on the "
    "notes below — do not introduce facts, examples, or definitions that "
    "are not present in them.\n\n"
    "Return ONLY a valid JSON array, no markdown code fences, no commentary "
    "before or after it. Each element must be an object with exactly these "
    "keys: \"question\" (string), \"answer\" (string, the correct answer), "
    "and \"explanation\" (one short sentence on why that's the answer).\n\n"
    "NOTES:\n{notes}"
)

_VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def _parse_quiz_json(raw: str) -> list[dict] | None:
    text = raw.strip()
    # Models sometimes wrap JSON in ```json ... ``` despite instructions not to.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list) or not data:
        return None
    for item in data:
        if not isinstance(item, dict) or not all(k in item for k in ("question", "answer", "explanation")):
            return None
    return data


def study_quiz(
    parameters: dict = None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}

    try:
        question_count = int(params.get("question_count", 5))
    except (TypeError, ValueError):
        question_count = 5
    question_count = max(1, min(question_count, 15))

    difficulty = str(params.get("difficulty", "medium")).strip().lower()
    if difficulty not in _VALID_DIFFICULTIES:
        difficulty = "medium"

    if player:
        player.write_log(f"[study_quiz] generating {question_count} {difficulty} questions")

    notes = get_last_notes()
    notes_text = notes.get("text", "")

    if not notes_text:
        notes_text, error = _capture_and_extract_notes()
        if error:
            return f"{error} I need some study material on screen or already noted to quiz you on."

    prompt = _QUIZ_PROMPT.format(count=question_count, difficulty=difficulty, notes=notes_text)

    try:
        raw = _text_query(prompt)
    except Exception as e:
        return f"Could not generate quiz questions: {e}"

    quiz = _parse_quiz_json(raw)
    if quiz is None:
        return "Could not generate a well-formed quiz from the notes — try again."

    lines = [f"Quiz ({len(quiz)} questions, {difficulty} difficulty). Ask ONE at a time and wait for the answer before revealing it:"]
    for i, item in enumerate(quiz, 1):
        lines.append(f"\nQ{i}: {item['question']}\nCorrect answer: {item['answer']}\nWhy: {item['explanation']}")

    return "\n".join(lines)


# ── Tool declaration (auto-discovered by core/action_loader.py) ─────────────
TOOL = {
    "name": "study_quiz",
    "description": (
        "Generates quiz questions from the last study notes (or a fresh "
        "screen capture if none exist yet) and returns them WITH answers "
        "for you to ask the user one at a time. Call when the user asks to "
        "be quizzed or tested on the material. After calling this, ask the "
        "questions yourself in the conversation — do not just read the "
        "tool's output aloud verbatim."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "question_count": {
                "type": "INTEGER",
                "description": "How many questions to generate (1-15). Default: 5",
            },
            "difficulty": {
                "type": "STRING",
                "description": "easy | medium | hard. Default: medium",
            },
        },
        "required": [],
    },
    "handler": study_quiz,
}
