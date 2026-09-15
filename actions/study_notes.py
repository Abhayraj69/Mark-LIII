"""
Study notes: capture the screen, extract structured study notes with a
one-shot vision call, and save them to a file.

Reuses _capture_screen and _vision_query from screen_processor.py rather than
opening the live multimodal session — this is a single structured-text read
of a screenshot, not a spoken interaction.
"""
from __future__ import annotations

from actions.screen_processor import _capture_screen, _vision_query
from actions.file_controller import create_file, _resolve_path
from actions.open_app import open_app

_NOTES_PROMPT = (
    "This is a screenshot of study material (a textbook page, slide, article, "
    "code, or diagram). Extract it into concise, structured study notes: a "
    "title line, then organized bullet points grouped under short headings, "
    "with any definitions or formulas copied verbatim. "
    "Bold each key term only once, the first time it is introduced or "
    "defined — never bold the same word or phrase again later in the notes, "
    "and do not bold ordinary nouns/verbs that are not themselves the term "
    "being taught. "
    "If the source material is already bullet points, do not just copy it — "
    "tighten wording and cut redundant phrasing while keeping every fact. "
    "If a diagram has labels but no accompanying caption or body text "
    "explaining what a labeled part does, list it as a labeled part only — "
    "do not invent or recall a definition/function for it from general "
    "knowledge; the notes must reflect only what is actually written or "
    "captioned in the image, not what you know about the subject. "
    "Do not add commentary, opinions, or information not present in the "
    "image, and ignore page chrome (titles bars, headers/footers, page "
    "numbers, watermarks) unless it is the actual subject heading. "
    "If there is no readable study content in the image, reply "
    "with exactly: NO_CONTENT"
)

# Session-local handoff so study_quiz can build questions from the last notes
# without re-capturing the screen. Cleared at process start; not persisted.
_last_notes: dict = {"topic": "", "text": ""}


def get_last_notes() -> dict:
    return dict(_last_notes)


def _capture_and_extract_notes(topic_hint: str = "") -> tuple[str, str]:
    """Capture the screen and extract structured notes from it.

    Shared by study_notes (which saves the result to a file) and study_quiz
    (which can fall back to a fresh capture when there are no notes yet in
    this session) so there is one place that owns the capture + vision call.

    Returns (notes_text, error). Exactly one of the two is non-empty:
    notes_text is set on success, error is a human-readable message
    (capture failure, vision failure, or no readable content found).
    """
    try:
        image_bytes, mime_type = _capture_screen()
    except Exception as e:
        return "", f"Could not capture the screen: {e}"

    prompt = _NOTES_PROMPT
    if topic_hint:
        prompt += f"\n\nThe user says this material is about: {topic_hint}."

    try:
        notes_text = _vision_query(image_bytes, mime_type, prompt)
    except Exception as e:
        return "", f"Could not analyze the screen: {e}"

    if not notes_text or notes_text.strip().upper() == "NO_CONTENT":
        return "", "I couldn't find any readable study material on the screen right now."

    _last_notes["topic"] = topic_hint
    _last_notes["text"]  = notes_text
    return notes_text, ""


def study_notes(
    parameters: dict = None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params       = parameters or {}
    topic_hint   = params.get("topic_hint", "").strip()
    save_path    = params.get("save_path", "").strip() or "desktop/JarvisNotes"
    open_notepad = params.get("open_notepad", True)

    target_dir = _resolve_path(save_path)

    if player:
        player.write_log(f"[study_notes] capturing screen{' — ' + topic_hint if topic_hint else ''}")

    notes_text, error = _capture_and_extract_notes(topic_hint)
    if error:
        return f"{error} Nothing was saved."

    from datetime import datetime
    filename = f"StudyNotes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    result = create_file(str(target_dir), name=filename, content=notes_text)

    if not result.startswith("File created"):
        return f"Generated the notes but could not save them: {result}"

    summary = f"Notes saved as {filename} in {target_dir}.\n\n{notes_text}"

    if open_notepad:
        saved_path = str(target_dir / filename)
        open_result = open_app(parameters={"app_name": "notepad", "file_path": saved_path})
        if not open_result.startswith("Opened"):
            summary += f"\n\n(Could not open Notepad automatically: {open_result})"

    return summary


# ── Tool declaration (auto-discovered by core/action_loader.py) ─────────────
TOOL = {
    "name": "study_notes",
    "description": (
        "Captures the screen, reads the study material on it (textbook page, "
        "slide, article, code, diagram), and generates concise structured "
        "notes. Call when the user asks to take notes on what's on screen, "
        "summarize this material, or make study notes. Does NOT require a "
        "prior screen_process call — it captures its own screenshot. Saves "
        "the notes to a text file."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "topic_hint": {
                "type": "STRING",
                "description": "What the material is about, if the user said it (helps focus the notes)",
            },
            "save_path": {
                "type": "STRING",
                "description": "Where to save, e.g. 'desktop', 'documents', or a full path. Default: desktop/JarvisNotes",
            },
            "open_notepad": {
                "type": "BOOLEAN",
                "description": "Open the saved notes in Notepad after writing. Default: true",
            },
        },
        "required": [],
    },
    "handler": study_notes,
}
