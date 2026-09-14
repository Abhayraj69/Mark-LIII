"""
Study mode: opt-in, continuous screen capture for studying — the one place
in this feature (and in JARVIS generally) where vision runs on a timer
instead of once per explicit request.

Every other vision path in this codebase (screen_process, study_notes,
computer_control's screen_find) is one-shot and user-triggered. A background
watcher that keeps looking at the screen without being asked has a
meaningfully different privacy posture, so this is never turned on by
JARVIS on its own initiative — only ever by an explicit user request to
start it — costs an API call every interval, and is scoped to the current
process only (no persisted setting to survive a restart, unlike the
wake-word toggle): the user has to ask for it again next time on purpose.

Runs as a daemon thread guarded by module-level state rather than through
main.py's live session loop, so this feature doesn't need to touch that
already-large, already-in-flux file to work.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

from actions.screen_processor import _capture_screen, _vision_query
from actions.study_notes import _NOTES_PROMPT
from actions.file_controller import _resolve_path
from core.adaptive_poll import AdaptiveInterval

_MIN_INTERVAL = 30
_MAX_INTERVAL = 600
_DEFAULT_INTERVAL = 90

# Consecutive ticks with no new material before the capture interval
# doubles (capped at _MAX_INTERVAL) — see AdaptiveInterval. A capture
# failure counts as "unchanged" too, so a stuck pipeline backs off instead
# of retrying at full speed forever.
_BACKOFF_PATIENCE = 3

_lock = threading.Lock()
_state = {
    "active": False,
    "thread": None,
    "stop_event": None,
    "interval": _DEFAULT_INTERVAL,
    "captures": 0,
    "file_path": None,
    "started_at": None,
}


def _clamp_interval(raw) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_INTERVAL
    return max(_MIN_INTERVAL, min(value, _MAX_INTERVAL))


def _run_loop(stop_event: threading.Event, interval: int, file_path: Path, topic_hint: str, player=None):
    last_notes = ""
    backoff = AdaptiveInterval(base=interval, max_interval=_MAX_INTERVAL,
                                patience=_BACKOFF_PATIENCE)
    with _lock:
        _state["poll_interval_seconds"] = backoff.current

    while not stop_event.wait(backoff.current):
        changed = False
        try:
            image_bytes, mime_type = _capture_screen()
            prompt = _NOTES_PROMPT
            if topic_hint:
                prompt += f"\n\nThe user says this material is about: {topic_hint}."
            notes_text = _vision_query(image_bytes, mime_type, prompt)

            if notes_text and notes_text.strip().upper() != "NO_CONTENT" \
                    and notes_text.strip() != last_notes.strip():
                changed = True
                last_notes = notes_text
                timestamp = datetime.now().strftime("%H:%M:%S")
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n---- {timestamp} ----\n{notes_text}\n")

                with _lock:
                    _state["captures"] += 1
                if player:
                    player.write_log(f"[study_mode] 🔴 captured new material at {timestamp} ({_state['captures']} total)")
        except Exception as e:
            print(f"[study_mode] capture/analyze/write failed: {e}")

        with _lock:
            _state["poll_interval_seconds"] = backoff.report(changed)


def _start(params: dict, player=None) -> str:
    with _lock:
        if _state["active"]:
            return (f"Study mode is already running (every {_state['interval']}s, "
                     f"{_state['captures']} capture(s) so far).")

        interval    = _clamp_interval(params.get("interval_seconds"))
        topic_hint  = params.get("topic_hint", "").strip()
        save_path   = params.get("save_path", "").strip() or "desktop/JarvisNotes"
        target_dir  = _resolve_path(save_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path   = target_dir / f"StudyMode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        stop_event = threading.Event()
        thread = threading.Thread(
            target=_run_loop,
            args=(stop_event, interval, file_path, topic_hint, player),
            daemon=True,
        )
        _state.update(
            active=True, thread=thread, stop_event=stop_event,
            interval=interval, captures=0, file_path=file_path,
            started_at=datetime.now(),
        )
        thread.start()

    if player:
        player.write_log(f"[study_mode] 🔴 STARTED — capturing every {interval}s → {file_path}")

    return (
        f"Study mode is ON — I'll re-read your screen every {interval} seconds "
        f"and log new material to {file_path.name} in {target_dir}. "
        f"Say 'stop study mode' whenever you want me to turn it off."
    )


def _stop(player=None) -> str:
    with _lock:
        if not _state["active"]:
            return "Study mode isn't running."
        stop_event = _state["stop_event"]
        thread     = _state["thread"]
        captures   = _state["captures"]
        file_path  = _state["file_path"]

    stop_event.set()
    thread.join(timeout=5)

    with _lock:
        _state["active"] = False

    if player:
        player.write_log("[study_mode] ⏹ stopped")

    if captures:
        return f"Study mode stopped. Logged {captures} update(s) to {file_path}."
    return "Study mode stopped. Nothing new was captured this session."


def _status() -> str:
    with _lock:
        if not _state["active"]:
            return "Study mode is off."
        elapsed = int((datetime.now() - _state["started_at"]).total_seconds())
        current = _state.get("poll_interval_seconds", _state["interval"])
        rate_txt = (
            f"currently every {current}s (backed off from {_state['interval']}s)"
            if current != _state["interval"]
            else f"every {current}s"
        )
        return (
            f"Study mode is ON — running for {elapsed}s, capturing {rate_txt}, "
            f"{_state['captures']} update(s) logged to {_state['file_path']}."
        )


def study_mode(
    parameters: dict = None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    action = str(params.get("action", "")).strip().lower()

    if action == "start":
        return _start(params, player=player)
    if action == "stop":
        return _stop(player=player)
    if action == "status":
        return _status()
    return "Unknown action. Use one of: start | stop | status."


# ── Tool declaration (auto-discovered by core/action_loader.py) ─────────────
TOOL = {
    "name": "study_mode",
    "description": (
        "Turns continuous background screen-watching for studying on or "
        "off. UNLIKE study_notes/study_quiz, this keeps re-reading the "
        "screen on a timer instead of once — call action='start' ONLY when "
        "the user explicitly asks for this (e.g. 'watch my screen while I "
        "study', 'turn on study mode', 'keep taking notes as I go'). NEVER "
        "start it on your own initiative or as a side effect of a one-off "
        "study_notes/study_quiz request. Call action='stop' when they ask "
        "to turn it off, and action='status' to check if it's running. "
        "Always say out loud, briefly, when you start or stop it."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": "start | stop | status",
            },
            "interval_seconds": {
                "type": "INTEGER",
                "description": f"Seconds between captures, for 'start'. {_MIN_INTERVAL}-{_MAX_INTERVAL}. Default: {_DEFAULT_INTERVAL}",
            },
            "topic_hint": {
                "type": "STRING",
                "description": "What the user is studying, if they said it (for 'start').",
            },
            "save_path": {
                "type": "STRING",
                "description": "Where to save the running notes file, for 'start'. Default: desktop/JarvisNotes",
            },
        },
        "required": ["action"],
    },
    "handler": study_mode,
}
