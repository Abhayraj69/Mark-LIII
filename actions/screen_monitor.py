"""
actions/screen_monitor.py — continuous, configurable screen monitoring.

Unlike the one-shot `screen_process` tool (main.py) which grabs a single
frame on demand, this runs a background loop that repeatedly captures the
screen every `interval_seconds`, asks Gemini vision whether the configured
`watch_for` condition is currently true, and speaks up only when it is —
so a long-running watch doesn't chatter on every tick. It stops itself after
`duration_minutes` if given, or runs until action="stop" is called.

Only one watch runs at a time, matching the single-user, single-desktop
assumption already made by actions/background_monitor.py (which does the
same thing for news topics instead of screen content). Starting a new watch
replaces whatever was running.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from actions.screen_processor import _capture_screen, _vision_query
from core.causal_reasoning import record_event as _record_causal_event

_DEFAULT_INTERVAL_S = 30
_MIN_INTERVAL_S = 5
_MAX_INTERVAL_S = 600
_MAX_DURATION_MIN = 240  # 4 hours — a safety ceiling, not a suggested value

_lock = threading.Lock()
_stop_event: Optional[threading.Event] = None
_thread: Optional[threading.Thread] = None
_state: dict = {}  # watch_for/interval/started/deadline/checks/alerts — UI-readable snapshot


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower().strip())[:40].strip("_")


def _run_loop(watch_for: str, interval_s: int, deadline: Optional[datetime],
              stop_event: threading.Event, speak) -> None:
    while not stop_event.is_set():
        if deadline and datetime.now() >= deadline:
            print("[ScreenMonitor] ⏰ Duration elapsed — stopping.")
            break
        try:
            img_bytes, mime_type = _capture_screen()
            prompt = (
                f"You are watching a user's screen for one thing: {watch_for}. "
                "Answer with exactly one word first — YES or NO — for whether that "
                "condition is currently visible, then (only if YES) one short sentence "
                "describing what you see."
            )
            reply = _vision_query(img_bytes, mime_type, prompt)
            _state["last_check"] = datetime.now().isoformat(timespec="seconds")
            _state["checks"] = _state.get("checks", 0) + 1

            if reply.strip().upper().startswith("YES"):
                detail = reply.split(None, 1)[1].strip() if " " in reply else ""
                alert = f"[SCREEN_MONITOR_ALERT] {watch_for}" + (f" — {detail}" if detail else "")
                _state["alerts"] = _state.get("alerts", 0) + 1
                _state["last_alert"] = alert
                print(f"[ScreenMonitor] 🔔 {alert}")
                try:
                    _record_causal_event(f"screen:{_slug(watch_for)}", detail=detail)
                except Exception as e:
                    print(f"[ScreenMonitor] ⚠️ causal record failed: {e}")
                if speak:
                    try:
                        speak(f"Heads up — I'm seeing {watch_for} on your screen. {detail}".strip())
                    except Exception as e:
                        print(f"[ScreenMonitor] ⚠️ speak() failed: {e}")
        except Exception as e:
            print(f"[ScreenMonitor] ⚠️ Check failed: {e}")

        stop_event.wait(interval_s)

    _state["running"] = False
    print("[ScreenMonitor] 🛑 Stopped.")


def _start(watch_for: str, interval_seconds: int, duration_minutes: int, speak) -> str:
    global _stop_event, _thread

    watch_for = (watch_for or "").strip()
    if not watch_for:
        return "Tell me what to watch for, e.g. 'an error message' or 'a new email popup'."

    interval_s = _clamp(int(interval_seconds or _DEFAULT_INTERVAL_S), _MIN_INTERVAL_S, _MAX_INTERVAL_S)
    duration_min = max(0, int(duration_minutes or 0))
    if duration_min:
        duration_min = min(duration_min, _MAX_DURATION_MIN)
    deadline = datetime.now() + timedelta(minutes=duration_min) if duration_min else None

    with _lock:
        if _thread is not None and _thread.is_alive():
            _stop_event.set()
            _thread.join(timeout=2)

        _stop_event = threading.Event()
        _state.clear()
        _state.update({
            "running": True,
            "watch_for": watch_for,
            "interval_seconds": interval_s,
            "started": datetime.now().isoformat(timespec="seconds"),
            "deadline": deadline.isoformat(timespec="seconds") if deadline else None,
            "checks": 0,
            "alerts": 0,
        })
        _thread = threading.Thread(
            target=_run_loop,
            args=(watch_for, interval_s, deadline, _stop_event, speak),
            daemon=True,
            name="screen-monitor",
        )
        _thread.start()

    duration_txt = f" for {duration_min} minute(s)" if duration_min else " until stopped"
    return f"Watching your screen for '{watch_for}' every {interval_s}s{duration_txt}."


def _stop() -> str:
    global _stop_event, _thread
    with _lock:
        if _thread is None or not _thread.is_alive():
            return "No screen watch is currently running."
        _stop_event.set()
        _thread.join(timeout=2)
        watch_for = _state.get("watch_for", "")
    return f"Stopped watching for '{watch_for}'." if watch_for else "Screen watch stopped."


def _status() -> str:
    with _lock:
        if not _state.get("running"):
            return "No screen watch is currently running."
        s = dict(_state)
    return (
        f"Watching for '{s.get('watch_for')}' every {s.get('interval_seconds')}s "
        f"since {s.get('started')}. Checks so far: {s.get('checks', 0)}, "
        f"alerts: {s.get('alerts', 0)}."
        + (f" Ends at {s['deadline']}." if s.get("deadline") else " Runs until stopped.")
    )


def screen_monitor(parameters: dict, speak=None) -> str:
    action = (parameters.get("action") or "start").strip().lower()
    if action == "start":
        return _start(
            parameters.get("watch_for", ""),
            parameters.get("interval_seconds", _DEFAULT_INTERVAL_S),
            parameters.get("duration_minutes", 0),
            speak,
        )
    if action == "stop":
        return _stop()
    if action == "status":
        return _status()
    return "Specify action: start | stop | status."


TOOL = {
    "name": "screen_monitor",
    "description": (
        "Continuously watches the screen in the background for a specific condition, "
        "checking on a timer instead of a single one-off look. Use action='start' when "
        "the user asks to be watched/alerted/notified about something appearing on "
        "screen (e.g. 'tell me when this download finishes', 'let me know if an error "
        "pops up', 'watch for a new Slack message') — pass 'watch_for' describing the "
        "condition in plain language, and optionally 'interval_seconds' (how often to "
        f"check, default {_DEFAULT_INTERVAL_S}) and 'duration_minutes' (0 or omitted = "
        "runs until stopped). Use action='stop' to end the current watch, and "
        "action='status' to report what is being watched and how it's going. "
        "Only one watch can run at a time; starting a new one replaces the old one."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "description": "start | stop | status"},
            "watch_for": {
                "type": "STRING",
                "description": "Plain-language description of what to detect on screen (start only).",
            },
            "interval_seconds": {
                "type": "INTEGER",
                "description": f"Seconds between checks, {_MIN_INTERVAL_S}-{_MAX_INTERVAL_S} (default {_DEFAULT_INTERVAL_S}).",
            },
            "duration_minutes": {
                "type": "INTEGER",
                "description": f"How long to watch, in minutes, up to {_MAX_DURATION_MIN}. 0 = until stopped.",
            },
        },
        "required": ["action"],
    },
    "handler": screen_monitor,
}
