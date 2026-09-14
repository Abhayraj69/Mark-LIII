"""
Local intent detection for typed commands — the "do it now" path.

A typed command like "volume up" normally makes the same round trip as a
question: text goes to the Live model, the model decides to call
`computer_settings(action="volume_up")`, the call comes back, and only then does
anything happen. That is a network round trip (plus model latency) spent
re-deriving a mapping that never varies.

This module recognises those fixed mappings locally and hands back the tool call
directly, so the action fires in roughly the time it takes to press a key.

Deliberately narrow, on two rules:

* **Actions only, never questions.** A match executes and reports a one-line
  confirmation; it does not produce spoken prose. So anything whose value *is*
  the prose — "what's the weather", "search for X", "summarise this" — is left
  to the model, which answers it properly. Only commands whose payoff is the
  action itself are matched here.
* **Whole-utterance matches only.** Every pattern is anchored with
  `re.fullmatch` against the normalised text, so a sentence that merely
  *contains* "volume up" ("why does volume up not work?") does not match. A
  miss is free — the text simply takes the normal model path — so the patterns
  stay strict rather than clever.

Disruptive or irreversible actions (restart, shutdown, Wi-Fi) are intentionally
absent: those should keep going through the model and its confirmation gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Intent:
    """A resolved local command: the tool to call, its arguments, and the line
    to show the user once it has run."""
    tool:  str
    args:  dict = field(default_factory=dict)
    reply: str  = "Done."


# Filler that carries no instruction. Stripped before matching so the patterns
# below only describe the command itself instead of every way of asking politely.
_PREFIX = re.compile(
    r"^(?:(?:hey|ok|okay)\s+)?(?:jarvis[\s,]+)?"
    r"(?:(?:please|can you|could you|would you|will you|go ahead and|"
    r"i want you to|i need you to|i'd like you to)\s+)*"
)
_SUFFIX = re.compile(r"(?:\s+(?:please|for me|right now|now|jarvis))*$")

_STOPWORDS = {"the", "a", "an", "my", "this", "that", "some", "up", "it"}

# App names that would otherwise be swallowed by the generic "open X" rule and
# routed to open_app, when a dedicated action below handles them better.
_NOT_AN_APP = {
    "tab", "new tab", "window", "settings", "file explorer", "windows explorer",
    "task manager", "run", "run dialog", "desktop", "screenshot", "dark mode",
    "volume", "brightness", "the door", "pod bay doors", "camera",
}


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace("’", "'")
    t = re.sub(r"[.!?]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    t = _PREFIX.sub("", t, count=1)
    t = _SUFFIX.sub("", t, count=1)
    return t.strip()


# ── computer_settings: (pattern, action, confirmation) ───────────────────────
# Actions are the literal enum values computer_settings accepts.
_SETTINGS = [
    (r"(?:turn (?:the )?)?volume up|turn up (?:the )?volume|(?:increase|raise) (?:the )?volume|louder|make it louder",
     "volume_up", "Volume up."),
    (r"(?:turn (?:the )?)?volume down|turn down (?:the )?volume|(?:decrease|lower|reduce) (?:the )?volume|quieter|make it quieter",
     "volume_down", "Volume down."),
    # Only speaker mute. Bare "mute" is left alone: in this app it far more
    # often means the microphone, which is a UI control, not this tool.
    (r"(?:un)?mute (?:the )?(?:volume|sound|audio|speakers?)",
     "mute", "Toggled audio mute."),

    (r"(?:turn (?:the )?)?brightness up|turn up (?:the )?brightness|(?:increase|raise) (?:the )?brightness|brighter",
     "brightness_up", "Brightness up."),
    (r"(?:turn (?:the )?)?brightness down|turn down (?:the )?brightness|(?:decrease|lower|reduce|dim) (?:the )?brightness|dimmer|dim (?:the )?screen",
     "brightness_down", "Brightness down."),

    (r"lock (?:the |my )?(?:screen|pc|computer|laptop)|lock it",
     "lock_screen", "Locking the screen."),
    (r"show (?:the )?desktop|minimi[sz]e all(?: windows)?",
     "show_desktop", "Showing the desktop."),
    (r"(?:open |show |launch )?task manager",
     "task_manager", "Opening Task Manager."),
    (r"(?:open |launch )?(?:file explorer|windows explorer|my computer|this pc)",
     "file_explorer", "Opening File Explorer."),
    (r"open (?:windows|system|pc) settings",
     "open_settings", "Opening Windows Settings."),
    (r"(?:take (?:a )?)?screen ?shot|capture (?:the )?screen",
     "screenshot", "Screenshot taken."),
    (r"(?:turn on |enable |switch to |toggle )?dark mode",
     "dark_mode", "Toggling dark mode."),

    (r"minimi[sz]e(?: (?:this|the) window)?",       "minimize",     "Minimised."),
    (r"maximi[sz]e(?: (?:this|the) window)?",       "maximize",     "Maximised."),
    (r"close (?:this |the |current )?window",       "close_window", "Window closed."),
    (r"(?:go )?full ?screen|toggle full ?screen",   "full_screen",  "Full screen."),
    (r"snap (?:this |the )?(?:window )?(?:to the )?left",  "snap_left",  "Snapped left."),
    (r"snap (?:this |the )?(?:window )?(?:to the )?right", "snap_right", "Snapped right."),
    (r"switch window|next window|alt tab",          "switch_window", "Switching window."),

    (r"(?:open (?:a )?)?new tab",                   "new_tab",   "New tab."),
    (r"close (?:this |the |current )?tab",          "close_tab", "Tab closed."),
    (r"next tab",                                   "next_tab",  "Next tab."),
    (r"(?:previous|prev|last) tab",                 "prev_tab",  "Previous tab."),
    (r"go back|navigate back",                      "go_back",    "Back."),
    (r"go forward|navigate forward",                "go_forward", "Forward."),
    (r"(?:refresh|reload)(?: (?:the |this )?page)?", "refresh_page", "Refreshed."),
    (r"find on(?: the)? page|search (?:this|the) page", "find_on_page", "Find on page."),

    (r"zoom in",                                    "zoom_in",    "Zoomed in."),
    (r"zoom out",                                   "zoom_out",   "Zoomed out."),
    (r"(?:reset|restore) (?:the )?zoom|zoom reset",  "zoom_reset", "Zoom reset."),
    (r"scroll up",                                  "scroll_up",   "Scrolled up."),
    (r"scroll down",                                "scroll_down", "Scrolled down."),
    (r"scroll to (?:the )?top|go to (?:the )?top",       "scroll_top",    "Top."),
    (r"scroll to (?:the )?bottom|go to (?:the )?bottom", "scroll_bottom", "Bottom."),
    (r"page up",                                    "page_up",   "Page up."),
    (r"page down",                                  "page_down", "Page down."),

    (r"pause(?: (?:the )?(?:video|music|song|playback))?"
     r"|resume(?: (?:the )?(?:video|music|song|playback))?"
     r"|play (?:the )?(?:video|music|song)",
     "pause_video", "Toggled playback."),

    (r"select all", "select_all", "Selected all."),
    (r"copy(?: (?:this|that|it))?",  "copy",  "Copied."),
    (r"paste(?: (?:this|that|it))?", "paste", "Pasted."),
    (r"cut(?: (?:this|that|it))?",   "cut",   "Cut."),

    (r"(?:turn off|switch off) (?:the )?(?:screen|display|monitor)|"
     r"sleep (?:the )?display|screen off|display off|monitor off",
     "sleep_display", "Turning off the display."),
    (r"open (?:the )?run(?: dialog| box)?|run dialog",
     "open_run", "Opening Run."),
]

_SETTINGS_COMPILED = [(re.compile(p), a, r) for p, a, r in _SETTINGS]

_VOLUME_SET = re.compile(
    r"(?:set |change |put |turn )?(?:the )?volume (?:to |at |on )?(\d{1,3})(?: ?%| ?percent)?"
)
_OPEN_APP = re.compile(r"(?:open|launch|start|fire up)(?: up)? (?:the )?([a-z0-9][a-z0-9 .+_-]{0,40})")
_YOUTUBE  = re.compile(r"(?:play|search|find|put on) (.+?) (?:on|in) (?:youtube|yt)")

# ── computer_control: single keypresses ──────────────────────────────────────
# A fixed, closed vocabulary of key names — never the raw captured text —
# passed straight to pyautogui.press() by computer_settings' press_key action.
# Deliberately excludes keys already covered above (page up/down, arrows used
# as scroll synonyms) so there is exactly one route to each of those.
_PRESS_KEY_MAP = {
    "enter": "enter", "return": "enter",
    "escape": "escape", "esc": "escape",
    "tab": "tab",
    "backspace": "backspace",
    "delete": "delete", "del": "delete",
    "space": "space", "spacebar": "space",
    "home": "home",
    "end": "end",
}
_PRESS_KEY = re.compile(
    r"(?:press|hit|tap)(?: the)? (" +
    "|".join(sorted(_PRESS_KEY_MAP, key=len, reverse=True)) +
    r")(?: key)?"
)

# ── computer_control: media transport keys ───────────────────────────────────
_MEDIA_KEY_MAP = {
    "next song": "nexttrack", "next track": "nexttrack", "skip song": "nexttrack",
    "skip track": "nexttrack", "skip this song": "nexttrack",
    "previous song": "prevtrack", "previous track": "prevtrack",
    "last song": "prevtrack", "go back a song": "prevtrack",
}
_MEDIA_KEY = re.compile(
    "|".join(re.escape(p) for p in sorted(_MEDIA_KEY_MAP, key=len, reverse=True))
)

_CLOSE_CAMERA = re.compile(r"(?:close|stop|turn off) (?:the )?camera")

# ── desktop_control: non-destructive file moves only (never "delete") ────────
_DESKTOP_CLEAN    = re.compile(r"(?:clean|tidy|clear)(?: up)? (?:my |the )?desktop")
_DESKTOP_ORGANIZE = re.compile(r"organi[sz]e (?:my |the )?desktop(?: by (type|date))?")


def detect(text: str) -> Intent | None:
    """Return the tool call this text unambiguously means, or None to let the
    model handle it. None is the safe, expected answer for most input."""
    t = _normalize(text)
    if not t or len(t) > 120:
        return None

    m = _VOLUME_SET.fullmatch(t)
    if m:
        level = max(0, min(100, int(m.group(1))))
        return Intent("computer_settings",
                      {"action": "volume_set", "value": str(level)},
                      f"Volume set to {level}%.")

    m = _YOUTUBE.fullmatch(t)
    if m:
        query = m.group(1).strip()
        if query:
            return Intent("youtube_video", {"action": "play", "query": query},
                          f"Playing '{query}' on YouTube.")

    m = _PRESS_KEY.fullmatch(t)
    if m:
        key = _PRESS_KEY_MAP[m.group(1)]
        return Intent("computer_settings", {"action": "press_key", "value": key},
                      f"Pressed {key}.")

    m = _MEDIA_KEY.fullmatch(t)
    if m:
        key = _MEDIA_KEY_MAP[m.group(0)]
        return Intent("computer_control", {"action": "press", "key": key},
                      "Skipped." if key == "nexttrack" else "Previous track.")

    if _CLOSE_CAMERA.fullmatch(t):
        return Intent("close_camera", {}, "Camera closed.")

    m = _DESKTOP_ORGANIZE.fullmatch(t)
    if m:
        mode = "by_date" if m.group(1) == "date" else "by_type"
        return Intent("desktop_control", {"action": "organize", "mode": mode},
                      "Organizing your desktop.")

    if _DESKTOP_CLEAN.fullmatch(t):
        return Intent("desktop_control", {"action": "clean"}, "Cleaning up your desktop.")

    for pattern, action, reply in _SETTINGS_COMPILED:
        if pattern.fullmatch(t):
            return Intent("computer_settings", {"action": action}, reply)

    # Generic "open <app>" runs last, so every specific rule above wins first
    # ("open a new tab" is a tab action, not an app called "new tab").
    m = _OPEN_APP.fullmatch(t)
    if m:
        name = m.group(1).strip()
        words = name.split()
        if (name not in _NOT_AN_APP and len(words) <= 3
                and words[0] not in _STOPWORDS):
            return Intent("open_app", {"app_name": name}, f"Opening {name}.")

    return None
