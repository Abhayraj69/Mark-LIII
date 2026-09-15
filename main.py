import platform as _platform
import subprocess as _subprocess

# ── Nuclear: force CREATE_NO_WINDOW on EVERY subprocess call on Windows ───────
# This patches Popen itself, so no per-file flag is needed anywhere.
if _platform.system() == "Windows":
    _OrigPopen = _subprocess.Popen

    class _Popen(_OrigPopen):
        def __init__(self, args, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _subprocess.CREATE_NO_WINDOW
            kw.pop("startupinfo", None)   # drop any stale/shared STARTUPINFO
            super().__init__(args, **                       kw)

    _subprocess.Popen = _Popen

# ─────────────────────────────────────────────────────────────────────────────

# ── Console encoding ─────────────────────────────────────────────────────────
# Status lines in this app carry emoji and arrows ("📤 file_controller → Moved:
# a.txt → Documents/"). On a non-UTF-8 console — cp1254 on a Turkish Windows,
# cp1251 on a Russian one, cp932 on a Japanese one — printing one of those
# raises UnicodeEncodeError, and because the print sits after the tool's own
# try/except, the exception escapes into the receive loop and takes the session
# down. The assistant dies on a log line.
#
# Reconfiguring costs nothing and makes the app start the same way in every
# locale. `errors="replace"` is the belt and braces — a console that genuinely
# cannot render a glyph shows a box instead of killing the process.
import sys as _sys

# Under pythonw.exe there is no console at all and sys.stdout/sys.stderr are
# None. print() quietly no-ops on that, but library code that *inspects* the
# stream does not: uvicorn's log formatter calls sys.stdout.isatty(), which
# raises AttributeError, which logging.config turns into "Unable to configure
# formatter 'default'" and the dashboard never starts. Handing those libraries
# a real (discarding) stream keeps every such caller on its normal path.
import io as _io
import os as _os
for _name in ("stdout", "stderr"):
    if getattr(_sys, _name, None) is None:
        setattr(_sys, _name, _io.TextIOWrapper(
            open(_os.devnull, "wb"), encoding="utf-8", errors="replace", write_through=True
        ))

for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import asyncio
import contextlib
import re
import threading
import time
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import sounddevice as sd
import numpy as np
from google import genai
from google.genai import types
from ui import JarvisUI
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
    save_session_summary, pop_last_session,
    peek_recent_sessions, get_session_count,
    search_memory, set_trim_notifier,
)

# The file-backed tools (open_app, web_search, browser_control, …) are no longer
# imported or declared here — they self-describe via a TOOL dict in their own
# actions/*.py file and are auto-discovered by core.action_loader at startup.
# Only tools that are tied to live-session state stay inline in this file
# (screen_process, close_camera, save_memory, manage_monitor, shutdown_jarvis,
# system_status).
from actions.screen_processor  import _capture_camera, _capture_screen
from actions.system_monitor    import SystemMonitor, get_system_status
from actions.proactive         import ProactiveEngine
from actions.background_monitor import (
    add_monitor, remove_monitor, list_monitors, check_all as monitor_check_all,
)
from memory.config_manager     import (
    get_brief_enabled, get_voice, get_wake_word_enabled, save_wake_word_enabled,    get_input_device, get_output_device,
    is_local_engine_enabled, get_plugin_config, get_plugin_setting, save_plugin_config,
)
from core.plugin_loader        import discover_plugins
from core                      import undo as undo_stack
from core                      import confirm as confirm_gate
from core                      import audio_devices
from core.action_loader        import discover_actions
from core.wake_word            import (
    WakeWordDetector, is_ready as wake_is_ready, install_and_download as wake_install,
)
from core.local_control        import LocalControlServer
from core                      import fast_intent
from core                      import predictive_assistant
from core                      import causal_reasoning
from core                      import context_manager
from core                      import sequence_memory
from core                      import sentiment_adapter
from core                      import telemetry
from tool_connectors.registry  import ToolRegistry

# How long the assistant stays awake with no user speech before it auto-sleeps
# again (wake-word mode only).
WAKE_SLEEP_TIMEOUT = 120.0   # seconds (2 minutes)

# _enter_standby() debounce: the mic keeps capturing in real time right through
# the awake->asleep transition, so the wake detector's very first frames are
# the acoustic tail of the "bye jarvis" the user just finished saying — room
# reverb and decay, not silence. Without a settle window, that tail alone can
# score as "Hey Jarvis" (same "...jarvis" sound) and re-wake JARVIS instantly.
WAKE_SETTLE_SECONDS = 1.0
# After a genuine wake, ignore a second shutdown_jarvis call for this long.
# Backstops the same echo/tail problem from the other direction: if a stray
# "jarvis"-ish sound both re-wakes the detector AND reaches the model as
# audio, the model can reflexively re-call shutdown_jarvis a moment later
# with no new, deliberate goodbye from the user. Short enough that a real,
# fast second "bye jarvis" from the user is not a realistic case it blocks.
STANDBY_REENTRY_GUARD_SECONDS = 2.5

def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

BASE_DIR        = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
LIVE_MODEL          = "models/gemini-3.1-flash-live-preview"
CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000 
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024

# RMS below which 16-bit PCM is treated as room silence; above _LEVEL_FULL it
# reads as a full-height waveform. Tuned so ordinary speech lands mid-range and
# the bars still move for a quiet talker — language- and device-independent.
_LEVEL_FLOOR = 60.0
_LEVEL_FULL  = 2600.0


def _pcm_level(samples) -> float:
    """Map a block of int16 PCM samples to a 0.0–1.0 loudness level for the HUD
    waveform. Returns 0.0 on empty/invalid input so it can never raise."""
    try:
        x = np.asarray(samples, dtype=np.float32)
        if x.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(x * x)))
    except Exception:
        return 0.0
    if rms <= _LEVEL_FLOOR:
        return 0.0
    return min(1.0, (rms - _LEVEL_FLOOR) / (_LEVEL_FULL - _LEVEL_FLOOR))


# Both files below are re-read on every Live session (re)connect — including
# transient reconnects (dropped packet, voice change, device switch) that can
# happen several times in one conversation. Neither changes mid-run in the
# common case, so each is cached by mtime: a cache hit is a stat() call
# instead of a full read + (for api_keys.json) a JSON parse, and an on-disk
# edit (e.g. pasting in a new API key) is still picked up on the next read.
_prompt_cache: tuple[float, str] | None = None
_api_config_cache: tuple[float, dict] | None = None


def _load_api_config() -> dict:
    global _api_config_cache
    mtime = API_CONFIG_PATH.stat().st_mtime
    if _api_config_cache is not None and _api_config_cache[0] == mtime:
        return _api_config_cache[1]
    with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    _api_config_cache = (mtime, cfg)
    return cfg


def _get_api_key() -> str:
    return _load_api_config()["gemini_api_key"]


def _load_system_prompt() -> str:
    global _prompt_cache
    try:
        mtime = PROMPT_PATH.stat().st_mtime
        if _prompt_cache is not None and _prompt_cache[0] == mtime:
            return _prompt_cache[1]
        text = PROMPT_PATH.read_text(encoding="utf-8")
        _prompt_cache = (mtime, text)
        return text
    except Exception:
        return (
            "You are JARVIS, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )

_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)

def _clean_transcript(text: str) -> str:    
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()

TOOL_DECLARATIONS = [
    # ── Inline tools ─────────────────────────────────────────────────────────
    # These stay here (rather than in an actions/*.py TOOL dict) because their
    # handling is woven into live-session state — vision capture/injection,
    # camera stream, memory writes, the monitor engine, and shutdown. All other
    # tools live in their own action file and are auto-discovered by
    # core.action_loader (see JarvisLive.__init__).
    {
        "name": "system_status",
        "description": (
            "Returns real-time system metrics: CPU usage, RAM, GPU load, CPU temperature, "
            "uptime, and process count. Use when the user asks about computer performance, "
            "temperature, memory, or resource usage."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "screen_process",
        "description": (
            "Captures the screen or webcam image and lets you analyze it. "
            "MUST be called when user asks what is on screen, what you see, "
            "look at camera, analyze my screen, etc. "
            "You have NO visual ability without this tool. "
            "After the image is captured it is sent directly to you — describe what you see and answer the user's question. "
            "When using camera: the live view stays open until user says close it or calls close_camera."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {"type": "STRING", "description": "'screen' to capture display, 'camera' for webcam. Default: 'screen'"},
                "text":  {"type": "STRING", "description": "The question or instruction about the captured image"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "close_camera",
        "description": (
            "Closes the live camera view shown on screen. "
            "Call when the user says (in ANY language): close camera, stop camera, "
            "turn off camera, that's creepy, etc."
        ),
        "parameters": {"type": "OBJECT", "properties": {}, "required": []}
    },
    {
        "name": "manage_monitor",
        "description": (
            "Add, remove, or list background monitoring topics. "
            "JARVIS checks these topics once a day and alerts the user when there is a new development. "
            "Use 'add' when the user says 'monitor X', 'track X', 'follow X'. "
            "Use 'remove' when the user says 'stop monitoring X'. "
            "Use 'list' when the user asks what is being monitored. "
            "Do NOT add crypto, financial, or trading topics."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type":        "STRING",
                    "description": "add | remove | list",
                },
                "topic": {
                    "type":        "STRING",
                    "description": "Topic to monitor or stop monitoring (e.g. 'space exploration', 'AI news')",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "shutdown_jarvis",
        "description": (
            "Goes quiet: stops speaking and stops treating audio or text as commands, "
            "WITHOUT closing the app. Call this when the user says goodbye, tells you to "
            "stop, or wants to end the conversation — e.g. 'bye Jarvis' — in ANY language. "
            "This does NOT exit or shut down the program. JARVIS keeps running in the "
            "background, muted, with all speech and on-screen activity suspended, and only "
            "resumes normal conversation once the user says 'Hey Jarvis' again."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "save_memory",
        "description": (
            "Save an important personal fact about the user to long-term memory. "
            "Call this silently whenever the user reveals something worth remembering: "
            "name, age, city, job, preferences, hobbies, relationships, projects, or future plans. "
            "Do NOT call for: weather, reminders, searches, or one-time commands. "
            "Do NOT announce that you are saving — just call it silently. "
            "Values must be in English regardless of the conversation language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": (
                        "identity — name, age, birthday, city, job, language, nationality | "
                        "preferences — favorite food/color/music/film/game/sport, hobbies | "
                        "projects — active projects, goals, things being built | "
                        "relationships — friends, family, partner, colleagues | "
                        "wishes — future plans, things to buy, travel dreams | "
                        "notes — habits, schedule, anything else worth remembering"
                    )
                },
                "key":   {"type": "STRING", "description": "Short snake_case key (e.g. name, favorite_food, sister_name)"},
                "value": {"type": "STRING", "description": "Concise value in English (e.g. Fatih, pizza, older sister)"},
            },
            "required": ["category", "key", "value"]
        }
    },
    {
        "name": "recall_memory",
        "description": (
            "Look up a fact you have stored about the user but which is NOT in "
            "the memory block of your system prompt. "
            "The prompt lists the keys it did not have room for under "
            "'[ALSO REMEMBERED]' — if the user asks about anything named there, "
            "call this FIRST. "
            "Also call it before saying you do not know something personal, and "
            "when the user asks what you remember about them (leave query empty "
            "for everything). "
            "This is a local file search: it is instant and costs nothing."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": (
                        "Keyword to search for — a name, a topic, a category "
                        "(e.g. 'ayse', 'coffee', 'projects'). "
                        "Leave empty to list everything stored."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "undo",
        "description": (
            "Reverse the last change YOU made to this computer — a file you "
            "moved, renamed, created or wrote, or a setting you changed such as "
            "volume, brightness, dark mode or WiFi. "
            "Call this whenever the user says undo, revert, take it back, put it "
            "back, cancel that, or tells you that you did the wrong thing, in ANY "
            "language. "
            "Use action='list' when they ask what can be undone. "
            "This only covers your own actions — it is not the Ctrl+Z of whatever "
            "application is on screen (that is computer_settings with action 'undo')."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "undo (default) — reverse the last change | list — show what can be undone",
                },
            },
            "required": [],
        },
    },
]

class _ReconnectSignal(Exception):
    """Raised inside the session TaskGroup to force a clean, voluntary reconnect
    (e.g. the user picked a new voice — the voice is fixed at connect time, so
    the session must be rebuilt).

    Carries `keep_context`: True for an ordinary rebuild, where the stored
    resumption handle is replayed and the conversation continues; False when the
    new session must genuinely start clean (see the voice-change note in
    _on_voice_change)."""

    def __init__(self, keep_context: bool = True):
        super().__init__()
        self.keep_context = keep_context


def _is_reconnect_signal(exc: BaseException) -> bool:
    """True if `exc` is a _ReconnectSignal, or a(n) (Base)ExceptionGroup that
    wraps one — TaskGroup bundles child exceptions into a group."""
    if isinstance(exc, _ReconnectSignal):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_reconnect_signal(sub) for sub in exc.exceptions)
    return False


def _keep_context_of(exc: BaseException) -> bool:
    """Read `keep_context` off a reconnect signal, unwrapping the group the
    TaskGroup put it in. Defaults to True: an unexpected shape must not silently
    wipe the conversation."""
    if isinstance(exc, _ReconnectSignal):
        return getattr(exc, "keep_context", True)
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            if _is_reconnect_signal(sub):
                return _keep_context_of(sub)
    return True


class JarvisLive:
    def __init__(self, ui: JarvisUI):
        self.ui             = ui
        self._asst_name     = "JARVI    S"   # updated each session from config
        self.session              = None
        self.audio_in_queue       = None
        self.out_queue            = None
        self._loop                     = None
        self._is_speaking         = False
        self._speaking_lock       = threading.Lock()
        self._phone_active        = False   # True while phone mic is streaming; pauses PC mic
        self._pending_vision       = None    # (img_bytes, mime_type, question, angle) to inject after tool response
        self._vision_cam_active    = False   # True if camera was opened for vision → auto-close after response
        self._vision_close_pending = False   # True after vision injected; next turn_complete closes camera
        self._vision_last_time     = 0.0     # monotonic time of last screen_process call (cooldown guard)
        self._vision_busy          = False   # True while a vision capture/inject cycle is in flight
        self._interrupted          = False   # True while draining audio after user interrupt
        self._local_stream_cancel  = threading.Event()  # set by interrupt() to stop a Local-Mode stream mid-flight
        self._local_tts_queue      = None    # queue.Queue[str|None] draining sentences to TTS during Local Mode
        self._local_tts_player     = None    # active TTSPlayer during Local Mode, so interrupt() can stop() it
        self._current_turn         = None    # core.telemetry.Turn for the in-flight Gemini Live exchange, if any
        self._active_suggestion    = None    # dict shown via ui.show_suggestion(), mirrored to the phone dashboard
        self.ui.on_text_command   = self._on_text_command
        self.ui.on_remote_clicked = self._make_remote_key
        self.ui.on_remote_tailscale_clicked = self._make_remote_key_tailscale
        self.ui.on_interrupt      = self.interrupt
        self.ui.on_voice_change   = self._on_voice_change     # voice picker → rebuild session
        self.ui.on_audio_device_change = self._on_audio_device_change
        self.ui.on_suggestion_decision = self._on_suggestion_decision
        self._suggestion_cooldown_until = 0.0  # monotonic time — throttles how often a hint can appear
        self._last_suggested_pattern    = None  # pattern_key of the hint currently on screen, or last shown
        self._reconnect_event: asyncio.Event | None = None
        self._reconnect_keep = True   # False → next rebuild drops the resumption handle

        # ── Session resumption ─────────────────────────────────────────
        # The server issues a resumption handle every few seconds and reissues
        # it as the conversation moves on. Before this, session_resumption was
        # switched ON in the config and the update was never read, so the handle
        # was thrown away and EVERY reconnect — a dropped packet, a voice change,
        # switching microphone — started an empty session. "Unlimited sessions"
        # leaked through exactly this hole.
        #
        # Deliberately in RAM only, never written to disk. Persisting it would
        # make a fresh launch continue yesterday's conversation, which sounds
        # appealing but breaks the session-summary flow: _save_session_summary
        # runs at shutdown and the morning briefing pops it the next day. A
        # conversation that never ends never produces a summary, and the
        # "yesterday we talked about…" line silently disappears.
        self._resume_handle: str | None = None
        self._turn_done_event: asyncio.Event | None = None
        self._dashboard     = None
        self._briefing_sent    = False          # morning briefing fires once per process
        self._sys_monitor      = SystemMonitor()  # persistent cooldown state
        self._proactive        = ProactiveEngine()
        self._last_user_speech = time.monotonic()  # updated on every user utterance
        self._session_log: list[str] = []          # conversation turns for end-of-session summary

        self._enhanced_live = True  # proactive audio; auto-disabled if the server rejects it

        # ── Engine mode: Cloud (Gemini Live) vs. Local (offline STT/LLM/TTS) ──
        # Decided once at startup, not re-read mid-run: the two pipelines are
        # structurally different (a streaming multimodal session vs. a
        # blocking record → transcribe → chat → speak loop), so switching
        # requires a restart — see ⚙ → PLUGIN SETTINGS → ENGINE in the UI.
        self._mode = "local" if is_local_engine_enabled() else "cloud"

        _base_dir = Path(__file__).resolve().parent
        _inline_names = {t["name"] for t in TOOL_DECLARATIONS}

        # File-backed tools: every actions/*.py with a TOOL dict, discovered the
        # same way plugins are. Reserved names = the inline tools above, so an
        # action can never shadow one.
        self._action_registry = discover_actions(
            actions_dir=_base_dir / "actions",
            reserved_names=_inline_names,
            logger=lambda msg: print(f"[Actions] {msg}"),
        )

        # Plugins must not collide with either an inline tool or a discovered action.
        _core_names = _inline_names | self._action_registry.names()
        self._plugin_registry = discover_plugins(
            plugins_dir=_base_dir / "plugins",
            core_tool_names=_core_names,
            logger=lambda msg: (print(f"[Plugins] {msg}"), self.ui.write_log(f"SYS: {msg}")),
        )
        self.ui.get_plugins = self._plugin_registry.list_for_ui
        self.ui.get_plugin_settings = self._settings_schemas  # ⚙ settings tab
        self.ui.reload_all_skills = self._reload_all_skills   # Plugin Manager: RELOAD ALL
        self.ui.request_say = self.plugin_say   # plugins: mid-task speech channel

        # tool_connectors/: a second registry (git/Docker/filesystem today),
        # published alongside actions and plugins with the same reserved-name
        # discipline. Declarations are namespaced "{connector}__{action}"
        # (see ToolRegistry.get_tool_declarations), so a collision here would
        # mean an action or plugin was deliberately named e.g. "git__commit" —
        # reserved_names below catches that rather than silently shadowing it.
        _names_incl_plugins = _core_names | {d["name"] for d in self._plugin_registry.get_tool_declarations()}
        self._tool_connector_registry = ToolRegistry(
            logger=lambda msg: (print(f"[Connectors] {msg}"), self.ui.write_log(f"SYS: {msg}")),
        ).discover()
        self._connector_declarations = self._tool_connector_registry.get_tool_declarations(
            reserved_names=_names_incl_plugins
        )

        # ── Skill hot reload (core/skill_watcher.py) ───────────────────────────
        # Off by default, and — like ENGINE mode above — read once at startup:
        # the watcher is a background thread, and toggling it live would need
        # its own start/stop plumbing for one setting nobody changes mid-run.
        # Local Mode already rebuilds its tool list fresh every turn (see
        # _run_local_loop), so only Live mode needs the reconnect nudge below.
        self._skill_watcher = None
        if bool(get_plugin_config("hot_reload").get("enabled", False)):
            from core.skill_watcher import SkillWatcher

            def _on_skills_reloaded(reason: str):
                if self._mode == "cloud":
                    self.request_reconnect(keep_context=True, reason=reason)

            self._skill_watcher = SkillWatcher(
                plugins_dir=_base_dir / "plugins",
                actions_dir=_base_dir / "actions",
                plugin_registry=self._plugin_registry,
                action_registry=self._action_registry,
                on_change=_on_skills_reloaded,
                logger=lambda msg: (print(msg), self.ui.write_log(f"SYS: {msg}")),
            )
            self._skill_watcher.start()

        # ── Wake word ────────────────────────────────────────────────────────
        # _awake gates the mic (see _listen_audio) and the background speakers.
        # It is True whenever wake word is OFF, so default behaviour is unchanged.
        self._wake_enabled     = get_wake_word_enabled()
        self._awake            = not self._wake_enabled
        self._wake_detector: WakeWordDetector | None = None
        self._wake_sleep_timeout = WAKE_SLEEP_TIMEOUT
        # Set only by _enter_standby() ("bye jarvis" on a session that never
        # turned wake-word mode on) so wake() knows to put _wake_enabled back
        # exactly as the user had it once they say "Hey Jarvis" again — the
        # standby is a one-off detour, not a silent, permanent settings change.
        self._standby_forced_wake         = False
        self._standby_restore_wake_enabled = False
        # See WAKE_SETTLE_SECONDS / STANDBY_REENTRY_GUARD_SECONDS above.
        self._wake_feed_gate_open_at  = 0.0
        self._standby_reentry_guard_until = 0.0
        # UI control surface for the Wake Word settings section.
        self.ui.wake_is_ready    = wake_is_ready          # () -> bool
        self.ui.wake_get_state   = self._wake_state       # () -> dict
        self.ui.on_wake_toggle   = self._ui_wake_toggle   # (enable: bool) -> str
        self.ui.on_wake_manual   = self._ui_wake_manual   # () -> toggle awake/asleep
        self.ui.on_wake_install  = self._ui_wake_install  # () -> (ok, msg)

    # ── Wake word: state machine ─────────────────────────────────────────────

    def _wake_state(self) -> dict:
        # A loaded, running detector is definitively ready; otherwise fall back
        # to the cheap on-disk model-file check (no Model construction).
        ready = bool(self._wake_detector and self._wake_detector.ready) or wake_is_ready()
        return {"enabled": self._wake_enabled, "awake": self._awake, "ready": ready}

    def _ensure_wake_detector(self) -> bool:
        """Load the detector once (model loads on first start). Idempotent."""
        if self._wake_detector is None:
            self._wake_detector = WakeWordDetector(
                on_detect=self._on_wake_detected,
                logger=lambda m: (print(f"[Wake] {m}"), self.ui.write_log(f"SYS: {m}")),
            )
        if not self._wake_detector.ready:
            return self._wake_detector.start()
        return True

    def _on_wake_detected(self) -> None:
        """Called from the detector thread when 'Hey Jarvis' is heard."""
        self.wake(reason="wake word")

    def wake(self, reason: str = "wake word") -> None:
        if self._awake:
            return
        self._awake = True
        # See STANDBY_REENTRY_GUARD_SECONDS: a stray echo of the acoustic
        # tail that just caused this wake can otherwise read as a second,
        # genuine "bye jarvis" a moment later and immediately re-sleep.
        self._standby_reentry_guard_until = time.monotonic() + STANDBY_REENTRY_GUARD_SECONDS
        if self._standby_forced_wake:
            # Undo _enter_standby()'s temporary override now that its job is
            # done, so a user who never turned wake-word mode on goes back to
            # always-listening instead of being left in permanent sleep-until-
            # "Hey Jarvis" mode by a single "bye jarvis".
            self._wake_enabled       = self._standby_restore_wake_enabled
            self._standby_forced_wake = False
        self._last_user_speech = time.monotonic()   # start the auto-sleep clock now
        if not self.ui.muted:
            self.ui.set_state("LISTENING")
        self.ui.write_log(f"SYS: Awake — {reason}.")

    def _enter_standby(self, reason: str = "bye jarvis") -> None:
        """'bye jarvis' (or any explicit goodbye) no longer ends the process —
        it mutes. Speech is cut off immediately, JARVIS stops treating mic
        audio or typed text as commands, and it waits silently for "Hey
        Jarvis" to resume — the exact same _awake gate that already drives
        the wake-word auto-sleep timeout (see sleep()/_listen_audio), so
        "muted" means one consistent thing everywhere in the app rather than
        a second, parallel mechanism.

        Works even if the user never turned wake-word mode on: it force-arms
        the detector and flips _wake_enabled on for the duration of standby
        (wake() restores it) so the existing awake/asleep gates, the WAKE NOW
        button, and a live reconnect all treat this session correctly either
        way.

        Guarded against the two ways this loop was observed re-triggering
        itself right after "bye jarvis": (1) STANDBY_REENTRY_GUARD_SECONDS
        ignores a second call arriving immediately after a wake — that's an
        echo, not a new goodbye; (2) WAKE_SETTLE_SECONDS (applied in the mic
        callback / _local_wake_wait, not here) keeps the just-armed detector
        from hearing the tail of THIS utterance as a fresh "Hey Jarvis"."""
        if time.monotonic() < self._standby_reentry_guard_until:
            self.ui.write_log(
                "SYS: Ignoring an immediate repeat 'bye jarvis' — likely an "
                "echo of the last one."
            )
            return
        self.interrupt()   # stop mid-sentence speech now — no trailing audio
        if not self._wake_enabled:
            self._standby_restore_wake_enabled = self._wake_enabled
            self._wake_enabled        = True
            self._standby_forced_wake = True
        detector_ready = self._ensure_wake_detector()
        if detector_ready and self._wake_detector is not None:
            # Without this, "bye jarvis" only behaves correctly the FIRST
            # time per process: the detector's internal audio buffer is left
            # frozen mid-"hey jarvis" from whatever wake last fed it, and a
            # few fresh frames layered on top of that stale buffer can score
            # as an immediate false "Hey Jarvis" the moment feeding resumes
            # (see WakeWordDetector.start()'s docstring for the mechanism).
            self._wake_detector.reset()
        self.sleep(reason=reason)
        self._wake_feed_gate_open_at = time.monotonic() + WAKE_SETTLE_SECONDS
        if not detector_ready:
            self.ui.write_log(
                "SYS: Note — the 'Hey Jarvis' wake model isn't downloaded, so "
                "I won't hear you to wake back up. Use the WAKE NOW button in "
                "the HUD, or set up wake word once in ⚙ → WAKE WORD."
            )

    def sleep(self, reason: str = "timeout") -> None:
        if not self._awake:
            return
        self._awake = False
        self.set_speaking(False)
        self.ui.set_state("SLEEPING")
        self.ui.write_log(f"SYS: Sleeping — {reason}. Say 'Hey Jarvis' to wake me.")

    async def _run_sleep_watch(self) -> None:
        """Auto-sleep after the configured silence window (wake-word mode only)."""
        while True:
            await asyncio.sleep(5)
            if not self._wake_enabled or not self._awake:
                continue
            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking:
                continue
            if (time.monotonic() - self._last_user_speech) > self._wake_sleep_timeout:
                self.sleep(reason="no speech for 2 minutes")

    # ── Wake word: UI callbacks (called from the Qt thread) ──────────────────

    def _ui_wake_toggle(self, enable: bool) -> str:
        """Enable/disable wake word from the settings UI. Returns a status token:
        'enabled' | 'disabled' | 'need_download'."""
        if enable:
            if not wake_is_ready():
                return "need_download"
            self._wake_enabled = True
            save_wake_word_enabled(True)
            self._ensure_wake_detector()
            self.sleep(reason="wake word enabled")
            return "enabled"
        else:
            self._wake_enabled = False
            save_wake_word_enabled(False)
            self.wake(reason="wake word disabled")
            return "disabled"

    def _ui_wake_manual(self) -> None:
        """Manual sleep/wake button in the UI."""
        if not self._wake_enabled:
            return
        if self._awake:
            self.sleep(reason="you tapped sleep")
        else:
            self.wake(reason="you tapped wake")

    def _ui_wake_install(self) -> tuple[bool, str]:
        """Download openwakeword + the model (runs in a UI worker thread)."""
        return wake_install(logger=lambda m: self.ui.write_log(f"SYS: {m}"))

    def plugin_say(self, instruction: str) -> None:
        """
        Thread-safe speech channel for plugins: lets a plugin ask JARVIS to
        say something short WHILE its run() is still executing (plugins block
        their executor thread, so they can't speak through the tool response
        until they finish). The instruction is injected into the Live session
        exactly like a proactive check-in; Gemini phrases it naturally in the
        user's language. Silently a no-op when no session is connected.
        """
        loop = getattr(self, "_loop", None)
        if not loop or not self.session:
            return

        async def _say():
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": instruction}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[PluginSay] {e}")

        try:
            asyncio.run_coroutine_threadsafe(_say(), loop)
        except Exception as e:
            print(f"[PluginSay] {e}")

    # ── Engine mode settings (Cloud vs. Local) ────────────────────────────────
    # Rendered by the existing, fully generic PluginSettingsOverlay (ui.py) —
    # it iterates whatever sections ui.get_plugin_settings() returns and knows
    # nothing about any specific plugin, so prepending a hand-built section
    # here needed no new Qt code. Persisted through the same
    # save_plugin_config("local_engine", …) path a real plugin's settings
    # would use.

    def _settings_schemas(self) -> list[dict]:
        return ([self._engine_settings_section(), self._claude_settings_section(),
                  self._fast_commands_section(), self._sentiment_settings_section(),
                  self._performance_settings_section(), self._routing_settings_section(),
                  self._mcp_settings_section(), self._hot_reload_settings_section()]
                + self._plugin_registry.settings_schemas())

    # ── Skill hot reload (core/skill_watcher.py) ───────────────────────────────
    # Same generic PluginSettingsOverlay rendering as every section above.
    # Like ENGINE mode, the toggle itself needs a restart to take effect (the
    # watcher thread is started once in __init__) — this section exists so
    # the setting is visible and persisted, not to apply it live.
    def _hot_reload_settings_section(self) -> dict:
        return {
            "plugin":    "hot_reload",
            "namespace": "hot_reload",
            "title":     "🔁 HOT RELOAD — reload plugins/actions on file save (restart to apply)",
            "fields": [
                {"key": "enabled", "type": "toggle",
                 "label": "Watch plugins/ and actions/ and reload changed files automatically",
                 "default": False},
            ],
            "values": get_plugin_config("hot_reload"),
            "action": {"label": "RELOAD ALL NOW", "run": self._reload_all_skills},
        }

    def _reload_all_skills(self, values: dict) -> tuple[bool, str]:
        """Manual one-shot reload — works whether or not the background
        watcher is running, since it drives the same registry.reload_all()
        the watcher itself calls per-file."""
        plugin_results = self._plugin_registry.reload_all()
        action_results = self._action_registry.reload_all()
        changed = [f"{name}: {msg}" for name, ok, msg in plugin_results + action_results if ok]
        failed  = [f"{name}: {msg}" for name, ok, msg in plugin_results + action_results if not ok
                   and "no TOOL dict" not in msg]
        if self._mode == "cloud" and changed:
            self.request_reconnect(keep_context=True, reason="skills reloaded")
        lines = [f"Reloaded {len(changed)} file(s)."]
        lines += changed[:10]
        if failed:
            lines.append(f"{len(failed)} failed:")
            lines += failed[:10]
        return (not failed, "\n".join(lines))

    # ── MCP connector (tool_connectors/connectors/mcp_connector.py) ───────────
    # Server list is edited as raw JSON in one text field rather than a
    # repeating add/remove row widget — same PluginSettingsOverlay rendering,
    # no new Qt code. CHECK SERVER HEALTH reuses the connector registry's own
    # health_report(), which already never raises per server (see
    # tool_connectors/mcp_connector_base.py's health_check()).
    def _mcp_settings_section(self) -> dict:
        return {
            "plugin":    "mcp_connector",
            "namespace": "mcp_connector",
            "title":     "🔌 MCP SERVERS — connect Model Context Protocol servers",
            "fields": [
                {"key": "mcp_servers_json", "type": "text", "label": 'Servers (JSON list — see tool_connectors/README.md)',
                 "default": "[]"},
            ],
            "values": {"mcp_servers_json": json.dumps(
                get_plugin_config("mcp_connector").get("mcp_servers", []))},
            "action": {"label": "CHECK SERVER HEALTH", "run": self._check_mcp_health},
        }

    def _check_mcp_health(self, values: dict) -> tuple[bool, str]:
        raw = values.get("mcp_servers_json", "[]")
        try:
            servers = json.loads(raw or "[]")
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        if not isinstance(servers, list):
            return False, "mcp_servers must be a JSON list of {name, transport, command|url} objects."

        # Persist under the key mcp_connector.py's config reader actually
        # expects (this section's own "mcp_servers_json" field is just this
        # UI's edit box), then re-discover so health reflects what was just typed.
        save_plugin_config("mcp_connector", {"mcp_servers": servers})
        registry = ToolRegistry(logger=lambda _msg: None).discover()
        if not registry.connectors():
            return (servers == [], "No MCP servers configured." if servers == [] else
                    "No servers registered — check names/transport in the JSON above.")

        report = registry.health_report()
        lines = [f"{'✓' if ok else '✗'} {name}" for name, ok in sorted(report.items())]
        return all(report.values()), "\n".join(lines)

    # ── Backend router (core/backend_router.py) ───────────────────────────────
    # One comma-separated text field per TaskKind rather than a new dropdown-
    # per-row widget — same PluginSettingsOverlay rendering as every section
    # above, no new Qt code. Read by core.backend_router.load_policy_from_config()
    # whenever a caller (e.g. actions/dev_agent.py, once migrated) asks for a
    # policy built from these saved values instead of DEFAULT_POLICY.
    def _routing_settings_section(self) -> dict:
        from core.backend_router import DEFAULT_POLICY, TaskKind
        saved = get_plugin_config("routing")
        return {
            "plugin":    "routing",
            "namespace": "routing",
            "title":     "🧭 ROUTING — backend order per task kind (comma-separated)",
            "fields": [
                {"key": kind.value, "type": "text", "label": kind.value.replace("_", " ").title(),
                 "default": ", ".join(DEFAULT_POLICY[kind])}
                for kind in TaskKind
            ],
            "values": saved,
            "action": {"label": "TEST ALL BACKENDS", "run": self._test_all_backends},
        }

    def _test_all_backends(self, values: dict) -> tuple[bool, str]:
        """Quick health probe for ollama/claude/gemini — independent of the
        saved routing order, since a backend's reachability doesn't depend on
        which task kinds are configured to use it."""
        import time as _time
        from core import llm_client
        from core.claude_bridge import get_claude_config, get_claude_settings

        results = []
        t0 = _time.monotonic()
        reachable = llm_client.ensure_ollama_running(timeout=3)
        results.append(f"ollama: {'reachable' if reachable else 'unreachable'} "
                        f"({(_time.monotonic()-t0)*1000:.0f}ms)")

        api_key, _, _ = get_claude_settings(get_claude_config())
        results.append(f"claude: {'configured' if api_key else 'no API key'}")

        gemini_key = _load_api_config().get("gemini_api_key", "")
        results.append(f"gemini: {'configured' if gemini_key else 'no API key'}")

        return True, "\n".join(results)

    # ── Per-turn telemetry (core/telemetry.py) ────────────────────────────────
    # Same generic PluginSettingsOverlay rendering as every section above — the
    # "days" field is the only persisted value, and REFRESH STATS reuses the
    # existing TEST CONNECTION mechanism (an (ok, message) tuple rendered into
    # the section's status QLabel) purely as a read-only report surface, so no
    # new Qt widget was needed for this.
    def _performance_settings_section(self) -> dict:
        return {
            "plugin":    "performance",
            "namespace": "performance",
            "title":     "📊 PERFORMANCE — per-turn latency & token telemetry",
            "fields": [
                {"key": "days", "type": "text", "label": "Window (days)", "default": "7"},
            ],
            "values": get_plugin_config("performance"),
            "action": {"label": "REFRESH STATS", "run": self._refresh_performance_stats},
        }

    def _refresh_performance_stats(self, values: dict) -> tuple[bool, str]:
        try:
            days = max(1, int(float(values.get("days") or 7)))
        except (TypeError, ValueError):
            days = 7
        try:
            summary = telemetry.summary(days=days)
        except Exception as e:
            return False, f"Could not read telemetry: {e}"

        lines = [f"Last {days}d:"]
        if not summary["backends"]:
            lines.append("No turns recorded yet.")
        for name, s in sorted(summary["backends"].items()):
            p50 = f"{s['p50_time_to_first_audio_ms']:.0f}ms" if s["p50_time_to_first_audio_ms"] is not None else "n/a"
            p95 = f"{s['p95_time_to_first_audio_ms']:.0f}ms" if s["p95_time_to_first_audio_ms"] is not None else "n/a"
            lines.append(
                f"{name}: {s['turns']} turns · p50 {p50} · p95 {p95} · "
                f"tok in/out {s['tokens_in']}/{s['tokens_out']} · "
                f"fast-intent {s['fast_intent_hit_rate']*100:.0f}% · "
                f"interrupted {s['interrupted_rate']*100:.0f}%"
            )
        if summary["tools"]:
            top = sorted(summary["tools"].items(), key=lambda kv: -kv[1]["avg_ms"])[:5]
            lines.append("Slowest tools: " + ", ".join(f"{n} {v['avg_ms']:.0f}ms" for n, v in top))
        return True, "\n".join(lines)

    # ── Claude collaboration mode (core/claude_bridge.py) ─────────────────────
    # Same generic PluginSettingsOverlay rendering as ENGINE/TONE ADAPTATION
    # above — no new Qt code. Unlike ENGINE, this toggle needs no restart: it's
    # read fresh via is_claude_engine_enabled() each time dev_agent/code_helper
    # pick a model, not cached at session start.
    def _claude_settings_section(self) -> dict:
        return {
            "plugin":    "claude_engine",
            "namespace": "claude_engine",
            "title":     "🤝 CLAUDE COLLAB MODE — Claude backs dev_agent/code_helper",
            "fields": [
                {"key": "enabled", "type": "toggle",
                 "label": "Use Claude instead of Gemini for dev_agent/code_helper text generation",
                 "default": False},
                {"key": "api_key", "type": "password", "label": "Anthropic API key",
                 "default": "", "placeholder": "sk-ant-..."},
                {"key": "model", "type": "text", "label": "Model",
                 "default": "claude-sonnet-5", "placeholder": "claude-sonnet-5"},
                {"key": "max_tokens", "type": "text", "label": "Max reply tokens",
                 "default": "1024"},
            ],
            "values": get_plugin_config("claude_engine"),
            "action": {"label": "TEST CONNECTION", "run": self._test_claude_engine},
        }

    def _test_claude_engine(self, values: dict) -> tuple[bool, str]:
        """Off-thread reachability probe for the settings panel's TEST button —
        checks the key the user just typed, before they even save, with the
        smallest possible real request (max_tokens=1) since Anthropic has no
        unauthenticated health endpoint to ping."""
        import requests
        from core.claude_bridge import ANTHROPIC_VERSION, API_URL

        api_key = str(values.get("api_key") or "").strip()
        model   = str(values.get("model") or "claude-sonnet-5").strip()
        if not api_key:
            return False, "No API key entered."
        try:
            resp = requests.post(
                API_URL,
                json={"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
                headers={
                    "x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                return True, f"Reachable — '{model}' responded."
            detail = ""
            try:
                detail = resp.json().get("error", {}).get("message", "")
            except Exception:
                pass
            return False, f"HTTP {resp.status_code}: {detail}".strip()
        except Exception as e:
            return False, f"Request failed: {e}"

    def _sentiment_settings_section(self) -> dict:
        return {
            "plugin":    "sentiment_adapter",
            "namespace": "sentiment_adapter",
            "title":     "🎭 TONE ADAPTATION — adjust style to your mood",
            "fields": [
                {"key": "enabled", "type": "toggle",
                 "label": "Adjust tone/verbosity to detected mood (never changes facts or safety)",
                 "default": True},
                {"key": "persist_history", "type": "toggle",
                 "label": "Remember detected mood signals across sessions (off = this session only)",
                 "default": False},
            ],
            "values": get_plugin_config("sentiment_adapter"),
        }

    def _fast_commands_section(self) -> dict:
        return {
            "plugin":    "fast_commands",
            "namespace": "fast_commands",
            "title":     "⚡ FAST COMMANDS — skip the model for fixed actions",
            "fields": [
                {"key": "enabled", "type": "toggle",
                 "label": "Run typed device commands locally (instant)",
                 "default": True},
            ],
            "values": get_plugin_config("fast_commands"),
        }

    def _engine_settings_section(self) -> dict:
        return {
            "plugin":    "engine",
            "namespace": "local_engine",
            "title":     "🧠 ENGINE — Local Mode (offline STT/LLM/TTS)",
            "fields": [
                {"key": "enabled", "type": "toggle", "label": "Run fully local (restart required)",
                 "default": False},
                {"key": "llm_provider", "type": "choice", "label": "Backend",
                 "options": ["ollama", "openai"], "default": "ollama"},
                {"key": "llm_url", "type": "text", "label": "Server URL",
                 "default": "http://localhost:11434", "placeholder": "http://localhost:11434"},
                {"key": "llm_model", "type": "text", "label": "Model name",
                 "default": "llama3.2", "placeholder": "llama3.2"},
                {"key": "llm_temperature", "type": "text", "label": "Temperature (0.0-2.0)",
                 "default": "0.7"},
                {"key": "llm_max_tokens", "type": "text", "label": "Max reply tokens",
                 "default": "300"},
            ],
            "values": get_plugin_config("local_engine"),
            "action": {"label": "TEST CONNECTION", "run": self._test_local_engine},
        }

    def _test_local_engine(self, values: dict) -> tuple[bool, str]:
        """Off-thread reachability probe for the settings panel's TEST button —
        checks the values the user just typed, before they even save, so a
        wrong URL/port is caught here rather than by the local pipeline
        failing silently later."""
        import requests
        provider = str(values.get("llm_provider") or "ollama").strip().lower()
        default_url = "http://localhost:1234" if provider == "openai" else "http://localhost:11434"
        url = (str(values.get("llm_url") or "").strip() or default_url).rstrip("/")
        try:
            health = f"{url}/v1/models" if provider == "openai" else f"{url}/api/tags"
            resp = requests.get(health, timeout=5)
            if resp.status_code == 200:
                return True, f"Reachable at {url}"
            return False, f"Server at {url} returned HTTP {resp.status_code}"
        except Exception as e:
            return False, f"Unreachable at {url}: {e}"

    def request_reconnect(self, keep_context: bool = True, reason: str = ""):
        """Thread-safe: ask the run loop to tear down and rebuild the Live
        session. Called from the Qt thread. No-op until the async loop and
        reconnect event exist.

        `keep_context=False` drops the resumption handle so the new session
        starts empty — only for changes the server cannot apply to a resumed
        session."""
        loop = getattr(self, "_loop", None)
        ev   = self._reconnect_event
        self._reconnect_keep   = keep_context
        self._reconnect_reason = reason
        if loop and ev is not None:
            loop.call_soon_threadsafe(ev.set)

    def _on_voice_change(self):
        """Voice picker applied.

        The voice is baked into the session at connect time, so a rebuild is
        required. It is rebuilt WITHOUT the resumption handle on purpose:
        resuming restores the server's own session state, and the safe reading
        is that it restores the voice with it — which would make the picker
        appear to do nothing. Losing context here is acceptable because changing
        voice is a deliberate, rare act; losing it on a dropped packet was not."""
        self.request_reconnect(keep_context=False, reason="new voice")

    def _on_audio_device_change(self):
        """Microphone or speaker changed. Both streams are opened inside the
        session TaskGroup, so they can only be re-opened by rebuilding it —
        but the conversation is kept, which is the whole reason resumption
        landed before this feature did."""
        self.request_reconnect(keep_context=True, reason="audio device")

    async def _watch_reconnect(self):
        """Session-scoped task: when a voluntary reconnect is requested, raise a
        signal that unwinds the TaskGroup so the run loop rebuilds the session."""
        assert self._reconnect_event is not None
        await self._reconnect_event.wait()
        self._reconnect_event.clear()
        keep   = self._reconnect_keep
        reason = getattr(self, "_reconnect_reason", "") or "settings"
        self.ui.write_log(
            f"SYS: Applying {reason} — reconnecting"
            + ("..." if keep else " (starting a fresh conversation)...")
        )
        raise _ReconnectSignal(keep_context=keep)

    def _log_dashboard_unavailable(self):
        """The dashboard can be absent for two very different reasons, and
        telling the user to pip-install packages they already have sends them
        down the wrong path. If it died while starting, show what actually
        killed it."""
        why = getattr(self, "_dashboard_error", "")
        self.ui.write_log(
            f"SYS: Dashboard unavailable — {why}" if why else
            "SYS: Dashboard unavailable. "
            "Run: pip install fastapi \"uvicorn[standard]\" cryptography"
        )

    def _make_remote_key(self):
        """Called from Qt main thread when user presses Remote Control."""
        if self._dashboard is None:
            self._log_dashboard_unavailable()
            return None
        key    = self._dashboard.new_key()
        url    = self._dashboard.get_url()
        manual = self._dashboard.get_manual_url()
        return url, key, f"{url}/auto-login?key={key}", manual

    def _make_remote_key_tailscale(self):
        """Reads this PC's Tailscale address and pairs the phone dashboard
        against it — same login flow (QR/6-digit key) as the LAN Remote
        Control, just reachable from anywhere with internet via the private
        tailnet instead of only this Wi-Fi. Unlike the Cloudflare Tunnel
        this replaced, there's no process to start or wait on here: once
        Tailscale reports an address it's already routable, so this only
        ever shells out to read status. Still called off the Qt thread (see
        ui.py's threaded button handler) since that's still a subprocess call."""
        if self._dashboard is None:
            self._log_dashboard_unavailable()
            return None
        from dashboard.server import PORT
        from dashboard import tailscale as ts

        state = ts.login_state()
        if state == "missing":
            self.ui.write_log(
                "SYS: Tailscale isn't installed. Install it once with:\n"
                f"    {ts.install_hint()}\n"
                "then sign in on this PC (tailscale up) and install/sign in "
                "on your phone too, then click this button again."
            )
            return None
        if state == "needs_login":
            self.ui.write_log(
                "SYS: Tailscale is installed but not signed in. Run 'tailscale up' "
                "and follow the login link, then click this button again."
            )
            return None
        if state == "unknown":
            self.ui.write_log("SYS: Could not read Tailscale status — is the service running?")
            return None

        ip, dns = ts.get_address()
        if not ip:
            self.ui.write_log("SYS: Tailscale has no address for this device yet — try again shortly.")
            return None

        host = dns or ip
        base = f"http://{host}:{PORT}"
        key  = self._dashboard.new_key()
        return base, key, f"{base}/auto-login?key={key}", base, True

    # ── fast local commands ─────────────────────────────────────────────────

    def _try_fast_intent(self, text: str) -> bool:
        """Run `text` locally if it is an unambiguous device command, skipping
        the model round trip entirely. Returns True if it was handled here.

        Thread-safe: called from the Qt thread (typed box) and from the async
        loop (phone). Detection itself is a handful of regex matches, so it is
        cheap enough to attempt on every typed command; a miss returns False
        and the caller takes the normal path unchanged."""
        if not self._loop:
            return False
        if not get_plugin_setting("fast_commands", "enabled", True):
            return False
        intent = fast_intent.detect(text)
        if intent is None:
            return False
        asyncio.run_coroutine_threadsafe(self._run_fast_intent(intent), self._loop)
        return True

    async def _run_fast_intent(self, intent: "fast_intent.Intent") -> None:
        turn = telemetry.start_turn("fast_intent")
        turn.set_fast_intent()
        self.ui.set_state("THINKING")
        with turn.tool_span(intent.tool):
            result = await self._dispatch_tool(intent.tool, dict(intent.args))
        turn.mark("model_done")   # no model call on this path — marks the shortcut's own latency
        turn.finish()
        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        failed = result.startswith(("Tool '", "Unknown tool:", "Action '"))
        self.ui.write_log(f"ERR: {result}" if failed else f"JARVIS: {intent.reply}")

        # Tell the model what just happened so follow-ups ("do that again",
        # "put it back") still make sense. turn_complete=False appends to the
        # conversation WITHOUT asking for a reply — the action has already run
        # and been confirmed on screen, so a spoken answer here would only add
        # back the latency this path exists to remove.
        if self.session:
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text":
                        f"[Executed locally] {intent.tool}({intent.args}) → {result}"}]},
                    turn_complete=False,
                )
            except Exception:
                pass

    def _on_text_command(self, text: str):
        if not self._loop or not self.session:
            return
        # Respect wake-word sleep: a typed command must not be answered while
        # asleep either (the sleep gate is not just for the mic). Wake first with
        # "Hey Jarvis" or the WAKE NOW button.
        if self._wake_enabled and not self._awake:
            self.ui.write_log("SYS: I'm asleep — say 'Hey Jarvis' or tap WAKE NOW first.")
            return
        if self._try_fast_intent(text):
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            self._is_speaking = value
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def _local_control_state(self) -> dict:
        """Polled by the Arc Sentinel widget (core/local_control.py) to keep
        its mic/interrupt buttons in sync with the real session — called from
        the control server's own thread, so only cheap, already-thread-safe
        reads belong here (a lock-guarded bool, a plain property)."""
        with self._speaking_lock:
            speaking = self._is_speaking
        return {
            "muted":    self.ui.muted,
            "speaking": speaking,
            "awake":    self._awake,
        }

    def interrupt(self) -> None:
        """Stop JARVIS mid-speech: drain queued audio and open mic immediately."""
        self._interrupted = True
        q = self.audio_in_queue
        if q:
            drained = 0
            while True:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break
            if drained:
                print(f"[JARVIS] ✋ Interrupted — {drained} audio chunks discarded")
        self.set_speaking(False)
        if self._turn_done_event:
            self._turn_done_event.clear()

        # Local Mode barge-in: tell the streaming consumer thread to stop
        # reading from the LLM, drop any sentences already queued for TTS,
        # and cut audio that's playing right now.
        self._local_stream_cancel.set()
        tts_q = self._local_tts_queue
        if tts_q is not None:
            while True:
                try:
                    tts_q.get_nowait()
                except Exception:
                    break
            try:
                tts_q.put_nowait(None)
            except Exception:
                pass
        if self._local_tts_player is not None:
            self._local_tts_player.stop()

        self.ui.write_log("SYS: Interrupted — listening...")

    def speak(self, text: str):
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def speak_error(self, tool_name: str, error: str):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Sir, {tool_name} encountered an error. {short}")

    def _assemble_system_prompt(self) -> str:
        """Build the full system-instruction text: current time, identity
        (assistant name / how to address the user), the memory block, then
        the static JARVIS protocol from prompt.txt.

        Extracted out of _build_config() so Local Mode's tool-calling loop
        (_run_local_loop) gets the exact same system prompt content the
        Gemini Live path builds, instead of a second, drifting copy of this
        assembly logic. _build_config()'s own output is unchanged by this
        split — it just calls this instead of inlining the same code."""
        from datetime import datetime

        # Load customization from config
        try:
            _cfg = _load_api_config()
            self._asst_name = (_cfg.get("assistant_name") or "JARVIS").strip()
            _user_name = (_cfg.get("user_name") or "").strip()
        except Exception:
            self._asst_name = "JARVIS"
            _user_name = ""

        memory     = load_memory()
        mem_str    = format_memory_for_prompt(memory)
        sys_prompt = _load_system_prompt()

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {time_str}\n"
            f"Use this to calculate exact times for reminders.\n\n"
        )

        # Identity injection — overrides any hardcoded name in prompt.txt
        _addr = (f"ADDRESS: Always call the user '{_user_name}'."
                 if _user_name
                 else "ADDRESS: Address the user with the ordinary respectful form "
                      "for a superior in the language you are currently speaking — "
                      "\"sir\" in English, its everyday equivalent in any other "
                      "language. Never an archaic or aristocratic form, and never "
                      "the form from a different language than the one you are "
                      "speaking in this sentence.")
        identity_ctx = (
            f"[IDENTITY]\n"
            f"Your name is {self._asst_name}. "
            f"Always refer to yourself as {self._asst_name}.\n"
            f"{_addr}\n\n"
        )

        parts = [time_ctx, identity_ctx]
        if mem_str:
            parts.append(mem_str)
        parts.append(sys_prompt)
        return "\n".join(parts)

    def _all_tool_declarations(self) -> list[dict]:
        """The same Gemini-shaped tool list _build_config() feeds to the Live
        API — TOOL_DECLARATIONS + every discovered action + every enabled
        plugin — for Local Mode to convert into OpenAI/Ollama's tool format
        (see core.tool_schema)."""
        return (
            TOOL_DECLARATIONS
            + self._action_registry.get_tool_declarations()
            + self._plugin_registry.get_tool_declarations()
            + self._connector_declarations
        )

    def _build_config(self) -> types.LiveConnectConfig:
        system_instruction = self._assemble_system_prompt()

        cfg = dict(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            system_instruction=system_instruction,
            tools=[{"function_declarations": self._all_tool_declarations()}],
            # Hand back the handle captured from the last session_resumption
            # update. `handle=None` is exactly the old behaviour (ask for
            # handles, start fresh), so the first connect of a run is unchanged.
            session_resumption=types.SessionResumptionConfig(
                handle=self._resume_handle
            ),
            # Sliding-window compression: session never dies from a full context
            # window — JARVIS can stay in one conversation for hours
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=get_voice()
                    )
                )
            ),
        )
        if self._enhanced_live:
            # Proactive audio: JARVIS stays silent when speech isn't addressed
            # to it (background chatter, talking to someone else in the room).
            # (Affective dialog was dropped: gemini-3.1-flash-live does not
            #  support it, and it never reliably detected tone in practice.
            #  To restore it on a 2.5 native-audio model, add back:
            #  cfg["enable_affective_dialog"] = True )
            cfg["proactivity"] = types.ProactivityConfig(proactive_audio=True)
        return types.LiveConnectConfig(**cfg)

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})

        print(f"[JARVIS] 🔧 {name}  {args}")
        self.ui.set_state("THINKING")

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 save_memory: {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True}
            )

        if name == "shutdown_jarvis":
            # `silent: True` (see save_memory above) tells the Live API not to
            # generate a spoken turn for this — a text instruction alone is
            # not a hard enough guarantee for "zero trailing speech".
            self._enter_standby(reason="bye jarvis")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "muted until 'Hey Jarvis'", "silent": True}
            )

        result = await self._dispatch_tool(name, args)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        print(f"[JARVIS] 📤 {name} → {str(result)[:80]}")
        return types.FunctionResponse(
            id=fc.id, name=name,
            response={"result": result}
        )

    async def _dispatch_tool(self, name: str, args: dict) -> str:
        """The actual tool router: everything except the Gemini-specific
        `save_memory` fast path above (which returns a `silent` FunctionResponse
        flag that only means something to the Live API and isn't worth
        threading through a second caller).

        Split out of _execute_tool so Local Mode's text tool-calling loop
        (_run_local_loop) can dispatch against the exact same action/plugin
        registry and inline tools as the Gemini Live path, instead of
        duplicating this router. Returns the plain string result — the two
        callers wrap it differently (a Gemini FunctionResponse here, a
        {"role": "tool", ...} message in local mode)."""
        loop    = asyncio.get_event_loop()
        result  = "Done."
        success = True

        # Vision needs a live multimodal session to inject the captured image
        # into (see _receive_audio's turn_complete handling) — Local Mode's
        # text-only LLM has nothing to send that image to, so pretending to
        # capture it would leave the model describing an image it never saw.
        if name in ("screen_process", "close_camera") and self._mode == "local":
            return ("Vision is not available in Local Mode — it requires the "
                    "Cloud (Gemini Live) engine. Switch engines in Settings "
                    "to use the camera or screen.")

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 save_memory: {category}/{key} = {value}")
            return "ok"

        try:
            if name == "recall_memory":
                # Local file search: no network, no second model. Kept out of
                # the executor deliberately — it is a dictionary scan over a few
                # hundred short strings, and a thread hop would cost more than
                # the work itself.
                result = search_memory(args.get("query", ""), limit=8)

            elif name == "undo":
                if str(args.get("action", "")).lower().strip() == "list":
                    items = undo_stack.history()
                    result = ("Things I can undo, most recent first:\n"
                              + "\n".join(f"{i+1}. {t}" for i, t in enumerate(items))
                              ) if items else "I have not changed anything I can undo yet."
                else:
                    result = await loop.run_in_executor(None, undo_stack.undo_last)

            elif name == "screen_process":
                import time as _t_mod
                _now = _t_mod.monotonic()
                _cooldown = 4.0  # seconds — covers echo window after speaking ends
                if self._vision_busy or (_now - self._vision_last_time) < _cooldown:
                    _wait = max(0, _cooldown - (_now - self._vision_last_time))
                    print(f"[Vision] ⏳ Cooldown active ({_wait:.1f}s remaining) — ignoring duplicate call")
                    result = "Vision is still processing the previous request. I will not call this again."
                else:
                    self._vision_busy      = True
                    self._vision_last_time = _now
                    angle     = args.get("angle", "screen").lower()
                    user_text = args.get("text", "What do you see?")
                    if angle == "camera":
                        img_b, mime_t = await loop.run_in_executor(None, _capture_camera)
                        self.ui.start_camera_stream()
                        self._vision_cam_active = True
                        print(f"[Vision] 📷 Camera: {len(img_b):,} bytes")
                        _stall = "camera"
                    else:
                        img_b, mime_t = await loop.run_in_executor(None, _capture_screen)
                        print(f"[Vision] 🖥️  Screen: {len(img_b):,} bytes")
                        _stall = "screen"
                    self._pending_vision = (img_b, mime_t, user_text, angle)
                    result = (
                        f"[VISION_ACTIVE] {_stall.capitalize()} captured. "
                        f"Immediately say ONE short natural sentence in the user's own language, "
                        f"telling them you are looking at their {_stall} right now. "
                        f"Do NOT describe or guess content — the actual image arrives in the NEXT message."
                    )

            elif name == "close_camera":
                self.ui.stop_camera_stream()
                result = "Camera closed."

            elif name == "system_status":
                r = await loop.run_in_executor(None, get_system_status)
                result = str(r)

            elif name == "manage_monitor":
                action = args.get("action", "").lower().strip()
                topic  = args.get("topic", "").strip()
                if action == "add" and topic:
                    result = await asyncio.to_thread(add_monitor, topic)
                elif action == "remove" and topic:
                    result = await asyncio.to_thread(remove_monitor, topic)
                elif action == "list":
                    topics = await asyncio.to_thread(list_monitors)
                    result = ("Monitoring: " + ", ".join(topics)) if topics else "No topics are being monitored."
                else:
                    result = "Specify action (add/remove/list) and a topic."

            elif self._action_registry.has(name):
                # file_processor: fall back to the currently-uploaded file when none is given
                if name == "file_processor" and not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file

                def _dispatch_sync(step_name: str, step_args: dict) -> str:
                    """Blocking (tool_name, args) -> str re-entry into this same
                    router, for actions that need to run other tools as steps
                    (e.g. sequence replay in actions/sequence_recall.py). Runs on
                    the caller's worker thread; hands the actual coroutine to the
                    event loop and blocks only this thread, not the loop."""
                    future = asyncio.run_coroutine_threadsafe(
                        self._dispatch_tool(step_name, dict(step_args or {})), loop
                    )
                    return future.result(timeout=120)

                _ctx = {"player": self.ui, "speak": self.speak,
                        "response": None, "session_memory": None,
                        "dispatch": _dispatch_sync}
                r = await loop.run_in_executor(None, lambda: self._action_registry.run(name, args, _ctx))
                result = r or "Done."
                # web_search: mirror results to the on-screen content panel
                if (name == "web_search" and r
                        and not r.startswith("No results")
                        and not r.startswith("Search failed")):
                    _mode  = args.get("mode", "search")
                    _query = args.get("query") or ", ".join(args.get("items", []))
                    _label = f"{_mode.upper()} — {_query[:38]}" if _query else _mode.upper()
                    self.ui.show_content(_label, r)

            else:
                _connector_target = self._tool_connector_registry.has_declaration(name)
                if self._plugin_registry.has(name):
                    r = await loop.run_in_executor(
                        None,
                        lambda: self._plugin_registry.run(name, args, player=self.ui, session_memory=None)
                    )
                    result = r or "Done."
                elif _connector_target:
                    # READ_ONLY runs and returns its result inline; REVERSIBLE/
                    # DESTRUCTIVE instead comes back as a "[CONFIRMATION_PENDING]"
                    # sentence for the model to relay — the registry itself
                    # decides which, via core.confirm — exactly like
                    # shutdown_jarvis above, just one layer further down.
                    connector_name, connector_action = _connector_target
                    r = await loop.run_in_executor(
                        None,
                        lambda: self._tool_connector_registry.execute(connector_name, connector_action, args),
                    )
                    result = r or "Done."
                else:
                    result = f"Unknown tool: {name}"

        except Exception as e:
            result  = f"Tool '{name}' failed: {e}"
            success = False
            traceback.print_exc()
            self.speak_error(name, e)

        # Feed the Predictive Assistant. Best-effort: a logging hiccup must
        # never surface as a dispatch failure, so it's swallowed rather than
        # propagated. save_memory/recall_memory return early above and are
        # deliberately not logged here — they aren't user-facing "actions" in
        # the sense the pattern detectors care about.
        try:
            await loop.run_in_executor(
                None,
                lambda: predictive_assistant.log_event(
                    action_type=name,
                    context=self.ui.current_file or "",
                    input_data=str(args)[:500],
                    output=str(result)[:500],
                    success=success,
                ),
            )
        except Exception:
            pass

        # Mirror the same dispatch into the causal-reasoning timeline (see
        # core/causal_reasoning.py) so tool calls and screen_monitor alerts
        # share one cause-and-effect graph instead of two disconnected logs.
        # Best-effort for the same reason as the block above: a reasoning
        # side-channel must never be able to fail a real tool dispatch.
        if success:
            try:
                await loop.run_in_executor(
                    None, lambda: causal_reasoning.record_event(f"tool:{name}")
                )
            except Exception:
                pass

        self._maybe_show_suggestion()

        # Record-mode macros (core/sequence_memory.py): a no-op unless the
        # user is actively recording one (start_recording/manage_sequence
        # with action="record_start"). manage_sequence's own steps are
        # excluded inside record_step itself, so a macro can't record itself.
        # "confirm" is set from the sentinel core/confirm.py's request()
        # returns — see manage_sequence's replay path re-gating on it.
        if success:
            sequence_memory.record_step(
                name, args, confirm=isinstance(result, str) and result.startswith("[CONFIRMATION_PENDING]")
            )

        return result

    def _log_context_turn(self, role: str, content: str) -> None:
        """Fire-and-forget persistence into context_manager's durable turn
        store (separate from self._session_log, which only lives for the
        current session/summary) — used from both the Live audio pipeline
        and Local Mode's text loop so semantic recall works across sessions
        no matter which engine produced the turn. Best-effort: a storage
        hiccup must never stall speech or transcription."""
        async def _do():
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: context_manager.log_turn(role, content)
                )
            except Exception as e:
                print(f"[ContextManager] ⚠️ log_turn failed: {e}")
        asyncio.ensure_future(_do())

    # ── Phone-dashboard parity: confirm / undo / suggestions ──────────────────
    # Gives the phone the same three safety controls the HUD has, over the
    # dashboard's existing /ws channel plus three POST endpoints — all routed
    # through the exact functions the HUD's own buttons call
    # (core.confirm.resolve, core.undo.undo_last, _on_suggestion_decision), so
    # behaviour is identical no matter which surface acted, and a confirmation
    # or suggestion resolved on one surface is invalid on the other because
    # core.confirm/core.undo's pending state is a single shared slot/stack.

    async def _broadcast_remote_state(self) -> None:
        if not self._dashboard:
            return
        try:
            await self._dashboard.broadcast({
                "type":       "state",
                "confirm":    confirm_gate.pending_info(),
                "undo":       undo_stack.history(),
                "suggestion": self._active_suggestion,
            })
        except Exception as e:
            print(f"[Dashboard] state broadcast failed: {e}")

    def _broadcast_remote_state_threadsafe(self) -> None:
        """Safe to call from any thread (a Qt button handler, an executor
        thread running a tool) — hops onto the asyncio loop to actually send."""
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._broadcast_remote_state(), self._loop)

    def _dashboard_confirm(self, confirm_id: str, accepted: bool) -> str | bool:
        """Called off the dashboard's own event loop thread (see
        dashboard/server.py's /api/confirm — it runs this in an executor).
        Rejects a stale/already-resolved id instead of blindly resolving,
        since core.confirm's pending slot is single-use across both surfaces."""
        current = confirm_gate.pending_info()
        if current is None or current["key"] != confirm_id:
            return False
        confirm_gate.resolve(accepted)
        return True

    def _dashboard_undo(self) -> str:
        result = undo_stack.undo_last()
        self._broadcast_remote_state_threadsafe()
        return result

    def _dashboard_suggestion(self, accepted: bool) -> None:
        if self._active_suggestion is None:
            return
        self._on_suggestion_decision(accepted, self._active_suggestion)

    def _maybe_show_suggestion(self) -> None:
        """Throttled check for a proactive hint worth surfacing. Runs the
        (cheap, frequency-based) pattern detectors off the Qt/asyncio thread
        and, if something above threshold turns up and isn't the same pattern
        already on screen, raises it as a dismissible SuggestionHint — never
        auto-executed, see _on_suggestion_decision below."""
        now = time.monotonic()
        if now < self._suggestion_cooldown_until:
            return
        # However this resolves, don't check again for a while — a hit keeps
        # the hint from being spammed, a miss avoids re-running the detectors
        # on every single tool call.
        self._suggestion_cooldown_until = now + 120.0

        def _check():
            try:
                return predictive_assistant.get_proactive_suggestions(
                    current_context=self.ui.current_file or ""
                )
            except Exception as e:
                print(f"[PredictiveAssistant] ⚠️ suggestion check failed: {e}")
                return []

        async def _run():
            suggestions = await asyncio.get_event_loop().run_in_executor(None, _check)
            if not suggestions:
                return
            top = suggestions[0]
            if top.pattern_key == self._last_suggested_pattern:
                return  # already shown (and presumably dismissed/ignored) recently
            self._last_suggested_pattern = top.pattern_key
            suggestion = {
                "action": top.action,
                "confidence_score": top.confidence_score,
                "reasoning": top.reasoning,
                "one_click_command": top.one_click_command,
                "pattern_key": top.pattern_key,
            }
            self.ui.show_suggestion(suggestion)
            self._active_suggestion = suggestion
            await self._broadcast_remote_state()

        asyncio.ensure_future(_run())

    def _on_suggestion_decision(self, accepted: bool, suggestion: dict) -> None:
        """UI callback for the SuggestionHint's RUN/DISMISS buttons. RUN is the
        explicit human confirmation the spec requires — nothing here ever
        fires without it. Runs the underlying tool through the exact same
        registry _dispatch_tool already uses, so an accepted suggestion behaves
        identically to the user asking for it out loud."""
        pattern_key = suggestion.get("pattern_key", "")
        action      = suggestion.get("action", "")

        if self._active_suggestion is suggestion or (
            self._active_suggestion and self._active_suggestion.get("pattern_key") == pattern_key
        ):
            self._active_suggestion = None
        self._broadcast_remote_state_threadsafe()

        async def _record():
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: predictive_assistant.record_suggestion_feedback(
                    pattern_key, action, accepted,
                    context=self.ui.current_file or "",
                ),
            )
        asyncio.ensure_future(_record())

        if not accepted:
            return

        if self._action_registry.has(action):
            asyncio.ensure_future(self._dispatch_tool(action, {}))
        elif pattern_key.startswith("manual:"):
            # A repeated-manual-steps suggestion names a sequence of steps
            # ("step1 then step2"), not a single registered tool. Reconstruct
            # the real {tool, args} calls behind it from the workflow log and
            # save them as a manage_sequence macro, so accepting the hint
            # gives the user a real one-click replay instead of dead-ending.
            async def _save_as_macro():
                steps = await asyncio.get_event_loop().run_in_executor(
                    None, predictive_assistant.get_manual_sequence_steps, pattern_key,
                )
                if not steps:
                    self.ui.write_log(
                        f"SYS: Couldn't find the steps behind '{action}' to save as a macro."
                    )
                    return
                name = suggestion.get("one_click_command") or "auto_sequence"
                result = await self._dispatch_tool("manage_sequence", {
                    "action": "save",
                    "name": name,
                    "steps": steps,
                    "description": suggestion.get("reasoning", ""),
                })
                self.ui.write_log(f"SYS: {result}")

            asyncio.ensure_future(_save_as_macro())
        else:
            # Neither a registered tool nor a reconstructable manual-step
            # sequence — nothing to run yet, so say so instead of silently
            # doing nothing.
            self.ui.write_log(f"SYS: '{action}' isn't wired to a runnable action yet.")

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            # Gemini 3.x Live rejects the old realtime_input.media_chunks field
            # (what `media=...` maps to) and closes the socket with a 1007. Send
            # mic / phone PCM through the new `audio` field instead. Queue items
            # are {"data": <bytes>, "mime_type": <str>} from _listen_audio and
            # the phone relay.
            await self.session.send_realtime_input(
                audio=types.Blob(
                    data=msg["data"],
                    mime_type=msg.get("mime_type", "audio/pcm"),
                )
            )

    async def _listen_audio(self):
        print("[JARVIS] 🎤 Mic started")
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time_info, status):
            # ── Wake-word gate ───────────────────────────────────────────────
            # While asleep, the mic audio NEVER goes to Gemini (nothing is
            # streamed, so JARVIS can't respond to speech not addressed to it and
            # nothing leaves the machine). Frames are instead handed to the local
            # detector, which runs its model in ITS OWN thread — the cost here is
            # only a queue push, so the audio path is never slowed. When wake word
            # is off (default) or we're awake, this is a single boolean check.
            if self._wake_enabled and not self._awake:
                det = self._wake_detector
                # WAKE_SETTLE_SECONDS: drop, don't feed, for a moment after
                # going to sleep — otherwise the detector's first frames are
                # the acoustic tail of the utterance that just put it to
                # sleep ("...jarvis" fading out), which can score as a fresh
                # "Hey Jarvis" and wake it right back up.
                if det is not None and time.monotonic() >= self._wake_feed_gate_open_at:
                    det.feed(indata)
                return
            with self._speaking_lock:
                jarvis_speaking = self._is_speaking
            if not jarvis_speaking and not self.ui.muted and not self._phone_active:
                data = indata.tobytes()
                loop.call_soon_threadsafe(
                    self.out_queue.put_nowait,
                    {"data": data, "mime_type": "audio/pcm"}
                )
                # Feed the live mic level to the HUD so the waveform reacts to
                # the user's actual voice while listening. Purely cosmetic — any
                # failure here must never disturb the mic.
                try:
                    self.ui.set_audio_level(_pcm_level(indata))
                except Exception:
                    pass

        try:
            def _open_mic(dev):
                return sd.InputStream(
                    samplerate=SEND_SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=CHUNK_SIZE,
                    device=dev,
                    callback=callback,
                )

            # Which microphone. resolve() returns None for "system default" and
            # for a saved device that is no longer present — so a headset
            # unplugged since the last run falls back to the built-in mic
            # instead of raising on startup and taking the session with it.
            _mic_name = get_input_device()
            _mic_dev  = audio_devices.resolve(_mic_name, "input")
            if _mic_dev is not None:
                print(f"[JARVIS] 🎤 Input device: {_mic_name}")
            try:
                _mic_stream = _open_mic(_mic_dev)
            except Exception as _e:
                # A device the picker listed but the driver will not open right
                # now — exclusive mode, a webcam already in use, a virtual mic
                # whose source went away. Chosen hardware failing must never
                # mean the assistant cannot hear at all.
                if _mic_dev is None:
                    raise
                print(f"[JARVIS] ⚠️  Mic '{_mic_name}' failed: {_e} — using default")
                self.ui.write_log(
                    f"SYS: Microphone '{_mic_name}' unavailable — using system default."
                )
                _mic_stream = _open_mic(None)

            with _mic_stream:
                print("[JARVIS] 🎤 Mic stream open")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[JARVIS] ❌ Mic: {e}")
            raise

    async def _receive_audio(self):
        print("[JARVIS] 👂 Recv started")
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():

                    # ── Session resumption ───────────────────────────────────
                    # The server sends this periodically. `resumable` goes false
                    # while a turn is mid-flight — replaying a handle from that
                    # moment is what the flag exists to prevent — so only
                    # resumable handles are kept. This is three lines and it is
                    # the entire fix for "every reconnect forgets everything".
                    _sru = getattr(response, "session_resumption_update", None)
                    if _sru is not None:
                        if getattr(_sru, "resumable", False) and getattr(_sru, "new_handle", None):
                            if self._resume_handle is None:
                                print("[JARVIS] 🔗 Session resumption armed")
                            self._resume_handle = _sru.new_handle

                    if response.data:
                        if self._interrupted:
                            pass  # discard: interrupted
                        else:
                            if self._turn_done_event and self._turn_done_event.is_set():
                                self._turn_done_event.clear()
                            if self._current_turn:
                                self._current_turn.mark_once("first_audio")
                            # Split into ~50 ms chunks so interrupt() stops audio within 50 ms
                            # (24000 Hz × 2 bytes/sample × 0.05 s = 2400 bytes per slice)
                            _audio_data = response.data
                            _SLICE = 2400
                            for _i in range(0, len(_audio_data), _SLICE):
                                self.audio_in_queue.put_nowait(_audio_data[_i : _i + _SLICE])

                    if response.server_content:
                        sc = response.server_content

                        if sc.output_transcription and sc.output_transcription.text:
                            txt = _clean_transcript(sc.output_transcription.text)
                            if txt and txt != (out_buf[-1] if out_buf else ""):
                                out_buf.append(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt:
                                if not in_buf and self._current_turn is None:
                                    self._current_turn = telemetry.start_turn("gemini")
                                in_buf.append(txt)
                                self._last_user_speech = time.monotonic()

                        if sc.turn_complete:
                            if self._turn_done_event:
                                self._turn_done_event.set()

                            if self._current_turn:
                                self._current_turn.mark("model_done")
                                usage = getattr(response, "usage_metadata", None)
                                if usage is not None:
                                    self._current_turn.tokens(
                                        tokens_in=getattr(usage, "prompt_token_count", None),
                                        tokens_out=getattr(usage, "response_token_count", None),
                                    )

                            # If this turn_complete ends an interrupted response, clear the
                            # flag and skip all further processing for that turn.
                            if self._interrupted:
                                self._interrupted = False
                                in_buf  = []
                                out_buf = []
                                if self._current_turn:
                                    self._current_turn.set_interrupted()
                                    self._current_turn.finish()
                                    self._current_turn = None
                                continue

                            full_in = " ".join(in_buf).strip()
                            if full_in:
                                self.ui.write_log(f"You: {full_in}")
                                self._session_log.append(f"User: {full_in}")
                                self._log_context_turn("user", full_in)
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "user",
                                        "text": full_in,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            in_buf = []

                            full_out = " ".join(out_buf).strip()
                            if full_out:
                                self.ui.write_log(f"{self._asst_name}: {full_out}")
                                self._session_log.append(f"{self._asst_name}: {full_out}")
                                self._log_context_turn("assistant", full_out)
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "jarvis",
                                        "text": full_out,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            out_buf = []

                            if self._current_turn:
                                self._current_turn.finish()
                                self._current_turn = None

                            # Vision injection: model finished tool-response turn → now send the image
                            if self._pending_vision and self.session:
                                import base64 as _b64
                                img_b, mime_t, question, angle = self._pending_vision
                                self._pending_vision = None
                                b64 = _b64.b64encode(img_b).decode("ascii")
                                print(f"[Vision] 📤 {len(img_b):,} bytes (angle={angle}) → main session")
                                await self.session.send_client_content(
                                    turns={"role": "user", "parts": [
                                        {"inline_data": {"mime_type": mime_t, "data": b64}},
                                        {"text": question},
                                    ]},
                                    turn_complete=True,
                                )
                                # Mark next turn_complete behaviour depending on angle
                                if self._vision_cam_active:
                                    # Camera: keep busy until JARVIS finishes speaking the answer
                                    self._vision_cam_active    = False
                                    self._vision_close_pending = True
                                else:
                                    # Screen-only: no camera to close; release busy flag now
                                    self._vision_busy = False
                            elif self._vision_close_pending:
                                # This turn_complete IS the vision answer — close camera + release busy flag
                                self._vision_close_pending = False
                                self._vision_busy = False
                                async def _cam_close():
                                    await asyncio.sleep(2.0)
                                    self.ui.stop_camera_stream()
                                asyncio.create_task(_cam_close())

                    if response.tool_call:
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            print(f"[JARVIS] 📞 {fc.name}")
                            span = (self._current_turn.tool_span(fc.name)
                                    if self._current_turn else contextlib.nullcontext())
                            with span:
                                fr = await self._execute_tool(fc)
                            fn_responses.append(fr)
                        await self.session.send_tool_response(
                            function_responses=fn_responses
                        )
        except Exception as e:
            print(f"[JARVIS] ❌ Recv: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        print("[JARVIS] 🔊 Play started")

        _spk_name = get_output_device()
        _spk_dev  = audio_devices.resolve(_spk_name, "output")
        if _spk_dev is not None:
            print(f"[JARVIS] 🔊 Output device: {_spk_name}")

        def _open_spk(dev):
            st = sd.RawOutputStream(
                samplerate=RECEIVE_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                device=dev,
            )
            st.start()
            return st

        try:
            stream = _open_spk(_spk_dev)
        except Exception as _e:
            # A chosen output that the host API accepts by name but refuses to
            # open (exclusive mode, wrong sample rate, device asleep) must not
            # cost the user their voice. Fall back to the default and say so.
            if _spk_dev is None:
                raise
            print(f"[JARVIS] ⚠️  Output device '{_spk_name}' failed: {_e} — using default")
            self.ui.write_log(f"SYS: Speaker '{_spk_name}' unavailable — using system default.")
            stream = _open_spk(None)

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self.audio_in_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    if (
                        self._turn_done_event
                        and self._turn_done_event.is_set()
                        and self.audio_in_queue.empty()
                    ):
                        self.set_speaking(False)
                        self._turn_done_event.clear()
                    continue

                self.set_speaking(True)

                # Batch all immediately-available chunks into one write to reduce
                # thread-pool round-trips (was one asyncio.to_thread per 50ms slice).
                # Cap at ~200 ms so interrupt() still stops audio within ~200 ms.
                batch = bytearray(chunk)
                while len(batch) < 9600:   # 9600 bytes ≈ 200 ms at 24 kHz / 16-bit mono
                    try:
                        batch.extend(self.audio_in_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                # Drive the HUD waveform from JARVIS's own voice while speaking.
                try:
                    self.ui.set_audio_level(_pcm_level(
                        np.frombuffer(bytes(batch), dtype=np.int16)))
                except Exception:
                    pass

                try:
                    await asyncio.to_thread(stream.write, bytes(batch))
                except (RuntimeError, asyncio.CancelledError):
                    break   # executor shutting down — exit cleanly
        except Exception as e:
            print(f"[JARVIS] ❌ Play: {e}")
            raise
        finally:
            self.set_speaking(False)
            stream.stop()
            stream.close()

    # ── Morning briefing ────────────────────────────────────────────────────────

    async def _send_startup_briefing(self) -> None:
        """
        Two-phase briefing optimized for speed:
          Phase 1 — instant greeting (no tools) → speech starts in <1s
          Phase 2 — the consolidated morning brief (build_morning_brief: monitored-
                    topic news, habit suggestions, recent-session context, and
                    causal patterns) is assembled in a background thread while
                    Phase 1 plays, then delivered as ready text (no Gemini tool-call
                    round-trip) and shown on the UI content panel. Waits for
                    turn_complete instead of a fixed sleep so there is no
                    unnecessary gap.
        """
        memory   = load_memory()
        identity = memory.get("identity", {})

        def _val(k: str) -> str:
            e = identity.get(k, {})
            return (e.get("value", "") if isinstance(e, dict) else str(e)).strip()

        lang = _val("language")
        name = _val("name")
        time_str = datetime.now().strftime("%H:%M")

        # Start assembling the morning brief immediately — runs in a background
        # thread (news lookups + DB queries) in parallel while phase 1 plays
        loop  = asyncio.get_event_loop()
        depth = await asyncio.to_thread(get_session_count)
        brief_future = loop.run_in_executor(None, self._proactive.get_morning_brief, memory, depth)

        await asyncio.sleep(0.3)
        if not self.session:
            return

        # ── Phase 1: instant greeting ─────────────────────────────────────────
        # The briefing fires before the user has said anything, so the
        # remembered language is the only signal there is. It is a starting
        # point, not a setting: the moment they reply, their language wins.
        lang_clause = (f" Speak this greeting in {lang}, then follow the "
                       f"user's own language from their first reply onward."
                       if lang else "")
        name_clause = f" Address the user as {name}." if name else ""

        # Inject last session context if available — pop removes it so it's never repeated
        last = await asyncio.to_thread(pop_last_session)
        session_clause = ""
        if last:
            try:
                _delta = (datetime.now() - datetime.strptime(last["date"], "%Y-%m-%d")).days
                _when  = "earlier today" if _delta == 0 else ("yesterday" if _delta == 1 else f"{_delta} days ago")
            except Exception:
                _when = "last time"
            session_clause = (
                f" Also briefly and naturally mention that {_when}: {last['summary']}"
            )

        p1 = (
            f"Greet the user warmly, mention it is {time_str}, and say you are fetching today's news now.{session_clause} "
            f"Keep it to 2 short sentences max. Do not call any tools.{lang_clause}{name_clause}"
        )

        # Clear the turn-done event so we can wait for Phase 1 to finish
        if self._turn_done_event:
            self._turn_done_event.clear()

        await self.session.send_client_content(
            turns={"role": "user", "parts": [{"text": p1}]},
            turn_complete=True,
        )
        self.ui.write_log("SYS: Briefing phase 1 (greeting) sent.")

        # ── Phase 2: fire as soon as Phase 1 audio is done ───────────────────
        async def _deliver_brief():
            try:
                lang_str = (f" Speak in {lang} unless the user has since "
                            f"spoken another language, in which case use theirs."
                            if lang else "")

                # Wait for the brief to finish assembling (already running) and
                # Phase 1 turn-complete in parallel — whichever takes longer
                # determines the wait time
                brief_done  = asyncio.wrap_future(brief_future)
                turn_waited = False
                if self._turn_done_event:
                    try:
                        await asyncio.wait_for(self._turn_done_event.wait(), timeout=6.0)
                        turn_waited = True
                    except asyncio.TimeoutError:
                        pass

                # Extra buffer: turn_complete fires when Gemini finishes *generating*
                # Phase 1, but audio may still be playing.  Waiting a beat here
                # prevents Phase 2 audio from arriving while Phase 1 is mid-sentence
                # (which sounds like a "repeated first response" to the user).
                if turn_waited:
                    await asyncio.sleep(0.8)
                else:
                    await asyncio.sleep(1.0)

                try:
                    brief_text = await asyncio.wait_for(brief_done, timeout=8.0)
                except Exception:
                    brief_text = ""

                if not self.session:
                    return

                if brief_text:
                    # Show on UI content panel immediately
                    self.ui.show_content("MORNING BRIEF", brief_text)
                    p2 = f"{brief_text}{lang_str}"
                else:
                    p2 = (
                        "Nothing new to report right now — no news, patterns, or "
                        f"context worth mentioning. Let the user know briefly.{lang_str}"
                    )

                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": p2}]},
                    turn_complete=True,
                )
                self.ui.write_log("SYS: Briefing phase 2 (morning brief) sent.")
            except Exception as e:
                print(f"[Briefing] Phase 2 error: {e}")
                self.ui.write_log(f"SYS: Briefing phase 2 failed: {e}")

        asyncio.create_task(_deliver_brief())

    # ── Session memory ──────────────────────────────────────────────────────────

    async def _save_session_summary(self) -> None:
        """Summarise the current session in 1-2 sentences and save to long_term.json."""
        log = self._session_log
        if len(log) < 3:          # need at least one exchange to be worth saving
            return
        if self._mode == "local":
            # This calls Gemini's API to write the summary — sending the local,
            # offline conversation to the cloud is exactly the privacy
            # regression Local Mode exists to avoid, so it's skipped rather
            # than silently done anyway. The morning-briefing callback this
            # feeds just won't have anything to reference after a local
            # session; the conversation itself isn't lost, only its summary.
            self._session_log = []
            self.ui.write_log("SYS: Session summary skipped — Local Mode makes no cloud calls.")
            return
        self._session_log = []    # reset immediately so the next session starts clean

        memory = load_memory()
        lang_entry = memory.get("identity", {}).get("language", {})
        lang = (lang_entry.get("value", "") if isinstance(lang_entry, dict) else str(lang_entry)).strip()
        lang = lang or "English"

        convo = "\n".join(log[-40:])   # cap at last 40 turns to stay within token budget
        prompt = (
            f"Summarize this conversation in 1-2 sentences in {lang}. "
            "Focus on what the user accomplished or discussed. "
            "Output ONLY the summary text, nothing else:\n\n" + convo
        )
        try:
            from google import genai as _genai
            client = _genai.Client(api_key=_get_api_key())
            resp   = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-flash-latest",
                contents=prompt,
            )
            summary = (resp.text or "").strip()
            if summary:
                save_session_summary(summary, lang)
        except Exception as e:
            print(f"[Memory] ⚠️ Session summary failed: {e}")

    # ── System monitor ──────────────────────────────────────────────────────────

    async def _run_system_monitor(self) -> None:
        """Background task: voice alerts when metrics exceed thresholds."""
        while True:
            await asyncio.sleep(10)
            alert = await asyncio.to_thread(self._sys_monitor.check)
            if not alert or not self.session or not self._awake:
                continue
            # Don't interrupt an active conversation
            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking or (time.monotonic() - self._last_user_speech) < 10:
                continue
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": alert}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[Monitor] ⚠️ Could not send alert: {e}")

    # ── Background monitor ──────────────────────────────────────────────────────

    async def _run_background_monitor(self) -> None:
        """Check user-configured topics once per day; speak alerts when new headlines appear."""
        await asyncio.sleep(300)          # wait 5 min after startup before first check
        while True:
            if self.session and self._awake:
                # Don't interrupt if user spoke recently or JARVIS is mid-sentence
                with self._speaking_lock:
                    speaking = self._is_speaking
                recent_speech = (time.monotonic() - self._last_user_speech) < 30
                if not speaking and not recent_speech:
                    try:
                        alerts = await asyncio.to_thread(monitor_check_all)
                        memory = load_memory()
                        lang_e = memory.get("identity", {}).get("language", {})
                        lang   = (lang_e.get("value", "") if isinstance(lang_e, dict) else str(lang_e)).strip() or "English"
                        for alert in alerts:
                            msg = (
                                f"{alert}\n\n"
                                f"Inform the user about this development naturally in {lang}. "
                                "One brief sentence only."
                            )
                            await self.session.send_client_content(
                                turns={"role": "user", "parts": [{"text": msg}]},
                                turn_complete=True,
                            )
                            self.ui.write_log(f"SYS: Monitor alert sent.")
                            await asyncio.sleep(6)   # gap between consecutive alerts
                    except Exception as e:
                        print(f"[Monitor] ⚠️ Background check error: {e}")
            await asyncio.sleep(1800)     # check every 30 minutes

    # ── Proactive mode ──────────────────────────────────────────────────────────

    async def _run_proactive_mode(self) -> None:
        """
        Background task: periodically checks if the user has been silent long enough,
        then hands time + memory context to Gemini so it can decide what (if anything)
        to say proactively. No hardcoded rules — Gemini makes the call.
        """
        while True:
            await asyncio.sleep(60)   # evaluate once per minute

            if not self.session or not self._awake:
                continue

            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking:
                continue

            if not self._proactive.should_trigger(self._last_user_speech):
                continue

            self._proactive.mark_triggered()

            try:
                memory        = await asyncio.to_thread(load_memory)
                monitors      = await asyncio.to_thread(list_monitors)
                recent_turns  = self._session_log[-8:] if self._session_log else []
                past_sessions = await asyncio.to_thread(peek_recent_sessions, 2)
                depth         = await asyncio.to_thread(get_session_count)
                prompt = self._proactive.build_prompt(
                    memory             = memory,
                    monitors           = monitors or None,
                    recent_turns       = recent_turns or None,
                    past_sessions      = past_sessions or None,
                    relationship_depth = depth,
                )
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": prompt}]},
                    turn_complete=True,
                )
                self.ui.write_log("SYS: Proactive check-in.")
            except Exception as e:
                print(f"[Proactive] ⚠️ {e}")

    # ── Phone audio relay ────────────────────────────────────────────────────────

    async def _relay_phone_audio(self) -> None:
        """Forward phone mic PCM chunks from dashboard queue into the Gemini Live session."""
        q = self._dashboard._phone_audio_queue
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No audio for 1 s → phone mic inactive, give PC mic back
                self._phone_active = False
                continue
            self._phone_active = True   # phone is streaming — silence PC mic
            with self._speaking_lock:
                speaking = self._is_speaking
            if not speaking and not self.ui.muted:
                try:
                    self.out_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass

    def _on_phone_connected(self) -> None:
        self.ui.write_log("SYS: Phone connected via Remote Dashboard.")
        self.ui.notify_phone_connected()

    # ── dashboard command relay ─────────────────────────────────────────────

    async def _process_dashboard_commands(self) -> None:
        while True:
            try:
                text = await asyncio.wait_for(
                    self._dashboard._command_queue.get(), timeout=0.5
                )
                if not text:
                    continue
                # Wait up to 8s for session to become ready after a wake
                for _ in range(80):
                    if self.session:
                        break
                    await asyncio.sleep(0.1)
                if self.session:
                    # A remote command is deliberate control and the phone user
                    # has no desktop WAKE button — so it wakes JARVIS if asleep.
                    if self._wake_enabled and not self._awake:
                        self.wake(reason="remote command")
                    self.ui.write_log(f"[Web]: {text}")
                    if self._try_fast_intent(text):
                        continue
                    await self.session.send_client_content(
                        turns={"role": "user", "parts": [{"text": text}]},
                        turn_complete=True,
                    )
                else:
                    print(f"[Dashboard] Dropped command (no session): {text}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"[Dashboard] Command error: {e}")
                await asyncio.sleep(0.5)

    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self):
        self._loop = asyncio.get_event_loop()
        self._reconnect_event = asyncio.Event()

        # ── Wire the shared core services to the interface ───────────────────
        # The confirmation gate is useless without a way to ask, and a memory
        # trim is invisible without a way to say so. Both are bound once here
        # rather than passed down through every action signature.
        def _show_confirm_and_broadcast(title, detail):
            self.ui.show_confirm(title, detail)
            self._broadcast_remote_state_threadsafe()

        def _hide_confirm_and_broadcast():
            self.ui.hide_confirm()
            self._broadcast_remote_state_threadsafe()

        confirm_gate.bind(
            show = _show_confirm_and_broadcast,
            hide = _hide_confirm_and_broadcast,
            log  = self.ui.write_log,
        )
        undo_stack.bind(self._broadcast_remote_state_threadsafe)
        set_trim_notifier(self.ui.write_log)

        # Tell the device picker the exact rates the streams open at, from the
        # constants that actually open them — so it can never list a device that
        # cannot be opened at them.
        audio_devices.configure(SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE)

        # Enumerate audio devices off-thread. The settings drawer must never pay
        # for host-API enumeration on the Qt thread.
        audio_devices.prefetch()

        # Start dashboard (optional — needs: pip install fastapi "uvicorn[standard]" cryptography)
        try:
            from dashboard.server import DashboardServer
            self._dashboard = DashboardServer()
            self._dashboard.set_connect_callback(self._on_phone_connected)
            self._dashboard.set_confirm_callback(self._dashboard_confirm)
            self._dashboard.set_undo_callback(self._dashboard_undo)
            self._dashboard.set_suggestion_callback(self._dashboard_suggestion)

            async def _run_dashboard():
                # asyncio.create_task() below only SCHEDULES this coroutine —
                # an exception raised once it's actually running (e.g.
                # uvicorn failing to bind the port because something else
                # already holds it) is not caught by this method's own
                # try/except, which only covers the synchronous setup above.
                # Uncaught, it would vanish into asyncio's default "Task
                # exception was never retrieved" handler — silent, because
                # JARVIS runs under pythonw.exe with no console to print it
                # to. The result: a phone user gets a QR code / remote link
                # pointing at a dashboard that was never actually listening,
                # with zero indication anything went wrong — "this site
                # can't be reached" and no clue why.
                try:
                    await self._dashboard.serve()
                except Exception as e:
                    self.ui.write_log(f"ERR: Dashboard failed to start — {e}")
                    self._dashboard_error = str(e)
                    self._dashboard = None

            asyncio.create_task(_run_dashboard())
            # Runs for the whole lifetime, not just inside an active session
            asyncio.create_task(self._process_dashboard_commands())
        except Exception as e:
            print(f"[Dashboard] Disabled: {e}")
            self._dashboard = None

        # Loopback-only control channel for local companion processes (the
        # Arc Sentinel widget's mic-mute / interrupt buttons) — stdlib only,
        # so it works whether or not the (optional) dashboard is installed.
        def _toggle_mute_and_settle():
            # toggle_mute() only queues a cross-thread Qt signal — it returns
            # before _toggle_mute() actually runs on the Qt thread. Without
            # this short wait, the POST /mute response (read right after)
            # would report the PRE-toggle state; the widget's next poll
            # would still self-correct, but this makes the click feel instant
            # instead of one 900ms poll cycle behind.
            self.ui.toggle_mute()
            time.sleep(0.05)

        self._local_control = LocalControlServer(
            get_state      = self._local_control_state,
            on_mute_toggle = _toggle_mute_and_settle,
            on_interrupt   = self.interrupt,
        )
        if not self._local_control.start():
            print("[LocalControl] Port already in use — probably another JARVIS instance.")

        if self._mode == "local":
            # Entirely different pipeline (blocking record → transcribe →
            # chat → speak loop, no Live session, no TaskGroup) — the cloud
            # while-loop below is untouched and simply never reached.
            await self._run_local_loop()
            return

        while True:
            try:
                print("[JARVIS] Connecting...")
                self.ui.set_state("THINKING")
                _resumed_with = self._resume_handle is not None
                config = self._build_config()

                # Fresh client on every reconnect — avoids stale HTTP session state
                # v1alpha carries proactive audio; if it gets rejected we fall
                # back to v1beta.
                client = genai.Client(
                    api_key=_get_api_key(),
                    http_options={"api_version": "v1alpha" if self._enhanced_live else "v1beta"}
                )

                async with (
                    client.aio.live.connect(model=LIVE_MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session          = session
                    self.audio_in_queue   = asyncio.Queue()
                    self.out_queue        = asyncio.Queue(maxsize=200)
                    self._turn_done_event = asyncio.Event()

                    # Reset transient state that must not carry over from a previous session
                    self._pending_vision       = None
                    self._vision_cam_active    = False
                    self._vision_close_pending = False
                    self._vision_busy          = False
                    self._vision_last_time     = 0.0
                    self._interrupted          = False

                    print("[JARVIS] Connected.")
                    if _resumed_with:
                        # Say it plainly: the difference between "it reconnected"
                        # and "it reconnected and still knows what we were doing"
                        # is the whole point, and it is invisible otherwise.
                        self.ui.write_log("SYS: Reconnected — conversation restored.")

                    # Wake word: if enabled, come up ASLEEP (mic gated, silent)
                    # until the user says "Hey Jarvis" or taps wake in the UI.
                    if self._wake_enabled:
                        self._ensure_wake_detector()
                        self._awake = False
                        self.ui.set_state("SLEEPING")
                        self.ui.write_log("SYS: JARVIS online — sleeping. Say 'Hey Jarvis' to wake me.")
                    else:
                        self._awake = True
                        self.ui.set_state("LISTENING")
                        self.ui.write_log("SYS: JARVIS online.")

                    if self._dashboard:
                        await self._dashboard.broadcast({"type": "status", "state": "active"})

                    self._reconnect_event.clear()  # ignore requests from before this session
                    tg.create_task(self._watch_reconnect())
                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())
                    tg.create_task(self._run_system_monitor())
                    tg.create_task(self._run_background_monitor())
                    tg.create_task(self._run_proactive_mode())
                    tg.create_task(self._run_sleep_watch())
                    if self._dashboard:
                        tg.create_task(self._relay_phone_audio())

                    # Morning briefing — fires once per process launch (if enabled).
                    # Skipped in wake-word mode: it comes up asleep, and a briefing
                    # would mean talking while "asleep".
                    if not self._briefing_sent and get_brief_enabled() and self._awake:
                        self._briefing_sent = True
                        tg.create_task(self._send_startup_briefing())

            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except BaseException as e:
                # Catches both Exception and BaseExceptionGroup (Python 3.11+
                # TaskGroup raises BaseExceptionGroup when tasks are cancelled
                # externally, which `except Exception` would miss, letting the
                # exception escape the while-loop and causing asyncio.run() to
                # start shutdown — resulting in "executor after shutdown" errors).
                # Voluntary reconnect (voice change) — not an error. Rebuild the
                # session immediately with no backoff and no scary logs.
                if _is_reconnect_signal(e):
                    print("[JARVIS] Voluntary reconnect requested.")
                    if not _keep_context_of(e):
                        # A deliberate clean slate (voice change) — drop the
                        # handle so the next connect really does start empty.
                        self._resume_handle = None
                    self._conn_backoff = 0
                    continue

                # A resumption handle the server will not accept — expired, or
                # belonging to a session it has since dropped. Without this, the
                # same dead handle would be replayed on every retry and the
                # assistant would never come back at all: the feature meant to
                # survive a reconnect would be the thing preventing one. Drop it
                # once and let the next attempt start clean.
                if _resumed_with and (
                    "resum" in str(e).lower()
                    or "handle" in str(e).lower()
                    or "INVALID_ARGUMENT" in str(e)
                    or "NOT_FOUND" in str(e)
                ):
                    print("[JARVIS] 🔗 Resumption handle rejected — starting a fresh session")
                    self.ui.write_log("SYS: Could not restore the conversation — starting fresh.")
                    self._resume_handle = None
                    self._conn_backoff = 0
                    continue

                err_str = str(e)
                print(f"[JARVIS] Error ({type(e).__name__}): {e}")
                traceback.print_exc()

                # Proactive audio rejected by the server (preview API drift) —
                # drop it and reconnect with the plain config.
                if self._enhanced_live and (
                    "INVALID_ARGUMENT" in err_str
                    or "proactiv" in err_str.lower()
                    or "Unknown name" in err_str
                    or "unexpected keyword" in err_str
                ):
                    self._enhanced_live = False
                    self.ui.write_log(
                        "SYS: Proactive audio unavailable — reconnecting without it."
                    )
                    continue

                # Invalid API key — stop hammering the API, prompt re-configuration
                if "API key not valid" in err_str or "1007" in err_str:
                    self.ui.write_log("ERR: API key invalid — please re-enter your key.")
                    self.ui.set_state("SLEEPING")
                    self.ui.prompt_reconfig()
                    while not self.ui._win._ready:
                        await asyncio.sleep(1)
                    print("[JARVIS] New API key saved — reconnecting...")
                    _conn_backoff = 3
                    continue

                # Network / timeout errors — log clearly and back off
                is_net_err = any(k in err_str for k in (
                    "TimeoutError", "timed out", "getaddrinfo", "CancelledError",
                    "ConnectionRefusedError", "OSError", "Cannot connect",
                ))
                if is_net_err:
                    _conn_backoff = min(getattr(self, "_conn_backoff", 3) * 2, 60)
                    self._conn_backoff = _conn_backoff
                    self.ui.write_log(
                        f"NET: Connection failed — retrying in {_conn_backoff}s. "
                        "(a VPN may be required)"
                    )
                else:
                    self._conn_backoff = 3
            finally:
                self.session = None
                # Only save if there was a real conversation (≥3 turns)
                if len(self._session_log) >= 3:
                    asyncio.create_task(self._save_session_summary())

            self.set_speaking(False)
            self.ui.set_state("SLEEPING")

            if self._dashboard:
                await self._dashboard.broadcast({"type": "status", "state": "sleeping"})

            delay = getattr(self, "_conn_backoff", 3)
            print(f"[JARVIS] Reconnecting in {delay}s...")
            await asyncio.sleep(delay)

    # ── Local Mode: offline STT → local LLM → offline TTS ─────────────────────
    # An additive second pipeline, selected via ⚙ → PLUGIN SETTINGS → ENGINE.
    # It reuses the same tool registry, system prompt assembly, and tool
    # dispatch (_dispatch_tool) as the Gemini Live path above — only the
    # transport (audio streaming vs. record/transcribe/chat/speak) differs.
    # Feature parity is intentionally NOT a goal: vision and Gemini-specific
    # behaviour (session resumption, proactive audio) are unavailable here,
    # and _dispatch_tool already says so rather than pretending to work.

    async def _local_wake_wait(self) -> None:
        """While asleep in Local Mode, actually listen for "Hey Jarvis"
        instead of just polling the _awake flag. Local Mode has no persistent
        mic stream the way the cloud path's _listen_audio does — normally
        each utterance opens its own short-lived InputStream via
        _record_utterance — so without this, a detector armed by
        _enter_standby()/_ui_wake_toggle would sit fed with silence forever
        and "Hey Jarvis" could never fire here. Opens one stream and waits
        (with a timeout so a wedged stream can't hang the loop) rather than
        looping open/close on every poll."""
        det = self._wake_detector
        if det is None or not det.ready:
            await asyncio.sleep(0.5)
            return
        woke = asyncio.Event()
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time_info, status):
            # Same settle window as the cloud path's mic callback (see
            # WAKE_SETTLE_SECONDS) — skip feeding the tail of the utterance
            # that just triggered sleep, so it can't immediately re-wake us.
            if time.monotonic() >= self._wake_feed_gate_open_at:
                det.feed(indata)
            if self._awake:
                loop.call_soon_threadsafe(woke.set)

        _mic_name = get_input_device()
        _mic_dev  = audio_devices.resolve(_mic_name, "input")
        try:
            stream = sd.InputStream(
                samplerate=SEND_SAMPLE_RATE, channels=CHANNELS, dtype="int16",
                blocksize=CHUNK_SIZE, device=_mic_dev, callback=callback,
            )
        except Exception as e:
            self.ui.write_log(f"ERR: Could not open mic to listen for 'Hey Jarvis': {e}")
            await asyncio.sleep(2.0)
            return
        with stream:
            try:
                await asyncio.wait_for(woke.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                pass

    async def _record_utterance(self, max_secs: float = 15.0) -> np.ndarray:
        """Record from the mic until ~700 ms of silence following detected
        speech, or `max_secs` as a hard cap. Returns float32 mono @16kHz,
        normalised to [-1, 1] for faster-whisper / Vosk. Reuses _pcm_level()
        (the same loudness measure the cloud path's HUD waveform uses) as a
        cheap, dependency-free voice-activity gate — no separate VAD model."""
        loop = asyncio.get_event_loop()
        chunks: list[np.ndarray] = []
        state = {"speech_started": False, "silence_run": 0.0}
        done = asyncio.Event()

        def callback(indata, frames, time_info, status):
            level = _pcm_level(indata)
            chunks.append(indata.copy())
            block_secs = frames / SEND_SAMPLE_RATE
            if level > 0.03:
                state["speech_started"] = True
                state["silence_run"] = 0.0
            elif state["speech_started"]:
                state["silence_run"] += block_secs
            if state["speech_started"] and state["silence_run"] > 0.7:
                loop.call_soon_threadsafe(done.set)

        _mic_name = get_input_device()
        _mic_dev  = audio_devices.resolve(_mic_name, "input")
        stream = sd.InputStream(
            samplerate=SEND_SAMPLE_RATE, channels=CHANNELS, dtype="int16",
            blocksize=CHUNK_SIZE, device=_mic_dev, callback=callback,
        )
        with stream:
            try:
                await asyncio.wait_for(done.wait(), timeout=max_secs)
            except asyncio.TimeoutError:
                pass

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio_i16 = np.concatenate(chunks).flatten()
        return audio_i16.astype(np.float32) / 32768.0

    async def _stream_round(self, messages: list, tools: list, tts_queue: "queue.Queue") -> dict:
        """Runs one streaming LLM turn in a background thread: forwards text
        deltas through a SentenceChunker onto tts_queue as sentences complete,
        and collects tool calls to return once the stream ends. Returns the
        same {"content", "tool_calls"} shape llm_client.call_llm() does, so
        the tool-calling loop in _run_local_loop doesn't need to know
        streaming is happening underneath it."""
        from core import llm_client
        from core.sentence_chunker import SentenceChunker

        self._local_stream_cancel.clear()

        def _run() -> dict:
            chunker = SentenceChunker()
            full_content = ""
            tool_calls: list = []
            usage: dict = {}
            for event in llm_client.stream_llm(messages, tools, cancel_event=self._local_stream_cancel):
                if "delta" in event:
                    full_content += event["delta"]
                    for sentence in chunker.feed(event["delta"]):
                        tts_queue.put(sentence)
                elif "tool_call" in event:
                    tool_calls.append(event["tool_call"])
                elif "done" in event:
                    usage = event["done"] or {}
                    break
            if not self._local_stream_cancel.is_set():
                remainder = chunker.flush()
                if remainder:
                    tts_queue.put(remainder)
            return {"content": full_content.strip(), "tool_calls": tool_calls, "usage": usage}

        return await asyncio.to_thread(_run)

    def _tts_consumer(self, tts_queue: "queue.Queue", tts_player, speech_profile, turn=None) -> None:
        """Drains sentences _stream_round pushes and speaks them one at a
        time, on its own thread for the whole turn — so speech overlaps with
        the model still generating (and any tool dispatch in between rounds)
        instead of waiting for the full reply. interrupt() stops this
        mid-sentence by draining tts_queue, pushing the sentinel, and calling
        tts_player.stop(). `turn` (core.telemetry.Turn), if given, is marked
        "first_audio" right before the first sentence is actually spoken."""
        first = True
        while True:
            item = tts_queue.get()
            if item is None:
                break
            try:
                if first and turn is not None:
                    turn.mark_once("first_audio")
                first = False
                tts_player.speak(item, profile=speech_profile)
            except Exception as e:
                print(f"[Local] TTS error: {e}")

    async def _run_local_loop(self) -> None:
        import queue
        from core import llm_client
        from core.tool_schema import gemini_tools_to_openai
        from core import tts as tts_mod

        cfg      = get_plugin_config("local_engine")
        provider = llm_client.get_llm_provider()

        self.ui.write_log(f"SYS: Starting Local Mode ({provider}) — no cloud calls will be made.")
        self.ui.set_state("THINKING")

        # Fail loudly and stop — never fall back to the cloud API the user
        # explicitly opted out of by choosing Local Mode.
        reachable = await asyncio.to_thread(llm_client.ensure_ollama_running)
        if not reachable:
            url, model = llm_client.get_llm_settings()
            self.ui.write_log(
                f"ERR: Local LLM backend unreachable at {url}. "
                f"Start it (or check the URL in Settings), or switch back to "
                f"Cloud mode in ⚙ → PLUGIN SETTINGS → ENGINE."
            )
            self.ui.set_state("SLEEPING")
            return

        try:
            static_prompt = _load_system_prompt()
            await asyncio.to_thread(llm_client.warmup_model, static_prompt)
        except Exception as e:
            print(f"[Local] Warmup skipped: {e}")

        stt_engine_name = str(cfg.get("local_stt_engine", "whisper")).lower()
        try:
            if stt_engine_name == "vosk":
                from core.stt import VoskSTT
                stt = await asyncio.to_thread(
                    VoskSTT, None, cfg.get("local_stt_language", "en-us"))
            else:
                from core.stt import WhisperSTT
                stt = await asyncio.to_thread(
                    WhisperSTT, cfg.get("local_stt_model", "base"), cfg.get("local_stt_language"))
        except Exception as e:
            self.ui.write_log(f"ERR: Local speech-to-text failed to load: {e}")
            self.ui.set_state("SLEEPING")
            return

        try:
            tts_player = await asyncio.to_thread(tts_mod.create_tts_player, cfg)
        except Exception as e:
            self.ui.write_log(f"ERR: Local text-to-speech failed to load: {e}")
            self.ui.set_state("SLEEPING")
            return
        self._local_tts_player = tts_player

        if self._wake_enabled:
            self._ensure_wake_detector()
            self._awake = False
            self.ui.set_state("SLEEPING")
            self.ui.write_log("SYS: JARVIS online (Local) — sleeping. Say 'Hey Jarvis' to wake me.")
        else:
            self._awake = True
            self.ui.write_log("SYS: JARVIS online (Local Mode).")

        messages: list[dict] = [
            {"role": "system", "content": self._assemble_system_prompt()},
            {"role": "system", "content": ""},  # reserved: refreshed with the ContextBundle each turn, not appended
        ]

        while True:
            if self._wake_enabled and not self._awake:
                await self._local_wake_wait()
                continue

            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            try:
                audio = await self._record_utterance()
            except Exception as e:
                print(f"[Local] Mic error: {e}")
                await asyncio.sleep(1.0)
                continue

            if audio.size < int(SEND_SAMPLE_RATE * 0.3):   # too short — no real speech
                continue

            self.ui.set_state("THINKING")
            try:
                if stt_engine_name == "vosk":
                    text, _ = await asyncio.to_thread(
                        stt.process_chunk, (audio * 32768.0).astype(np.int16).tobytes())
                else:
                    text = await asyncio.to_thread(stt.transcribe, audio)
            except Exception as e:
                self.ui.write_log(f"ERR: Transcription failed: {e}")
                continue

            text = (text or "").strip()
            if not text:
                continue

            self.ui.write_log(f"You: {text}")
            self._session_log.append(f"User: {text}")
            self._log_context_turn("user", text)
            self._last_user_speech = time.monotonic()

            # Refresh the reserved context slot (messages[1]) right before
            # this turn is sent — session recency, project facts, and
            # semantically-relevant past turns, structured so the model can
            # tell them apart from the live conversation instead of getting
            # one undifferentiated dump. Best-effort: a context-build failure
            # falls back to the empty placeholder rather than blocking the turn.
            try:
                bundle = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: context_manager.build_context(text, session_log=self._session_log),
                )
                combined = bundle.to_prompt()
            except Exception as e:
                print(f"[ContextManager] ⚠️ build_context failed: {e}")
                combined = ""

            # Tone/verbosity/proactivity modifier for the prompt, plus a
            # SpeechProfile (rate/pitch/stability) for the TTS engine once the
            # reply comes back — one detect+log call covers both, via
            # evaluate() (see core/sentiment_adapter.py). recent_texts lets
            # detection notice "I already told you" style repeats. Returns
            # ("", None) outright when the user has disabled adaptation in
            # Settings, so a disabled toggle really means nothing touches the
            # prompt or the voice, not "always neutral".
            speech_profile = None
            try:
                style_text, speech_profile = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: sentiment_adapter.evaluate(text, recent_texts=self._session_log),
                )
                if style_text:
                    combined = f"{combined}\n\n{style_text}" if combined else style_text
            except Exception as e:
                print(f"[SentimentAdapter] ⚠️ evaluate failed: {e}")

            messages[1]["content"] = combined

            messages.append({"role": "user", "content": text})

            turn = telemetry.start_turn("local")

            tts_queue: "queue.Queue" = queue.Queue()
            self._local_tts_queue = tts_queue
            speaker = threading.Thread(
                target=self._tts_consumer, args=(tts_queue, tts_player, speech_profile, turn), daemon=True,
            )
            speaker.start()
            self.set_speaking(True)

            try:
                resp = await self._stream_round(messages, gemini_tools_to_openai(self._all_tool_declarations()), tts_queue)
                usage = resp.get("usage") or {}
                turn.tokens(tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"))
            except Exception as e:
                self.ui.write_log(f"ERR: Local LLM call failed: {e}")
                tts_queue.put(None)
                await asyncio.to_thread(speaker.join)
                self.set_speaking(False)
                self._local_tts_queue = None
                turn.finish()
                continue

            # Tool-calling loop — capped so a model stuck calling tools can
            # never spin forever.
            standby_entered = False
            for _ in range(5):
                tool_calls = resp.get("tool_calls") or []
                if not tool_calls:
                    break
                messages.append({
                    "role": "assistant",
                    "content": resp.get("content", ""),
                    "tool_calls": tool_calls,
                })
                for tc in tool_calls:
                    fn   = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}

                    if name == "shutdown_jarvis":
                        # Mute, don't exit — see _enter_standby(). No LLM call
                        # and no TTS after this: skip straight back to the top
                        # of the outer loop, where the _awake gate takes over.
                        self._enter_standby(reason="bye jarvis")
                        messages.append({
                            "role": "tool", "tool_call_id": tc.get("id", ""),
                            "name": name, "content": "muted until 'Hey Jarvis'",
                        })
                        standby_entered = True
                        break

                    print(f"[JARVIS] 🔧 {name}  {args}")
                    self.ui.set_state("THINKING")
                    with turn.tool_span(name):
                        tool_result = await self._dispatch_tool(name, args)
                    print(f"[JARVIS] 📤 {name} → {str(tool_result)[:80]}")
                    messages.append({
                        "role": "tool", "tool_call_id": tc.get("id", ""),
                        "name": name, "content": str(tool_result),
                    })
                if standby_entered:
                    break
                try:
                    resp = await self._stream_round(messages, gemini_tools_to_openai(self._all_tool_declarations()), tts_queue)
                    usage = resp.get("usage") or {}
                    turn.tokens(tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"))
                except Exception as e:
                    self.ui.write_log(f"ERR: Local LLM call failed: {e}")
                    resp = {"content": "", "tool_calls": []}
                    break

            turn.mark("model_done")
            if self._local_stream_cancel.is_set():
                turn.set_interrupted()
            tts_queue.put(None)
            await asyncio.to_thread(speaker.join)
            self.set_speaking(False)
            self._local_tts_queue = None
            turn.finish()

            if standby_entered:
                continue

            reply = (resp.get("content") or "").strip()
            if reply:
                self.ui.write_log(f"{self._asst_name}: {reply}")
                self._session_log.append(f"{self._asst_name}: {reply}")
                self._log_context_turn("assistant", reply)
                messages.append({"role": "assistant", "content": reply})

            if not self.ui.muted:
                self.ui.set_state("LISTENING")

def main():
    ui = JarvisUI("face.png")

    def runner():
        ui.wait_for_api_key()
        jarvis = JarvisLive(ui)
        try:
            asyncio.run(jarvis.run())
        except KeyboardInterrupt:
            print("\n🔴 Shutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

if __name__ == "__main__":
    main()