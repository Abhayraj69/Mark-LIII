"""
core/skill_watcher.py — polls plugins/ and actions/ for file changes and
hot-reloads them, so a skill can be edited without restarting JARVIS.

WHY THIS EXISTS
    Self-describing skills are the headline of Mark LIII, but the loop was
    still "edit -> restart the whole assistant -> re-say the wake word."
    Neither core/plugin_loader.py nor core/action_loader.py had any reload
    path, and the Live session is exactly the expensive thing you don't want
    to tear down to test a three-line plugin change.

HOW IT WORKS
    A daemon thread polls the mtime of every *.py file in both directories
    every POLL_INTERVAL_S. A changed or removed file is only acted on once
    it has been stable for DEBOUNCE_S (an editor's save-as-you-type, or a
    multi-file write, would otherwise trigger a reload mid-edit). Each
    touched file is handed to the owning registry's reload(path) — see
    PluginRegistry.reload / ActionRegistry.reload in the loader modules —
    which re-imports just that file and swaps its entry in atomically,
    keeping the previous version live if the new one fails to import or
    validate.

    Stdlib-only polling (no `watchdog` dependency), matching the rest of
    core/'s stance on new dependencies — 2s latency on noticing a change is
    an acceptable trade for not adding a filesystem-events library.

TOOL DECLARATIONS AFTER A RELOAD
    Local Mode already rebuilds its OpenAI-shaped tool list from
    JarvisLive._all_tool_declarations() on every turn (see _run_local_loop),
    so a reload is picked up there for free — no callback needed. The Gemini
    Live session is different: its tool set is fixed for the life of the
    session, so on_change (below) is used to trigger
    JarvisLive.request_reconnect(keep_context=True) only when Live mode is
    the active engine and a reload actually changed a declaration.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

POLL_INTERVAL_S = 2.0
DEBOUNCE_S = 0.5


class SkillWatcher:
    def __init__(
        self,
        plugins_dir: Path,
        actions_dir: Path,
        plugin_registry,
        action_registry,
        on_change: Optional[Callable[[str], None]] = None,
        logger: Callable[[str], None] = print,
    ):
        self._plugins_dir = plugins_dir
        self._actions_dir = actions_dir
        self._plugin_registry = plugin_registry
        self._action_registry = action_registry
        self._on_change = on_change
        self._logger = logger

        self._mtimes: dict[Path, float] = {}
        self._pending: dict[Path, float] = {}   # path -> monotonic time first seen changed
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._mtimes = self._snapshot()
        self._thread = threading.Thread(target=self._run, name="SkillWatcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _snapshot(self) -> dict[Path, float]:
        snap: dict[Path, float] = {}
        for d in (self._plugins_dir, self._actions_dir):
            if not d.exists():
                continue
            for p in d.glob("*.py"):
                if p.name.startswith("_"):
                    continue
                try:
                    snap[p] = p.stat().st_mtime
                except OSError:
                    pass
        return snap

    def _run(self) -> None:
        while not self._stop.wait(POLL_INTERVAL_S):
            try:
                self._poll_once()
            except Exception as e:
                self._logger(f"[SkillWatcher] poll failed: {e}")

    def _poll_once(self) -> None:
        current = self._snapshot()
        now = time.monotonic()

        for path, mtime in current.items():
            if self._mtimes.get(path) != mtime:
                self._pending.setdefault(path, now)
        for path in set(self._mtimes) - set(current):
            self._pending.setdefault(path, now)

        ready = [p for p, first_seen in self._pending.items() if now - first_seen >= DEBOUNCE_S]
        self._mtimes = current
        if not ready:
            return

        changed_declarations = False
        for path in ready:
            self._pending.pop(path, None)
            registry = self._plugin_registry if path.parent == self._plugins_dir else self._action_registry
            ok, msg = registry.reload(path)
            self._logger(f"[SkillWatcher] {msg}")
            changed_declarations = changed_declarations or ok

        if changed_declarations and self._on_change is not None:
            try:
                self._on_change("skills reloaded")
            except Exception as e:
                self._logger(f"[SkillWatcher] on_change callback failed: {e}")
