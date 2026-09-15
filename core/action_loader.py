"""
Action discovery, validation, and dispatch — the built-in twin of plugin_loader.

Every actions/*.py that exposes a module-level ``TOOL`` dict is auto-discovered
here, exactly like a drop-in plugin, so main.py never has to hardcode a tool
declaration or a dispatch branch for it. Adding a new bundled action is then the
same one-file operation as writing a plugin: define ``TOOL`` and a handler.

``TOOL`` shape (see actions/open_app.py for a live example):

    TOOL = {
        "name":        "open_app",              # unique, ^[a-zA-Z_][a-zA-Z0-9_]{0,63}$
        "description":  "...",                   # what Gemini reads to route the call
        "parameters":  {"type": "OBJECT", ...}, # Gemini function-declaration schema
        "handler":      open_app,                # the callable to run
    }

The handler is invoked through signature introspection: it receives ``parameters``
plus whichever of ``player`` / ``speak`` / ``response`` / ``session_memory`` /
``dispatch`` it actually declares — so existing action signatures work unchanged.
``dispatch`` is a ``(tool_name, args) -> str`` callable that re-enters main.py's
own tool router, letting one action (e.g. sequence replay) invoke other tools
by name without a second dispatch mechanism.

Discovery runs once at startup; import errors, validation errors, and name
collisions are logged and the offending file is skipped — they NEVER raise out
of discover_actions() and never abort the scan of the remaining files.
"""
from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")
_DEFAULT_PARAMS = {"type": "OBJECT", "properties": {}}
_CTX_KEYS = ("player", "speak", "response", "session_memory", "dispatch")


@dataclass
class ActionRecord:
    name: str
    description: str = ""
    parameters: dict = field(default_factory=lambda: dict(_DEFAULT_PARAMS))
    handler: Optional[Callable] = None
    file: str = ""
    valid: bool = False
    error: str = ""


class ActionRegistry:
    def __init__(self, actions: dict[str, ActionRecord], logger: Callable[[str], None]):
        self._actions = actions          # name -> ActionRecord, VALID entries only
        self._all_records: list[ActionRecord] = []
        self._logger = logger
        self._lock = threading.Lock()
        # Set by discover_actions() right after construction — see
        # PluginRegistry's identical fields for why.
        self._actions_dir: Optional[Path] = None
        self._reserved_names: set[str] = set()

    # -- called by main.py at LiveConnectConfig build time --
    def get_tool_declarations(self) -> list[dict]:
        return [
            {"name": rec.name, "description": rec.description, "parameters": rec.parameters}
            for rec in self._actions.values()
        ]

    def has(self, name: str) -> bool:
        return name in self._actions

    def names(self) -> set[str]:
        return set(self._actions.keys())

    # -- called by main.py from _execute_tool --
    def run(self, name: str, parameters: dict, ctx: dict | None = None) -> str:
        rec = self._actions.get(name)
        if rec is None or not rec.valid:
            return f"Action '{name}' is not available."
        try:
            return _call_handler(rec.handler, parameters, ctx or {}) or "Done."
        except Exception as e:
            self._logger(f"Action '{name}' crashed during run(): {e}")
            traceback.print_exc()
            return f"Tool '{name}' failed: {e}"

    # -- called by core/skill_watcher.py on a detected file change, and by
    # the "RELOAD ALL" button in ui.py's Plugin Manager (via reload_all) --
    def reload(self, path: Path) -> tuple[bool, str]:
        """Re-imports a single actions/*.py file after a change on disk. On
        import or validation failure the previous version stays registered
        and (False, message) is returned. Deleting the file, or editing a
        TOOL dict out of it, unregisters its tool. Returns (True, message)
        only when the registered tool set actually changed."""
        with self._lock:
            old_rec = next((r for r in self._all_records if r.file == path.name), None)
            old_name = old_rec.name if old_rec and old_rec.valid else None

            if not path.exists():
                if old_name and old_name in self._actions:
                    del self._actions[old_name]
                self._all_records = [r for r in self._all_records if r.file != path.name]
                if old_name:
                    self._logger(f"Action removed: {old_name} ({path.name})")
                return (bool(old_name), f"{old_name or path.name} removed")

            module_name = f"actions.{path.stem}"
            try:
                # Always a fresh spec (not importlib.reload) — see the
                # identical comment in PluginRegistry.reload for why.
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    raise ImportError("could not build import spec")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except Exception as e:
                msg = f"{path.name} failed to reload: {e} — previous version kept."
                self._logger(f"Action reload failed: {msg}")
                return (False, msg)

            if getattr(module, "TOOL", None) is None:
                # Not (or no longer) an action file. If it used to be one,
                # this is effectively a removal; otherwise nothing to do.
                if old_name and old_name in self._actions:
                    del self._actions[old_name]
                    self._all_records = [r for r in self._all_records if r.file != path.name]
                    self._logger(f"Action removed: {old_name} ({path.name}) — TOOL dict deleted.")
                    return (True, f"{old_name} removed (TOOL dict deleted)")
                return (False, f"{path.name} has no TOOL dict — not an action, ignored.")

            rec = _validate(module, path.name)
            if rec.valid and rec.name in self._reserved_names:
                rec = ActionRecord(name=rec.name, file=path.name,
                                    error=f"Name '{rec.name}' collides with a reserved core tool — rejected.")
            elif (rec.valid and rec.name in self._actions
                  and self._actions[rec.name].file != path.name):
                other = self._actions[rec.name].file
                rec = ActionRecord(name=rec.name, file=path.name,
                                    error=f"Name '{rec.name}' already used by action '{other}' — rejected.")

            if not rec.valid:
                msg = f"{path.name} failed to reload: {rec.error} — previous version kept."
                self._logger(f"Action reload failed: {msg}")
                return (False, msg)

            if old_name and old_name != rec.name and old_name in self._actions:
                del self._actions[old_name]
            self._actions[rec.name] = rec
            self._all_records = [r for r in self._all_records if r.file != path.name] + [rec]
            self._logger(f"Action reloaded: {rec.name} ({path.name})")
            return (True, f"Reloaded {rec.name}")

    def reload_all(self) -> list[tuple[str, bool, str]]:
        """Rescans self._actions_dir and reloads every file found (plus drops
        any registered file no longer on disk)."""
        if self._actions_dir is None:
            return []
        on_disk = {p.name: p for p in sorted(self._actions_dir.glob("*.py"))
                   if not p.name.startswith("_")}
        known = {r.file for r in self._all_records}
        results = []
        for name, path in on_disk.items():
            results.append((name, *self.reload(path)))
        for name in known - set(on_disk.keys()):
            results.append((name, *self.reload(self._actions_dir / name)))
        return results


def _call_handler(fn: Callable, parameters: dict, ctx: dict) -> str:
    """Invoke the handler passing only the context kwargs it actually declares
    (or all of them if it has **kwargs), so each action's existing signature
    works unchanged."""
    sig = inspect.signature(fn)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    kwargs = {}
    for key in _CTX_KEYS:
        if has_var_kw or key in sig.parameters:
            kwargs[key] = ctx.get(key)
    return fn(parameters=parameters, **kwargs)


def _validate(module, filename: str) -> ActionRecord:
    """Returns an ActionRecord; .valid=False + .error set on any problem. Never raises."""
    tool = getattr(module, "TOOL", None)
    if not isinstance(tool, dict):
        return ActionRecord(name=Path(filename).stem, file=filename,
                            error="No module-level TOOL dict (not a discoverable action).")

    name = tool.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        return ActionRecord(name=str(name or Path(filename).stem), file=filename,
                            error="TOOL['name'] missing or not a valid identifier.")

    description = tool.get("description")
    if not isinstance(description, str) or not description.strip():
        return ActionRecord(name=name, file=filename,
                            error="TOOL['description'] missing or empty.")

    parameters = tool.get("parameters", _DEFAULT_PARAMS)
    if not isinstance(parameters, dict) or parameters.get("type") != "OBJECT":
        return ActionRecord(name=name, file=filename,
                            error="TOOL['parameters'] must be a dict with \"type\": \"OBJECT\".")

    handler = tool.get("handler")
    if not callable(handler):
        return ActionRecord(name=name, file=filename,
                            error="TOOL['handler'] missing or not callable.")

    return ActionRecord(name=name, description=description.strip(), parameters=parameters,
                        handler=handler, file=filename, valid=True, error="")


def discover_actions(actions_dir: Path, reserved_names: set[str] | None = None,
                     logger: Callable[[str], None] = print) -> ActionRegistry:
    """
    Scans actions_dir for *.py files (skips files starting with '_'). A file is
    only treated as an action if it exposes a module-level TOOL dict; files
    without one (shared helpers, capture-only modules) are silently ignored.
    Import/validation errors and name collisions are logged and the file is
    skipped — they NEVER raise out of this function.
    """
    reserved = reserved_names or set()
    actions_dir.mkdir(parents=True, exist_ok=True)
    valid: dict[str, ActionRecord] = {}
    all_records: list[ActionRecord] = []

    files = sorted(actions_dir.glob("*.py"), key=lambda p: p.name)  # deterministic order
    for path in files:
        if path.name.startswith("_"):
            continue
        try:
            module_name = f"actions.{path.stem}"
            # Reuse the already-imported module when present so handlers are the
            # same objects the rest of the app holds.
            module = sys.modules.get(module_name)
            if module is None:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    raise ImportError("could not build import spec")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop(module_name, None)
                    raise

            if getattr(module, "TOOL", None) is None:
                continue   # not an action file — a helper/capture-only module

            rec = _validate(module, path.name)

            if rec.valid and rec.name in reserved:
                rec = ActionRecord(name=rec.name, file=path.name,
                                   error=f"Name '{rec.name}' collides with a reserved core tool — rejected.")
            elif rec.valid and rec.name in valid:
                other = valid[rec.name].file
                rec = ActionRecord(name=rec.name, file=path.name,
                                   error=f"Name '{rec.name}' already used by action '{other}' — rejected.")

        except Exception as e:
            rec = ActionRecord(name=path.stem, file=path.name,
                               error=f"Failed to load: {e}")
            traceback.print_exc()

        all_records.append(rec)
        if rec.valid:
            valid[rec.name] = rec
            logger(f"Action loaded: {rec.name} ({path.name})")
        else:
            # Only log a rejection if the file actually tried to be an action.
            logger(f"Action rejected: {path.name} — {rec.error}")

    registry = ActionRegistry(valid, logger)
    registry._all_records = all_records
    registry._actions_dir = actions_dir
    registry._reserved_names = set(reserved)
    logger(f"Action discovery complete: {len(valid)} active.")
    return registry
