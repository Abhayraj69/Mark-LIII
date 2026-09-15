"""
tool_connectors/registry.py — discovers connectors, publishes their
capabilities as function-calling schemas, and is the one place a
REVERSIBLE/DESTRUCTIVE action can actually run from.

CONFIRMATION, NOT A `confirmed` PARAMETER
    core/confirm.py exists precisely because a boolean tool parameter can be
    set by the model itself — it is not a real gate. This registry reuses
    that same module: execute() on anything above READ_ONLY hands the real
    call to core.confirm.request() and returns immediately with a sentence
    for the model to say, exactly like shutdown_jarvis does today. The action
    only actually runs if a human presses CONFIRM on the HUD. If no UI has
    bound itself yet (headless import, an early call), core.confirm already
    refuses gracefully — this module doesn't need to special-case that.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

from tool_connectors.base import (
    ActionSafety,
    Capability,
    ExecutionResult,
    ToolConnector,
    ToolExecutionError,
)
from tool_connectors import audit

try:
    from core import confirm as confirm_gate
except Exception:  # pragma: no cover — package used outside the JARVIS tree
    class _NoConfirmGate:
        @staticmethod
        def request(key, title, detail, run):
            return (f"I cannot confirm '{title}' right now because no confirmation "
                    f"interface is available, so I have not done it.")

    confirm_gate = _NoConfirmGate()


CONNECTORS_DIR = Path(__file__).resolve().parent / "connectors"


class ToolRegistry:
    def __init__(self, logger: Callable[[str], None] = print):
        self._connectors: dict[str, ToolConnector] = {}
        self._logger = logger

    # -- discovery --------------------------------------------------------

    def discover(self, connectors_dir: Path = CONNECTORS_DIR) -> "ToolRegistry":
        """Imports every *.py in connectors_dir (skipping '_'-prefixed files)
        and instantiates any ToolConnector subclass defined directly in that
        module. Import errors and constructor failures are logged and the
        file skipped — one bad connector never blocks the rest, mirroring
        core/action_loader.py's discovery contract."""
        connectors_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(connectors_dir.glob("*.py"), key=lambda p: p.name):
            if path.name.startswith("_"):
                continue
            try:
                module_name = f"tool_connectors.connectors.{path.stem}"
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

                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, ToolConnector)
                        and obj is not ToolConnector
                        and obj.__module__ == module_name
                    ):
                        instance = obj()
                        self.register(instance)
            except Exception as e:
                self._logger(f"Connector load failed: {path.name} — {e}")
                traceback.print_exc()
        return self

    def register(self, connector: ToolConnector) -> None:
        if connector.name in self._connectors:
            self._logger(f"Connector name collision — '{connector.name}' already registered, overwriting.")
        self._connectors[connector.name] = connector
        self._logger(f"Connector registered: {connector.name}")

    def get(self, name: str) -> Optional[ToolConnector]:
        return self._connectors.get(name)

    def connectors(self) -> list[ToolConnector]:
        return list(self._connectors.values())

    # -- capability publishing ---------------------------------------------

    def get_tool_declarations(self, reserved_names: set[str] | None = None) -> list[dict]:
        """Every connector capability as a flat function-calling schema, the
        same shape core/action_loader.py's ActionRegistry publishes — so the
        planning/reasoning layer can treat a connector action exactly like
        any other tool. Tool name is "{connector}__{action}" to keep the two
        namespaces from colliding with each other.

        reserved_names lets a caller (main.py, wiring this alongside the
        inline tools / actions / plugins that already exist) reject any
        declaration that collides with a name already claimed elsewhere —
        the same "reserved_names" discipline core/action_loader.py and
        core/plugin_loader.py already apply to each other."""
        reserved = reserved_names or set()
        declarations = []
        for connector in self._connectors.values():
            try:
                caps = connector.list_capabilities()
            except Exception as e:
                self._logger(f"list_capabilities failed for '{connector.name}': {e}")
                continue
            for cap in caps:
                tool_name = f"{connector.name}__{cap.name}"
                if tool_name in reserved:
                    self._logger(f"Connector capability rejected: '{tool_name}' collides with an existing tool name.")
                    continue
                declarations.append(
                    {
                        "name": tool_name,
                        "description": f"[{cap.safety.value}] {cap.description}",
                        "parameters": cap.parameters,
                    }
                )
        return declarations

    def has_declaration(self, tool_name: str) -> tuple[str, str] | None:
        """Reverse-maps a published tool name ('git__commit') back to
        (connector_name, action) for dispatch, or None if it isn't one of
        this registry's declarations."""
        if "__" not in tool_name:
            return None
        connector_name, _, action = tool_name.partition("__")
        connector = self._connectors.get(connector_name)
        if connector is None:
            return None
        try:
            connector.capability(action)
        except ToolExecutionError:
            return None
        return connector_name, action

    def health_report(self) -> dict[str, bool]:
        report = {}
        for name, connector in self._connectors.items():
            try:
                report[name] = bool(connector.health_check())
            except Exception:
                report[name] = False
        return report

    # -- execution ----------------------------------------------------------

    def execute(self, connector_name: str, action: str, params: Optional[dict] = None) -> str:
        """Looks up the capability's safety tier and either runs it now
        (READ_ONLY) or parks it behind the confirmation gate (REVERSIBLE /
        DESTRUCTIVE). Always returns a string — either the result, or a
        sentence for the model to relay while confirmation is pending —
        never raises: connector/parameter problems come back as a
        ToolExecutionError's human_message instead."""
        params = params or {}
        connector = self._connectors.get(connector_name)
        if connector is None:
            return f"Connector '{connector_name}' is not registered."

        try:
            cap = connector.capability(action)
        except ToolExecutionError as e:
            return e.human_message

        if cap.safety == ActionSafety.READ_ONLY:
            result = self._run(connector, action, params, cap.safety)
            return result.message or str(result.output)

        title = f"{connector.name}.{action}"
        detail = self._describe(params)
        audit.record(
            connector.name, action, cap.safety.value,
            json.dumps(params, default=str), "", True, stage="confirmation_requested",
        )

        def _do() -> str:
            result = self._run(connector, action, params, cap.safety)
            return result.message or str(result.output)

        return confirm_gate.request(
            key=f"connector:{connector.name}:{action}",
            title=title,
            detail=detail,
            run=_do,
        )

    def _run(self, connector: ToolConnector, action: str, params: dict, safety: ActionSafety) -> ExecutionResult:
        params_json = json.dumps(params, default=str)
        try:
            result = connector.execute(action, params)
            audit.record(connector.name, action, safety.value, params_json, str(result.output)[:2000], result.success)
            return result
        except ToolExecutionError as e:
            audit.record(connector.name, action, safety.value, params_json, e.human_message, False)
            return ExecutionResult(success=False, output=None, message=e.human_message)
        except Exception as e:
            wrapped = ToolExecutionError(connector.name, action, str(e), original=e)
            audit.record(connector.name, action, safety.value, params_json, wrapped.human_message, False)
            traceback.print_exc()
            return ExecutionResult(success=False, output=None, message=wrapped.human_message)

    @staticmethod
    def _describe(params: dict) -> str:
        if not params:
            return "No parameters."
        try:
            return ", ".join(f"{k}={v}" for k, v in params.items())[:280]
        except Exception:
            return str(params)[:280]


def build_default_registry(logger: Callable[[str], None] = print) -> ToolRegistry:
    return ToolRegistry(logger=logger).discover()
