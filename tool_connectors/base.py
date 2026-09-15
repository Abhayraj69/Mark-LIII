"""
tool_connectors/base.py — the contract every connector implements.

WHY A SEPARATE LAYER FROM actions/
    actions/*.py (see core/action_loader.py) already gives Gemini/Local Mode a
    flat namespace of callable tools. Connectors are different in kind: each
    one wraps an external system that has its own identity, auth, and health
    (git, Docker, a task tracker) and exposes several related actions rather
    than one. list_capabilities() lets the registry publish those actions as
    ordinary function-calling schemas, so from the model's point of view a
    connector action looks exactly like any other tool — the safety
    classification and confirmation gating happen underneath, in the registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class ActionSafety(Enum):
    """How much a connector action can hurt if it runs on a whim.

    READ_ONLY   — inspects state, changes nothing. Safe to auto-run.
    REVERSIBLE  — changes state, but the change can be undone (git commit,
                  moving a file). Requires a human's explicit confirmation
                  before it runs, but is not the same tier of danger as...
    DESTRUCTIVE — changes or removes state with no straightforward way back
                  (force-push, `rm`, dropping a container's volumes). Also
                  requires confirmation, and connectors should say so plainly
                  in the action's description.
    """

    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    DESTRUCTIVE = "destructive"


@dataclass
class Capability:
    """One action a connector exposes, described the way a function-calling
    schema describes a tool — so ToolRegistry can hand these to the
    planning/reasoning layer with no translation step."""

    name: str
    description: str
    safety: ActionSafety
    parameters: dict = field(default_factory=lambda: {"type": "OBJECT", "properties": {}})


@dataclass
class ExecutionResult:
    """What execute() returns on success."""

    success: bool
    output: object
    message: str = ""


class ToolExecutionError(Exception):
    """Every connector failure — a bad param, a missing binary, a network
    error, the underlying system rejecting the call — collapses to this, so
    the assistant always has one predictable shape to relay back out loud
    instead of a different exception type per connector."""

    def __init__(self, connector: str, action: str, message: str, original: Exception | None = None):
        self.connector = connector
        self.action = action
        self.original = original
        super().__init__(message)

    @property
    def human_message(self) -> str:
        return f"{self.connector}.{self.action} failed: {self}"


class ToolConnector(ABC):
    """Base class for one integration. Subclasses set `name` and implement
    the four methods below; everything else (safety gating, confirmation,
    audit logging) is the registry's job, not the connector's."""

    name: str = "connector"

    @abstractmethod
    def authenticate(self) -> bool:
        """Establish/verify whatever credentials or connection this connector
        needs. Returns True if the connector is usable. Connectors with
        nothing to authenticate (a local filesystem, a local CLI) should
        return True unconditionally rather than treating it as N/A."""

    @abstractmethod
    def list_capabilities(self) -> list[Capability]:
        """The fixed list of actions this connector supports, each tagged
        with its ActionSafety."""

    @abstractmethod
    def execute(self, action: str, params: dict) -> ExecutionResult:
        """Run one capability by name. Must raise ToolExecutionError (not
        let a raw exception escape) on any failure — the registry does not
        catch anything else."""

    @abstractmethod
    def health_check(self) -> bool:
        """Cheap liveness probe (binary on PATH, daemon reachable, directory
        exists) — separate from authenticate() because a registry may want to
        report "git not installed" without attempting a full auth handshake."""

    # -- shared helper, not part of the ABC contract --
    def capability(self, action: str) -> Capability:
        for cap in self.list_capabilities():
            if cap.name == action:
                return cap
        raise ToolExecutionError(self.name, action, f"'{action}' is not a capability of {self.name}.")
