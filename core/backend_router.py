"""
core/backend_router.py — per-task-kind backend selection with health-based
failover.

WHY THIS EXISTS
    Three backends are wired into this app — Gemini (google-genai), Ollama or
    an OpenAI-compatible server (core/llm_client.py), and Claude
    (core/claude_bridge.py) — but which one answers a given request has been
    one global setting. That means a trivial voice command pays for whatever
    the heaviest configured engine costs, a sub-task like dev_agent's code
    generation runs through whichever engine happens to be selected for
    voice, and if Ollama is down, everything that depended on it just fails
    instead of quietly trying something else. This router picks a backend
    per TASK KIND from an explicit ordered policy, skips a backend for 60s
    after it fails (a circuit breaker, so a downed server isn't retried on
    every single call) and skips it entirely if it isn't configured at all.

    The Gemini Live voice session is NOT routed through this — it's a
    persistent bidirectional audio stream, not a one-shot completion, and it
    stays directly on Gemini. This module is for everything else: one-shot
    sub-task text generation such as actions/dev_agent.py and
    actions/code_helper.py use today via their own hand-rolled Claude/Gemini
    switch, and any future one-shot model call that wants a task-appropriate
    backend with a fallback instead of a single hardcoded choice.

USAGE
    from core.backend_router import TaskKind, complete

    result = complete(TaskKind.CODE_GEN, messages=[{"role": "user", "content": "..."}])
    # {"content": str, "tool_calls": list, "usage": dict, "backend": "claude"}

    # For callers using the .generate_content(prompt).text convention that
    # actions/dev_agent.py and actions/code_helper.py already share:
    model = get_text_model(TaskKind.CODE_GEN)
    model.generate_content("write a haiku").text

POLICY
    DEFAULT_POLICY maps each TaskKind to an ordered list of backend names.
    Pass a custom `policy` dict to complete() to override it per call (the
    ROUTING settings section builds one from user choices this way).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


class TaskKind(Enum):
    VOICE_TURN  = "voice_turn"
    INTENT      = "intent"
    CODE_GEN    = "code_gen"
    CODE_REVIEW = "code_review"
    SUMMARIZE   = "summarize"
    VISION      = "vision"
    CHAT        = "chat"


DEFAULT_POLICY: dict[TaskKind, list[str]] = {
    TaskKind.CODE_GEN:    ["claude", "ollama", "gemini"],
    TaskKind.CODE_REVIEW: ["claude", "ollama", "gemini"],
    TaskKind.INTENT:      ["ollama", "gemini", "claude"],
    TaskKind.SUMMARIZE:   ["ollama", "gemini", "claude"],
    TaskKind.VISION:      ["gemini"],
    TaskKind.VOICE_TURN:  ["gemini"],
    TaskKind.CHAT:        ["ollama", "gemini", "claude"],
}

BREAKER_COOLDOWN_S = 60.0
GEMINI_MODEL       = "gemini-flash-latest"

_lock           = threading.Lock()
_breaker_until: dict[str, float] = {}          # backend name -> monotonic time it's skipped until
_logged_trips:  set[str]         = set()       # backend names already logged for the CURRENT trip


def _base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _get_api_config() -> dict:
    try:
        return json.loads((_base_dir() / "config" / "api_keys.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _breaker_open(name: str) -> bool:
    with _lock:
        until = _breaker_until.get(name)
        if until is None:
            return False
        if time.monotonic() >= until:
            del _breaker_until[name]
            _logged_trips.discard(name)
            return False
        return True


def _trip_breaker(kind: TaskKind, name: str, error: Exception) -> None:
    with _lock:
        _breaker_until[name] = time.monotonic() + BREAKER_COOLDOWN_S
        already_logged = name in _logged_trips
        _logged_trips.add(name)
    if not already_logged:
        print(f"[Router] {kind.value} — {name} failed, skipping it for {BREAKER_COOLDOWN_S:.0f}s: {error}")


def reset_breakers() -> None:
    """Test/ops hook: clear all circuit-breaker state immediately."""
    with _lock:
        _breaker_until.clear()
        _logged_trips.clear()


def _is_configured(name: str) -> bool:
    if name == "ollama":
        return True   # no auth needed; reachability is checked at call time by the adapter itself
    if name == "claude":
        from core.claude_bridge import get_claude_config, get_claude_settings
        api_key, _, _ = get_claude_settings(get_claude_config())
        return bool(api_key)
    if name == "gemini":
        return bool(_get_api_config().get("gemini_api_key"))
    return False


# ── Adapters ──────────────────────────────────────────────────────────────
# Each adapter takes (messages, tools, images, timeout) — the same flat
# {"role", "content"} message shape core/llm_client.py's call_llm() and
# core/claude_bridge.py's call_claude() already use — and returns
# {"content", "tool_calls", "usage", "backend"}. Kept in a plain dict (not
# hardcoded into complete()) so tests can substitute fakes without touching
# real network/config, per tests/test_backend_router.py.

def _call_ollama(messages: list, tools: list | None, images: list | None, timeout: int) -> dict:
    if images:
        raise RuntimeError("ollama backend does not accept images")
    from core import llm_client
    if not llm_client.ensure_ollama_running(timeout=3):
        raise RuntimeError(f"Ollama unreachable at {llm_client.get_llm_settings()[0]}")
    resp = llm_client.call_llm(messages, tools, timeout=timeout)
    return {"content": resp.get("content", ""), "tool_calls": resp.get("tool_calls") or [],
            "usage": {}, "backend": "ollama"}


def _call_claude(messages: list, tools: list | None, images: list | None, timeout: int) -> dict:
    if images:
        raise RuntimeError("claude adapter does not accept images")
    from core import claude_bridge

    system = None
    msgs = messages
    if msgs and msgs[0].get("role") == "system":
        system, msgs = msgs[0].get("content"), msgs[1:]

    anthropic_tools = None
    if tools:
        from core.tool_schema import gemini_tools_to_anthropic
        # Gemini-shaped declarations (the convention everywhere else in this
        # codebase) have a bare "parameters" key; Anthropic tools already in
        # {"name", "input_schema"} shape pass straight through unconverted.
        anthropic_tools = (gemini_tools_to_anthropic(tools) if tools[0].get("parameters") is not None
                           else tools)

    resp = claude_bridge.call_claude(msgs, anthropic_tools, system=system, timeout=timeout)
    return {"content": resp.get("content", ""), "tool_calls": resp.get("tool_calls") or [],
            "usage": {}, "backend": "claude"}


def _messages_to_prompt(messages: list) -> str:
    """Gemini's one-shot generate_content() takes a prompt, not a chat-message
    list — flatten system/user/assistant turns into one block. Only the
    non-streaming, non-Live path uses this (VOICE_TURN stays on Live)."""
    parts = []
    for m in messages:
        role, content = m.get("role", "user"), m.get("content", "")
        if role == "system":
            parts.append(str(content))
        else:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _call_gemini(messages: list, tools: list | None, images: list | None, timeout: int) -> dict:
    api_key = _get_api_config().get("gemini_api_key")
    if not api_key:
        raise RuntimeError("no gemini_api_key configured")
    from google import genai

    client  = genai.Client(api_key=api_key)
    content = [_messages_to_prompt(messages)]
    for img_bytes, mime in (images or []):
        content.append({"inline_data": {"mime_type": mime, "data": img_bytes}})
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=content)
    return {"content": (getattr(resp, "text", None) or "").strip(), "tool_calls": [],
            "usage": {}, "backend": "gemini"}


_ADAPTERS: dict[str, Callable[[list, Optional[list], Optional[list], int], dict]] = {
    "ollama": _call_ollama,
    "claude": _call_claude,
    "gemini": _call_gemini,
}


def complete(
    kind:     TaskKind,
    messages: list,
    tools:    list | None = None,
    images:   list | None = None,
    timeout:  int = 60,
    policy:   dict[TaskKind, list[str]] | None = None,
) -> dict:
    """Tries each backend in the policy order for `kind` until one succeeds.
    A backend is skipped if its circuit breaker is open or it isn't
    configured. Raises RuntimeError only if every backend in the order was
    skipped or failed."""
    order = (policy or DEFAULT_POLICY).get(kind, ["ollama", "gemini", "claude"])
    last_error: Exception | None = None
    attempted: list[str] = []

    for name in order:
        if _breaker_open(name) or not _is_configured(name):
            continue
        adapter = _ADAPTERS.get(name)
        if adapter is None:
            continue
        attempted.append(name)
        try:
            return adapter(messages, tools, images, timeout)
        except Exception as e:
            last_error = e
            _trip_breaker(kind, name, e)

    if not attempted:
        raise RuntimeError(f"No backend available for {kind.value} (policy: {order}, all "
                            f"skipped — unconfigured or in cooldown)")
    raise RuntimeError(f"All backends failed for {kind.value} (tried {attempted}): {last_error}")


@dataclass
class _TextResult:
    text: str


class _RoutedTextModel:
    """`.generate_content(prompt).text` adapter over complete(), for callers
    using the small google-genai-shaped wrapper convention (see
    core.claude_bridge.ClaudeTextModel) instead of the chat-messages shape."""

    def __init__(self, kind: TaskKind, timeout: int = 120,
                 policy: dict[TaskKind, list[str]] | None = None):
        self.kind    = kind
        self.timeout = timeout
        self.policy  = policy

    def generate_content(self, contents) -> _TextResult:
        prompt = contents if isinstance(contents, str) else str(contents)
        result = complete(self.kind, [{"role": "user", "content": prompt}],
                           timeout=self.timeout, policy=self.policy)
        return _TextResult(result["content"])


def get_text_model(kind: TaskKind, timeout: int = 120,
                    policy: dict[TaskKind, list[str]] | None = None) -> _RoutedTextModel:
    return _RoutedTextModel(kind, timeout, policy)


def load_policy_from_config(raw: dict[str, str]) -> dict[TaskKind, list[str]]:
    """Builds a policy dict from the ROUTING settings section's saved values
    — `raw` is {task_kind.value: "backend1, backend2, ..."} (see main.py's
    _routing_settings_section). Blank or missing entries fall back to
    DEFAULT_POLICY for that kind; unrecognised backend names are dropped
    rather than raising, since a saved value should never crash a real call."""
    policy: dict[TaskKind, list[str]] = {}
    for kind in TaskKind:
        raw_value = (raw or {}).get(kind.value, "").strip()
        if not raw_value:
            policy[kind] = DEFAULT_POLICY[kind]
            continue
        names = [n.strip() for n in raw_value.split(",") if n.strip() in _ADAPTERS]
        policy[kind] = names or DEFAULT_POLICY[kind]
    return policy
