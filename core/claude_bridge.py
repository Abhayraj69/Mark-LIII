"""
core/claude_bridge.py — Anthropic Messages API bridge, the Claude-backed twin
of core/llm_client.py's Ollama/OpenAI-compatible path.

WHY RAW REQUESTS INSTEAD OF THE `anthropic` SDK
    Every other LLM backend in this codebase (Ollama, LM Studio, and any other
    OpenAI-compatible server) talks over plain HTTP via `requests` — no SDK,
    just a REST call. The Messages API is exactly as simple (one endpoint, one
    header pair), so pulling in the `anthropic` package would add a dependency
    used in exactly one file for something `requests` — already a hard
    dependency — already does. Same reasoning core/context_manager.py and
    core/predictive_assistant.py already used to skip a vector-DB / ML
    dependency for something a few dozen lines can do directly.

CONFIG
    Settings live under the "claude_engine" plugin_config namespace, read
    through memory.config_manager the same way "local_engine" is — see
    core/llm_client.py's _load_config() for the identical pattern. Nothing
    here is wired into main.py's live dispatch yet (that's Phase 2); this
    module is a standalone, independently testable bridge.

NORMALIZED RETURN SHAPE
    call_claude() returns {"content": str, "tool_calls": [...]} in exactly
    the shape core/llm_client.py's call_llm() returns, so a dispatch loop
    written against one backend needs no changes to run against the other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()

API_URL            = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION  = "2023-06-01"

_DEFAULTS = {
    "model":      "claude-sonnet-5",
    "max_tokens": 1024,
}

# One keep-alive connection for the process instead of a fresh TCP+TLS
# handshake per call — dev_agent/code_helper can fire several requests in a
# single planner->writer sequence, and config_manager's own file cache
# (memory/config_manager.py) is per-load, not per-connection, so this is the
# one pooling layer that has to live here.
_session = requests.Session()


def _load_config() -> dict:
    # Same soft-dependency pattern as core/llm_client.py:_load_config() — the
    # namespaced plugin_config store is the source of truth; a bare read
    # attempt against api_keys.json only matters if config_manager itself
    # can't be imported (e.g. this module run standalone outside the app).
    try:
        from memory.config_manager import get_plugin_config
        cfg = get_plugin_config("claude_engine")
        if cfg:
            return cfg
    except Exception:
        pass
    return {}


def get_claude_config() -> dict:
    """Public entry point for callers (actions/dev_agent.py, actions/code_helper.py)
    that want to load the config once and reuse it across is_claude_engine_enabled()
    and ClaudeTextModel(cfg=...) instead of triggering _load_config() per call."""
    return _load_config()


def is_claude_engine_enabled(cfg: dict | None = None) -> bool:
    return bool((cfg if cfg is not None else _load_config()).get("enabled", False))


def get_claude_settings(cfg: dict | None = None) -> tuple[str, str, int]:
    """Returns (api_key, model, max_tokens)."""
    cfg = cfg if cfg is not None else _load_config()
    api_key = (cfg.get("api_key") or "").strip()
    model = cfg.get("model") or _DEFAULTS["model"]
    try:
        max_tokens = int(float(cfg.get("max_tokens", _DEFAULTS["max_tokens"])))
    except (TypeError, ValueError):
        max_tokens = _DEFAULTS["max_tokens"]
    return api_key, model, max(1, max_tokens)


def to_claude_messages(history: list[dict]) -> list[dict]:
    """
    Converts the flat {"role": "system"|"user"|"assistant"|"tool", "content": ...}
    history shape used elsewhere in this codebase (core/llm_client.py's
    call_llm(), main.py's `messages` list in _run_local_loop) into Anthropic's
    content-block message list.

    Three shape differences Anthropic's API forces:
      - No "system" role inside `messages` — system prompt is a separate
        top-level `system` parameter, so system-role entries are dropped here
        and must be passed to call_claude(system=...) instead.
      - No "tool" role — a tool result is a "user" message containing a
        {"type": "tool_result", "tool_use_id": ...} content block.
      - An assistant message that called a tool carries {"type": "tool_use"}
        content blocks instead of a sibling "tool_calls" key.
    """
    out: list[dict] = []
    for msg in history:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "user":
            out.append({"role": "user", "content": msg.get("content", "")})
        elif role == "assistant":
            blocks = []
            text = msg.get("content") or ""
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                blocks.append({
                    "type":  "tool_use",
                    "id":    tc.get("id", ""),
                    "name":  fn.get("name", ""),
                    "input": fn.get("arguments") or {},
                })
            out.append({"role": "assistant", "content": blocks or ""})
        elif role == "tool":
            out.append({
                "role": "user",
                "content": [{
                    "type":        "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content":     str(msg.get("content", "")),
                }],
            })
    return out


def call_claude(
    messages: list[dict],
    tools:    list[dict] | None = None,
    system:   str | None = None,
    timeout:  int = 120,
    cfg:      dict | None = None,
) -> dict:
    """
    Non-streaming call to the Anthropic Messages API.

    `messages` uses the SAME flat {"role", "content"} shape core/llm_client.py's
    call_llm() accepts — converted internally via to_claude_messages() so a
    caller doesn't need to hold two different history formats depending on
    which backend it's talking to.

    `tools` must already be in Anthropic's {"name", "description",
    "input_schema"} shape — see core/tool_schema.py:gemini_tools_to_anthropic().

    Returns the SAME normalized shape as core/llm_client.py's call_llm():
        {"content": str, "tool_calls": [{"id", "function": {"name","arguments"}}]}
    """
    cfg = cfg if cfg is not None else _load_config()
    api_key, model, max_tokens = get_claude_settings(cfg)
    if not api_key:
        raise RuntimeError(
            "No Claude API key configured. Set one under "
            "⚙ → PLUGIN SETTINGS → CLAUDE ENGINE, or directly in "
            "config/api_keys.json → plugin_config.claude_engine.api_key."
        )

    payload: dict = {
        "model":      model,
        "max_tokens": max_tokens,
        "messages":   to_claude_messages(messages),
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = tools

    headers = {
        "x-api-key":         api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type":      "application/json",
    }

    try:
        resp = _session.post(API_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError("Claude API request timed out.")
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        raise RuntimeError(f"Claude API HTTP error: {e.response.status_code} {detail}".strip())
    except Exception as e:
        raise RuntimeError(f"Claude API call failed: {e}")

    data = resp.json()
    content_blocks = data.get("content") or []

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in content_blocks:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "function": {
                    "name":      block.get("name", ""),
                    "arguments": block.get("input") or {},
                },
            })

    return {
        "content":    "".join(text_parts).strip(),
        "tool_calls": tool_calls,
    }


def call_claude_text(
    prompt: str,
    system: str | None = None,
    timeout: int = 120,
    cfg: dict | None = None,
) -> str:
    """
    Simple text-only generation (no tools) — the Claude-backed twin of
    core/llm_client.py's call_llm_text(). Delegates to call_claude() rather
    than duplicating the HTTP/error handling; callers that just want a
    single prompt->text round trip (actions/dev_agent.py, actions/code_helper.py)
    don't need to build a tool-calling loop for it.

    Pass `cfg` when the caller already loaded it (e.g. to check
    is_claude_engine_enabled()) so this doesn't trigger a second
    _load_config() round trip for the same logical request.
    """
    result = call_claude([{"role": "user", "content": prompt}], system=system, timeout=timeout, cfg=cfg)
    return result["content"]


class _SimpleResponse:
    """Mimics the one attribute (`.text`) every call site in this codebase
    reads off a google-genai response object."""

    def __init__(self, text: str):
        self.text = text


class ClaudeTextModel:
    """
    Drop-in replacement for the tiny google-genai wrapper classes actions/
    dev_agent.py and actions/code_helper.py build around `genai.Client` — same
    `.generate_content(prompt).text` shape, backed by Claude instead of
    Gemini. Swapping the model object at the call site (behind
    is_claude_engine_enabled()) means none of the surrounding prompt-building,
    retry, or JSON-parsing logic in those two actions needs to know or care
    which backend actually answered.
    """

    def __init__(self, cfg: dict | None = None):
        # Callers that already loaded the plugin config to check
        # is_claude_engine_enabled() can pass it here so each generate_content()
        # call doesn't re-trigger _load_config() for the same config.
        self._cfg = cfg

    def generate_content(self, contents) -> _SimpleResponse:
        prompt = contents if isinstance(contents, str) else str(contents)
        return _SimpleResponse(call_claude_text(prompt, cfg=self._cfg))
