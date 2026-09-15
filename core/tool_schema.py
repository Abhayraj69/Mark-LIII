"""
Converts Gemini's function-declaration schema — the shape used everywhere
else in this codebase (main.py's TOOL_DECLARATIONS, ActionRegistry and
PluginRegistry's get_tool_declarations()) — into the wire format each
tool-calling backend expects:

  gemini_tools_to_openai()    → OpenAI-style `tools` array (Ollama's
                                 /api/chat, any OpenAI-compatible server).
  gemini_tools_to_anthropic() → Anthropic Messages API `tools` array
                                 (core/claude_bridge.py).

Local Mode (main.py: JarvisLive._run_local_loop) and the Claude bridge both
dispatch against the exact same tool registry Gemini Live uses; only the wire
format the model sees differs, so this is a pure format translation with no
behavioural logic of its own.
"""
from __future__ import annotations

# Gemini's parameter schema spells JSON-Schema types in UPPERCASE
# ("OBJECT", "STRING", …); OpenAI/JSON-Schema wants them lowercase.
_TYPE_MAP = {
    "OBJECT": "object", "STRING": "string", "NUMBER": "number",
    "INTEGER": "integer", "BOOLEAN": "boolean", "ARRAY": "array",
}


def _convert_schema(node):
    """Recursively lower-case `type` values and walk into `properties` /
    `items`; everything else is passed through unchanged."""
    if not isinstance(node, dict):
        return node
    out = dict(node)
    t = out.get("type")
    if isinstance(t, str):
        out["type"] = _TYPE_MAP.get(t.upper(), t.lower())
    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: _convert_schema(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _convert_schema(out["items"])
    return out


def gemini_tools_to_openai(tools: list[dict]) -> list[dict]:
    """`tools` is the same list shape passed into Gemini's
    `function_declarations` — main.py's TOOL_DECLARATIONS plus whatever
    ActionRegistry / PluginRegistry .get_tool_declarations() return. Returns
    the equivalent OpenAI/Ollama `tools=[...]` array."""
    out = []
    for t in tools or []:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        out.append({
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t.get("description", ""),
                "parameters":  _convert_schema(
                    t.get("parameters") or {"type": "object", "properties": {}}
                ),
            },
        })
    return out


def gemini_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """`tools` is the same list shape passed into Gemini's
    `function_declarations` — main.py's TOOL_DECLARATIONS plus whatever
    ActionRegistry / PluginRegistry .get_tool_declarations() return. Returns
    the equivalent Anthropic Messages API `tools=[...]` array — Anthropic
    wants each entry flat, as {"name", "description", "input_schema"}, with
    no wrapper object (unlike OpenAI's nested "function" key above)."""
    out = []
    for t in tools or []:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        out.append({
            "name":        t["name"],
            "description": t.get("description", ""),
            "input_schema": _convert_schema(
                t.get("parameters") or {"type": "object", "properties": {}}
            ),
        })
    return out
