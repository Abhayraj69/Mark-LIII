# Tool Connector Layer

Lets JARVIS call specialized project tools (git, Docker, the local
filesystem, a task tracker, ...) directly instead of asking the user to run
them by hand — while keeping every non-read-only action behind the same
human-in-the-loop confirmation gate the rest of the app already uses
([`core/confirm.py`](../core/confirm.py)).

## How it fits together

```
tool_connectors/
  base.py        ToolConnector ABC, ActionSafety, Capability, ExecutionResult, ToolExecutionError
  registry.py     ToolRegistry — discovery, capability publishing, safety-gated execution, audit logging
  audit.py        SQLite audit trail (memory/tool_connector_audit.db)
  connectors/
    git_connector.py         status / diff / branch_info (read-only), commit (reversible)
    docker_connector.py      container_status / container_logs (read-only only — no lifecycle actions)
    filesystem_connector.py  read_file / list_dir / search_files, sandboxed to the project root
    mcp_connector.py         one connector per configured Model Context Protocol server (see below)
  mcp_client.py           minimal JSON-RPC 2.0 client (stdio + streamable HTTP), no MCP SDK
  mcp_connector_base.py   the ToolConnector every configured MCP server becomes an instance of
```

`ToolRegistry.discover()` imports every `*.py` in `connectors/` and
instantiates any `ToolConnector` subclass it finds — the same "drop a file
in, it just appears" model `core/action_loader.py` uses for `actions/*.py`.
Each connector's `list_capabilities()` is published through
`get_tool_declarations()` as flat function-calling schemas
(`{"name": "git__commit", "description": "...", "parameters": {...}}`), so
the planning/reasoning layer sees a connector action exactly like any other
tool.

## Safety tiers

Every `Capability` is tagged `READ_ONLY`, `REVERSIBLE`, or `DESTRUCTIVE`
(`tool_connectors.base.ActionSafety`):

- **READ_ONLY** runs immediately when `ToolRegistry.execute()` is called.
- **REVERSIBLE** / **DESTRUCTIVE** never run inline. `execute()` instead
  calls `core.confirm.request()` — the same gate `shutdown_jarvis` uses —
  and returns a sentence for the model to say out loud
  (`"[CONFIRMATION_PENDING] ..."`). The action only actually runs if a human
  presses CONFIRM on the HUD; a model can't forge that by passing a
  `confirmed=true` parameter, because there is no such parameter to forge.

This means a connector's *capability declaration* is what decides whether an
action can auto-run — not a runtime check inside the handler — so get the
tier right when you add one.

## Auditability

Every execution attempt is recorded to `memory/tool_connector_audit.db`
(table `tool_connector_events`) via `tool_connectors/audit.py`, including:

- the connector, action, safety tier, and (JSON) params
- the output or `ToolExecutionError.human_message` on failure
- `stage`: `"confirmation_requested"` for a REVERSIBLE/DESTRUCTIVE action
  parked behind the gate, `"executed"` once it actually ran — so the trail
  shows a proposed-but-cancelled action too, not just completed ones.

## Error handling

Connectors never let a raw exception escape `execute()` — they raise
`ToolExecutionError(connector, action, message)` instead, and the registry
wraps anything that leaks through anyway. `ToolExecutionError.human_message`
is always a single sentence safe to speak back to the user
(`"git.commit failed: nothing to commit"`), so the assistant never has to
improvise a phrasing for a stack trace.

## Adding a new connector

1. Create `tool_connectors/connectors/my_tool_connector.py`.
2. Subclass `ToolConnector`, set a unique `name`, and implement:
   - `authenticate() -> bool` — verify/establish whatever the tool needs;
     return `True` unconditionally if there's nothing to authenticate.
   - `health_check() -> bool` — cheap liveness probe (binary on PATH,
     directory exists, daemon reachable). Called before `execute()` runs
     the real thing in most connectors' own `execute()`.
   - `list_capabilities() -> list[Capability]` — one entry per action, each
     with an honest `ActionSafety` tier and a `parameters` schema shaped
     like a Gemini function-declaration (`{"type": "OBJECT", "properties": {...}}`).
   - `execute(action: str, params: dict) -> ExecutionResult` — dispatch on
     `action`, raise `ToolExecutionError` for anything unknown or failed.
3. Nothing else to register — `ToolRegistry.discover()` (called by
   `build_default_registry()`) finds it automatically on next startup.

```python
from tool_connectors.base import ActionSafety, Capability, ExecutionResult, ToolConnector, ToolExecutionError

class MyToolConnector(ToolConnector):
    name = "my_tool"

    def authenticate(self) -> bool:
        return True

    def health_check(self) -> bool:
        return True  # e.g. check the binary/daemon is reachable

    def list_capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="do_thing",
                description="Does the thing.",
                safety=ActionSafety.READ_ONLY,
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    def execute(self, action: str, params: dict) -> ExecutionResult:
        if action == "do_thing":
            return ExecutionResult(True, "ok", "Did the thing.")
        raise ToolExecutionError(self.name, action, f"Unknown action '{action}'.")
```

## Using the registry

```python
from tool_connectors import build_default_registry

registry = build_default_registry()
declarations = registry.get_tool_declarations()   # hand these to the model
result = registry.execute("git", "status", {})     # READ_ONLY: runs now
pending = registry.execute("git", "commit", {"message": "wip"})  # returns a
                                                                  # CONFIRMATION_PENDING sentence
```

## MCP servers

`tool_connectors/connectors/mcp_connector.py` turns any configured [Model
Context Protocol](https://modelcontextprotocol.io) server into a
`ToolConnector` — no code needed per server, just config:

```json
{
  "plugin_config": {
    "mcp_connector": {
      "mcp_servers": [
        {
          "name": "filesystem",
          "transport": "stdio",
          "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\you\\Documents"]
        },
        {
          "name": "notes-api",
          "transport": "http",
          "url": "http://localhost:8931/mcp"
        }
      ]
    }
  }
}
```

Each server becomes a connector named `mcp_<name>`, publishing its
`tools/list` output as `mcp_<name>__<tool>` capabilities. MCP tool
`annotations` map onto `ActionSafety`: `readOnlyHint` → `READ_ONLY` (runs
immediately), `destructiveHint` → `DESTRUCTIVE` (goes through the CONFIRM
gate), anything else → `REVERSIBLE` (also gated, just not flagged as
irreversible). An unreachable server is skipped with one log line — never a
crash — exactly like a plugin that fails to import.

Edit the list from ⚙ → PLUGIN SETTINGS → 🔌 MCP SERVERS (raw JSON, same shape
as above) and use CHECK SERVER HEALTH to re-discover and ping every
configured server without restarting the app.

## Wired into the live model session

`main.py` builds one `ToolRegistry` at startup (`self._tool_connector_registry`,
alongside `self._action_registry` and `self._plugin_registry`) and publishes
its capabilities the same way actions and plugins are published:

- Declaration names are checked against every inline tool, action, and plugin
  name already claimed (`get_tool_declarations(reserved_names=...)`); a
  collision is logged and that one capability is dropped rather than
  silently shadowing (or being shadowed by) an existing tool. In practice
  this only bites if something else is literally named e.g. `git__commit` —
  the `{connector}__{action}` shape keeps connector names out of the flat
  tool namespace everything else lives in.
- `_dispatch_tool`'s fallback chain (`core/action_loader` → plugins →
  connectors → "Unknown tool") reverse-maps a called tool name back to
  `(connector, action)` via `ToolRegistry.has_declaration()` and calls
  `ToolRegistry.execute()` — so a connector action goes through the exact
  same safety gating (`READ_ONLY` runs inline, `REVERSIBLE`/`DESTRUCTIVE`
  returns a `"[CONFIRMATION_PENDING]"` sentence via `core.confirm`) whether
  it was called from Gemini Live or Local Mode's text tool-calling loop,
  since both share `_dispatch_tool`.
- Which registry "wins" a name never has to be decided at runtime: actions
  and plugins are discovered first and reserve their names before connectors
  are, so the reservation is resolved once at startup, not per call.
