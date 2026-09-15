"""
tool_connectors/mcp_connector_base.py — the ToolConnector implementation
shared by every configured MCP server.

WHY THIS IS ITS OWN FILE
    tool_connectors/registry.py's discover() finds every ToolConnector
    subclass DEFINED DIRECTLY IN a connectors/*.py module (checked via
    `obj.__module__ == module_name`) and instantiates each with zero
    arguments. _MCPServerConnector below needs per-server construction
    arguments (its transport, command/url, env), so it cannot be one of
    those auto-instantiated classes itself — see
    tool_connectors/connectors/mcp_connector.py, which builds one small
    zero-arg subclass of this per configured server instead. Keeping this
    base class in a separate module (rather than mcp_connector.py) is what
    keeps discover() from also trying to instantiate IT directly: its
    __module__ is "tool_connectors.mcp_connector_base", never
    "tool_connectors.connectors.mcp_connector", so the equality check in
    discover() naturally skips it.
"""
from __future__ import annotations

from typing import Optional

from tool_connectors.base import ActionSafety, Capability, ExecutionResult, ToolConnector, ToolExecutionError
from tool_connectors.mcp_client import MCPClient


def _safety_for(tool: dict) -> ActionSafety:
    """Maps MCP tool annotations onto ActionSafety — readOnlyHint means the
    registry can run it immediately, destructiveHint means it needs the same
    CONFIRM gate as shutdown_jarvis, and anything else defaults to
    REVERSIBLE (confirmation required, but not flagged as irreversible)."""
    annotations = tool.get("annotations") or {}
    if annotations.get("destructiveHint"):
        return ActionSafety.DESTRUCTIVE
    if annotations.get("readOnlyHint"):
        return ActionSafety.READ_ONLY
    return ActionSafety.REVERSIBLE


def _mcp_schema_to_capability_params(input_schema: dict | None) -> dict:
    """MCP's inputSchema is already JSON Schema — the same shape
    Capability.parameters (and core.tool_schema's conversion) expect — so
    this only needs to guard against a missing or malformed schema."""
    if isinstance(input_schema, dict) and input_schema.get("type") == "object":
        return input_schema
    return {"type": "object", "properties": {}}


class _MCPServerConnector(ToolConnector):
    """One instance per configured MCP server. `name` becomes the
    "{name}__{action}" tool-declaration prefix registry.py already uses for
    every connector, so two servers with different names never collide."""

    def __init__(self, server_name: str, transport: str, command: list[str] | None,
                 url: str | None, env: dict | None, timeout: float = 15.0):
        self.name          = f"mcp_{server_name}"
        self._server_name  = server_name
        self._transport    = transport
        self._command      = command
        self._url          = url
        self._env          = env
        self._timeout      = timeout
        self._client: Optional[MCPClient] = None
        self._tools_cache: Optional[list[dict]] = None

    def _ensure_client(self) -> MCPClient:
        if self._client is None:
            client = MCPClient(self._transport, command=self._command,
                                url=self._url, env=self._env, timeout=self._timeout)
            client.initialize()
            self._client = client
        return self._client

    def authenticate(self) -> bool:
        try:
            self._ensure_client()
            return True
        except Exception as e:
            print(f"[MCP:{self._server_name}] authenticate failed: {e}")
            return False

    def health_check(self) -> bool:
        """Never raises or crashes the process — an unreachable server is
        skipped with one log line, matching plugin_loader.py's crash-isolation
        stance, not treated as a fatal error for the whole registry."""
        try:
            self._ensure_client().list_tools()
            return True
        except Exception as e:
            print(f"[MCP:{self._server_name}] unreachable, skipping: {e}")
            return False

    def list_capabilities(self) -> list[Capability]:
        if self._tools_cache is None:
            try:
                self._tools_cache = self._ensure_client().list_tools()
            except Exception as e:
                print(f"[MCP:{self._server_name}] list_tools failed, publishing no capabilities: {e}")
                return []
        return [
            Capability(
                name=t["name"],
                description=t.get("description") or f"MCP tool from '{self._server_name}'.",
                safety=_safety_for(t),
                parameters=_mcp_schema_to_capability_params(t.get("inputSchema")),
            )
            for t in self._tools_cache if t.get("name")
        ]

    def execute(self, action: str, params: dict) -> ExecutionResult:
        try:
            result = self._ensure_client().call_tool(action, params)
        except Exception as e:
            raise ToolExecutionError(self.name, action, str(e), original=e) from e

        text_parts = []
        for block in (result.get("content") or []):
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype in ("image", "audio"):
                text_parts.append(f"[{btype} content returned — not shown]")
            elif btype == "resource":
                uri = (block.get("resource") or {}).get("uri", "")
                text_parts.append(f"[resource: {uri}]" if uri else "[resource content returned]")

        message = "\n".join(p for p in text_parts if p).strip() or "Done."
        return ExecutionResult(success=not bool(result.get("isError")), output=result, message=message)
