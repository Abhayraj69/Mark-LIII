"""
tool_connectors/connectors/mcp_connector.py — turns every configured Model
Context Protocol (MCP) server into a ToolConnector, so hundreds of existing
MCP servers (GitHub, Slack, Postgres, Home Assistant, browsers, ...) become
JARVIS tools without writing a line of connector code per server.

Configuration (config/api_keys.json → plugin_config.mcp_connector.mcp_servers):

    "mcp_connector": {
        "mcp_servers": [
            {"name": "filesystem", "transport": "stdio",
             "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\you\\Documents"]},
            {"name": "notes-api", "transport": "http",
             "url": "http://localhost:8931/mcp"}
        ]
    }

WHY ONE CLASS PER SERVER, BUILT AT IMPORT TIME
    tool_connectors/registry.py's discover() finds every ToolConnector
    subclass defined directly in this module and instantiates each with zero
    arguments — there's normally one class per hand-written connector (git,
    Docker, filesystem). An MCP server is user-configured and there can be
    any number of them, each needing its own command/URL. Rather than change
    discover()'s one-class-one-instance contract, _build_connector_classes()
    below reads the config once at import time and creates one small
    zero-arg subclass of _MCPServerConnector (tool_connectors/
    mcp_connector_base.py) per server, bound into THIS module's globals — so
    by the time discover() runs inspect.getmembers() on the already-executed
    module, it sees one ready-made class per server, indistinguishable from
    a hand-written one.
"""
from __future__ import annotations

from tool_connectors.mcp_connector_base import _MCPServerConnector

try:
    from memory.config_manager import get_plugin_config
except Exception:   # pragma: no cover — package importable outside the JARVIS tree
    def get_plugin_config(_namespace: str) -> dict:
        return {}


def _class_name_for(server_name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in server_name).strip("_") or "Server"
    return f"MCP_{safe[0].upper()}{safe[1:]}_Connector"


def _build_connector_classes() -> None:
    try:
        cfg = get_plugin_config("mcp_connector") or {}
        servers = cfg.get("mcp_servers") or []
    except Exception as e:
        print(f"[MCP] Failed to read mcp_connector config: {e}")
        return
    if not isinstance(servers, list):
        print("[MCP] plugin_config.mcp_connector.mcp_servers must be a list — ignoring.")
        return

    seen_names: set[str] = set()
    for entry in servers:
        if not isinstance(entry, dict):
            continue
        server_name = str(entry.get("name") or "").strip()
        if not server_name:
            print(f"[MCP] Skipping server entry with no 'name': {entry}")
            continue
        if server_name in seen_names:
            print(f"[MCP] Duplicate server name '{server_name}' — only the first is used.")
            continue
        seen_names.add(server_name)

        transport = str(entry.get("transport") or "stdio").strip()
        command   = entry.get("command")
        url       = entry.get("url")
        env       = entry.get("env") or {}

        def _init(self, _name=server_name, _transport=transport, _command=command, _url=url, _env=env):
            _MCPServerConnector.__init__(self, _name, _transport, _command, _url, _env)

        cls = type(_class_name_for(server_name), (_MCPServerConnector,), {
            "__init__":   _init,
            "__module__": __name__,
        })
        globals()[cls.__name__] = cls


_build_connector_classes()
