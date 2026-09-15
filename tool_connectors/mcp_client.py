"""
tool_connectors/mcp_client.py — minimal JSON-RPC 2.0 client for Model
Context Protocol (MCP) servers, over stdio or streamable HTTP.

WHY A HAND-ROLLED CLIENT
    No official MCP SDK is added as a dependency here — core/claude_bridge.py
    already set this codebase's precedent for a wire-protocol integration:
    talk to it directly with requests + stdlib rather than pull in a client
    library for one endpoint shape. MCP's handshake is three JSON-RPC calls
    (initialize, tools/list, tools/call) plus one notification, framed either
    as one JSON object per line over stdin/stdout, or as a plain HTTP POST
    returning one JSON object — small enough to implement directly.

STDIO FRAMING
    The server is spawned as a subprocess; each request is one JSON object
    followed by "\\n" written to its stdin, and each response is one JSON
    object read as a line from its stdout. A background reader thread feeds
    lines into a queue so _request() can apply a real timeout even though
    subprocess pipes have no native one on Windows.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from typing import Optional

import requests

MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPError(Exception):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


class MCPClient:
    """One connection to one MCP server. Call initialize() once before
    list_tools()/call_tool(). Not thread-safe — callers serialize their own
    access (matches how ToolConnector.execute() is invoked, one call at a
    time through tool_connectors/registry.py)."""

    def __init__(self, transport: str, command: list[str] | None = None,
                 url: str | None = None, env: dict | None = None, timeout: float = 15.0):
        if transport not in ("stdio", "http"):
            raise ValueError(f"unknown MCP transport: {transport!r} (expected 'stdio' or 'http')")
        self.transport = transport
        self.timeout   = timeout
        self._id       = 0

        if transport == "stdio":
            if not command:
                raise ValueError("stdio transport requires 'command'")
            full_env = {**os.environ, **(env or {})}
            self._proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1, env=full_env,
            )
            self._out_q: "queue.Queue[str]" = queue.Queue()
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()
        else:
            if not url:
                raise ValueError("http transport requires 'url'")
            self.url = url.rstrip("/")

    def _read_loop(self) -> None:
        try:
            for line in self._proc.stdout:
                if line.strip():
                    self._out_q.put(line)
        except Exception:
            pass   # process died or pipe closed — _request()'s own timeout surfaces this

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _send(self, payload: dict) -> None:
        if self.transport == "stdio":
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except Exception as e:
                raise MCPError(f"failed to write to MCP server: {e}") from e
        else:
            try:
                resp = requests.post(self.url, json=payload, timeout=self.timeout,
                                      headers={"content-type": "application/json"})
                resp.raise_for_status()
                self._last_http_response = resp
            except requests.exceptions.RequestException as e:
                raise MCPError(f"MCP HTTP request failed: {e}") from e

    def _request(self, method: str, params: dict | None = None) -> dict:
        req_id = self._next_id()
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        self._send(payload)

        if self.transport == "http":
            try:
                msg = self._last_http_response.json()
            except ValueError as e:
                raise MCPError(f"MCP server returned non-JSON: {e}") from e
            return self._unwrap(msg)

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                line = self._out_q.get(timeout=remaining)
            except queue.Empty:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == req_id:
                return self._unwrap(msg)
            # a notification, or a response to a request we're not waiting on — ignore and keep reading
        raise MCPError(f"timed out waiting for a response to '{method}'")

    @staticmethod
    def _unwrap(msg: dict) -> dict:
        if "error" in msg and msg["error"] is not None:
            err = msg["error"]
            raise MCPError(err.get("message", "unknown MCP error"), code=err.get("code"))
        return msg.get("result") or {}

    def _notify(self, method: str, params: dict | None = None) -> None:
        """A JSON-RPC notification — no id, no response expected. MCP requires
        one right after a successful initialize()."""
        try:
            self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})
        except MCPError:
            pass   # notifications are best-effort; a dead server surfaces on the next real request

    def initialize(self) -> dict:
        result = self._request("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "mk3-jarvis", "version": "1.0"},
        })
        self._notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict]:
        return self._request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self._request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        if self.transport == "stdio":
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass
