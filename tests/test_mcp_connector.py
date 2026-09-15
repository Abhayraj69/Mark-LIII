"""Unit tests for the MCP client/connector — tool_connectors/mcp_client.py
and tool_connectors/mcp_connector_base.py. Spawns a tiny fake MCP server
(a standalone Python script, no third-party MCP SDK) over stdio so this
exercises the real JSON-RPC wire protocol, not a mocked transport."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool_connectors.base import ActionSafety  # noqa: E402
from tool_connectors.mcp_client import MCPClient  # noqa: E402
from tool_connectors.mcp_connector_base import _MCPServerConnector  # noqa: E402
from tool_connectors.registry import ToolRegistry  # noqa: E402

_FAKE_SERVER = r'''
import json
import sys

MARKER = sys.argv[1]

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

TOOLS = [
    {
        "name": "read_thing",
        "description": "Reads a thing.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "delete_thing",
        "description": "Deletes a thing.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"destructiveHint": True},
    },
]

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    method = req.get("method")
    req_id = req.get("id")

    if req_id is None:
        continue  # a notification — nothing to respond to

    if method == "initialize":
        send({"jsonrpc": "2.0", "id": req_id,
              "result": {"protocolVersion": "2024-11-05", "serverInfo": {"name": "fake"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        name = (req.get("params") or {}).get("name")
        if name == "delete_thing":
            with open(MARKER, "w", encoding="utf-8") as f:
                f.write("called")
            send({"jsonrpc": "2.0", "id": req_id,
                  "result": {"content": [{"type": "text", "text": "deleted"}]}})
        else:
            send({"jsonrpc": "2.0", "id": req_id,
                  "result": {"content": [{"type": "text", "text": "read ok"}]}})
    else:
        send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "no such method"}})
'''


class TestMCPClientAndConnector(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self._tmpdir.name)
        self.server_path = tmp / "fake_mcp_server.py"
        self.server_path.write_text(_FAKE_SERVER, encoding="utf-8")
        self.marker_path = tmp / "delete_called.marker"
        self.command = [sys.executable, str(self.server_path), str(self.marker_path)]

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_client_initialize_list_call(self):
        client = MCPClient("stdio", command=self.command, timeout=10)
        try:
            result = client.initialize()
            self.assertEqual(result["protocolVersion"], "2024-11-05")

            tools = client.list_tools()
            self.assertEqual({t["name"] for t in tools}, {"read_thing", "delete_thing"})

            call = client.call_tool("read_thing", {})
            self.assertEqual(call["content"][0]["text"], "read ok")
        finally:
            client.close()

    def test_connector_maps_annotations_to_safety(self):
        connector = _MCPServerConnector("fake", "stdio", self.command, None, None, timeout=10)
        try:
            caps = {c.name: c for c in connector.list_capabilities()}
            self.assertEqual(caps["read_thing"].safety, ActionSafety.READ_ONLY)
            self.assertEqual(caps["delete_thing"].safety, ActionSafety.DESTRUCTIVE)
        finally:
            connector._client and connector._client.close()

    def test_read_only_capability_executes_directly(self):
        connector = _MCPServerConnector("fake", "stdio", self.command, None, None, timeout=10)
        try:
            registry = ToolRegistry(logger=lambda _msg: None)
            registry.register(connector)
            result = registry.execute("mcp_fake", "read_thing", {})
            self.assertIn("read ok", result)
        finally:
            connector._client and connector._client.close()

    def test_destructive_capability_is_refused_without_confirmation(self):
        """No UI is bound in this test process (core.confirm._show_cb is None),
        so the registry must refuse to run the destructive tool at all —
        exactly the same refusal path shutdown_jarvis and every other
        DESTRUCTIVE action already goes through."""
        connector = _MCPServerConnector("fake", "stdio", self.command, None, None, timeout=10)
        try:
            registry = ToolRegistry(logger=lambda _msg: None)
            registry.register(connector)
            result = registry.execute("mcp_fake", "delete_thing", {})
            self.assertIn("cannot confirm", result.lower())
            self.assertFalse(self.marker_path.exists(), "delete_thing must not have actually run")
        finally:
            connector._client and connector._client.close()


if __name__ == "__main__":
    unittest.main()
