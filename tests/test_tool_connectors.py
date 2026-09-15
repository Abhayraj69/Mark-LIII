"""Smoke tests for the Tool Connector Layer: discovery, safety-tier gating,
and the filesystem sandbox boundary."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool_connectors.base import ActionSafety, Capability, ExecutionResult, ToolConnector, ToolExecutionError  # noqa: E402
from tool_connectors.connectors.filesystem_connector import FileSystemConnector  # noqa: E402
from tool_connectors.registry import ToolRegistry  # noqa: E402


class _StubReadOnly(ToolConnector):
    name = "stub_ro"

    def authenticate(self) -> bool:
        return True

    def health_check(self) -> bool:
        return True

    def list_capabilities(self) -> list[Capability]:
        return [Capability("ping", "Replies pong.", ActionSafety.READ_ONLY)]

    def execute(self, action: str, params: dict) -> ExecutionResult:
        if action == "ping":
            return ExecutionResult(True, "pong", "pong")
        raise ToolExecutionError(self.name, action, "unknown action")


class _StubReversible(ToolConnector):
    name = "stub_rev"

    def __init__(self):
        self.ran = False

    def authenticate(self) -> bool:
        return True

    def health_check(self) -> bool:
        return True

    def list_capabilities(self) -> list[Capability]:
        return [Capability("mutate", "Changes something.", ActionSafety.REVERSIBLE)]

    def execute(self, action: str, params: dict) -> ExecutionResult:
        self.ran = True
        return ExecutionResult(True, "mutated", "mutated")


class TestReadOnlyExecutesImmediately(unittest.TestCase):
    def test_read_only_runs_without_confirmation(self):
        registry = ToolRegistry()
        registry.register(_StubReadOnly())
        result = registry.execute("stub_ro", "ping", {})
        self.assertEqual(result, "pong")

    def test_unknown_action_raises_tool_execution_error_shape(self):
        registry = ToolRegistry()
        registry.register(_StubReadOnly())
        result = registry.execute("stub_ro", "nonexistent", {})
        self.assertIn("failed", result)


class TestReversibleRequiresConfirmation(unittest.TestCase):
    def test_reversible_does_not_run_inline(self):
        connector = _StubReversible()
        registry = ToolRegistry()
        registry.register(connector)
        registry.execute("stub_rev", "mutate", {})
        # Without a bound UI, core.confirm.request() refuses rather than
        # running the callable — either way, mutate() must not have executed
        # synchronously inside execute().
        self.assertFalse(connector.ran)


class TestCapabilityPublishing(unittest.TestCase):
    def test_declarations_are_namespaced_and_tagged(self):
        registry = ToolRegistry()
        registry.register(_StubReadOnly())
        declarations = registry.get_tool_declarations()
        self.assertEqual(len(declarations), 1)
        self.assertEqual(declarations[0]["name"], "stub_ro__ping")
        self.assertIn("read_only", declarations[0]["description"])

    def test_reserved_names_are_excluded_from_declarations(self):
        registry = ToolRegistry()
        registry.register(_StubReadOnly())
        declarations = registry.get_tool_declarations(reserved_names={"stub_ro__ping"})
        self.assertEqual(declarations, [])

    def test_has_declaration_reverse_maps_to_connector_and_action(self):
        registry = ToolRegistry()
        registry.register(_StubReadOnly())
        self.assertEqual(registry.has_declaration("stub_ro__ping"), ("stub_ro", "ping"))
        self.assertIsNone(registry.has_declaration("stub_ro__nonexistent"))
        self.assertIsNone(registry.has_declaration("no_dunder_here"))
        self.assertIsNone(registry.has_declaration("unregistered__ping"))


class TestFilesystemSandbox(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        (self.root / "inside.txt").write_text("hello from inside", encoding="utf-8")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "nested.txt").write_text("needle in a haystack", encoding="utf-8")
        # A sibling directory OUTSIDE the sandboxed root, to prove escape is blocked.
        self.outside = self.root.parent / f"{self.root.name}_outside_secret.txt"
        self.outside.write_text("should never be reachable", encoding="utf-8")
        self.connector = FileSystemConnector(project_root=self.root)

    def tearDown(self):
        self._tmpdir.cleanup()
        self.outside.unlink(missing_ok=True)

    def test_read_file_inside_root(self):
        result = self.connector.execute("read_file", {"path": "inside.txt"})
        self.assertEqual(result.output, "hello from inside")

    def test_path_traversal_is_rejected(self):
        with self.assertRaises(ToolExecutionError):
            self.connector.execute("read_file", {"path": f"../{self.outside.name}"})

    def test_absolute_path_escape_is_rejected(self):
        with self.assertRaises(ToolExecutionError):
            self.connector.execute("read_file", {"path": str(self.outside)})

    def test_search_finds_needle_in_subdirectory(self):
        result = self.connector.execute("search_files", {"query": "needle"})
        self.assertTrue(any("nested.txt" in hit[0] for hit in result.output))


if __name__ == "__main__":
    unittest.main()
