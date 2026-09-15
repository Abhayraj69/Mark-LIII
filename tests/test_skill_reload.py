"""Unit tests for hot reload — core/plugin_loader.py's and
core/action_loader.py's reload()/reload_all(), and core/skill_watcher.py's
change detection. Uses temp directories with real files (reload() re-imports
from disk), never touching the real plugins/ or actions/ folders."""

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.action_loader import discover_actions  # noqa: E402
from core.plugin_loader import discover_plugins  # noqa: E402
from core.skill_watcher import SkillWatcher  # noqa: E402


_PLUGIN_V1 = '''
PLUGIN = {"name": "greet", "description": "Says hello.",
          "parameters": {"type": "OBJECT", "properties": {}}}

def run(parameters, **kwargs):
    return "hello v1"
'''

_PLUGIN_V2 = '''
PLUGIN = {"name": "greet", "description": "Says hello, differently.",
          "parameters": {"type": "OBJECT", "properties": {}}}

def run(parameters, **kwargs):
    return "hello v2"
'''

_PLUGIN_BROKEN = '''
this is not valid python (((
'''

_ACTION_V1 = '''
TOOL = {"name": "count_things", "description": "Counts.",
        "parameters": {"type": "OBJECT", "properties": {}}, "handler": None}

def _handler(parameters, **kwargs):
    return "1"

TOOL["handler"] = _handler
'''

_ACTION_V2 = '''
TOOL = {"name": "count_things", "description": "Counts, better.",
        "parameters": {"type": "OBJECT", "properties": {}}, "handler": None}

def _handler(parameters, **kwargs):
    return "2"

TOOL["handler"] = _handler
'''


class TestPluginReload(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.plugins_dir = Path(self._tmp.name)
        self.path = self.plugins_dir / "greet.py"
        self.path.write_text(_PLUGIN_V1, encoding="utf-8")
        self.registry = discover_plugins(self.plugins_dir, core_tool_names=set(), logger=lambda m: None)

    def test_initial_discovery(self):
        self.assertTrue(self.registry.has("greet"))
        self.assertEqual(self.registry.run("greet", {}), "hello v1")

    def test_reload_picks_up_edit(self):
        self.path.write_text(_PLUGIN_V2, encoding="utf-8")
        ok, msg = self.registry.reload(self.path)
        self.assertTrue(ok, msg)
        self.assertEqual(self.registry.run("greet", {}), "hello v2")

    def test_reload_keeps_previous_version_on_broken_edit(self):
        self.path.write_text(_PLUGIN_BROKEN, encoding="utf-8")
        ok, msg = self.registry.reload(self.path)
        self.assertFalse(ok)
        self.assertIn("previous version kept", msg)
        # Old version still answers.
        self.assertEqual(self.registry.run("greet", {}), "hello v1")

    def test_reload_removal_unregisters(self):
        self.path.unlink()
        ok, msg = self.registry.reload(self.path)
        self.assertTrue(ok, msg)
        self.assertFalse(self.registry.has("greet"))

    def test_reload_all_covers_new_and_removed_files(self):
        second = self.plugins_dir / "second.py"
        second.write_text(_PLUGIN_V1.replace("greet", "greet2"), encoding="utf-8")
        self.path.unlink()

        results = self.registry.reload_all()
        names_ok = {name for name, ok, _ in results if ok}
        self.assertIn("second.py", names_ok)
        self.assertIn("greet.py", names_ok)   # the removal
        self.assertTrue(self.registry.has("greet2"))
        self.assertFalse(self.registry.has("greet"))


class TestActionReload(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.actions_dir = Path(self._tmp.name)
        self.path = self.actions_dir / "count_things.py"
        self.path.write_text(_ACTION_V1, encoding="utf-8")
        self.registry = discover_actions(self.actions_dir, reserved_names=set(), logger=lambda m: None)

    def test_initial_discovery(self):
        self.assertTrue(self.registry.has("count_things"))
        self.assertEqual(self.registry.run("count_things", {}), "1")

    def test_reload_picks_up_edit(self):
        self.path.write_text(_ACTION_V2, encoding="utf-8")
        ok, msg = self.registry.reload(self.path)
        self.assertTrue(ok, msg)
        self.assertEqual(self.registry.run("count_things", {}), "2")

    def test_reload_removing_tool_dict_unregisters(self):
        self.path.write_text("X = 1\n", encoding="utf-8")
        ok, msg = self.registry.reload(self.path)
        self.assertTrue(ok, msg)
        self.assertFalse(self.registry.has("count_things"))


class TestSkillWatcher(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.plugins_dir = Path(self._tmp.name) / "plugins"
        self.actions_dir = Path(self._tmp.name) / "actions"
        self.plugins_dir.mkdir()
        self.actions_dir.mkdir()
        self.path = self.plugins_dir / "greet.py"
        self.path.write_text(_PLUGIN_V1, encoding="utf-8")

        self.plugin_registry = discover_plugins(self.plugins_dir, core_tool_names=set(), logger=lambda m: None)
        self.action_registry = discover_actions(self.actions_dir, reserved_names=set(), logger=lambda m: None)
        self.changes: list[str] = []
        self.watcher = SkillWatcher(
            plugins_dir=self.plugins_dir, actions_dir=self.actions_dir,
            plugin_registry=self.plugin_registry, action_registry=self.action_registry,
            on_change=self.changes.append, logger=lambda m: None,
        )
        self.watcher._mtimes = self.watcher._snapshot()

    def test_unchanged_file_triggers_no_reload(self):
        self.watcher._poll_once()
        self.assertEqual(self.changes, [])
        self.assertEqual(self.plugin_registry.run("greet", {}), "hello v1")

    def test_edit_is_debounced_then_applied(self):
        self.path.write_text(_PLUGIN_V2, encoding="utf-8")
        os_stat_bump = time.time() + 1
        import os
        os.utime(self.path, (os_stat_bump, os_stat_bump))

        # First poll: change detected but not yet past the debounce window.
        self.watcher._poll_once()
        self.assertEqual(self.changes, [])
        self.assertEqual(self.plugin_registry.run("greet", {}), "hello v1")

        # Force the pending entry to look old enough, then poll again.
        for path in self.watcher._pending:
            self.watcher._pending[path] -= 10
        self.watcher._poll_once()

        self.assertEqual(self.changes, ["skills reloaded"])
        self.assertEqual(self.plugin_registry.run("greet", {}), "hello v2")


if __name__ == "__main__":
    unittest.main()
