"""Unit tests for core/sequence_memory.py — the permanent multi-step
sequence store — and actions/sequence_recall.py's tool handler on top of it."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.sequence_memory import (  # noqa: E402
    delete_sequence,
    get_sequence,
    list_sequences,
    record_run,
    save_sequence,
)
import actions.sequence_recall as sr  # noqa: E402


class TestSequenceMemory(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.db = Path(self._tmp) / "sequences.db"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_save_and_get_roundtrip(self):
        msg = save_sequence(
            "morning",
            [{"tool": "open_app", "args": {"name": "chrome"}}, {"tool": "web_search", "args": {"query": "news"}}],
            description="Start the day",
            db_path=self.db,
        )
        self.assertIn("Saved", msg)

        seq = get_sequence("morning", db_path=self.db)
        self.assertIsNotNone(seq)
        self.assertEqual(seq.description, "Start the day")
        self.assertEqual(len(seq.steps), 2)
        self.assertEqual(seq.steps[0].tool, "open_app")
        self.assertEqual(seq.steps[0].args, {"name": "chrome"})

    def test_save_overwrites_existing(self):
        save_sequence("x", [{"tool": "a"}], db_path=self.db)
        msg = save_sequence("x", [{"tool": "a"}, {"tool": "b"}], db_path=self.db)
        self.assertIn("Updated", msg)
        seq = get_sequence("x", db_path=self.db)
        self.assertEqual(len(seq.steps), 2)

    def test_save_rejects_empty_name(self):
        with self.assertRaises(ValueError):
            save_sequence("", [{"tool": "a"}], db_path=self.db)

    def test_save_rejects_no_steps(self):
        with self.assertRaises(ValueError):
            save_sequence("x", [], db_path=self.db)

    def test_save_rejects_step_without_tool(self):
        with self.assertRaises(ValueError):
            save_sequence("x", [{"args": {}}], db_path=self.db)

    def test_get_missing_returns_none(self):
        self.assertIsNone(get_sequence("nope", db_path=self.db))

    def test_list_sequences(self):
        save_sequence("a", [{"tool": "t"}], db_path=self.db)
        save_sequence("b", [{"tool": "t"}], db_path=self.db)
        names = {s.name for s in list_sequences(db_path=self.db)}
        self.assertEqual(names, {"a", "b"})

    def test_delete_sequence(self):
        save_sequence("a", [{"tool": "t"}], db_path=self.db)
        self.assertTrue(delete_sequence("a", db_path=self.db))
        self.assertIsNone(get_sequence("a", db_path=self.db))
        self.assertFalse(delete_sequence("a", db_path=self.db))

    def test_record_run_bumps_counter(self):
        save_sequence("a", [{"tool": "t"}], db_path=self.db)
        record_run("a", db_path=self.db)
        record_run("a", db_path=self.db)
        seq = get_sequence("a", db_path=self.db)
        self.assertEqual(seq.run_count, 2)
        self.assertIsNotNone(seq.last_run)


class TestSequenceRecallHandler(unittest.TestCase):
    """Exercises actions/sequence_recall.py against the real DB module but
    patched to a throwaway file, matching test_study_mode.py's pattern of
    patching module internals rather than the DB layer twice over."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.db = Path(self._tmp) / "sequences.db"
        import core.sequence_memory as sm
        self._orig_db_path = sm.DB_PATH
        sm.DB_PATH = self.db

    def tearDown(self):
        import core.sequence_memory as sm
        sm.DB_PATH = self._orig_db_path
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_save_via_tool(self):
        result = sr.manage_sequence({
            "action": "save", "name": "greet",
            "steps": [{"tool": "web_search", "args": {"query": "hi"}}],
        })
        self.assertIn("Saved", result)

    def test_save_rejects_manage_sequence_as_a_step(self):
        result = sr.manage_sequence({
            "action": "save", "name": "loopy",
            "steps": [{"tool": "manage_sequence", "args": {"action": "run"}}],
        })
        self.assertIn("cannot contain", result)

    def test_run_replays_steps_via_dispatch(self):
        sr.manage_sequence({
            "action": "save", "name": "two_step",
            "steps": [{"tool": "a"}, {"tool": "b"}],
        })
        calls = []

        def fake_dispatch(tool, args):
            calls.append(tool)
            return "ok"

        result = sr.manage_sequence({"action": "run", "name": "two_step"}, dispatch=fake_dispatch)
        self.assertEqual(calls, ["a", "b"])
        self.assertIn("complete", result)

    def test_run_missing_sequence(self):
        result = sr.manage_sequence({"action": "run", "name": "ghost"}, dispatch=lambda t, a: "ok")
        self.assertIn("No saved sequence", result)

    def test_run_without_dispatch_reports_unavailable(self):
        sr.manage_sequence({"action": "save", "name": "x", "steps": [{"tool": "a"}]})
        result = sr.manage_sequence({"action": "run", "name": "x"})
        self.assertIn("unavailable", result)

    def test_run_stops_at_first_failure(self):
        sr.manage_sequence({
            "action": "save", "name": "brittle",
            "steps": [{"tool": "a"}, {"tool": "b"}, {"tool": "c"}],
        })

        def flaky_dispatch(tool, args):
            if tool == "b":
                raise RuntimeError("boom")
            return "ok"

        result = sr.manage_sequence({"action": "run", "name": "brittle"}, dispatch=flaky_dispatch)
        self.assertIn("failed", result)
        self.assertNotIn("3. c", result)

    def test_list_and_delete_via_tool(self):
        sr.manage_sequence({"action": "save", "name": "x", "steps": [{"tool": "a"}]})
        listing = sr.manage_sequence({"action": "list"})
        self.assertIn("x", listing)
        deleted = sr.manage_sequence({"action": "delete", "name": "x"})
        self.assertIn("Deleted", deleted)


if __name__ == "__main__":
    unittest.main()
