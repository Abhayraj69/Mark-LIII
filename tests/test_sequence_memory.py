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
    discard_recording,
    get_sequence,
    is_recording,
    list_sequences,
    parametrize,
    record_run,
    record_step,
    render_steps,
    required_params,
    save_sequence,
    start_recording,
    stop_recording,
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


class TestRecordMode(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.db = Path(self._tmp) / "sequences.db"
        import core.sequence_memory as sm
        self._sm = sm
        sm._recording = None  # ensure no leftover recording from another test

    def tearDown(self):
        self._sm._recording = None
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_record_round_trip(self):
        self.assertFalse(is_recording())
        msg = start_recording("morning", "Start the day")
        self.assertIn("Recording started", msg)
        self.assertTrue(is_recording())

        record_step("open_app", {"name": "chrome"})
        record_step("web_search", {"query": "news"})

        result = stop_recording(db_path=self.db)
        self.assertIn("Saved", result)
        self.assertFalse(is_recording())

        seq = get_sequence("morning", db_path=self.db)
        self.assertEqual(len(seq.steps), 2)
        self.assertEqual(seq.steps[0].tool, "open_app")
        self.assertEqual(seq.steps[1].tool, "web_search")

    def test_cannot_start_two_recordings(self):
        start_recording("a")
        with self.assertRaises(ValueError):
            start_recording("b")

    def test_self_recording_excluded(self):
        start_recording("loopy")
        record_step("manage_sequence", {"action": "run"})
        record_step("open_app", {"name": "chrome"})
        result = stop_recording(db_path=self.db)
        self.assertIn("Saved", result)
        seq = get_sequence("loopy", db_path=self.db)
        self.assertEqual(len(seq.steps), 1)
        self.assertEqual(seq.steps[0].tool, "open_app")

    def test_discard_recording_saves_nothing(self):
        start_recording("scratch")
        record_step("open_app", {"name": "chrome"})
        msg = discard_recording()
        self.assertIn("Discarded", msg)
        self.assertFalse(is_recording())
        self.assertIsNone(get_sequence("scratch", db_path=self.db))

    def test_stop_without_start_raises(self):
        with self.assertRaises(ValueError):
            stop_recording(db_path=self.db)

    def test_discard_without_start_raises(self):
        with self.assertRaises(ValueError):
            discard_recording()

    def test_record_step_confirm_flag_preserved(self):
        start_recording("gated")
        record_step("shutdown_jarvis", {}, confirm=True)
        stop_recording(db_path=self.db)
        seq = get_sequence("gated", db_path=self.db)
        self.assertTrue(seq.steps[0].confirm)

    def test_record_step_noop_when_not_recording(self):
        record_step("open_app", {"name": "chrome"})  # must not raise
        self.assertFalse(is_recording())


class TestPlaceholdersAndParametrize(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.db = Path(self._tmp) / "sequences.db"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_required_params_found(self):
        save_sequence("greet_file", [{"tool": "open_app", "args": {"name": "{file}"}}], db_path=self.db)
        seq = get_sequence("greet_file", db_path=self.db)
        self.assertEqual(required_params(seq), {"file"})

    def test_render_steps_substitutes(self):
        save_sequence("greet_file", [{"tool": "open_app", "args": {"name": "{file}"}}], db_path=self.db)
        seq = get_sequence("greet_file", db_path=self.db)
        rendered, missing = render_steps(seq, {"file": "report.pdf"})
        self.assertEqual(missing, set())
        self.assertEqual(rendered[0].args, {"name": "report.pdf"})

    def test_render_steps_reports_missing(self):
        save_sequence("greet_file", [{"tool": "open_app", "args": {"name": "{file}"}}], db_path=self.db)
        seq = get_sequence("greet_file", db_path=self.db)
        rendered, missing = render_steps(seq, {})
        self.assertEqual(rendered, [])
        self.assertEqual(missing, {"file"})

    def test_parametrize_replaces_literal(self):
        save_sequence("open_report", [{"tool": "open_app", "args": {"name": "report_Q3.pdf"}}], db_path=self.db)
        msg = parametrize("open_report", "report_Q3.pdf", "file", db_path=self.db)
        self.assertIn("Replaced 1", msg)
        seq = get_sequence("open_report", db_path=self.db)
        self.assertEqual(seq.steps[0].args, {"name": "{file}"})

    def test_parametrize_no_match(self):
        save_sequence("open_report", [{"tool": "open_app", "args": {"name": "report_Q3.pdf"}}], db_path=self.db)
        msg = parametrize("open_report", "nope.pdf", "file", db_path=self.db)
        self.assertIn("nothing changed", msg)

    def test_parametrize_rejects_bad_placeholder(self):
        save_sequence("open_report", [{"tool": "open_app", "args": {"name": "report_Q3.pdf"}}], db_path=self.db)
        with self.assertRaises(ValueError):
            parametrize("open_report", "report_Q3.pdf", "not a valid name!", db_path=self.db)


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

    def test_record_start_stop_via_tool(self):
        import core.sequence_memory as sm
        sm._recording = None
        try:
            start = sr.manage_sequence({"action": "record_start", "name": "rec1"})
            self.assertIn("Recording started", start)

            sr.manage_sequence({"action": "record_stop"})  # not a real dispatch; just confirms no crash
        finally:
            sm._recording = None

    def test_record_stop_saves_captured_steps(self):
        import core.sequence_memory as sm
        sm._recording = None
        try:
            sr.manage_sequence({"action": "record_start", "name": "rec2"})
            sm.record_step("open_app", {"name": "chrome"})
            result = sr.manage_sequence({"action": "record_stop"})
            self.assertIn("Saved", result)
            seq = get_sequence("rec2", db_path=self.db)
            self.assertEqual(len(seq.steps), 1)
        finally:
            sm._recording = None

    def test_record_discard_via_tool(self):
        import core.sequence_memory as sm
        sm._recording = None
        try:
            sr.manage_sequence({"action": "record_start", "name": "rec3"})
            sm.record_step("open_app", {"name": "chrome"})
            result = sr.manage_sequence({"action": "record_discard"})
            self.assertIn("Discarded", result)
            self.assertIsNone(get_sequence("rec3", db_path=self.db))
        finally:
            sm._recording = None

    def test_replay_substitutes_params(self):
        sr.manage_sequence({
            "action": "save", "name": "open_named",
            "steps": [{"tool": "open_app", "args": {"name": "{file}"}}],
        })
        calls = []

        def fake_dispatch(tool, args):
            calls.append((tool, args))
            return "ok"

        result = sr.manage_sequence(
            {"action": "replay", "name": "open_named", "params": {"file": "report.pdf"}},
            dispatch=fake_dispatch,
        )
        self.assertIn("complete", result)
        self.assertEqual(calls, [("open_app", {"name": "report.pdf"})])

    def test_replay_reports_missing_params(self):
        sr.manage_sequence({
            "action": "save", "name": "open_named2",
            "steps": [{"tool": "open_app", "args": {"name": "{file}"}}],
        })
        result = sr.manage_sequence(
            {"action": "run", "name": "open_named2"}, dispatch=lambda t, a: "ok"
        )
        self.assertIn("file", result)
        self.assertIn("params", result)

    def test_parametrize_via_tool(self):
        sr.manage_sequence({
            "action": "save", "name": "open_report",
            "steps": [{"tool": "open_app", "args": {"name": "report_Q3.pdf"}}],
        })
        result = sr.manage_sequence({
            "action": "parametrize", "name": "open_report",
            "value": "report_Q3.pdf", "placeholder": "file",
        })
        self.assertIn("Replaced 1", result)


if __name__ == "__main__":
    unittest.main()
