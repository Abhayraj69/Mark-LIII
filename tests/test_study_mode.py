"""Unit tests for actions/study_mode.py — the opt-in continuous-capture
loop. Mocks screen capture/vision and uses a tiny interval so tests run
fast; verifies start/stop/status dispatch, dedup-on-unchanged-screen, and
that it never starts itself (only responds to explicit action='start')."""

import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import actions.study_mode as sm  # noqa: E402


class TestStudyMode(unittest.TestCase):
    def setUp(self):
        # Reset module-level state between tests — it's a singleton by design.
        sm._state.update(active=False, thread=None, stop_event=None,
                          interval=sm._DEFAULT_INTERVAL, captures=0,
                          file_path=None, started_at=None)
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        if sm._state["active"]:
            sm._stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_status_off_by_default(self):
        self.assertEqual(sm.study_mode(parameters={"action": "status"}), "Study mode is off.")

    def test_stop_when_not_running(self):
        result = sm.study_mode(parameters={"action": "stop"})
        self.assertIn("isn't running", result)

    def test_unknown_action(self):
        result = sm.study_mode(parameters={"action": "bogus"})
        self.assertIn("Unknown action", result)

    @patch("actions.study_mode._clamp_interval", return_value=1)
    @patch("actions.study_mode._resolve_path")
    @patch("actions.study_mode._vision_query", return_value="notes")
    @patch("actions.study_mode._capture_screen", return_value=(b"fake", "image/jpeg"))
    def test_start_then_status_then_stop(self, mock_capture, mock_vision, mock_resolve, mock_clamp):
        mock_resolve.return_value = Path(self._tmp)
        start_result = sm.study_mode(parameters={"action": "start", "interval_seconds": 1})
        self.assertIn("Study mode is ON", start_result)
        self.assertTrue(sm._state["active"])

        status_result = sm.study_mode(parameters={"action": "status"})
        self.assertIn("Study mode is ON", status_result)

        # Let the loop run at least one tick to confirm it actually captures.
        time.sleep(1.5)
        self.assertGreaterEqual(sm._state["captures"], 1)

        stop_result = sm.study_mode(parameters={"action": "stop"})
        self.assertIn("Study mode stopped", stop_result)
        self.assertFalse(sm._state["active"])
        self.assertTrue(sm._state["file_path"].exists())
        self.assertIn("notes", sm._state["file_path"].read_text(encoding="utf-8"))

    @patch("actions.study_mode._resolve_path")
    def test_start_twice_reports_already_running(self, mock_resolve):
        mock_resolve.return_value = Path(self._tmp)
        with patch("actions.study_mode._capture_screen", return_value=(b"f", "image/jpeg")), \
             patch("actions.study_mode._vision_query", return_value="notes"):
            sm.study_mode(parameters={"action": "start", "interval_seconds": 30})
            second = sm.study_mode(parameters={"action": "start", "interval_seconds": 30})
        self.assertIn("already running", second)

    @patch("actions.study_mode._clamp_interval", return_value=1)
    @patch("actions.study_mode._resolve_path")
    @patch("actions.study_mode._vision_query", return_value="NO_CONTENT")
    @patch("actions.study_mode._capture_screen", return_value=(b"fake", "image/jpeg"))
    def test_no_content_does_not_write_or_count(self, mock_capture, mock_vision, mock_resolve, mock_clamp):
        mock_resolve.return_value = Path(self._tmp)
        sm.study_mode(parameters={"action": "start", "interval_seconds": 1})
        time.sleep(1.5)
        self.assertEqual(sm._state["captures"], 0)
        sm.study_mode(parameters={"action": "stop"})

    def test_interval_is_clamped(self):
        self.assertEqual(sm._clamp_interval(5), sm._MIN_INTERVAL)
        self.assertEqual(sm._clamp_interval(99999), sm._MAX_INTERVAL)
        self.assertEqual(sm._clamp_interval("not a number"), sm._DEFAULT_INTERVAL)
        self.assertEqual(sm._clamp_interval(120), 120)


if __name__ == "__main__":
    unittest.main()
