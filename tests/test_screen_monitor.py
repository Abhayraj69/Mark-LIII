"""Unit tests for actions/screen_monitor.py — the configurable continuous
screen-watch loop. Mocks capture/vision and uses a short interval so tests
run fast, mirroring tests/test_study_mode.py's approach for the sibling
continuous-capture feature."""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import actions.screen_monitor as sm  # noqa: E402


class TestScreenMonitor(unittest.TestCase):
    def setUp(self):
        sm._state.clear()

    def tearDown(self):
        if sm._thread is not None and sm._thread.is_alive():
            sm._stop_event.set()
            sm._thread.join(timeout=2)

    def test_status_off_by_default(self):
        self.assertIn("No screen watch", sm.screen_monitor({"action": "status"}))

    def test_stop_when_not_running(self):
        self.assertIn("No screen watch", sm.screen_monitor({"action": "stop"}))

    def test_start_requires_watch_for(self):
        result = sm.screen_monitor({"action": "start"})
        self.assertIn("Tell me what to watch for", result)

    def test_unknown_action(self):
        self.assertIn("Specify action", sm.screen_monitor({"action": "bogus"}))

    @patch("actions.screen_monitor._capture_screen", return_value=(b"fake", "image/jpeg"))
    @patch("actions.screen_monitor._vision_query", return_value="NO")
    def test_start_then_status_then_stop(self, mock_vision, mock_capture):
        start_result = sm.screen_monitor({
            "action": "start", "watch_for": "an error dialog", "interval_seconds": 1,
        })
        self.assertIn("Watching your screen", start_result)

        time.sleep(1.5)
        status_result = sm.screen_monitor({"action": "status"})
        self.assertIn("an error dialog", status_result)
        self.assertGreaterEqual(sm._state["checks"], 1)

        stop_result = sm.screen_monitor({"action": "stop"})
        self.assertIn("Stopped watching", stop_result)
        self.assertFalse(sm._state["running"])

    @patch("actions.screen_monitor._capture_screen", return_value=(b"fake", "image/jpeg"))
    @patch("actions.screen_monitor._vision_query", return_value="YES the download finished")
    def test_alert_fires_speak_and_counts(self, mock_vision, mock_capture):
        spoken = []
        sm.screen_monitor({
            "action": "start", "watch_for": "download finished", "interval_seconds": 1,
        }, speak=lambda text: spoken.append(text))
        time.sleep(1.5)
        sm.screen_monitor({"action": "stop"})
        self.assertGreaterEqual(sm._state["alerts"], 1)
        self.assertTrue(any("download finished" in s for s in spoken))

    def test_interval_and_duration_are_clamped(self):
        self.assertEqual(sm._clamp(1, sm._MIN_INTERVAL_S, sm._MAX_INTERVAL_S), sm._MIN_INTERVAL_S)
        self.assertEqual(sm._clamp(99999, sm._MIN_INTERVAL_S, sm._MAX_INTERVAL_S), sm._MAX_INTERVAL_S)
        self.assertEqual(sm._clamp(60, sm._MIN_INTERVAL_S, sm._MAX_INTERVAL_S), 60)

    @patch("actions.screen_monitor._capture_screen", return_value=(b"fake", "image/jpeg"))
    @patch("actions.screen_monitor._vision_query", return_value="NO")
    def test_starting_again_replaces_previous_watch(self, mock_vision, mock_capture):
        sm.screen_monitor({"action": "start", "watch_for": "A", "interval_seconds": 30})
        first_thread = sm._thread
        sm.screen_monitor({"action": "start", "watch_for": "B", "interval_seconds": 30})
        self.assertFalse(first_thread.is_alive())
        self.assertIn("B", sm._status())
        sm.screen_monitor({"action": "stop"})


if __name__ == "__main__":
    unittest.main()
