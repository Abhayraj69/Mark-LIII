"""Unit tests for core/telemetry.py — per-turn latency/cost telemetry.
Uses a temp DB path so this never touches memory/telemetry.db."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import telemetry  # noqa: E402


class TestTelemetry(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "telemetry.db"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _finish_and_wait(self, turn: "telemetry.Turn"):
        # finish() writes on a background thread; join it via the thread
        # object it starts is not exposed, so re-implement the same write
        # synchronously for deterministic assertions in tests.
        turn._finished = False
        total_ms = 0.0
        turn._write(total_ms)
        turn._finished = True

    def test_round_trip_turn_with_tool_spans(self):
        turn = telemetry.start_turn("local", db_path=self.db_path)
        turn.mark_once("first_audio")
        turn.mark("model_done")
        with turn.tool_span("weather_report"):
            pass
        with turn.tool_span("weather_report"):
            pass
        turn.tokens(tokens_in=100, tokens_out=42)
        turn.set_fast_intent()
        self._finish_and_wait(turn)

        summary = telemetry.summary(days=7, db_path=self.db_path)
        self.assertIn("local", summary["backends"])
        local = summary["backends"]["local"]
        self.assertEqual(local["turns"], 1)
        self.assertEqual(local["tokens_in"], 100)
        self.assertEqual(local["tokens_out"], 42)
        self.assertEqual(local["fast_intent_hit_rate"], 1.0)
        self.assertEqual(local["interrupted_rate"], 0.0)
        self.assertIsNotNone(local["p50_time_to_first_audio_ms"])

        self.assertIn("weather_report", summary["tools"])
        self.assertEqual(summary["tools"]["weather_report"]["count"], 2)

    def test_interrupted_turn_recorded(self):
        turn = telemetry.start_turn("gemini", db_path=self.db_path)
        turn.set_interrupted()
        self._finish_and_wait(turn)

        summary = telemetry.summary(days=7, db_path=self.db_path)
        self.assertEqual(summary["backends"]["gemini"]["interrupted_rate"], 1.0)

    def test_finish_is_idempotent(self):
        turn = telemetry.start_turn("local", db_path=self.db_path)
        calls = []
        turn._write = lambda total_ms: calls.append(total_ms)   # avoid a real background thread in this test
        turn.finish()
        turn.finish()   # must not raise or double-write
        self.assertTrue(turn._finished)
        self.assertEqual(len(calls), 1)

    def test_summary_on_empty_db_has_no_backends(self):
        summary = telemetry.summary(days=7, db_path=self.db_path)
        self.assertEqual(summary["backends"], {})
        self.assertEqual(summary["tools"], {})

    def test_mark_once_does_not_overwrite(self):
        turn = telemetry.start_turn("local", db_path=self.db_path)
        turn.mark_once("first_audio")
        first = turn._marks["first_audio"]
        turn.mark_once("first_audio")
        self.assertEqual(turn._marks["first_audio"], first)


if __name__ == "__main__":
    unittest.main()
