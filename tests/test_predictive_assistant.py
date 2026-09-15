"""Unit tests for core/predictive_assistant.py — pattern detection + feedback."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.predictive_assistant import (  # noqa: E402
    detect_repeated_manual_steps,
    detect_sequential_patterns,
    detect_time_based_patterns,
    get_proactive_suggestions,
    log_event,
    record_suggestion_feedback,
)


class _Row(dict):
    """Minimal stand-in for sqlite3.Row supporting the ["key"] access the
    detectors use, built without touching a real database."""

    def __getitem__(self, key):
        return dict.__getitem__(self, key)


def make_event(action_type, timestamp, context="proj", success=True):
    return _Row(
        {
            "timestamp": timestamp,
            "action_type": action_type,
            "context": context,
            "input": "",
            "output": "",
            "success": success,
        }
    )


class TestSequentialPatterns(unittest.TestCase):
    def test_detects_frequent_bigram(self):
        events = [
            make_event("git_add", "2026-01-01T09:00:00"),
            make_event("git_commit", "2026-01-01T09:01:00"),
            make_event("git_add", "2026-01-01T10:00:00"),
            make_event("git_commit", "2026-01-01T10:01:00"),
            make_event("git_add", "2026-01-01T11:00:00"),
            make_event("git_commit", "2026-01-01T11:01:00"),
        ]
        patterns = detect_sequential_patterns(events, min_count=2)
        keys = {p["pattern_key"] for p in patterns}
        self.assertIn("seq:git_add->git_commit", keys)
        match = next(p for p in patterns if p["pattern_key"] == "seq:git_add->git_commit")
        self.assertEqual(match["support"], 3)
        self.assertAlmostEqual(match["confidence"], 1.0)

    def test_below_min_count_is_excluded(self):
        events = [
            make_event("open_chrome", "2026-01-01T09:00:00"),
            make_event("lock_screen", "2026-01-01T09:01:00"),
        ]
        patterns = detect_sequential_patterns(events, min_count=2)
        self.assertEqual(patterns, [])

    def test_ignores_failed_events(self):
        events = [
            make_event("deploy", "2026-01-01T09:00:00", success=False),
            make_event("rollback", "2026-01-01T09:01:00", success=False),
            make_event("deploy", "2026-01-01T10:00:00", success=False),
            make_event("rollback", "2026-01-01T10:01:00", success=False),
        ]
        patterns = detect_sequential_patterns(events, min_count=2)
        self.assertEqual(patterns, [])


class TestTimeBasedPatterns(unittest.TestCase):
    def test_detects_recurring_morning_action(self):
        events = [
            make_event("check_email", "2026-01-05T08:05:00"),  # Monday
            make_event("check_email", "2026-01-06T08:10:00"),  # Tuesday
            make_event("check_email", "2026-01-07T08:07:00"),  # Wednesday
        ]
        patterns = detect_time_based_patterns(events, min_count=3, hour_window=1)
        self.assertTrue(any(p["suggested_action"] == "check_email" for p in patterns))

    def test_scattered_times_not_flagged(self):
        events = [
            make_event("check_email", "2026-01-05T08:00:00"),
            make_event("check_email", "2026-01-06T14:00:00"),
            make_event("check_email", "2026-01-07T20:00:00"),
        ]
        patterns = detect_time_based_patterns(events, min_count=3, hour_window=1)
        self.assertEqual(patterns, [])


class TestRepeatedManualSteps(unittest.TestCase):
    def test_detects_repeated_ngram(self):
        events = [
            make_event("resize_image", "2026-01-01T09:00:00"),
            make_event("upload_image", "2026-01-01T09:01:00"),
            make_event("resize_image", "2026-01-02T09:00:00"),
            make_event("upload_image", "2026-01-02T09:01:00"),
            make_event("resize_image", "2026-01-03T09:00:00"),
            make_event("upload_image", "2026-01-03T09:01:00"),
        ]
        patterns = detect_repeated_manual_steps(events, min_count=3, ngram_size=2)
        keys = {p["pattern_key"] for p in patterns}
        self.assertIn("manual:resize_image>upload_image", keys)

    def test_automation_prefixed_actions_are_excluded(self):
        events = [
            make_event("automation:resize_image", "2026-01-01T09:00:00"),
            make_event("automation:upload_image", "2026-01-01T09:01:00"),
        ] * 3
        patterns = detect_repeated_manual_steps(events, min_count=3, ngram_size=2)
        self.assertEqual(patterns, [])


class TestSuggestionEngineAndFeedback(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "workflow.db"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_end_to_end_suggestion_above_threshold(self):
        for _ in range(4):
            log_event("git_add", context="proj", db_path=self.db_path)
            log_event("git_commit", context="proj", db_path=self.db_path)

        suggestions = get_proactive_suggestions(
            current_context="proj", threshold=0.6, days=30, db_path=self.db_path
        )
        self.assertTrue(any(s.action == "git_commit" for s in suggestions))
        top = next(s for s in suggestions if s.action == "git_commit")
        self.assertGreaterEqual(top.confidence_score, 0.6)
        self.assertEqual(top.one_click_command, "git_commit")

    def test_dismiss_feedback_lowers_future_confidence(self):
        for _ in range(4):
            log_event("git_add", context="proj", db_path=self.db_path)
            log_event("git_commit", context="proj", db_path=self.db_path)

        before = get_proactive_suggestions(
            current_context="proj", threshold=0.0, days=30, db_path=self.db_path
        )
        before_score = next(s.confidence_score for s in before if s.action == "git_commit")

        for _ in range(3):
            record_suggestion_feedback(
                "seq:git_add->git_commit", "git_commit", accepted=False,
                context="proj", db_path=self.db_path,
            )

        after = get_proactive_suggestions(
            current_context="proj", threshold=0.0, days=30, db_path=self.db_path
        )
        after_score = next(s.confidence_score for s in after if s.action == "git_commit")

        self.assertLess(after_score, before_score)


if __name__ == "__main__":
    unittest.main()
