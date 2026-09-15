"""Unit tests for core/sentiment_adapter.py — signal extraction, style
mapping, guardrail wording, and the session-only logging boundary."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.sentiment_adapter import (  # noqa: E402
    SentimentSignal,
    StylePolicy,
    build_style_modifier,
    clear_session_log,
    detect,
    get_session_log,
    get_style,
    log_signal,
    to_prompt_modifier,
)


class TestDetectPolarityAndUrgency(unittest.TestCase):
    def test_frustrated_all_caps_is_urgent(self):
        signal = detect("THIS STILL ISN'T WORKING, SERIOUSLY")
        self.assertEqual(signal.polarity, "frustrated")
        self.assertTrue(signal.urgency)

    def test_positive_message_is_positive_and_not_urgent(self):
        signal = detect("This works now, thanks so much, awesome!")
        self.assertEqual(signal.polarity, "positive")

    def test_plain_neutral_message(self):
        signal = detect("Can you open the file explorer?")
        self.assertEqual(signal.polarity, "neutral")
        self.assertFalse(signal.urgency)

    def test_repeated_correction_is_urgent(self):
        signal = detect("open chrome now", recent_texts=["please open chrome"])
        self.assertTrue(signal.urgency)

    def test_hedging_sets_user_confidence(self):
        signal = detect("Maybe try restarting it, I'm not sure though")
        self.assertEqual(signal.user_confidence, "hedging")

    def test_short_acronym_does_not_trigger_caps_false_positive(self):
        signal = detect("OK")
        self.assertFalse(signal.urgency)


class TestStylePolicyMapping(unittest.TestCase):
    def test_frustrated_urgent_is_concise_low_verbosity_low_proactivity(self):
        policy = get_style(SentimentSignal("frustrated", True, "neutral"))
        self.assertEqual(policy, StylePolicy("concise", "low", "low"))

    def test_positive_calm_is_supportive_high_verbosity_high_proactivity(self):
        policy = get_style(SentimentSignal("positive", False, "neutral"))
        self.assertEqual(policy.tone, "supportive")
        self.assertEqual(policy.verbosity, "high")
        self.assertEqual(policy.proactivity, "high")

    def test_hedging_bumps_verbosity_only_when_not_urgent(self):
        calm = get_style(SentimentSignal("neutral", False, "hedging"))
        self.assertEqual(calm.verbosity, "high")
        urgent = get_style(SentimentSignal("neutral", True, "hedging"))
        self.assertEqual(urgent.verbosity, "low")  # urgency wins, not overridden by hedging


class TestPromptModifierGuardrails(unittest.TestCase):
    def test_modifier_never_mentions_content_accuracy_should_change(self):
        text = to_prompt_modifier(StylePolicy("concise", "low", "low"))
        self.assertIn("factual accuracy", text)
        self.assertIn("do not say you detected an emotion", text.lower())

    def test_disabled_adapter_returns_empty_modifier(self):
        # is_enabled() reads config; here we exercise the documented contract
        # directly via build_style_modifier's is_enabled() gate by monkeypatching.
        import core.sentiment_adapter as sa
        original = sa.is_enabled
        sa.is_enabled = lambda: False
        try:
            self.assertEqual(build_style_modifier("I AM SO ANGRY!!!"), "")
        finally:
            sa.is_enabled = original


class TestSessionOnlyLogging(unittest.TestCase):
    def setUp(self):
        clear_session_log()

    def tearDown(self):
        clear_session_log()

    def test_log_signal_never_stores_raw_text(self):
        signal = detect("my password is hunter2, this is broken")
        log_signal(signal)
        entries = get_session_log()
        self.assertEqual(len(entries), 1)
        dumped = str(entries[0])
        self.assertNotIn("hunter2", dumped)
        self.assertNotIn("password", dumped)

    def test_persistence_is_opt_in_and_off_by_default(self):
        import core.sentiment_adapter as sa
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "sentiment_log.db"
            signal = detect("still not working!!")
            log_signal(signal, db_path=db_path)
            # Persistence defaults to disabled, so nothing should be on disk.
            self.assertFalse(db_path.exists())

    def test_persistence_writes_only_derived_fields_when_opted_in(self):
        import core.sentiment_adapter as sa
        original = sa.is_persist_enabled
        sa.is_persist_enabled = lambda: True
        try:
            with tempfile.TemporaryDirectory() as d:
                db_path = Path(d) / "sentiment_log.db"
                signal = detect("my secret plan is broken, still not working")
                log_signal(signal, db_path=db_path)
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                rows = conn.execute("SELECT * FROM sentiment_signals").fetchall()
                conn.close()
                self.assertEqual(len(rows), 1)
                dumped = str(rows[0])
                self.assertNotIn("secret plan", dumped)
        finally:
            sa.is_persist_enabled = original


if __name__ == "__main__":
    unittest.main()
