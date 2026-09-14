"""Unit tests for core/causal_reasoning.py — the cause-and-effect graph —
and actions/causal_insight.py's tool handler on top of it."""

import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.causal_reasoning import (  # noqa: E402
    explain,
    get_links,
    predict_effects,
    record_event,
)
import actions.causal_insight as ci  # noqa: E402


class TestCausalReasoning(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.db = Path(self._tmp) / "causal_graph.db"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _seed_pair(self, cause: str, effect: str, times: int, lag_seconds: float = 5.0,
                    base_time: datetime = None):
        """Repeats cause->effect `times`, each pair `lag_seconds` apart and
        each occurrence far enough from the next pair to avoid cross-linking."""
        t = base_time or datetime(2026, 1, 1, 9, 0, 0)
        for i in range(times):
            pair_t = t + timedelta(minutes=5 * i)
            record_event(cause, timestamp=pair_t, db_path=self.db)
            record_event(effect, timestamp=pair_t + timedelta(seconds=lag_seconds), db_path=self.db)

    def test_no_link_below_min_support(self):
        self._seed_pair("tool:a", "tool:b", times=1)
        links = get_links(min_support=2, db_path=self.db)
        self.assertEqual(links, [])

    def test_link_forms_after_repeated_cooccurrence(self):
        self._seed_pair("tool:a", "tool:b", times=3)
        links = get_links(min_support=2, db_path=self.db)
        self.assertEqual(len(links), 1)
        link = links[0]
        self.assertEqual(link.cause, "tool:a")
        self.assertEqual(link.effect, "tool:b")
        self.assertEqual(link.support, 3)
        self.assertAlmostEqual(link.confidence, 1.0, places=2)
        self.assertGreater(link.lift, 1.0)

    def test_event_outside_lag_window_not_linked(self):
        t = datetime(2026, 1, 1, 9, 0, 0)
        record_event("tool:a", timestamp=t, db_path=self.db)
        record_event("tool:b", timestamp=t + timedelta(seconds=200), db_path=self.db)
        record_event("tool:a", timestamp=t + timedelta(minutes=10), db_path=self.db)
        record_event("tool:b", timestamp=t + timedelta(minutes=10, seconds=200), db_path=self.db)
        links = get_links(min_support=1, db_path=self.db)
        self.assertEqual(links, [])

    def test_repeated_cause_within_window_counts_once(self):
        # tool:a fires twice quickly, then tool:b fires once — should not
        # inflate support to 2 for a single effect occurrence.
        t = datetime(2026, 1, 1, 9, 0, 0)
        record_event("tool:a", timestamp=t, db_path=self.db)
        record_event("tool:a", timestamp=t + timedelta(seconds=5), db_path=self.db)
        record_event("tool:b", timestamp=t + timedelta(seconds=10), db_path=self.db)
        links = get_links(min_support=1, db_path=self.db)
        matching = [l for l in links if l.cause == "tool:a" and l.effect == "tool:b"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].support, 1)

    def test_frequent_effect_gets_lower_lift(self):
        # "tool:b" happens after everything (high base rate) -> low lift even
        # with decent confidence, distinguishing coincidence from a real driver.
        self._seed_pair("tool:a", "tool:rare_effect", times=3, base_time=datetime(2026, 1, 1, 9, 0))
        # Make tool:common fire constantly, unrelated to anything.
        for i in range(20):
            record_event("tool:common", timestamp=datetime(2026, 1, 2, 9, i, 0), db_path=self.db)
        links = {(l.cause, l.effect): l for l in get_links(min_support=2, db_path=self.db)}
        self.assertIn(("tool:a", "tool:rare_effect"), links)
        self.assertGreater(links[("tool:a", "tool:rare_effect")].lift, 1.0)

    def test_explain_both_directions(self):
        self._seed_pair("tool:a", "tool:b", times=3)
        result = explain("tool:b", min_support=2, db_path=self.db)
        self.assertEqual(len(result["causes_of"]), 1)
        self.assertEqual(result["causes_of"][0].cause, "tool:a")
        self.assertEqual(result["effects_of"], [])

    def test_predict_effects(self):
        self._seed_pair("screen:download_finished", "tool:open_app", times=3)
        preds = predict_effects("screen:download_finished", top_k=3, min_support=2, db_path=self.db)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].effect, "tool:open_app")

    def test_record_event_never_raises_on_bad_input(self):
        record_event("", db_path=self.db)  # empty type — silently ignored
        record_event("tool:a", timestamp=None, db_path=self.db)  # should just work


class TestCausalInsightHandler(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.db = Path(self._tmp) / "causal_graph.db"
        import core.causal_reasoning as cr
        self._orig_db_path = cr.DB_PATH
        cr.DB_PATH = self.db

    def tearDown(self):
        import core.causal_reasoning as cr
        cr.DB_PATH = self._orig_db_path
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_stats_empty(self):
        result = ci.causal_insight({"action": "stats"})
        self.assertIn("No causal patterns", result)

    def test_explain_requires_event(self):
        result = ci.causal_insight({"action": "explain"})
        self.assertIn("Specify 'event'", result)

    def test_predict_and_stats_after_seeding(self):
        from core.causal_reasoning import record_event
        t = datetime(2026, 1, 1, 9, 0, 0)
        for i in range(3):
            pair_t = t + timedelta(minutes=5 * i)
            record_event("screen:download_finished", timestamp=pair_t)
            record_event("tool:open_app", timestamp=pair_t + timedelta(seconds=5))

        predict_result = ci.causal_insight({"action": "predict", "event": "download finished"})
        self.assertIn("open_app", predict_result)

        stats_result = ci.causal_insight({"action": "stats"})
        self.assertIn("download_finished", stats_result)

    def test_explain_no_pattern(self):
        result = ci.causal_insight({"action": "explain", "event": "something_never_seen"})
        self.assertIn("No causal pattern learned yet", result)


if __name__ == "__main__":
    unittest.main()
