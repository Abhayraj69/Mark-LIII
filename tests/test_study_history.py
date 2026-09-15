"""Unit tests for memory/study_history.py — uses a temp SQLite file per test
(via db_path) rather than the real memory/study_history.db, matching the
db_path-override pattern already used by core/context_manager.py's own
tests/usage."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory.study_history as sh  # noqa: E402


class TestStudyHistory(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "study_history.db"

    def tearDown(self):
        self._tmp.cleanup()

    def test_log_and_stats(self):
        sh.log_answer("Q1", "A1", True, topic="bio", db_path=self.db_path)
        sh.log_answer("Q2", "A2", False, topic="bio", db_path=self.db_path)
        stats = sh.get_stats(db_path=self.db_path)
        self.assertEqual(stats["total_answered"], 2)
        self.assertEqual(stats["total_correct"], 1)
        self.assertEqual(stats["accuracy"], 0.5)

    def test_empty_question_not_logged(self):
        row_id = sh.log_answer("", "A1", True, db_path=self.db_path)
        self.assertEqual(row_id, -1)
        self.assertEqual(sh.get_stats(db_path=self.db_path)["total_answered"], 0)

    def test_weak_topics_respects_min_attempts(self):
        sh.log_answer("Q1", "A1", False, topic="chemistry", db_path=self.db_path)
        # Only one attempt so far — below the default min_attempts of 2.
        weak = sh.get_weak_topics(db_path=self.db_path)
        self.assertEqual(weak, [])
        sh.log_answer("Q2", "A2", False, topic="chemistry", db_path=self.db_path)
        weak = sh.get_weak_topics(db_path=self.db_path)
        self.assertEqual(len(weak), 1)
        self.assertEqual(weak[0]["topic"], "chemistry")
        self.assertEqual(weak[0]["accuracy"], 0.0)

    def test_weak_topics_ranks_lowest_accuracy_first(self):
        for _ in range(2):
            sh.log_answer("Q", "A", True, topic="easy_topic", db_path=self.db_path)
        for _ in range(2):
            sh.log_answer("Q", "A", False, topic="hard_topic", db_path=self.db_path)
        weak = sh.get_weak_topics(db_path=self.db_path)
        self.assertEqual(weak[0]["topic"], "hard_topic")

    def test_missed_questions_filters_by_topic(self):
        sh.log_answer("Q1", "A1", False, topic="bio", db_path=self.db_path)
        sh.log_answer("Q2", "A2", False, topic="chem", db_path=self.db_path)
        sh.log_answer("Q3", "A3", True, topic="bio", db_path=self.db_path)
        missed_bio = sh.get_missed_questions(topic="bio", db_path=self.db_path)
        self.assertEqual(len(missed_bio), 1)
        self.assertEqual(missed_bio[0]["question"], "Q1")
        missed_all = sh.get_missed_questions(db_path=self.db_path)
        self.assertEqual(len(missed_all), 2)

    def test_clear_history(self):
        sh.log_answer("Q1", "A1", True, db_path=self.db_path)
        n = sh.clear_history(db_path=self.db_path)
        self.assertEqual(n, 1)
        self.assertEqual(sh.get_stats(db_path=self.db_path)["total_answered"], 0)


if __name__ == "__main__":
    unittest.main()
