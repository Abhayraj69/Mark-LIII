"""Unit tests for actions/study_progress.py — mocks memory/study_history.py's
functions so these test the handler's dispatch/formatting logic rather than
real SQLite behavior (covered separately in test_study_history.py)."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import actions.study_progress as sp  # noqa: E402


class TestStudyProgress(unittest.TestCase):
    def test_unknown_action(self):
        result = sp.study_progress(parameters={"action": "nonsense"})
        self.assertIn("Unknown action", result)

    @patch("actions.study_progress.log_answer", return_value=1)
    def test_log_result(self, mock_log):
        result = sp.study_progress(parameters={
            "action": "log_result",
            "topic": "bio",
            "question": "What is the powerhouse of the cell?",
            "correct_answer": "Mitochondria",
            "user_answer": "the mitochondria",
            "correct": True,
        })
        mock_log.assert_called_once_with(
            question="What is the powerhouse of the cell?",
            correct_answer="Mitochondria",
            correct=True,
            topic="bio",
            user_answer="the mitochondria",
        )
        self.assertEqual(result, "Logged.")

    def test_log_result_without_question(self):
        result = sp.study_progress(parameters={"action": "log_result"})
        self.assertIn("No question provided", result)

    @patch("actions.study_progress.get_weak_topics", return_value=[
        {"topic": "chemistry", "attempts": 4, "correct": 1, "accuracy": 0.25}
    ])
    def test_weak_topics_formats_percentage(self, mock_weak):
        result = sp.study_progress(parameters={"action": "weak_topics"})
        self.assertIn("chemistry", result)
        self.assertIn("25%", result)

    @patch("actions.study_progress.get_weak_topics", return_value=[])
    def test_weak_topics_empty(self, mock_weak):
        result = sp.study_progress(parameters={"action": "weak_topics"})
        self.assertIn("Not enough quiz history", result)

    @patch("actions.study_progress.get_missed_questions", return_value=[
        {"topic": "bio", "question": "Q1", "correct_answer": "A1", "user_answer": "wrong", "timestamp": "t"}
    ])
    def test_missed_questions(self, mock_missed):
        result = sp.study_progress(parameters={"action": "missed_questions"})
        self.assertIn("Q1", result)
        self.assertIn("A1", result)

    @patch("actions.study_progress.get_stats", return_value={"total_answered": 10, "total_correct": 7, "accuracy": 0.7})
    def test_stats(self, mock_stats):
        result = sp.study_progress(parameters={"action": "stats"})
        self.assertIn("7/10", result)
        self.assertIn("70%", result)

    @patch("actions.study_progress.get_stats", return_value={"total_answered": 0, "total_correct": 0, "accuracy": None})
    def test_stats_empty(self, mock_stats):
        result = sp.study_progress(parameters={"action": "stats"})
        self.assertIn("No quiz history", result)

    @patch("actions.study_progress.clear_history", return_value=3)
    def test_clear(self, mock_clear):
        result = sp.study_progress(parameters={"action": "clear"})
        self.assertIn("Cleared 3", result)


if __name__ == "__main__":
    unittest.main()
