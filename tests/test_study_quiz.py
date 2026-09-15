"""Unit tests for actions/study_quiz.py — mocks the text-generation call and
notes lookup so these test the handler's own logic (JSON parsing/validation,
falling back to a fresh capture, parameter clamping) rather than real model
output quality."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import actions.study_quiz as sq  # noqa: E402

_GOOD_JSON = (
    '[{"question": "What is the powerhouse of the cell?", '
    '"answer": "Mitochondria", "explanation": "It generates chemical energy."}, '
    '{"question": "What contains the DNA?", '
    '"answer": "Nucleus", "explanation": "It stores hereditary material."}]'
)


class TestStudyQuiz(unittest.TestCase):
    @patch("actions.study_quiz._text_query", return_value=_GOOD_JSON)
    @patch("actions.study_quiz.get_last_notes", return_value={"topic": "biology", "text": "Cell notes..."})
    def test_builds_quiz_from_existing_notes(self, mock_notes, mock_text):
        result = sq.study_quiz(parameters={"question_count": 2, "difficulty": "easy"})
        mock_text.assert_called_once()
        self.assertIn("Mitochondria", result)
        self.assertIn("Nucleus", result)
        self.assertIn("2 questions", result)
        self.assertIn("easy", result)

    @patch("actions.study_quiz._text_query", return_value=_GOOD_JSON)
    @patch("actions.study_quiz.get_last_notes", return_value={"topic": "", "text": ""})
    @patch("actions.study_quiz._capture_and_extract_notes", return_value=("Fresh notes text", ""))
    def test_falls_back_to_fresh_capture_when_no_notes(self, mock_capture, mock_notes, mock_text):
        result = sq.study_quiz(parameters={})
        mock_capture.assert_called_once()
        mock_text.assert_called_once()
        self.assertIn("Mitochondria", result)

    @patch("actions.study_quiz.get_last_notes", return_value={"topic": "", "text": ""})
    @patch("actions.study_quiz._capture_and_extract_notes", return_value=("", "I couldn't find any readable study material on the screen right now."))
    def test_no_notes_and_capture_fails_returns_error(self, mock_capture, mock_notes):
        result = sq.study_quiz(parameters={})
        self.assertIn("couldn't find any readable study material", result)

    @patch("actions.study_quiz._text_query", return_value="not json at all")
    @patch("actions.study_quiz.get_last_notes", return_value={"topic": "", "text": "notes"})
    def test_malformed_json_is_reported_not_crashed(self, mock_notes, mock_text):
        result = sq.study_quiz(parameters={})
        self.assertIn("Could not generate a well-formed quiz", result)

    @patch("actions.study_quiz._text_query", return_value=_GOOD_JSON)
    @patch("actions.study_quiz.get_last_notes", return_value={"topic": "", "text": "notes"})
    def test_fenced_json_is_parsed(self, mock_notes, mock_text):
        mock_text.return_value = f"```json\n{_GOOD_JSON}\n```"
        result = sq.study_quiz(parameters={})
        self.assertIn("Mitochondria", result)

    @patch("actions.study_quiz._text_query", return_value=_GOOD_JSON)
    @patch("actions.study_quiz.get_last_notes", return_value={"topic": "", "text": "notes"})
    def test_question_count_is_clamped(self, mock_notes, mock_text):
        sq.study_quiz(parameters={"question_count": 999})
        prompt_used = mock_text.call_args[0][0]
        self.assertIn("exactly 15 quiz questions", prompt_used)

    @patch("actions.study_quiz._text_query", return_value=_GOOD_JSON)
    @patch("actions.study_quiz.get_last_notes", return_value={"topic": "", "text": "notes"})
    def test_invalid_difficulty_falls_back_to_medium(self, mock_notes, mock_text):
        sq.study_quiz(parameters={"difficulty": "impossible"})
        prompt_used = mock_text.call_args[0][0]
        self.assertIn("medium difficulty", prompt_used)


if __name__ == "__main__":
    unittest.main()
