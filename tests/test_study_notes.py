"""Unit tests for actions/study_notes.py — mocks the screen capture, vision
call, and file write so these test the handler's own logic (NO_CONTENT
handling, save path passthrough) rather than real vision output quality."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import actions.study_notes as sn  # noqa: E402


class TestStudyNotes(unittest.TestCase):
    @patch("actions.study_notes.open_app", return_value="Opened notepad with StudyNotes_x.txt.")
    @patch("actions.study_notes.create_file", return_value="File created: StudyNotes_x.txt")
    @patch("actions.study_notes._vision_query", return_value="# Title\n- **Term**: def")
    @patch("actions.study_notes._capture_screen", return_value=(b"fake", "image/jpeg"))
    def test_saves_notes_on_success(self, mock_capture, mock_vision, mock_create, mock_open):
        result = sn.study_notes(parameters={"topic_hint": "biology"})
        mock_capture.assert_called_once()
        mock_vision.assert_called_once()
        mock_create.assert_called_once()
        self.assertIn("Notes saved", result)
        self.assertIn("Term", result)
        self.assertEqual(sn.get_last_notes()["topic"], "biology")

    @patch("actions.study_notes._vision_query", return_value="NO_CONTENT")
    @patch("actions.study_notes._capture_screen", return_value=(b"fake", "image/jpeg"))
    def test_no_content_does_not_write_file(self, mock_capture, mock_vision):
        with patch("actions.study_notes.create_file") as mock_create:
            result = sn.study_notes(parameters={})
            mock_create.assert_not_called()
        self.assertIn("couldn't find any readable study material", result)

    @patch("actions.study_notes._capture_screen", side_effect=RuntimeError("no screen"))
    def test_capture_failure_is_reported(self, mock_capture):
        result = sn.study_notes(parameters={})
        self.assertIn("Could not capture the screen", result)

    @patch("actions.study_notes.open_app", return_value="Opened notepad.")
    @patch("actions.study_notes.create_file", return_value="File created: StudyNotes_x.txt")
    @patch("actions.study_notes._vision_query", return_value="notes")
    @patch("actions.study_notes._capture_screen", return_value=(b"fake", "image/jpeg"))
    def test_custom_save_path_is_passed_through(self, mock_capture, mock_vision, mock_create, mock_open):
        sn.study_notes(parameters={"save_path": "documents/School"})
        args, kwargs = mock_create.call_args
        self.assertTrue(args[0].replace("\\", "/").endswith("Documents/School"))

    @patch("actions.study_notes.open_app")
    @patch("actions.study_notes.create_file", return_value="File created: StudyNotes_x.txt")
    @patch("actions.study_notes._vision_query", return_value="notes")
    @patch("actions.study_notes._capture_screen", return_value=(b"fake", "image/jpeg"))
    def test_opens_notepad_by_default(self, mock_capture, mock_vision, mock_create, mock_open):
        mock_open.return_value = "Opened notepad with x.txt."
        result = sn.study_notes(parameters={})
        mock_open.assert_called_once()
        call_kwargs = mock_open.call_args.kwargs["parameters"]
        self.assertEqual(call_kwargs["app_name"], "notepad")
        self.assertIn("file_path", call_kwargs)
        self.assertNotIn("Could not open Notepad", result)

    @patch("actions.study_notes.open_app")
    @patch("actions.study_notes.create_file", return_value="File created: StudyNotes_x.txt")
    @patch("actions.study_notes._vision_query", return_value="notes")
    @patch("actions.study_notes._capture_screen", return_value=(b"fake", "image/jpeg"))
    def test_open_notepad_false_skips_launch(self, mock_capture, mock_vision, mock_create, mock_open):
        sn.study_notes(parameters={"open_notepad": False})
        mock_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
