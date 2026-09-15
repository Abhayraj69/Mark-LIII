"""Unit tests for actions/open_app.py's file_path extension — opening a
specific file with an app (e.g. Notepad with a saved notes file) instead of
just launching it blank, and the safe-path guard on that argument."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import actions.open_app as oa  # noqa: E402


class TestOpenAppFilePath(unittest.TestCase):
    @patch("actions.file_controller._is_safe_path", return_value=True)
    def test_safe_file_path_accepts_existing_file_under_home(self, mock_safe):
        with patch("pathlib.Path.exists", return_value=True):
            result = oa._safe_file_path(str(Path.home() / "notes.txt"))
        self.assertTrue(result)

    def test_safe_file_path_rejects_missing_file(self):
        result = oa._safe_file_path(str(Path.home() / "definitely_missing_xyz.txt"))
        self.assertEqual(result, "")

    def test_safe_file_path_empty_input(self):
        self.assertEqual(oa._safe_file_path(""), "")

    @patch("actions.open_app._safe_file_path", return_value="")
    @patch("actions.open_app._normalize", return_value="notepad.exe")
    def test_unsafe_file_path_is_dropped_not_passed_through(self, mock_norm, mock_safe):
        with patch.object(oa, "_OS_LAUNCHERS", {oa._SYSTEM: lambda app, fp: fp == ""}):
            result = oa.open_app(parameters={"app_name": "notepad", "file_path": "/etc/passwd"})
        self.assertIn("Opened", result)

    @patch("actions.open_app._safe_file_path", return_value="C:/Users/x/notes.txt")
    @patch("actions.open_app._normalize", return_value="notepad.exe")
    def test_valid_file_path_reaches_launcher(self, mock_norm, mock_safe):
        captured = {}
        def fake_launcher(app, fp):
            captured["app"] = app
            captured["fp"] = fp
            return True
        with patch.object(oa, "_OS_LAUNCHERS", {oa._SYSTEM: fake_launcher}):
            result = oa.open_app(parameters={"app_name": "notepad", "file_path": "C:/Users/x/notes.txt"})
        self.assertEqual(captured["fp"], "C:/Users/x/notes.txt")
        self.assertIn("notes.txt", result)


if __name__ == "__main__":
    unittest.main()
