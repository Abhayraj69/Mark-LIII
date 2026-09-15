"""Unit tests for actions/computer_control.py's typing path — specifically
that non-ASCII text (accents, curly quotes, emoji) is routed through the
clipboard instead of crashing pyautogui.typewrite(), which only knows a
fixed ASCII+control-key table."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import actions.computer_control as cc  # noqa: E402


class TestTypeIntoFocus(unittest.TestCase):
    @patch("actions.computer_control.pyautogui")
    def test_ascii_text_uses_typewrite(self, mock_pyautogui):
        mode = cc._type_into_focus("hello world")
        self.assertEqual(mode, "typed")
        mock_pyautogui.typewrite.assert_called_once()
        mock_pyautogui.hotkey.assert_not_called()

    @patch("actions.computer_control._get_os", return_value="windows")
    @patch("actions.computer_control.pyperclip")
    @patch("actions.computer_control.pyautogui")
    @patch("actions.computer_control._PYPERCLIP", True)
    def test_non_ascii_text_uses_clipboard(self, mock_pyautogui, mock_pyperclip, _mock_os):
        mode = cc._type_into_focus("café — 🎉")
        self.assertEqual(mode, "pasted")
        mock_pyperclip.copy.assert_called_once_with("café — 🎉")
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "v")
        mock_pyautogui.typewrite.assert_not_called()

    @patch("actions.computer_control.pyautogui")
    @patch("actions.computer_control._PYPERCLIP", False)
    def test_non_ascii_without_pyperclip_falls_back_to_ascii_only(self, mock_pyautogui):
        mode = cc._type_into_focus("café")
        self.assertIn("ascii-only fallback", mode)
        mock_pyautogui.typewrite.assert_called_once_with("caf", interval=0.03)


if __name__ == "__main__":
    unittest.main()
