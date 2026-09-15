"""Regression test for actions/file_controller.py's _resolve_path: a compound
path like "desktop/JarvisNotes" must resolve under the real Desktop folder,
not silently become a path relative to the current working directory (the
bug that broke study_notes' default save location)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import actions.file_controller as fc  # noqa: E402


class TestResolvePath(unittest.TestCase):
    def test_bare_shortcut(self):
        self.assertEqual(fc._resolve_path("desktop"), fc._get_desktop())

    def test_compound_shortcut_path(self):
        result = fc._resolve_path("desktop/JarvisNotes")
        self.assertEqual(result, fc._get_desktop() / "JarvisNotes")

    def test_compound_shortcut_path_backslash(self):
        result = fc._resolve_path("documents\\School")
        self.assertEqual(result, fc._get_documents() / "School")

    def test_compound_path_stays_under_home(self):
        result = fc._resolve_path("desktop/JarvisNotes")
        self.assertTrue(fc._is_safe_path(result))

    def test_non_shortcut_path_untouched(self):
        result = fc._resolve_path("C:/some/other/path")
        self.assertEqual(result, Path("C:/some/other/path").expanduser())


if __name__ == "__main__":
    unittest.main()
