"""Unit tests for actions/dev_agent.py and actions/code_helper.py's text-model
selection, now routed through core/backend_router.py (see
tests/test_backend_router.py for the router's own policy/failover contract).
Both modules build a `core.backend_router._RoutedTextModel` and defer the
actual claude/ollama/gemini choice to `backend_router.complete()` at
`.generate_content()` time, rather than picking a backend class up front —
so these tests assert the TaskKind each call site requests and that the
returned object defers to `complete()`, with no network calls."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actions import code_helper, dev_agent  # noqa: E402
from core.backend_router import TaskKind, _RoutedTextModel  # noqa: E402


class TestDevAgentBackendSelection(unittest.TestCase):
    def test_returns_routed_text_model(self):
        model = dev_agent._get_model()
        self.assertIsInstance(model, _RoutedTextModel)
        self.assertEqual(model.kind, TaskKind.CODE_GEN)

    @patch("core.backend_router.complete")
    def test_generate_content_defers_to_router(self, mock_complete):
        mock_complete.return_value = {"content": "hi", "tool_calls": [], "usage": {}, "backend": "claude"}
        model = dev_agent._get_model()
        result = model.generate_content("write a haiku")
        self.assertEqual(result.text, "hi")
        self.assertEqual(mock_complete.call_args.args[0], TaskKind.CODE_GEN)


class TestCodeHelperBackendSelection(unittest.TestCase):
    def test_returns_routed_text_model(self):
        model = code_helper._get_gemini()
        self.assertIsInstance(model, _RoutedTextModel)
        self.assertEqual(model.kind, TaskKind.CODE_GEN)

    def test_explain_action_uses_code_review_kind(self):
        model = code_helper._get_gemini(TaskKind.CODE_REVIEW)
        self.assertEqual(model.kind, TaskKind.CODE_REVIEW)

    @patch("core.backend_router.complete")
    def test_generate_content_defers_to_router(self, mock_complete):
        mock_complete.return_value = {"content": "hi", "tool_calls": [], "usage": {}, "backend": "ollama"}
        model = code_helper._get_gemini()
        result = model.generate_content("explain this")
        self.assertEqual(result.text, "hi")
        self.assertEqual(mock_complete.call_args.args[0], TaskKind.CODE_GEN)


if __name__ == "__main__":
    unittest.main()
