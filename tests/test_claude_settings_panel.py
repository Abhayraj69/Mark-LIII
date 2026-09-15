"""Unit tests for the Phase 3 settings panel wiring — main.py's
JarvisLive._claude_settings_section() / _test_claude_engine() render the same
generic PluginSettingsOverlay schema the existing ENGINE and TONE ADAPTATION
sections use (see main.py: _engine_settings_section, _sentiment_settings_section).

Both methods are exercised on a bare, un-initialized JarvisLive instance
(object.__new__) since neither touches anything set up in __init__ — the
section builder only reads module-level get_plugin_config() and looks up its
own bound _test_claude_engine method, and the test-connection probe only
reads the values dict the settings panel would have passed it. This avoids
constructing the real app (Qt widgets, audio devices, a live Gemini session)
just to test two pure-ish methods."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402


def _bare_jarvis():
    return object.__new__(main.JarvisLive)


class TestClaudeSettingsSection(unittest.TestCase):
    def test_schema_shape_matches_overlay_contract(self):
        section = main.JarvisLive._claude_settings_section(_bare_jarvis())
        self.assertEqual(section["namespace"], "claude_engine")
        self.assertIn("title", section)
        self.assertEqual(
            [f["key"] for f in section["fields"]],
            ["enabled", "api_key", "model", "max_tokens"],
        )

    def test_enabled_field_is_a_toggle_defaulting_off(self):
        section = main.JarvisLive._claude_settings_section(_bare_jarvis())
        enabled_field = section["fields"][0]
        self.assertEqual(enabled_field["type"], "toggle")
        self.assertFalse(enabled_field["default"])

    def test_api_key_field_is_masked(self):
        section = main.JarvisLive._claude_settings_section(_bare_jarvis())
        api_key_field = section["fields"][1]
        self.assertEqual(api_key_field["type"], "password")

    def test_action_wired_to_test_claude_engine(self):
        section = main.JarvisLive._claude_settings_section(_bare_jarvis())
        self.assertTrue(callable(section["action"]["run"]))

    def test_included_in_settings_schemas(self):
        # _settings_schemas also calls self._plugin_registry.settings_schemas(),
        # which needs a real registry — stub it rather than constructing one.
        jarvis = _bare_jarvis()
        jarvis._plugin_registry = MagicMock(settings_schemas=MagicMock(return_value=[]))
        namespaces = [s["namespace"] for s in main.JarvisLive._settings_schemas(jarvis)]
        self.assertIn("claude_engine", namespaces)


def _fake_resp(status_code=200, body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body or {}
    return resp


class TestTestClaudeEngine(unittest.TestCase):
    def test_missing_api_key_short_circuits_without_network_call(self):
        with patch("requests.post") as mock_post:
            ok, msg = main.JarvisLive._test_claude_engine(_bare_jarvis(), {"api_key": ""})
        self.assertFalse(ok)
        self.assertIn("No API key", msg)
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_reachable_key_reports_success(self, mock_post):
        mock_post.return_value = _fake_resp(200)
        ok, msg = main.JarvisLive._test_claude_engine(
            _bare_jarvis(), {"api_key": "sk-ant-test", "model": "claude-sonnet-5"}
        )
        self.assertTrue(ok)
        self.assertIn("claude-sonnet-5", msg)

    @patch("requests.post")
    def test_bad_key_reports_http_status_and_error_detail(self, mock_post):
        mock_post.return_value = _fake_resp(401, {"error": {"message": "invalid x-api-key"}})
        ok, msg = main.JarvisLive._test_claude_engine(_bare_jarvis(), {"api_key": "sk-ant-bad"})
        self.assertFalse(ok)
        self.assertIn("401", msg)
        self.assertIn("invalid x-api-key", msg)

    @patch("requests.post", side_effect=ConnectionError("no route to host"))
    def test_network_failure_reports_error_without_raising(self, _mock_post):
        ok, msg = main.JarvisLive._test_claude_engine(_bare_jarvis(), {"api_key": "sk-ant-test"})
        self.assertFalse(ok)
        self.assertIn("no route to host", msg)


if __name__ == "__main__":
    unittest.main()
