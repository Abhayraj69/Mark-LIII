"""Unit tests for core/claude_bridge.py — history-shape conversion, config
resolution, and the Messages API round trip (network mocked; no live key or
API calls)."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.claude_bridge import (  # noqa: E402
    ClaudeTextModel,
    call_claude,
    call_claude_text,
    get_claude_settings,
    is_claude_engine_enabled,
    to_claude_messages,
)


class TestConfigResolution(unittest.TestCase):
    @patch("core.claude_bridge._load_config", return_value={})
    def test_disabled_and_defaults_when_unconfigured(self, _mock_cfg):
        self.assertFalse(is_claude_engine_enabled())
        api_key, model, max_tokens = get_claude_settings()
        self.assertEqual(api_key, "")
        self.assertEqual(model, "claude-sonnet-5")
        self.assertEqual(max_tokens, 1024)

    @patch("core.claude_bridge._load_config")
    def test_reads_stored_values(self, mock_cfg):
        mock_cfg.return_value = {
            "enabled": True, "api_key": "sk-ant-test", "model": "claude-opus-5",
            "max_tokens": "500",
        }
        self.assertTrue(is_claude_engine_enabled())
        api_key, model, max_tokens = get_claude_settings()
        self.assertEqual(api_key, "sk-ant-test")
        self.assertEqual(model, "claude-opus-5")
        self.assertEqual(max_tokens, 500)

    @patch("core.claude_bridge._load_config", return_value={"max_tokens": "not-a-number"})
    def test_unparseable_max_tokens_falls_back_to_default(self, _mock_cfg):
        _, _, max_tokens = get_claude_settings()
        self.assertEqual(max_tokens, 1024)


class TestToClaudeMessages(unittest.TestCase):
    def test_system_role_dropped(self):
        out = to_claude_messages([{"role": "system", "content": "you are jarvis"}])
        self.assertEqual(out, [])

    def test_user_message_passthrough(self):
        out = to_claude_messages([{"role": "user", "content": "open chrome"}])
        self.assertEqual(out, [{"role": "user", "content": "open chrome"}])

    def test_plain_assistant_text(self):
        out = to_claude_messages([{"role": "assistant", "content": "done"}])
        self.assertEqual(out, [{"role": "assistant", "content": [{"type": "text", "text": "done"}]}])

    def test_assistant_tool_call_becomes_tool_use_block(self):
        history = [{
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "toolu_1", "function": {"name": "move_file", "arguments": {"a": 1}}}],
        }]
        out = to_claude_messages(history)
        self.assertEqual(out, [{
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "move_file", "input": {"a": 1}}],
        }])

    def test_tool_result_becomes_user_tool_result_block(self):
        history = [{"role": "tool", "tool_call_id": "toolu_1", "name": "move_file", "content": "moved"}]
        out = to_claude_messages(history)
        self.assertEqual(out, [{
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "moved"}],
        }])

    def test_empty_assistant_message_with_no_tool_calls_has_empty_content(self):
        out = to_claude_messages([{"role": "assistant", "content": ""}])
        self.assertEqual(out, [{"role": "assistant", "content": ""}])


def _fake_response(content_blocks, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"content": content_blocks}
    resp.raise_for_status.return_value = None
    return resp


class TestCallClaude(unittest.TestCase):
    @patch("core.claude_bridge._load_config", return_value={})
    def test_missing_api_key_raises_without_network_call(self, _mock_cfg):
        with self.assertRaises(RuntimeError):
            call_claude([{"role": "user", "content": "hi"}])

    @patch("core.claude_bridge._session.post")
    @patch("core.claude_bridge._load_config", return_value={"enabled": True, "api_key": "sk-ant-test"})
    def test_text_only_response_normalized(self, _mock_cfg, mock_post):
        mock_post.return_value = _fake_response([{"type": "text", "text": "Sure thing."}])
        result = call_claude([{"role": "user", "content": "hi"}])
        self.assertEqual(result, {"content": "Sure thing.", "tool_calls": []})

    @patch("core.claude_bridge._session.post")
    @patch("core.claude_bridge._load_config", return_value={"enabled": True, "api_key": "sk-ant-test"})
    def test_tool_use_response_normalized(self, _mock_cfg, mock_post):
        mock_post.return_value = _fake_response([
            {"type": "text", "text": "Let me check that."},
            {"type": "tool_use", "id": "toolu_9", "name": "move_file", "input": {"source": "a", "destination": "b"}},
        ])
        result = call_claude([{"role": "user", "content": "move a to b"}])
        self.assertEqual(result["content"], "Let me check that.")
        self.assertEqual(result["tool_calls"], [{
            "id": "toolu_9",
            "function": {"name": "move_file", "arguments": {"source": "a", "destination": "b"}},
        }])

    @patch("core.claude_bridge._session.post")
    @patch("core.claude_bridge._load_config", return_value={"enabled": True, "api_key": "sk-ant-test"})
    def test_sends_system_and_tools_when_provided(self, _mock_cfg, mock_post):
        mock_post.return_value = _fake_response([{"type": "text", "text": "ok"}])
        tools = [{"name": "move_file", "description": "", "input_schema": {"type": "object", "properties": {}}}]
        call_claude([{"role": "user", "content": "hi"}], tools=tools, system="be concise")
        sent_payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(sent_payload["system"], "be concise")
        self.assertEqual(sent_payload["tools"], tools)
        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(headers["x-api-key"], "sk-ant-test")

    @patch("core.claude_bridge._session.post", side_effect=__import__("requests").exceptions.Timeout())
    @patch("core.claude_bridge._load_config", return_value={"enabled": True, "api_key": "sk-ant-test"})
    def test_timeout_raises_runtime_error(self, _mock_cfg, _mock_post):
        with self.assertRaises(RuntimeError):
            call_claude([{"role": "user", "content": "hi"}])


class TestCallClaudeText(unittest.TestCase):
    @patch("core.claude_bridge._session.post")
    @patch("core.claude_bridge._load_config", return_value={"enabled": True, "api_key": "sk-ant-test"})
    def test_returns_plain_text_content(self, _mock_cfg, mock_post):
        mock_post.return_value = _fake_response([{"type": "text", "text": "print('hi')"}])
        self.assertEqual(call_claude_text("write a hello world"), "print('hi')")

    @patch("core.claude_bridge._session.post")
    @patch("core.claude_bridge._load_config", return_value={"enabled": True, "api_key": "sk-ant-test"})
    def test_passes_system_prompt_through(self, _mock_cfg, mock_post):
        mock_post.return_value = _fake_response([{"type": "text", "text": "ok"}])
        call_claude_text("hi", system="be terse")
        self.assertEqual(mock_post.call_args.kwargs["json"]["system"], "be terse")


class TestClaudeTextModel(unittest.TestCase):
    @patch("core.claude_bridge._session.post")
    @patch("core.claude_bridge._load_config", return_value={"enabled": True, "api_key": "sk-ant-test"})
    def test_generate_content_returns_object_with_text_attribute(self, _mock_cfg, mock_post):
        mock_post.return_value = _fake_response([{"type": "text", "text": "def foo(): pass"}])
        response = ClaudeTextModel().generate_content("write a no-op function")
        self.assertEqual(response.text, "def foo(): pass")

    @patch("core.claude_bridge._session.post")
    @patch("core.claude_bridge._load_config", return_value={"enabled": True, "api_key": "sk-ant-test"})
    def test_non_string_contents_stringified(self, _mock_cfg, mock_post):
        mock_post.return_value = _fake_response([{"type": "text", "text": "ok"}])
        ClaudeTextModel().generate_content(["part one", "part two"])
        sent_prompt = mock_post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("part one", sent_prompt)


if __name__ == "__main__":
    unittest.main()
