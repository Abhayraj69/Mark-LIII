"""Unit tests for core.llm_client.stream_llm() — the raw streaming chat
request for Ollama's NDJSON /api/chat and OpenAI-compatible SSE
/v1/chat/completions. Network is mocked; no live Ollama/LM Studio server
required."""

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import llm_client  # noqa: E402


def _fake_stream_response(lines: list[bytes]) -> MagicMock:
    """A MagicMock standing in for `with requests.post(..., stream=True) as resp`."""
    resp = MagicMock()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = lines
    return resp


class TestOllamaStream(unittest.TestCase):
    @patch("core.llm_client.requests.post")
    def test_yields_deltas_then_done(self, mock_post):
        lines = [
            b'{"message": {"content": "Hello"}, "done": false}',
            b'{"message": {"content": " world."}, "done": false}',
            b'{"message": {}, "done": true, "prompt_eval_count": 5, "eval_count": 3}',
        ]
        mock_post.return_value = _fake_stream_response(lines)

        events = list(llm_client.stream_llm([{"role": "user", "content": "hi"}]))

        self.assertEqual(events[0], {"delta": "Hello"})
        self.assertEqual(events[1], {"delta": " world."})
        self.assertEqual(
            events[2],
            {"done": {"prompt_tokens": 5, "completion_tokens": 3}},
        )

    @patch("core.llm_client.requests.post")
    def test_tool_call_is_yielded_whole(self, mock_post):
        lines = [
            b'{"message": {"tool_calls": [{"id": "1", "function": '
            b'{"name": "get_weather", "arguments": {"city": "NYC"}}}]}, "done": false}',
            b'{"message": {}, "done": true}',
        ]
        mock_post.return_value = _fake_stream_response(lines)

        events = list(llm_client.stream_llm([{"role": "user", "content": "weather?"}]))

        tool_events = [e for e in events if "tool_call" in e]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(
            tool_events[0]["tool_call"],
            {"id": "1", "function": {"name": "get_weather", "arguments": {"city": "NYC"}}},
        )

    @patch("core.llm_client.requests.post")
    def test_cancel_event_set_before_start_yields_nothing(self, mock_post):
        lines = [b'{"message": {"content": "too late"}, "done": false}']
        fake_resp = _fake_stream_response(lines)
        mock_post.return_value = fake_resp

        cancel = threading.Event()
        cancel.set()
        events = list(llm_client.stream_llm(
            [{"role": "user", "content": "hi"}], cancel_event=cancel,
        ))

        self.assertEqual(events, [])
        self.assertTrue(fake_resp.__exit__.called)


class TestOpenAIStream(unittest.TestCase):
    @patch("core.llm_client.get_llm_provider", return_value="openai")
    @patch("core.llm_client.requests.post")
    def test_yields_deltas_then_done(self, mock_post, _mock_provider):
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            b'data: [DONE]',
        ]
        mock_post.return_value = _fake_stream_response(lines)

        events = list(llm_client.stream_llm([{"role": "user", "content": "hi"}]))

        self.assertIn({"delta": "Hi"}, events)
        self.assertEqual(events[-1], {"done": {}})

    @patch("core.llm_client.get_llm_provider", return_value="openai")
    @patch("core.llm_client.requests.post")
    def test_chunked_tool_call_fragments_reassembled(self, mock_post, _mock_provider):
        # OpenAI streams tool-call name/arguments in pieces keyed by index —
        # this is the case stream_llm must accumulate before yielding.
        lines = [
            b'data: {"choices":[{"delta":{"content":"Sure, "}}]}',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            b'"function":{"name":"get_","arguments":""}}]}}]}',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            b'"function":{"name":"weather","arguments":"{\\"city\\""}}]}}]}',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            b'"function":{"arguments":": \\"NYC\\"}"}}]}}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            b'data: [DONE]',
        ]
        mock_post.return_value = _fake_stream_response(lines)

        events = list(llm_client.stream_llm(
            [{"role": "user", "content": "weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        ))

        tool_events = [e for e in events if "tool_call" in e]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(
            tool_events[0]["tool_call"],
            {"id": "call_1", "function": {"name": "get_weather", "arguments": {"city": "NYC"}}},
        )
        self.assertIn({"delta": "Sure, "}, events)

    @patch("core.llm_client.get_llm_provider", return_value="openai")
    @patch("core.llm_client.requests.post")
    def test_cancel_event_set_before_start_yields_nothing(self, mock_post, _mock_provider):
        lines = [b'data: {"choices":[{"delta":{"content":"too late"}}]}']
        fake_resp = _fake_stream_response(lines)
        mock_post.return_value = fake_resp

        cancel = threading.Event()
        cancel.set()
        events = list(llm_client.stream_llm(
            [{"role": "user", "content": "hi"}], cancel_event=cancel,
        ))

        self.assertEqual(events, [])
        self.assertTrue(fake_resp.__exit__.called)


if __name__ == "__main__":
    unittest.main()
