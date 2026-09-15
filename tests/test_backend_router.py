"""Unit tests for core/backend_router.py — per-task-kind backend selection,
circuit breaker, and failover. Uses fake adapters (no network, no real
Ollama/Claude/Gemini config needed)."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import backend_router as router  # noqa: E402
from core.backend_router import TaskKind  # noqa: E402


def _ok(name):
    def _adapter(messages, tools, images, timeout):
        return {"content": f"ok from {name}", "tool_calls": [], "usage": {}, "backend": name}
    return _adapter


def _fail(name, exc=RuntimeError("boom")):
    def _adapter(messages, tools, images, timeout):
        raise exc
    return _adapter


class TestBackendRouter(unittest.TestCase):
    def setUp(self):
        router.reset_breakers()
        self._orig_adapters = dict(router._ADAPTERS)
        self._configured_patch = patch.object(router, "_is_configured", return_value=True)
        self._configured_patch.start()

    def tearDown(self):
        router._ADAPTERS.clear()
        router._ADAPTERS.update(self._orig_adapters)
        router.reset_breakers()
        self._configured_patch.stop()

    def test_policy_order_respected(self):
        router._ADAPTERS["claude"] = _ok("claude")
        router._ADAPTERS["ollama"] = _ok("ollama")
        router._ADAPTERS["gemini"] = _ok("gemini")

        result = router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])
        self.assertEqual(result["backend"], "claude")   # first in DEFAULT_POLICY[CODE_GEN]

        result = router.complete(TaskKind.INTENT, [{"role": "user", "content": "hi"}])
        self.assertEqual(result["backend"], "ollama")   # first in DEFAULT_POLICY[INTENT]

    def test_falls_over_to_next_backend_on_failure(self):
        router._ADAPTERS["claude"] = _fail("claude")
        router._ADAPTERS["ollama"] = _ok("ollama")
        router._ADAPTERS["gemini"] = _ok("gemini")

        result = router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])
        self.assertEqual(result["backend"], "ollama")

    def test_breaker_opens_after_failure_and_skips_on_next_call(self):
        calls = {"claude": 0}

        def _flaky(messages, tools, images, timeout):
            calls["claude"] += 1
            raise RuntimeError("down")

        router._ADAPTERS["claude"] = _flaky
        router._ADAPTERS["ollama"] = _ok("ollama")
        router._ADAPTERS["gemini"] = _ok("gemini")

        router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])
        router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi again"}])

        self.assertEqual(calls["claude"], 1)   # second call skipped claude — breaker open

    def test_breaker_closes_after_cooldown_window(self):
        router._ADAPTERS["claude"] = _fail("claude")
        router._ADAPTERS["ollama"] = _ok("ollama")
        router._ADAPTERS["gemini"] = _ok("gemini")

        router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])
        self.assertTrue(router._breaker_open("claude"))

        with patch("time.monotonic", return_value=__import__("time").monotonic() + 61):
            self.assertFalse(router._breaker_open("claude"))

    def test_unconfigured_backend_is_skipped(self):
        def _configured(name):
            return name != "claude"

        with patch.object(router, "_is_configured", side_effect=_configured):
            router._ADAPTERS["claude"] = _ok("claude")   # would succeed if tried
            router._ADAPTERS["ollama"] = _ok("ollama")
            router._ADAPTERS["gemini"] = _ok("gemini")

            result = router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])
            self.assertEqual(result["backend"], "ollama")

    def test_all_backends_failing_raises(self):
        router._ADAPTERS["claude"] = _fail("claude")
        router._ADAPTERS["ollama"] = _fail("ollama")
        router._ADAPTERS["gemini"] = _fail("gemini")

        with self.assertRaises(RuntimeError):
            router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])

    def test_all_backends_unconfigured_raises(self):
        with patch.object(router, "_is_configured", return_value=False):
            with self.assertRaises(RuntimeError):
                router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])

    def test_usage_and_content_pass_through(self):
        def _adapter(messages, tools, images, timeout):
            return {"content": "hello", "tool_calls": [{"id": "1"}], "usage": {"tokens": 5}, "backend": "claude"}
        router._ADAPTERS["claude"] = _adapter

        result = router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}])
        self.assertEqual(result["content"], "hello")
        self.assertEqual(result["usage"], {"tokens": 5})
        self.assertEqual(result["tool_calls"], [{"id": "1"}])

    def test_get_text_model_wraps_complete(self):
        router._ADAPTERS["claude"] = _ok("claude")
        model = router.get_text_model(TaskKind.CODE_GEN)
        result = model.generate_content("write a haiku")
        self.assertEqual(result.text, "ok from claude")

    def test_custom_policy_overrides_default(self):
        router._ADAPTERS["ollama"] = _ok("ollama")
        router._ADAPTERS["claude"] = _ok("claude")
        custom_policy = {TaskKind.CODE_GEN: ["ollama", "claude"]}

        result = router.complete(TaskKind.CODE_GEN, [{"role": "user", "content": "hi"}],
                                  policy=custom_policy)
        self.assertEqual(result["backend"], "ollama")


if __name__ == "__main__":
    unittest.main()
