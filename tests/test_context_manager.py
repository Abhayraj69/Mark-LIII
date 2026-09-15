"""Unit tests for core/context_manager.py — turn storage, semantic ranking,
budget truncation, and the forget/clear privacy controls."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context_manager import (  # noqa: E402
    ContextBundle,
    build_context,
    clear_turns,
    count_turns,
    forget_turns,
    log_turn,
    semantic_search,
)
from core.context_manager import _truncate_to_budget  # noqa: E402


class _IsolatedDBTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "context_store.db"

    def tearDown(self):
        self._tmpdir.cleanup()


class TestTurnStorage(_IsolatedDBTestCase):
    def test_log_and_count(self):
        log_turn("user", "hello there", db_path=self.db_path)
        log_turn("assistant", "hi, how can I help", db_path=self.db_path)
        self.assertEqual(count_turns(db_path=self.db_path), 2)

    def test_blank_content_is_not_stored(self):
        log_turn("user", "   ", db_path=self.db_path)
        self.assertEqual(count_turns(db_path=self.db_path), 0)


class TestSemanticSearch(_IsolatedDBTestCase):
    def test_related_turn_ranks_above_unrelated(self):
        log_turn("user", "how do I fix the git commit confirmation dialog", db_path=self.db_path)
        log_turn("user", "what's the weather forecast for tomorrow", db_path=self.db_path)
        hits = semantic_search("git commit confirmation issue", top_k=5, db_path=self.db_path)
        self.assertTrue(hits)
        self.assertIn("git commit confirmation", hits[0]["content"])

    def test_empty_query_returns_nothing(self):
        log_turn("user", "some content", db_path=self.db_path)
        self.assertEqual(semantic_search("", db_path=self.db_path), [])

    def test_empty_store_returns_nothing(self):
        self.assertEqual(semantic_search("anything", db_path=self.db_path), [])


class TestPrivacyControls(_IsolatedDBTestCase):
    def test_forget_removes_matching_turns_only(self):
        log_turn("user", "my api key is secret123", db_path=self.db_path)
        log_turn("user", "what's the capital of France", db_path=self.db_path)
        removed = forget_turns("api key", db_path=self.db_path)
        self.assertEqual(removed, 1)
        self.assertEqual(count_turns(db_path=self.db_path), 1)

    def test_clear_removes_everything(self):
        log_turn("user", "one", db_path=self.db_path)
        log_turn("user", "two", db_path=self.db_path)
        removed = clear_turns(db_path=self.db_path)
        self.assertEqual(removed, 2)
        self.assertEqual(count_turns(db_path=self.db_path), 0)


class TestBudgetTruncation(unittest.TestCase):
    def test_truncation_drops_retrieved_before_session(self):
        bundle = ContextBundle(
            session=["User: turn one", "JARVIS: reply one", "User: turn two"],
            project={"recent_files": ["a.py", "b.py"], "open_tasks": ["finish the report"]},
            preferences={"tone": {"value": "concise"}},
            retrieved=[{"content": "old relevant thing", "role": "user", "score": 0.5, "timestamp": "t"}],
        )
        full_len = len(bundle)
        _truncate_to_budget(bundle, max_chars=full_len - 1)
        # Retrieved (lowest priority) should be gone first; session/preferences kept.
        self.assertEqual(bundle.retrieved, [])
        self.assertTrue(bundle.session)
        self.assertTrue(bundle.preferences)

    def test_preferences_survive_extreme_truncation(self):
        bundle = ContextBundle(
            session=["User: a"] * 20,
            project={"recent_files": ["x.py"] * 10, "open_tasks": ["task"] * 10},
            preferences={"tone": {"value": "concise"}},
            retrieved=[{"content": f"item {i}", "role": "user", "score": 0.1, "timestamp": "t"} for i in range(10)],
        )
        _truncate_to_budget(bundle, max_chars=1)
        self.assertTrue(bundle.preferences)  # never dropped


class TestBuildContext(_IsolatedDBTestCase):
    def test_session_log_is_capped_to_session_turns(self):
        session_log = [f"User: turn {i}" for i in range(20)]
        bundle = build_context(
            "irrelevant query", session_log=session_log, session_turns=3, db_path=self.db_path
        )
        self.assertEqual(len(bundle.session), 3)
        self.assertEqual(bundle.session, session_log[-3:])

    def test_retrieved_excludes_turns_already_in_session(self):
        log_turn("user", "duplicate turn text", db_path=self.db_path)
        bundle = build_context(
            "duplicate turn text",
            session_log=["User: duplicate turn text"],
            db_path=self.db_path,
        )
        contents = [r["content"] for r in bundle.retrieved]
        self.assertNotIn("duplicate turn text", contents)

    def test_sections_are_labeled_not_a_raw_dump(self):
        bundle = build_context("hello", session_log=["User: hello"], db_path=self.db_path)
        prompt = bundle.to_prompt()
        self.assertIn("[RECENT CONVERSATION]", prompt)


if __name__ == "__main__":
    unittest.main()
