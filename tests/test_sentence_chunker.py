"""Unit tests for core/sentence_chunker.py — incremental sentence splitting
for streaming LLM output. Pure string logic, no mocking required."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.sentence_chunker import SentenceChunker  # noqa: E402


class TestSentenceChunker(unittest.TestCase):
    def test_splits_on_full_stop_followed_by_whitespace(self):
        chunker = SentenceChunker()
        sentences = chunker.feed("Hello world. How are you? Great!")
        self.assertEqual(sentences, ["Hello world.", "How are you?"])
        self.assertEqual(chunker.flush(), "Great!")

    def test_incremental_feed_matches_single_shot(self):
        text = "This is one. This is two. This is three."
        whole = SentenceChunker()
        one_shot = whole.feed(text)
        one_shot.append(whole.flush())

        incremental = SentenceChunker()
        pieces = []
        for ch in text:
            pieces.extend(incremental.feed(ch))
        tail = incremental.flush()
        if tail:
            pieces.append(tail)

        self.assertEqual(one_shot, pieces)
        self.assertEqual(one_shot, ["This is one.", "This is two.", "This is three."])

    def test_does_not_split_on_decimal_numbers(self):
        chunker = SentenceChunker()
        sentences = chunker.feed("The value is 3.5 exactly. Next sentence.")
        self.assertEqual(sentences, ["The value is 3.5 exactly."])
        self.assertEqual(chunker.flush(), "Next sentence.")

    def test_does_not_split_on_abbreviations(self):
        chunker = SentenceChunker()
        sentences = chunker.feed(
            "Bring supplies, e.g. water and food. Ask Dr. Lee for help. Thanks all."
        )
        self.assertEqual(
            sentences,
            ["Bring supplies, e.g. water and food.", "Ask Dr. Lee for help."],
        )
        self.assertEqual(chunker.flush(), "Thanks all.")

    def test_does_not_split_on_bare_initials(self):
        chunker = SentenceChunker()
        sentences = chunker.feed("J. Smith arrived early. He left late.")
        self.assertEqual(sentences, ["J. Smith arrived early."])
        self.assertEqual(chunker.flush(), "He left late.")

    def test_newline_is_always_a_boundary(self):
        chunker = SentenceChunker()
        sentences = chunker.feed("Line one\nLine two")
        self.assertEqual(sentences, ["Line one"])
        self.assertEqual(chunker.flush(), "Line two")

    def test_long_run_on_falls_back_to_comma(self):
        chunker = SentenceChunker(max_buffer=40)
        # No terminal punctuation at all — must not stall forever.
        long_text = "this is a very long run on clause without end, and it keeps going"
        sentences = chunker.feed(long_text)
        self.assertTrue(sentences)
        self.assertTrue(sentences[0].endswith(","))

    def test_flush_returns_none_when_buffer_empty(self):
        chunker = SentenceChunker()
        chunker.feed("Complete sentence. ")
        self.assertEqual(chunker.flush(), None)

    def test_empty_delta_is_noop(self):
        chunker = SentenceChunker()
        self.assertEqual(chunker.feed(""), [])


if __name__ == "__main__":
    unittest.main()
