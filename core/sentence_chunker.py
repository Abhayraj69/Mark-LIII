"""
Incremental sentence chunker for streaming LLM output.

WHY this exists
────────────────
Streaming TTS needs to start speaking as soon as the first sentence of a
reply is complete, not wait for the whole response. But a naive split on
every "." would butcher "3.5", "e.g." and "Dr. Smith" mid-thought, and a
single run-on sentence with no terminal punctuation would block speech
indefinitely.

SentenceChunker buffers text deltas as they arrive and releases a sentence
only at a genuine boundary: a period/question mark/exclamation point
followed by whitespace, where the token immediately before it isn't a known
abbreviation and isn't a bare initial (so "3.5" and "e.g." never split,
since nothing follows the "." in "3.5" and "e.g" is on the abbreviation
list). A newline is always a boundary. As a last resort — so one long
run-on sentence doesn't stall speech — a comma boundary is used once the
buffered text exceeds ~200 characters.

Pure and I/O-free: it only ever manipulates strings passed to it, so it can
be unit tested without a network or audio stack.
"""
import re

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "approx", "no", "fig", "inc", "ltd", "co", "corp",
    "u.s", "u.k", "a.m", "p.m", "mt", "gen", "rev", "capt", "cmdr",
    "lt", "col", "maj", "sgt", "dept", "univ", "assoc",
}

# Trailing run of letters (allowing internal dots, so "e.g" survives as one
# token) immediately before the punctuation mark being evaluated.
_WORD_BEFORE = re.compile(r'([A-Za-z]+(?:\.[A-Za-z]+)*)$')

_DEFAULT_MAX_BUFFER = 200


class SentenceChunker:
    """Feed text deltas in; get complete sentences out as they finish."""

    def __init__(self, max_buffer: int = _DEFAULT_MAX_BUFFER):
        self._buf = ""
        self._max_buffer = max_buffer

    def feed(self, delta: str) -> list[str]:
        """Append a delta and return zero or more sentences it completed."""
        if not delta:
            return []
        self._buf += delta
        return self._drain()

    def flush(self) -> str | None:
        """Return and clear any leftover buffered text (call at stream end)."""
        text = self._buf.strip()
        self._buf = ""
        return text or None

    def _drain(self) -> list[str]:
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            sentence = self._buf[:cut].strip()
            self._buf = self._buf[cut:]
            if sentence:
                out.append(sentence)
        return out

    def _find_cut(self) -> int | None:
        candidates = []

        nl = self._buf.find("\n")
        if nl != -1:
            candidates.append(nl + 1)

        punct_cut = self._find_punct_cut()
        if punct_cut is not None:
            candidates.append(punct_cut)

        if candidates:
            return min(candidates)

        if len(self._buf) > self._max_buffer:
            comma = self._buf.find(",")
            if comma != -1:
                return comma + 1

        return None

    def _find_punct_cut(self) -> int | None:
        for m in re.finditer(r'[.!?]', self._buf):
            end = m.end()
            if end >= len(self._buf):
                continue   # nothing after it yet — wait for more input
            if not self._buf[end].isspace():
                continue   # e.g. the "." in "3.5" — not a boundary
            if self._is_abbreviation(self._buf[:m.start()]):
                continue
            j = end
            while j < len(self._buf) and self._buf[j].isspace():
                j += 1
            return j
        return None

    @staticmethod
    def _is_abbreviation(before: str) -> bool:
        m = _WORD_BEFORE.search(before)
        if not m:
            return False
        word = m.group(1)
        if len(word) == 1 and word.isupper():
            return True   # a bare initial, e.g. "J. Smith"
        return word.lower() in _ABBREVIATIONS
