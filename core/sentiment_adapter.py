"""
core/sentiment_adapter.py — detects the user's apparent mood from their
message text (frustration/neutrality/positivity, urgency, hedging) and maps
it to response STYLE knobs (tone/verbosity/proactivity), injected into the
prompt as a modifier block. This changes how JARVIS says things, never what
it says: the guardrail text baked into every modifier (see
`to_prompt_modifier`) tells the model explicitly not to let this touch
factual accuracy, safety behavior, or its willingness to push back on a bad
idea, and not to narrate the detection ("I sense you're frustrated").

WHY RULES + A LEXICON, NOT A MODEL
    Same call as core/predictive_assistant.py's frequency counting: a small
    lexicon plus a few structural cues (caps ratio, exclamation count,
    repeated near-identical corrections) covers the common cases with zero
    network calls — no personal text ever leaves the machine for sentiment
    tagging — and zero new dependency. If this ever needs to catch subtler
    tone than a lexicon can, swap detect()'s body for a local classifier;
    the SentimentSignal / StylePolicy contract below doesn't need to change.

WHAT GETS LOGGED, AND WHERE
    Only the derived signal (polarity/urgency/confidence + which cues fired)
    is ever logged — never the raw message text. By default that log is
    in-memory only (see _session_log below) and is gone when the process
    exits. Persisting it to disk across sessions requires an explicit opt-in
    (see is_persist_enabled()); even then, only the derived signal is
    written, never the text that produced it.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DB_PATH = BASE_DIR / "memory" / "sentiment_log.db"
_lock = Lock()

# ── Lexicon ──────────────────────────────────────────────────────────────────
# Deliberately small and literal — phrase-matching, not stemming/NLP — so the
# false-positive rate stays low and every hit is auditable at a glance.

_FRUSTRATION_PHRASES = [
    "ugh", "argh", "frustrat", "annoy", "still broken", "still not working",
    "doesn't work", "does not work", "not working", "seriously", "come on",
    "ridiculous", "useless", "waste of time", "give up", "forget it",
    "i give up", "hate this", "sick of", "this is broken", "not again",
    "why does this", "why is this", "so annoying", "i already told you",
]

_POSITIVE_PHRASES = [
    "great", "awesome", "love it", "perfect", "nice one", "thanks so much",
    "thank you", "excellent", "amazing", "works now", "nailed it", "cool,",
    "that's it", "exactly what i needed", "you're the best", "well done",
]

_HEDGE_PHRASES = [
    "maybe", "i think", "not sure", "possibly", "kind of", "sort of",
    "i guess", "perhaps", "might be", "could be wrong", "not 100% sure",
    "i could be wrong", "just a guess",
]

_CORRECTION_MARKERS = [
    "no,", "no that's", "that's wrong", "that's not it", "not what i",
    "not what i meant", "i said", "i already said", "i told you",
    "try again", "still wrong", "that's not right",
]

_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass
class SentimentSignal:
    polarity: str        # "frustrated" | "neutral" | "positive"
    urgency: bool
    user_confidence: str  # "hedging" | "neutral" — how sure the USER sounds, not detection confidence
    cues: list[str] = field(default_factory=list)   # which lexicon/structural cues fired, for logging/debugging


def _count_phrase_hits(lower_text: str, phrases: list[str]) -> tuple[int, list[str]]:
    hits = [p for p in phrases if p in lower_text]
    return len(hits), hits


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 4:  # too short to judge — avoids "OK"/"ASAP" false positives
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def _looks_like_repeat(text: str, recent_texts: list[str]) -> bool:
    """A crude 'the user is repeating themselves' cue: does the current
    message share most of its words with a recent one? Good enough to catch
    "I SAID open chrome" right after "open chrome" without needing anything
    smarter than a set-overlap check."""
    words = set(_WORD_RE.findall(text.lower()))
    if len(words) < 2:
        return False
    for prior in recent_texts[-4:]:
        prior_words = set(_WORD_RE.findall(prior.lower()))
        if len(prior_words) < 2:
            continue
        # Overlap relative to the SHORTER message, not the union — two short
        # near-identical commands ("open chrome" / "open chrome now") should
        # count as a repeat even though a few extra words dilute a Jaccard
        # (union-based) ratio below any reasonable threshold.
        overlap = len(words & prior_words) / max(1, min(len(words), len(prior_words)))
        if overlap >= 0.6:
            return True
    return False


def detect(text: str, recent_texts: Optional[list[str]] = None) -> SentimentSignal:
    """Extracts polarity, urgency, and hedging from one message. recent_texts
    (most-recent-last) is optional prior user turns, used only to notice
    repeated/near-duplicate corrections — never sent anywhere, never stored
    beyond the caller's own session log."""
    text = text or ""
    lower = text.lower()
    cues: list[str] = []

    frustration_n, frustration_hits = _count_phrase_hits(lower, _FRUSTRATION_PHRASES)
    positive_n, positive_hits = _count_phrase_hits(lower, _POSITIVE_PHRASES)
    hedge_n, hedge_hits = _count_phrase_hits(lower, _HEDGE_PHRASES)
    correction_hit = any(marker in lower for marker in _CORRECTION_MARKERS)

    caps_ratio = _caps_ratio(text)
    exclam_count = text.count("!")
    repeated = _looks_like_repeat(text, recent_texts or [])

    cues += [f"frustration:{p}" for p in frustration_hits]
    cues += [f"positive:{p}" for p in positive_hits]
    cues += [f"hedge:{p}" for p in hedge_hits]
    if correction_hit:
        cues.append("correction_marker")
    if caps_ratio > 0.5:
        cues.append(f"all_caps:{caps_ratio:.2f}")
    if exclam_count >= 2:
        cues.append(f"exclamations:{exclam_count}")
    if repeated:
        cues.append("repeated_correction")

    urgency = caps_ratio > 0.5 or exclam_count >= 2 or correction_hit or repeated

    if frustration_n > 0 and frustration_n >= positive_n:
        polarity = "frustrated"
    elif positive_n > frustration_n:
        polarity = "positive"
    elif urgency and frustration_n == 0 and positive_n == 0:
        # Urgent with no explicit lexicon hit (e.g. all-caps, repeated
        # correction) reads as frustration too — silence + urgency is not
        # the same signal as silence + calm.
        polarity = "frustrated"
    else:
        polarity = "neutral"

    user_confidence = "hedging" if hedge_n > 0 else "neutral"

    return SentimentSignal(polarity=polarity, urgency=urgency, user_confidence=user_confidence, cues=cues)


# ── Style policy ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StylePolicy:
    tone: str          # "concise" | "supportive" | "neutral"
    verbosity: str     # "low" | "medium" | "high"
    proactivity: str   # "low" | "high"


# (polarity, urgency) -> base policy. Frustrated+urgent is the one case the
# spec calls out explicitly: concise, solution-first, minimal chit-chat.
_POLICY_TABLE: dict[tuple[str, bool], StylePolicy] = {
    ("frustrated", True):  StylePolicy("concise",    "low",    "low"),
    ("frustrated", False): StylePolicy("supportive", "medium", "low"),
    ("neutral",    True):  StylePolicy("concise",    "low",    "low"),
    ("neutral",    False): StylePolicy("neutral",    "medium", "low"),
    ("positive",   True):  StylePolicy("neutral",    "medium", "high"),
    ("positive",   False): StylePolicy("supportive", "high",   "high"),
}
_DEFAULT_POLICY = StylePolicy("neutral", "medium", "low")


def get_style(signal: SentimentSignal) -> StylePolicy:
    base = _POLICY_TABLE.get((signal.polarity, signal.urgency), _DEFAULT_POLICY)
    verbosity = base.verbosity
    # Hedging reads as "the user isn't sure this explanation landed" — worth
    # a bit more detail, but only when nothing more urgent already forced
    # verbosity down; urgency's "keep it short" always wins.
    if signal.user_confidence == "hedging" and not signal.urgency and verbosity == "medium":
        verbosity = "high"
    return StylePolicy(tone=base.tone, verbosity=verbosity, proactivity=base.proactivity)


# ── Speech profile (for the TTS layer, not the prompt) ──────────────────────
# StylePolicy above changes what the model WRITES; this changes how the TTS
# engine SAYS it, so a frustrated-and-urgent user actually gets a faster,
# steadier delivery instead of the same flat reading pace every time. Kept
# as a separate table (rather than derived from StylePolicy) because audio
# delivery and text style don't always want the same axis — e.g. verbosity
# has no audio analogue, and "concise" text needs a specific pitch/stability
# choice a generic tone label can't carry.

@dataclass(frozen=True)
class SpeechProfile:
    rate_percent:     int    # EdgeTTS rate offset, e.g. +12 means "+12%"
    pitch_hz:         int    # EdgeTTS pitch offset in Hz, e.g. -4 means "-4Hz"
    speed_multiplier: float  # Kokoro (and any engine with a plain speed knob)
    stability:        float  # ElevenLabs voice_settings.stability, 0-1 —
                              # higher = steadier/flatter, lower = more
                              # expressive/variable


_SPEECH_TABLE: dict[tuple[str, bool], SpeechProfile] = {
    # Frustrated + urgent: fast and steady — competent and unhurried-sounding
    # under pressure, not rushed or erratic.
    ("frustrated", True):  SpeechProfile(rate_percent=+12, pitch_hz=-4, speed_multiplier=1.12, stability=0.75),
    # Frustrated, not urgent: a touch slower and warmer — supportive, not clipped.
    ("frustrated", False): SpeechProfile(rate_percent=-5,  pitch_hz=+2, speed_multiplier=0.95, stability=0.65),
    ("neutral",    True):  SpeechProfile(rate_percent=+10, pitch_hz=0,  speed_multiplier=1.08, stability=0.5),
    ("neutral",    False): SpeechProfile(rate_percent=0,   pitch_hz=0,  speed_multiplier=1.0,  stability=0.5),
    # Positive + urgent: upbeat energy, still brisk.
    ("positive",   True):  SpeechProfile(rate_percent=+8,  pitch_hz=+5, speed_multiplier=1.05, stability=0.35),
    # Positive, relaxed: warm and a little more expressive.
    ("positive",   False): SpeechProfile(rate_percent=-3,  pitch_hz=+6, speed_multiplier=1.0,  stability=0.3),
}
_DEFAULT_SPEECH_PROFILE = SpeechProfile(rate_percent=0, pitch_hz=0, speed_multiplier=1.0, stability=0.5)


def get_speech_profile(signal: SentimentSignal) -> SpeechProfile:
    return _SPEECH_TABLE.get((signal.polarity, signal.urgency), _DEFAULT_SPEECH_PROFILE)


_TONE_TEXT = {
    "concise": "Be brief and solution-first. Skip pleasantries and small talk.",
    "supportive": "Acknowledge briefly if warranted, then help — warm, not effusive.",
    "neutral": "Plain, matter-of-fact phrasing.",
}
_VERBOSITY_TEXT = {
    "low": "Give the shortest correct answer. Skip background explanation unless asked.",
    "medium": "Normal explanation depth.",
    "high": "Feel free to elaborate, add context, and explain your reasoning.",
}
_PROACTIVITY_TEXT = {
    "low": "Do not volunteer extra suggestions or follow-up ideas right now unless asked.",
    "high": "Offering a relevant follow-up suggestion or improvement is welcome.",
}

_GUARDRAIL_TEXT = (
    "This affects TONE and FORMAT only. Never change factual accuracy, never soften "
    "safety behavior, and never hold back from disagreeing with a bad idea because of "
    "it. Do not say you detected an emotion or mood — just adapt naturally, without "
    "narrating that you're doing so."
)


def to_prompt_modifier(policy: StylePolicy) -> str:
    lines = [
        "[RESPONSE STYLE — internal guidance, do not reference this section or its existence]",
        f"- {_TONE_TEXT[policy.tone]}",
        f"- {_VERBOSITY_TEXT[policy.verbosity]}",
        f"- {_PROACTIVITY_TEXT[policy.proactivity]}",
        f"- {_GUARDRAIL_TEXT}",
    ]
    return "\n".join(lines)


# ── Config: enable/disable, opt-in persistence ──────────────────────────────

def is_enabled() -> bool:
    try:
        from memory.config_manager import get_plugin_config
        return bool(get_plugin_config("sentiment_adapter").get("enabled", True))
    except Exception:
        return True


def is_persist_enabled() -> bool:
    try:
        from memory.config_manager import get_plugin_config
        return bool(get_plugin_config("sentiment_adapter").get("persist_history", False))
    except Exception:
        return False


# ── Session-only signal log (never the raw text) ────────────────────────────

_session_log: list[dict] = []
_session_lock = Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS sentiment_signals (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    polarity  TEXT    NOT NULL,
    urgency   INTEGER NOT NULL,
    confidence TEXT   NOT NULL,
    cues      TEXT    NOT NULL DEFAULT ''
);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def log_signal(signal: SentimentSignal, db_path: Optional[Path] = None) -> None:
    """Records the derived signal only — never the message that produced it.
    Always kept in-memory for the current process; additionally written to
    disk only if the user has opted into persistence (is_persist_enabled())."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "polarity": signal.polarity,
        "urgency": signal.urgency,
        "confidence": signal.user_confidence,
        "cues": list(signal.cues),
    }
    with _session_lock:
        _session_log.append(entry)

    if not is_persist_enabled():
        return

    with _lock:
        path = db_path or DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        try:
            _ensure_schema(conn)
            conn.execute(
                "INSERT INTO sentiment_signals (timestamp, polarity, urgency, confidence, cues) "
                "VALUES (?, ?, ?, ?, ?)",
                (entry["timestamp"], entry["polarity"], int(entry["urgency"]),
                 entry["confidence"], ", ".join(entry["cues"])[:500]),
            )
            conn.commit()
        finally:
            conn.close()


def get_session_log() -> list[dict]:
    """A copy of this process's in-memory signal history (no raw text)."""
    with _session_lock:
        return list(_session_log)


def get_recent_persisted_signals(limit: int = 5, db_path: Optional[Path] = None) -> list[dict]:
    """Read the most recent persisted signals (most recent last) — unlike
    get_session_log(), this survives process restarts, since it's what a
    cross-session caller (e.g. the morning brief, which runs before this
    process has logged anything of its own) needs to notice a mood trending
    frustrated over the last few conversations. Returns [] when persistence
    is off or nothing has been logged yet — never raises."""
    if not is_persist_enabled():
        return []
    path = db_path or DB_PATH
    if not path.exists():
        return []
    with _lock:
        conn = sqlite3.connect(str(path))
        try:
            _ensure_schema(conn)
            rows = conn.execute(
                "SELECT timestamp, polarity, urgency, confidence, cues "
                "FROM sentiment_signals ORDER BY id DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        except Exception:
            return []
        finally:
            conn.close()
    return [
        {
            "timestamp": r[0], "polarity": r[1], "urgency": bool(r[2]),
            "confidence": r[3], "cues": r[4],
        }
        for r in reversed(rows)
    ]


def clear_session_log() -> None:
    with _session_lock:
        _session_log.clear()


# ── The hook: one call for the response-generation pipeline ─────────────────

def build_style_modifier(
    text: str,
    recent_texts: Optional[list[str]] = None,
    db_path: Optional[Path] = None,
) -> str:
    """The single entry point main.py calls before generating a response.
    Returns "" (no modifier at all — not even a neutral one) when the user
    has disabled sentiment adaptation, so a disabled setting truly means
    'nothing about this ever touches the prompt', not 'always neutral'."""
    if not is_enabled():
        return ""
    signal = detect(text, recent_texts=recent_texts)
    log_signal(signal, db_path=db_path)
    return to_prompt_modifier(get_style(signal))


def evaluate(
    text: str,
    recent_texts: Optional[list[str]] = None,
    db_path: Optional[Path] = None,
) -> tuple[str, Optional[SpeechProfile]]:
    """Like build_style_modifier(), but also returns the SpeechProfile for
    the same signal, so a caller that needs to steer both the prompt AND
    the TTS delivery (see core/tts.py) detects and logs the signal once
    instead of twice. Returns ("", None) when adaptation is disabled."""
    if not is_enabled():
        return "", None
    signal = detect(text, recent_texts=recent_texts)
    log_signal(signal, db_path=db_path)
    return to_prompt_modifier(get_style(signal)), get_speech_profile(signal)
