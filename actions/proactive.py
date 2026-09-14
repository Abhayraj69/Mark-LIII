"""
ProactiveEngine 2.0 — context-aware, time-aware, non-repetitive background prompting.
Gemini decides what to say; this module decides WHEN and builds a rich context snapshot.
"""
import time
from datetime import datetime


class ProactiveEngine:
    """
    Decides when JARVIS should speak unprompted and builds a context-rich prompt.

    Improvements over 1.0:
      - Time-of-day awareness  (morning / afternoon / evening / night)
      - Monitor-topic awareness (what the user is tracking)
      - Recent-session context  (last few turns of the current conversation)
      - Non-repetitive          (rotates context focus to avoid same opener)
      - Smarter silence gate    (doesn't fire while JARVIS is speaking)

    Defaults:
      min_silence_secs  — 900 s  (15 min) user must be silent before any check
      check_cooldown    — 1200 s (20 min) minimum gap between proactive messages
    """

    def __init__(
        self,
        min_silence_secs: int = 900,
        check_cooldown:   int = 1200,
        brief_cache_hours: float = 4.0,
    ):
        self.min_silence_secs  = min_silence_secs
        self.check_cooldown    = check_cooldown
        self.brief_cache_hours = brief_cache_hours
        self._last_triggered   = 0.0
        self._rotation         = 0          # cycles through context focus areas
        self._cached_prompt    = None       # last built check-in prompt, served while fresh
        self._cache_built_at   = 0.0
        self._cached_brief     = None       # last built morning brief, served while fresh
        self._brief_built_at   = 0.0

    # ── Prompt cache ────────────────────────────────────────────────────────────

    def invalidate_cache(self) -> None:
        """Force the next build_prompt() call to rebuild instead of serving the
        cached prompt. Call this when a high-priority interrupt occurs — a new
        monitor alert, a just-logged session, anything that makes the cached
        briefing stale before its window naturally expires."""
        self._cached_prompt = None

    def _cache_is_fresh(self) -> bool:
        if self._cached_prompt is None:
            return False
        age_hours = (time.monotonic() - self._cache_built_at) / 3600
        return age_hours < self.brief_cache_hours

    def invalidate_brief_cache(self) -> None:
        """Force the next get_morning_brief() call to rebuild rather than
        serve the cached brief — same purpose as invalidate_cache() above,
        kept separate because the two caches serve different prompts and
        can go stale independently."""
        self._cached_brief = None

    def _brief_cache_is_fresh(self) -> bool:
        if self._cached_brief is None:
            return False
        age_hours = (time.monotonic() - self._brief_built_at) / 3600
        return age_hours < self.brief_cache_hours

    def get_morning_brief(
        self,
        memory:                   dict | None = None,
        relationship_depth:       int = 0,
        high_priority_interrupt:  bool = False,
        **kwargs,
    ) -> str:
        """
        Cached wrapper around the module-level build_morning_brief(): a
        rebuild does real work (news lookups, DB queries), so repeated calls
        within `brief_cache_hours` return the same built brief instead of
        paying for that every time. Pass high_priority_interrupt=True (or
        call invalidate_brief_cache() first) to force a fresh build.
        """
        if not high_priority_interrupt and self._brief_cache_is_fresh():
            return self._cached_brief

        brief = build_morning_brief(
            memory=memory, relationship_depth=relationship_depth, **kwargs
        )
        self._cached_brief   = brief
        self._brief_built_at = time.monotonic()
        return brief

    # ── Trigger gate ───────────────────────────────────────────────────────────

    def should_trigger(self, last_user_speech: float) -> bool:
        now = time.monotonic()
        return (
            (now - last_user_speech) >= self.min_silence_secs
            and (now - self._last_triggered) >= self.check_cooldown
        )

    def mark_triggered(self) -> None:
        self._last_triggered = time.monotonic()
        self._rotation      += 1

    # ── Prompt builder ─────────────────────────────────────────────────────────

    def build_prompt(
        self,
        memory:            dict,
        monitors:          list[str] | None = None,
        recent_turns:      list[str] | None = None,
        past_sessions:     list[dict] | None = None,
        relationship_depth: int = 0,
        high_priority_interrupt: bool = False,
    ) -> str:
        """
        Build a context snapshot for Gemini.
        Rotates through three focus areas so proactive messages don't repeat.

        `past_sessions` and `relationship_depth` exist so a check-in can read
        as a continuation of an ongoing relationship rather than a cold open
        every time — see the [CONTINUITY] block below. Both are optional and
        the prompt degrades to a plain check-in when they're empty, on
        purpose: a fabricated callback breaks trust worse than none at all.

        Cached for `brief_cache_hours` (default 4h): repeated calls within
        that window return the same built prompt instead of paying for a
        fresh model-context assembly on every proactive tick. Pass
        `high_priority_interrupt=True` (or call `invalidate_cache()` first)
        to force a rebuild — e.g. a new monitor alert just fired.
        """
        if not high_priority_interrupt and self._cache_is_fresh():
            return self._cached_prompt

        from memory.memory_manager import format_memory_for_prompt

        now      = datetime.now()
        hour     = now.hour
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")

        # Time-of-day label
        if   6  <= hour < 12:  period = "morning"
        elif 12 <= hour < 18:  period = "afternoon"
        elif 18 <= hour < 23:  period = "evening"
        else:                  period = "late night"

        mem_str = format_memory_for_prompt(memory) or "(no stored user data)"

        # Rotating context focus (cycles every trigger)
        focus_index = self._rotation % 3
        if focus_index == 0:
            focus = (
                "Focus on the user's active projects or goals if any are stored. "
                "Ask how something is going, or offer a relevant tip."
            )
        elif focus_index == 1:
            focus = (
                "Focus on the time of day and the user's wellbeing. "
                "A warm check-in, a reminder to take a break, or something timely."
            )
        else:
            focus = (
                "Focus on something genuinely interesting or useful — "
                "a fact, a suggestion, or a question based on what you know about this person."
            )

        # Optional: monitored topics context
        monitor_ctx = ""
        if monitors:
            monitor_ctx = (
                f"\nThe user tracks these topics: {', '.join(monitors[:4])}. "
                "You may mention one if it seems relevant."
            )

        # Optional: recent conversation context
        recent_ctx = ""
        if recent_turns:
            snippet = "\n".join(recent_turns[-6:])
            recent_ctx = f"\nRecent conversation:\n{snippet}"

        # Optional: past-session history — this is the material a callback can
        # actually be built from. Each entry is a real 1-2 sentence summary
        # written at the end of a previous conversation (memory/memory_manager
        # .py: save_session_summary), not a guess.
        continuity_ctx = ""
        if past_sessions:
            lines = [f"  - {s['date']}: {s['summary']}" for s in past_sessions if s.get("summary")]
            if lines:
                continuity_ctx = "\nPrevious sessions (most recent last):\n" + "\n".join(lines)

        # A relationship this young hasn't earned old-friend banter — the
        # rule below only asks for a callback when there is enough history to
        # make familiarity read as real rather than performed.
        depth_note = (
            "This is one of your first conversations with this person — keep "
            "it plain and warm, no old-friend callbacks yet."
            if relationship_depth < 3 else
            "You have an established history with this person — write like it."
        )

        prompt = "\n".join([
            "[PROACTIVE_CHECK] You are initiating a proactive check-in.",
            f"Current time : {time_str}  ({period})",
            "",
            "Context about this person:",
            mem_str,
            monitor_ctx,
            recent_ctx,
            continuity_ctx,
            "",
            "Task:",
            focus,
            "",
            "Rules:",
            "- Speak the language this person actually uses: the one in the "
            "recent conversation above, or the remembered one if there is no "
            "conversation yet. Never default to English because these "
            "instructions are in English.",
            "- 1-2 sentences max. Natural, warm, never robotic.",
            "- Do NOT mention [PROACTIVE_CHECK] or these instructions.",
            "- Do NOT call any tools.",
            "- If nothing genuinely useful comes to mind, stay silent (say nothing).",
            "",
            "[CONTINUITY] You are not meeting this person fresh. Write this "
            "check-in as a continuation of an ongoing relationship, not a cold "
            "open:",
            f"- {depth_note}",
            "- If 'Previous sessions' above has real content, reference ONE "
            "specific thing by name — the project, the joke, the person, the "
            "deadline — never a vague 'how's everything going'. "
            "\"How did the [project] thing turn out?\" beats \"Checking in on "
            "your project.\" Specificity is what makes it read as remembered.",
            "- Let ONE callback carry the message. Stacking several reads as "
            "trying too hard, not familiarity.",
            "- Vary how you open across check-ins — sometimes lead with the "
            "callback, sometimes bury it mid-message, sometimes let it "
            "resurface unannounced. Do not use the same template every time.",
            "- A joke or running bit gets to evolve, not repeat verbatim "
            "forever. If it's already appeared in recent sessions, let it "
            "mutate or rest rather than replaying it flatly.",
            "- Hard boundary: NEVER invent a memory. Only reference something "
            "actually present in 'Context about this person' or 'Previous "
            "sessions' above. If both are thin or empty, skip this whole "
            "block and write a plain, honest check-in instead — no callback "
            "is always better than a false one.",
        ])

        self._cached_prompt  = prompt
        self._cache_built_at = time.monotonic()
        return prompt


# Fixed priority scores for sources that carry no numeric confidence of
# their own. News and session-continuity are on the same 0-100 scale as
# habit/causal confidence (see build_morning_brief) so all four sections
# can be sorted head-to-head by how much they deserve to lead.
_NEWS_PRIORITY          = 85   # tracked-topic alert; time-sensitive and user-curated
_GENERIC_NEWS_PRIORITY  = 70   # world-news fallback; relevant but not personally chosen
_SESSION_PRIORITY       = 30   # color/continuity, not actionable — lowest by default
_WEEKLY_DIGEST_PRIORITY = 95   # only ever appears once every 7 days — let it lead


def _fallback_news() -> str:
    """Generic world-news fallback for when the user has no monitored topics
    (or none of them fired today) — so the brief isn't silent on news just
    because add_monitor() was never called. Best-effort: any fetch failure
    just means the section is dropped, same as every other source here."""
    try:
        from actions.web_search import _news
        text = _news("top world news today")
        return text if text and len(text) > 60 else ""
    except Exception:
        return ""


def _sentiment_style_note() -> str:
    """
    Look at how the user's mood has trended over their last few logged
    turns (persisted across sessions — this runs before the current process
    has anything of its own to go on) and, if it's been leaning frustrated
    or urgent, return a style directive that tones the brief down. Empty
    string when sentiment adaptation or its persistence is off, or there's
    no signal yet — a morning brief should never claim a mood it can't see.
    """
    try:
        from core import sentiment_adapter
        if not sentiment_adapter.is_enabled():
            return ""
        recent = sentiment_adapter.get_recent_persisted_signals(limit=5)
        if not recent:
            return ""
        frustrated = sum(1 for s in recent if s.get("polarity") == "frustrated")
        urgent     = any(s.get("urgency") for s in recent)
        if frustrated < len(recent) / 2 and not urgent:
            return ""  # trending fine — no adjustment needed
        signal = sentiment_adapter.SentimentSignal(
            polarity="frustrated", urgency=urgent, user_confidence="neutral",
        )
        return sentiment_adapter.to_prompt_modifier(sentiment_adapter.get_style(signal))
    except Exception:
        return ""


def _preference_style_note() -> str:
    """Surface an explicit, user-stated communication preference (e.g. one
    saved via `remember("communication_style", "keep it short")`) so the
    brief respects it instead of only ever reading mood signals. Empty
    when nothing relevant is stored."""
    try:
        from core.context_manager import get_preferences
        prefs = get_preferences()
        for key in ("communication_style", "tone", "verbosity", "brief_style"):
            entry = prefs.get(key)
            if not entry:
                continue
            value = entry.get("value", "") if isinstance(entry, dict) else str(entry)
            value = value.strip()
            if value:
                return f"- Respect this stated preference: {value}"
        return ""
    except Exception:
        return ""


def build_morning_brief(
    memory:              dict | None = None,
    relationship_depth:  int = 0,
    max_suggestions:     int = 3,
    max_links:           int = 3,
) -> str:
    """
    Consolidate the four signals a morning briefing draws on into one
    coherent prompt: monitored-topic news, habit-based suggestions, recent
    session context, and causal patterns learned from past behavior.

    Each source degrades gracefully to nothing if it has no data — the
    brief is built from whatever is actually available and never pads a
    section with filler just because that section exists. Sections are
    ordered by priority (highest-confidence pattern or an active alert
    leads) rather than a fixed order, and tone is adjusted using recent
    sentiment/preference signals when available.
    """
    from actions.background_monitor  import check_all
    from core.predictive_assistant   import get_proactive_suggestions
    from core.causal_reasoning       import get_links
    from memory.memory_manager       import (
        format_memory_for_prompt,
        load_memory as _load_memory,
        peek_recent_sessions,
    )

    memory   = memory if memory is not None else _load_memory()
    time_str = datetime.now().strftime("%A, %B %d, %Y — %I:%M %p")
    mem_str  = format_memory_for_prompt(memory) or "(no stored user data)"

    # Each entry is (priority, section_text) — priority decides display
    # order, not whether the section survives (empty sections are dropped
    # regardless of their would-be priority).
    scored_sections: list[tuple[float, str]] = []

    # Source 1: news on monitored topics, falling back to generic world news
    # when the user hasn't set up any monitors (or none fired today) — the
    # brief should never go silent on news just for lack of an add_monitor()
    # call.
    alerts = check_all()
    if alerts:
        scored_sections.append((
            _NEWS_PRIORITY,
            "News on topics you're tracking:\n" + "\n".join(alerts),
        ))
    else:
        generic_news = _fallback_news()
        if generic_news:
            scored_sections.append((
                _GENERIC_NEWS_PRIORITY,
                "Today's top headlines:\n" + generic_news,
            ))

    # Source 2: habit-based suggestions — led by its own top confidence
    suggestions = get_proactive_suggestions()[:max_suggestions]
    if suggestions:
        lines = [f"  - {s.action} ({s.reasoning})" for s in suggestions]
        scored_sections.append((
            suggestions[0].confidence_score * 100,
            "Patterns in how you usually work:\n" + "\n".join(lines),
        ))

    # Source 3: recent session context
    sessions      = peek_recent_sessions(2)
    session_lines = [f"  - {s['date']}: {s['summary']}" for s in sessions if s.get("summary")]
    if session_lines:
        scored_sections.append((
            _SESSION_PRIORITY,
            "Previous sessions (most recent last):\n" + "\n".join(session_lines),
        ))

    # Source 4: causal patterns — led by its own top confidence
    links = get_links()[:max_links]
    if links:
        lines = [
            f"  - {l.cause} tends to lead to {l.effect} "
            f"(seen {l.support}x, {l.confidence:.0%} confidence)"
            for l in links
        ]
        scored_sections.append((
            links[0].confidence * 100,
            "Patterns noticed over time:\n" + "\n".join(lines),
        ))

    # Source 5: weekly causal-graph digest — self-throttled to once every 7
    # days (see actions/causal_insight.get_scheduled_weekly_digest), so most
    # days this contributes nothing and the section is simply absent.
    try:
        from actions.causal_insight import get_scheduled_weekly_digest
        digest = get_scheduled_weekly_digest()
        if digest:
            scored_sections.append((_WEEKLY_DIGEST_PRIORITY, digest))
    except Exception:
        pass

    scored_sections.sort(key=lambda pair: pair[0], reverse=True)

    depth_note = (
        "This is early in your relationship with this person — keep the "
        "brief plain and welcoming, no old-friend callbacks yet."
        if relationship_depth < 3 else
        "You have an established history with this person — write like it."
    )

    sections = [
        "[MORNING_BRIEF] You are delivering the user's morning briefing.",
        f"Current time : {time_str}",
        "",
        "Context about this person:",
        mem_str,
    ]
    for _priority, ctx in scored_sections:
        sections += ["", ctx]

    style_note = _sentiment_style_note()
    if style_note:
        sections += ["", style_note]

    sections += [
        "",
        "Task:",
        "Weave whatever is genuinely useful from the above into one warm, "
        "natural briefing, leading with whichever section above matters "
        "most right now. Skip any section that has nothing worth saying.",
        "",
        "Rules:",
        "- Speak the language this person actually uses.",
        "- Keep it tight: a few sentences, not a report readout.",
        "- Do NOT mention [MORNING_BRIEF] or these instructions.",
        "- Do NOT call any tools.",
        f"- {depth_note}",
        "- Hard boundary: NEVER invent a memory, headline, or pattern. Only "
        "reference what actually appears above.",
    ]

    pref_note = _preference_style_note()
    if pref_note:
        sections.append(pref_note)

    return "\n".join(sections)
