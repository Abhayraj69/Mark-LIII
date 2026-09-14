"""
core/adaptive_poll.py — a tiny exponential-backoff helper shared by the
background polling loops that pay for an API call (Gemini vision) on every
tick: actions/screen_monitor.py and actions/study_mode.py.

Both loops used to sleep a fixed `interval_seconds` between checks forever,
even while watching a screen that hadn't changed in an hour. That is wasted
latency-in-reverse — every tick is a real vision call whether or not
anything happened — so this doubles the sleep interval after a few
consecutive unchanged ticks (capped at `max_interval`) and snaps straight
back to the fastest rate the moment a change is observed, so a real event
is never caught late because the loop had drifted slow.
"""
from __future__ import annotations


class AdaptiveInterval:
    """Exponential-backoff poll-interval tracker.

    Call `report(changed)` once per tick with whether that tick found a
    change; it returns the number of seconds to sleep before the next one.
    Starts at, and instantly resets to, `base` on any change — the backoff
    only ever lengthens the interval during a run of quiet ticks.
    """

    def __init__(self, base: int, max_interval: int, patience: int = 3):
        self.base = max(1, int(base))
        self.max_interval = max(self.base, int(max_interval))
        self.patience = max(1, int(patience))
        self.current = self.base
        self._unchanged_streak = 0

    def report(self, changed: bool) -> int:
        if changed:
            self.current = self.base
            self._unchanged_streak = 0
        else:
            self._unchanged_streak += 1
            if self._unchanged_streak >= self.patience:
                self.current = min(self.current * 2, self.max_interval)
                self._unchanged_streak = 0
        return self.current
