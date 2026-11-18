"""Monotonic work counters extend idle deadlines, never the absolute limit."""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping


class ProgressDeadline:
    def __init__(self, stall_s: float, maximum_s: float, *,
                 monotonic: Callable[[], float] = time.monotonic):
        self.stall_s = stall_s
        self.maximum_s = maximum_s
        self.clock = monotonic
        self.started = self.last_progress = monotonic()
        self.high_water: dict[str, int] = {}
        self.steps = 0

    def observe(self, counters: Mapping[str, int]) -> None:
        advanced = False
        for key, value in counters.items():
            if type(value) is not int or value < 0:
                continue
            old = self.high_water.get(key)
            # First observations establish a baseline. Disappearing, malformed
            # or regressing counters cannot buy more shutdown time.
            if old is None or value > old:
                self.high_water[key] = value
                advanced |= old is not None
        if advanced:
            self.last_progress = self.clock()

    def completed_step(self) -> None:
        self.steps += 1
        self.last_progress = self.clock()

    def remaining_s(self) -> float:
        now = self.clock()
        return max(0.0, min(self.started + self.maximum_s, self.last_progress + self.stall_s) - now)

    def report(self) -> dict[str, object]:
        now = self.clock()
        return {
            "elapsed_s": round(now - self.started, 3),
            "idle_s": round(now - self.last_progress, 3),
            "stall_timeout_s": self.stall_s,
            "maximum_s": self.maximum_s,
            "counters": dict(self.high_water),
            "completed_steps": self.steps,
            "limit": ("overall" if now >= self.started + self.maximum_s else
                      "stall" if now >= self.last_progress + self.stall_s else None),
        }


def wait_for_completion(wait: Callable[[float], bool], budget: ProgressDeadline, *,
                        progress: Callable[[], Mapping[str, int]] = dict,
                        poll_s: float = 0.1) -> bool:
    while True:
        # Check expiry before sampling: work noticed after the idle deadline
        # must not resurrect a stage already judged stalled.
        if budget.remaining_s() <= 0:
            return wait(0)
        budget.observe(progress())
        if wait(min(poll_s, budget.stall_s / 4, budget.remaining_s())):
            return True
