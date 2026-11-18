"""Contain Python faults before returning through a native streaming callback."""
from __future__ import annotations

import threading


class NativeCallbackGuard:
    def __init__(self, context, on_fatal):
        self.context = context
        self.on_fatal = on_fatal
        self.failure: str | None = None
        self._lock = threading.Lock()

    def call(self, callback) -> bool:
        if self.failure is not None:
            return True  # Preserve video while the lifecycle owner stops triggers.
        try:
            result = callback()
            return result if type(result) is bool else True
        except BaseException as exc:
            try:
                detail = f"{self.context}: {type(exc).__name__}: {exc}"
            except BaseException:
                detail = f"{self.context}: callback failed"
            with self._lock:
                if self.failure is not None:
                    return True
                self.failure = detail
            # Reporting is outside writer/telemetry locks, and must never let
            # another Python exception cross the native callback boundary.
            try:
                if self.on_fatal is not None:
                    self.on_fatal(detail)
            except BaseException:
                pass
        return True

    def check(self):
        if self.failure is not None:
            raise RuntimeError(self.failure)
