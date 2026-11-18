"""Fail-closed liveness supervision for the scientific recording path."""

from __future__ import annotations

import os
import math
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Callable, Protocol

from .recording_activity import RecordingActivity


class RecordingActivitySource(Protocol):
    stream_id: int

    def try_activity(self) -> RecordingActivity | None: ...


@dataclass(frozen=True, slots=True)
class RecordingLivenessPolicy:
    """Resolved timeout policy, retaining both configured and effective values."""

    configured_timeout_s: float
    effective_timeout_ns: int
    poll_interval_s: float = 0.2


def resolve_recording_liveness_policy(
    fps: int,
    *,
    failure_injection: bool,
    environ: Mapping[str, str] | None = None,
) -> RecordingLivenessPolicy:
    """Resolve the existing environment/default policy without runtime coupling."""

    values = os.environ if environ is None else environ
    try:
        configured = float(
            values.get("SQUEAKVIEW_RECORDING_STALL_TIMEOUT_S", "")
            or (2.0 if failure_injection else max(30.0, 5.0 / float(fps)))
        )
        if not math.isfinite(configured):
            raise ValueError("timeout must be finite")
    except (ValueError, ZeroDivisionError):
        configured = 30.0
    return RecordingLivenessPolicy(
        configured_timeout_s=configured,
        effective_timeout_ns=int(max(1.0, configured) * 1_000_000_000),
    )


class RecordingLivenessMonitor:
    """Inspect constant-memory activity snapshots until stopped or stalled."""

    def __init__(
        self,
        sources: Iterable[RecordingActivitySource],
        stop_event: threading.Event,
        on_fatal: Callable[[str], None],
        policy: RecordingLivenessPolicy,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._sources = tuple(sources)
        self._stop_event = stop_event
        self._on_fatal = on_fatal
        self.policy = policy
        self._monotonic_ns = monotonic_ns

    def run(self) -> None:
        unavailable_since: dict[int, int] = {}
        while not self._stop_event.wait(self.policy.poll_interval_s):
            now_ns = self._monotonic_ns()
            for source in self._sources:
                try:
                    activity = source.try_activity()
                except Exception as exc:
                    self._on_fatal(
                        "recording liveness telemetry failed on stream "
                        f"{source.stream_id}: {type(exc).__name__}: {exc}"
                    )
                    return
                if activity is None:
                    since = unavailable_since.setdefault(source.stream_id, now_ns)
                    if now_ns - since >= self.policy.effective_timeout_ns:
                        self._on_fatal(
                            f"recording telemetry snapshot unavailable for {self.policy.configured_timeout_s:.1f}s "
                            f"on stream {source.stream_id}; a recording callback may be blocked"
                        )
                        return
                    continue
                unavailable_since.pop(source.stream_id, None)
                last_source = activity.last_source_monotonic_ns
                if (
                    activity.source_count > 0
                    and last_source is not None
                    and now_ns - last_source >= self.policy.effective_timeout_ns
                ):
                    self._on_fatal(
                        "recording source stopped making progress for "
                        f"{self.policy.configured_timeout_s:.1f}s on stream "
                        f"{source.stream_id}; source={activity.source_count} "
                        f"admitted={activity.admission_count} "
                        f"encoded={activity.egress_count}"
                    )
                    return
