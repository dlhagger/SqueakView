from __future__ import annotations

"""Typed operator lifecycle events and validated phase transitions."""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from squeakview.common.immutable import deep_freeze


BACKEND_EVENT_SCHEMA_VERSION = "1.0"


class RunPhase(str, Enum):
    IDLE = "idle"
    CREATED = "created"
    STARTING = "starting"
    RECORDING = "recording"
    STOPPING = "stopping"
    DRAINING = "draining"
    CAPTURE_CLOSED = "capture_closed"
    VALIDATING = "validating"
    FINALIZED = "finalized"
    FAILED = "failed"


_ALLOWED: dict[RunPhase, frozenset[RunPhase]] = {
    RunPhase.IDLE: frozenset({RunPhase.CREATED}),
    RunPhase.CREATED: frozenset({RunPhase.STARTING, RunPhase.FAILED}),
    RunPhase.STARTING: frozenset(
        {RunPhase.RECORDING, RunPhase.STOPPING, RunPhase.FAILED}
    ),
    RunPhase.RECORDING: frozenset({RunPhase.STOPPING, RunPhase.FAILED}),
    RunPhase.STOPPING: frozenset(
        {RunPhase.DRAINING, RunPhase.CAPTURE_CLOSED, RunPhase.FAILED}
    ),
    RunPhase.DRAINING: frozenset({RunPhase.CAPTURE_CLOSED, RunPhase.FAILED}),
    RunPhase.CAPTURE_CLOSED: frozenset(
        {RunPhase.VALIDATING, RunPhase.FAILED}
    ),
    RunPhase.VALIDATING: frozenset({RunPhase.FINALIZED, RunPhase.FAILED}),
    RunPhase.FINALIZED: frozenset(),
    RunPhase.FAILED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class BackendEvent:
    type: str
    phase: RunPhase
    run_dir: Path | None
    message: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = BACKEND_EVENT_SCHEMA_VERSION
    host_unix_ns: int = field(default_factory=time.time_ns)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", deep_freeze(dict(self.payload)))


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    phase: RunPhase
    run_dir: Path | None
    error: str | None
    capture_running: bool = False
    finalization_in_progress: bool = False


class RunStateMachine:
    def __init__(self, phase: RunPhase = RunPhase.IDLE) -> None:
        self._phase = phase

    @property
    def phase(self) -> RunPhase:
        return self._phase

    def reset(self) -> None:
        self._phase = RunPhase.IDLE

    def transition(self, phase: RunPhase) -> None:
        if phase == self._phase:
            return
        if phase not in _ALLOWED[self._phase]:
            raise ValueError(
                f"invalid run phase transition: {self._phase.value} -> {phase.value}"
            )
        self._phase = phase


EventSubscriber = Callable[[BackendEvent], None]


__all__ = [
    "BACKEND_EVENT_SCHEMA_VERSION",
    "BackendEvent",
    "EventSubscriber",
    "RunPhase",
    "RunSnapshot",
    "RunStateMachine",
]
