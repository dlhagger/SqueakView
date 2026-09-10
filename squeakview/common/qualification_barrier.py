"""Explicit, bounded barriers for destructive supervisor qualification only."""

from __future__ import annotations

import os
import math
import time
from pathlib import Path
from typing import Callable, Mapping

from squeakview.common import run_context


GATE_ENV = "SQUEAKVIEW_ENABLE_SUPERVISOR_FAILURE_INJECTION"
BARRIER_ENV = "SQUEAKVIEW_SUPERVISOR_FAILURE_BARRIER"
TIMEOUT_ENV = "SQUEAKVIEW_SUPERVISOR_FAILURE_BARRIER_TIMEOUT_S"
MAX_TIMEOUT_S = 60.0
DEFAULT_TIMEOUT_S = 15.0
BARRIERS = frozenset(
    {
        "pre_capture",
        "after_spawn_before_ready",
        "finalizer:capture_reconciliation",
        "finalizer:inference_admission",
        "finalizer:recording_validation",
        "finalizer:streaming_alignment",
    }
)


def configured_barrier(environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    value = str(env.get(BARRIER_ENV, "")).strip()
    if not value:
        return None
    if value not in BARRIERS:
        raise RuntimeError(f"unsupported qualification barrier: {value!r}")
    if env.get(GATE_ENV) != "1":
        raise RuntimeError(f"{BARRIER_ENV} requires {GATE_ENV}=1")
    return value


def wait_at_barrier(
    run_dir: Path,
    name: str,
    *,
    stop_requested: Callable[[], bool] = lambda: False,
    environ: Mapping[str, str] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Wait at one selected barrier, self-releasing within a strict bound."""

    env = os.environ if environ is None else environ
    if configured_barrier(env) != name:
        return False
    try:
        timeout_s = float(env.get(TIMEOUT_ENV, str(DEFAULT_TIMEOUT_S)))
    except ValueError as exc:
        raise RuntimeError(f"{TIMEOUT_ENV} must be numeric") from exc
    if not math.isfinite(timeout_s) or not 0.1 <= timeout_s <= MAX_TIMEOUT_S:
        raise RuntimeError(
            f"{TIMEOUT_ENV} must be between 0.1 and {MAX_TIMEOUT_S:g} seconds"
        )
    deadline = monotonic() + timeout_s
    waiting = {
        "schema_version": "1.0",
        "name": name,
        "state": "waiting",
        "timeout_s": timeout_s,
    }
    run_context.update_status(run_dir, supervisor_failure_barrier=waiting)
    reason = "timeout"
    while monotonic() < deadline:
        if stop_requested():
            reason = "stop_requested"
            break
        sleep(min(0.05, max(0.0, deadline - monotonic())))
    released = dict(waiting, state="released", release_reason=reason)
    run_context.update_status(run_dir, supervisor_failure_barrier=released)
    return True


__all__ = [
    "BARRIER_ENV",
    "BARRIERS",
    "DEFAULT_TIMEOUT_S",
    "GATE_ENV",
    "MAX_TIMEOUT_S",
    "TIMEOUT_ENV",
    "configured_barrier",
    "wait_at_barrier",
]
