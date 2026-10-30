from __future__ import annotations

"""Deterministic capture-ledger drain checks used before MP4 finalization."""

from squeakview.common.run_layout import artifact_path, open_artifact
from squeakview.common.capture_health import read_capture_health, source_path, health_path

import json
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from squeakview.apps.operator.backend.events import RunPhase


DEFAULT_DRAIN_QUIET_S = 0.35
DEFAULT_DRAIN_TIMEOUT_S = 5.0
MAX_DRAIN_QUIET_S = 60.0
MAX_DRAIN_TIMEOUT_S = 600.0


@dataclass(frozen=True, slots=True)
class DrainResult:
    passed: bool
    ledger_sizes: tuple[int, ...]
    ledger_frame_counts: tuple[int | None, ...]
    expected_ttl_count: int | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CaptureDrainRequest:
    run_dir: Path
    camera_count: int
    phase: RunPhase
    expected_ttl_count: int | None = None


@dataclass(frozen=True, slots=True)
class CaptureDrainHooks:
    write_status: Callable[..., Any]
    transition: Callable[[RunPhase], None]


def ledger_paths(run_dir: Path, camera_count: int) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index in range(max(1, int(camera_count))):
        paths.append(artifact_path(Path(run_dir), "frames.csv" if index == 0 else f"frames_cam{index}.csv")
                     if health_path(run_dir, index).exists() else source_path(Path(run_dir), index))
        paths.append(
            artifact_path(Path(run_dir), "record_admission.csv"
                if index == 0
                else f"record_admission_cam{index}.csv")
        )
    return tuple(paths)


def ledger_size_snapshot(run_dir: Path, camera_count: int) -> tuple[int, ...]:
    sizes: list[int] = []
    for path in ledger_paths(run_dir, camera_count):
        try:
            sizes.append(int(path.stat().st_size))
        except FileNotFoundError:
            sizes.append(-1)
    return tuple(sizes)


def last_complete_ledger_line(path: Path) -> str | None:
    from squeakview.common.ledger_tail import last_complete_line
    try:
        return last_complete_line(path)
    except (OSError, ValueError):
        return None


def ledger_frame_counts(
    run_dir: Path, camera_count: int
) -> tuple[int | None, ...]:
    counts: list[int | None] = []
    for index in range(max(1, int(camera_count))):
        native = health_path(run_dir, index).exists()
        path = (artifact_path(Path(run_dir), "frames.csv" if index == 0 else f"frames_cam{index}.csv")
                if native else source_path(Path(run_dir), index))
        capture_line = last_complete_ledger_line(path)
        try:
            if native:
                fields = next(csv.reader([capture_line or ""]))
                capture_count = int(fields[3]) + 1
            else:
                capture_count = int(json.loads(capture_line or "")["source_sequence_index"]) + 1
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError, StopIteration):
            capture_count = 0 if capture_line is None else None
        admission_path = artifact_path(Path(run_dir), "record_admission.csv"
            if index == 0
            else f"record_admission_cam{index}.csv")
        admission_line = last_complete_ledger_line(admission_path)
        try:
            admission_count = int(admission_line.split(",")[1]) + 1
        except (AttributeError, IndexError, TypeError, ValueError):
            admission_count = 0 if admission_line in (None, "") else None
        counts.extend((capture_count, admission_count))
    return tuple(counts)


def wait_for_capture_drain(
    run_dir: Path,
    camera_count: int,
    *,
    expected_ttl_count: int | None,
    quiet_s: float,
    timeout_s: float,
    poll_s: float = 0.05,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> DrainResult:
    """Wait for equal source/admission counts and a stable quiet window."""

    deadline = monotonic() + timeout_s
    previous: tuple[int, ...] | None = None
    stable_since: float | None = None
    snapshot: tuple[int, ...] = ()
    counts: tuple[int | None, ...] = ()
    while monotonic() < deadline:
        snapshot = ledger_size_snapshot(run_dir, camera_count)
        counts = ledger_frame_counts(run_dir, camera_count)
        now = monotonic()
        ledgers_exist = bool(snapshot) and all(size >= 0 for size in snapshot)
        counts_valid = bool(counts) and all(count is not None for count in counts)
        pairs_match = counts_valid and all(
            counts[offset] == counts[offset + 1]
            for offset in range(0, len(counts), 2)
        )
        target_reached = expected_ttl_count is None or (
            counts_valid and all(int(count) >= expected_ttl_count for count in counts)
        )
        if ledgers_exist and pairs_match and target_reached and snapshot == previous:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= quiet_s:
                return DrainResult(
                    True, snapshot, counts, expected_ttl_count
                )
        else:
            stable_since = now if ledgers_exist else None
        previous = snapshot
        sleep(poll_s)
    return DrainResult(
        False,
        snapshot,
        counts,
        expected_ttl_count,
        "capture ledgers did not reconcile and become quiet before timeout",
    )


class CaptureDrainCoordinator:
    """Own the fail-closed status protocol around ledger reconciliation."""

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        waiter: Callable[..., DrainResult] = wait_for_capture_drain,
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._waiter = waiter

    def wait(
        self,
        request: CaptureDrainRequest,
        hooks: CaptureDrainHooks,
    ) -> bool:
        quiet_s, timeout_s = self._timings()
        hooks.write_status(
            request.run_dir,
            "capture_draining",
            quiet_period_s=quiet_s,
            timeout_s=timeout_s,
            expected_ttl_count=request.expected_ttl_count,
        )
        if request.phase == RunPhase.STOPPING:
            hooks.transition(RunPhase.DRAINING)

        result = self._waiter(
            request.run_dir,
            int(request.camera_count),
            expected_ttl_count=request.expected_ttl_count,
            quiet_s=quiet_s,
            timeout_s=timeout_s,
        )
        common = {
            "ledger_sizes": list(result.ledger_sizes),
            "ledger_frame_counts": list(result.ledger_frame_counts),
        }
        if result.passed:
            hooks.write_status(
                request.run_dir,
                "capture_drained",
                **common,
                expected_ttl_count=request.expected_ttl_count,
            )
            return True

        hooks.write_status(
            request.run_dir,
            "capture_drain_timeout",
            **common,
            drain_error=result.reason,
        )
        return False

    def _timings(self) -> tuple[float, float]:
        try:
            quiet_s = float(
                self._environ.get(
                    "SQUEAKVIEW_CAPTURE_DRAIN_QUIET_S",
                    str(DEFAULT_DRAIN_QUIET_S),
                )
            )
            timeout_s = float(
                self._environ.get(
                    "SQUEAKVIEW_CAPTURE_DRAIN_TIMEOUT_S",
                    str(DEFAULT_DRAIN_TIMEOUT_S),
                )
            )
        except (TypeError, ValueError):
            return DEFAULT_DRAIN_QUIET_S, DEFAULT_DRAIN_TIMEOUT_S
        if (
            not math.isfinite(quiet_s)
            or not math.isfinite(timeout_s)
            or not 0.1 <= quiet_s <= MAX_DRAIN_QUIET_S
            or not quiet_s <= timeout_s <= MAX_DRAIN_TIMEOUT_S
        ):
            return DEFAULT_DRAIN_QUIET_S, DEFAULT_DRAIN_TIMEOUT_S
        return quiet_s, timeout_s
