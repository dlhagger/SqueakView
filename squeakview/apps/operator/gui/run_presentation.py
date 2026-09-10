from __future__ import annotations

"""Qt-free presentation policy and bounded readers for live run artifacts."""

import csv
import io
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from squeakview.apps.operator.backend.events import RunPhase


@dataclass(frozen=True, slots=True)
class RunPresentation:
    badge_state: str
    preview_status: str
    run_enabled: bool
    stop_enabled: bool
    configure_enabled: bool


@dataclass(frozen=True, slots=True)
class FinalizationPresentation:
    passed: bool
    badge_state: str
    preview_status: str
    health_text: str
    health_level: str


@dataclass(frozen=True, slots=True)
class CaptureHealthSnapshot:
    frames: int
    gaps: int
    incomplete: int
    dropped: int
    lost: int
    waiting: int
    in_flight: int
    sensor_temperature_c: float | None


@dataclass(frozen=True, slots=True)
class CaptureHealthPresentation:
    text: str
    level: str


def present_run_phase(phase: RunPhase) -> RunPresentation:
    if phase in {RunPhase.CREATED, RunPhase.STARTING}:
        return RunPresentation("starting", "Starting", False, True, False)
    if phase == RunPhase.RECORDING:
        return RunPresentation("recording", "Recording", False, True, False)
    if phase in {
        RunPhase.STOPPING,
        RunPhase.DRAINING,
        RunPhase.CAPTURE_CLOSED,
        RunPhase.VALIDATING,
    }:
        return RunPresentation("finalizing", "Finalizing", False, False, False)
    if phase == RunPhase.FINALIZED:
        return RunPresentation("complete", "Complete", True, False, True)
    if phase == RunPhase.FAILED:
        return RunPresentation("failed", "Failed", True, False, True)
    return RunPresentation("ready", "Ready", True, False, True)


def present_finalization(status: dict[str, Any]) -> FinalizationPresentation:
    """Fail closed unless terminal recording validation explicitly passed."""

    validation = status.get("recording_validation")
    passed = isinstance(validation, dict) and validation.get("passed") is True
    terminal_success = str(status.get("state") or "") in {
        "post_run_complete",
        "analysis_complete",
        "finalized",
    }
    if passed and terminal_success:
        return FinalizationPresentation(
            True,
            "complete",
            "Complete",
            "Capture saved · frame-count validation PASS",
            "ok",
        )
    return FinalizationPresentation(
        False,
        "failed",
        "Failed",
        "Run ended with a validation failure — open Events",
        "error",
    )


def _tail_text_line(path: Path, *, max_bytes: int = 131_072) -> str:
    """Return the last nonblank line while bounding I/O for multi-day runs."""

    lines = _tail_text_lines(path, max_bytes=max_bytes)
    return lines[-1] if lines else ""


def _tail_text_lines(
    path: Path,
    *,
    max_bytes: int = 131_072,
    complete_only: bool = False,
) -> list[str]:
    """Return a bounded nonblank tail for readers that can skip a partial row."""

    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            tail = handle.read()
    except OSError:
        return []
    lines = [line for line in tail.decode(errors="replace").splitlines() if line.strip()]
    if complete_only and tail and not tail.endswith((b"\n", b"\r")) and lines:
        lines.pop()
    return lines


def _last_csv_row(path: Path) -> dict[str, str]:
    """Read a CSV header and its latest row without scanning the whole file."""

    return _csv_tail_row(path, complete_only=False)


def _last_complete_csv_row(path: Path) -> dict[str, str]:
    """Read the latest newline-terminated row, skipping a concurrent partial write."""

    return _csv_tail_row(path, complete_only=True)


def _csv_tail_row(path: Path, *, complete_only: bool) -> dict[str, str]:

    try:
        with Path(path).open(newline="") as handle:
            header = next(csv.reader(handle))
    except (OSError, StopIteration, csv.Error):
        return {}
    for raw in reversed(_tail_text_lines(path, complete_only=complete_only)):
        if raw == ",".join(header):
            continue
        try:
            values = next(csv.reader(io.StringIO(raw)))
        except (StopIteration, csv.Error):
            continue
        if len(values) == len(header):
            return dict(zip(header, values))
    return {}


def _elapsed_text(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _read_ds_batch_size(cfg_path: Path) -> int | None:
    from squeakview.apps.inference.contracts import read_config_value

    try:
        value = read_config_value(Path(cfg_path), "batch-size")
        return int(value) if value is not None else None
    except (OSError, UnicodeError, ValueError, IndexError, TypeError):
        return None


def _effective_batch_camera_count(config: Any) -> int:
    return max(1, int(getattr(config, "num_cameras", 1)))


def disk_free_text(path: Path) -> str:
    try:
        return f"{shutil.disk_usage(path).free / (1024**3):,.0f} GB"
    except OSError:
        return "--"


def _nonnegative_int(value: object) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, parsed)


def scan_capture_health(run_path: Path) -> CaptureHealthSnapshot:
    """Read only the latest bounded capture and admission diagnostics."""

    capture: dict[str, Any] = {}
    for raw in reversed(
        _tail_text_lines(Path(run_path) / "capture_cam0.jsonl", complete_only=True)
    ):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            capture = payload
            break
    recording = _last_complete_csv_row(Path(run_path) / "diagnostics" / "recording.csv")
    raw_sequence = capture.get("source_sequence_index")
    try:
        parsed_sequence = int(raw_sequence)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        parsed_sequence = -1
    temperature = capture.get("sensor_temperature_c")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        temperature_value = None
    else:
        temperature_value = float(temperature)
    return CaptureHealthSnapshot(
        frames=parsed_sequence + 1 if parsed_sequence >= 0 else 0,
        gaps=_nonnegative_int(capture.get("total_frame_gap_events")),
        incomplete=_nonnegative_int(capture.get("total_incomplete")),
        dropped=_nonnegative_int(capture.get("stream_dropped_frames")),
        lost=_nonnegative_int(capture.get("stream_lost_frames")),
        waiting=_nonnegative_int(recording.get("waiting_for_record_admission")),
        in_flight=_nonnegative_int(recording.get("encoder_in_flight")),
        sensor_temperature_c=temperature_value,
    )


def present_capture_health(
    snapshot: CaptureHealthSnapshot,
    *,
    disk_text: str,
) -> CaptureHealthPresentation:
    issues = snapshot.gaps + snapshot.incomplete + snapshot.dropped + snapshot.lost
    level = "error" if issues else "warning" if snapshot.waiting >= 30 else "ok"
    frame_text = f"{snapshot.frames:,}" if snapshot.frames > 0 else "--"
    temp_text = (
        f"{snapshot.sensor_temperature_c:.1f}°C"
        if snapshot.sensor_temperature_c is not None
        else "--"
    )
    return CaptureHealthPresentation(
        f"Frames {frame_text}  ·  Gaps {snapshot.gaps}  ·  "
        f"Queue {snapshot.waiting}/{snapshot.in_flight}  ·  "
        f"Cam {temp_text}  ·  Disk {disk_text}",
        level,
    )


__all__ = [
    "CaptureHealthPresentation",
    "CaptureHealthSnapshot",
    "FinalizationPresentation",
    "RunPresentation",
    "_effective_batch_camera_count",
    "_elapsed_text",
    "_last_csv_row",
    "_read_ds_batch_size",
    "_tail_text_line",
    "disk_free_text",
    "present_capture_health",
    "present_finalization",
    "present_run_phase",
    "scan_capture_health",
]
