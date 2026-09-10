"""Streaming reconciliation of camera capture and recording-admission ledgers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


MAX_LEDGER_LINE_BYTES = 64 * 1024


def _bounded_text_lines(path: Path) -> Iterator[str]:
    """Yield strict UTF-8 lines while bounding every individual record."""

    with path.open("rb") as handle:
        line_number = 0
        while True:
            raw = handle.readline(MAX_LEDGER_LINE_BYTES + 1)
            if not raw:
                return
            line_number += 1
            if len(raw) > MAX_LEDGER_LINE_BYTES:
                raise RuntimeError(
                    f"ledger line exceeds {MAX_LEDGER_LINE_BYTES} byte limit at "
                    f"{path.name}:{line_number}"
                )
            try:
                yield raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise RuntimeError(
                    f"ledger is not strict UTF-8 at {path.name}:{line_number}"
                ) from exc


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _json_uint(payload: dict, name: str) -> int:
    value = payload[name]
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _csv_uint(value: object, name: str) -> int:
    raw = value if isinstance(value, str) else ""
    if not raw or not raw.isascii() or not raw.isdigit():
        raise ValueError(f"{name} must be an unsigned decimal integer")
    return int(raw)


@dataclass(slots=True)
class StreamStats:
    """Mutable counters accumulated while one stream is reconciled."""

    source_frames: int = 0
    recorded_frames: int = 0
    unmatched_admissions: int = 0


def iter_capture_payloads(path: Path, stream_id: int) -> Iterator[dict]:
    """Yield validated capture records without buffering the ledger.

    Camera frame identity is a scientific integrity gate: every record must
    provide an integer ID and IDs must advance by exactly one.
    """

    if not path.exists():
        raise RuntimeError(f"capture ledger is missing: {path.name}")
    previous_pts: int | None = None
    previous_camera_frame_id: int | None = None
    expected_sequence = 0
    try:
        lines = _bounded_text_lines(path)
        for line_number, line in enumerate(lines, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(
                    raw,
                    object_pairs_hook=_strict_json_object,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"invalid capture ledger {path.name}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"invalid capture ledger object in {path.name}:{line_number}"
                )
            try:
                camera_index = _json_uint(payload, "camera_index")
                sequence = _json_uint(payload, "source_sequence_index")
                pts_ns = _json_uint(payload, "gst_pts_ns")
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"invalid capture ledger fields {path.name}:{line_number}"
                ) from exc
            if camera_index != stream_id:
                raise RuntimeError(
                    f"capture ledger stream mismatch at {path.name}:{line_number}: "
                    f"expected {stream_id}, got {camera_index}"
                )
            if sequence != expected_sequence:
                raise RuntimeError(
                    "capture ledger source sequence is not contiguous at "
                    f"{path.name}:{line_number}: expected {expected_sequence}, got {sequence}"
                )
            if previous_pts is not None and pts_ns <= previous_pts:
                raise RuntimeError(
                    f"capture ledger PTS is not strictly increasing at {path.name}:{line_number}"
                )
            previous_pts = pts_ns
            camera_frame_id = payload.get("camera_frame_id")
            if isinstance(camera_frame_id, bool) or not isinstance(camera_frame_id, int):
                raise RuntimeError(
                    "capture ledger camera frame identity is unavailable at "
                    f"{path.name}:{line_number}"
                )
            if camera_frame_id < 0:
                raise RuntimeError(
                    f"capture ledger has negative camera frame ID at {path.name}:{line_number}"
                )
            if (
                previous_camera_frame_id is not None
                and camera_frame_id != previous_camera_frame_id + 1
            ):
                raise RuntimeError(
                    "capture ledger camera frame IDs are not contiguous at "
                    f"{path.name}:{line_number}: expected "
                    f"{previous_camera_frame_id + 1}, got {camera_frame_id}"
                )
            previous_camera_frame_id = camera_frame_id
            expected_sequence += 1
            yield payload
    except OSError as exc:
        raise RuntimeError(f"could not read capture ledger {path.name}: {exc}") from exc


def iter_admission_pts(path: Path, stream_id: int) -> Iterator[int]:
    """Yield validated recording-admission timestamps for one stream."""

    if not path.exists():
        raise RuntimeError(f"recording admission ledger is missing: {path.name}")
    previous_pts: int | None = None
    expected_index = 0
    try:
        reader = csv.DictReader(_bounded_text_lines(path))
        required = {"stream_id", "record_frame_index", "pts_ns"}
        missing = required.difference(reader.fieldnames or ())
        duplicate_columns = len(reader.fieldnames or ()) != len(set(reader.fieldnames or ()))
        if missing or duplicate_columns:
            raise RuntimeError(
                f"recording admission ledger {path.name} has invalid columns: "
                + (
                    f"missing {', '.join(sorted(missing))}"
                    if missing
                    else "duplicate column names"
                )
            )
        for line_number, row in enumerate(reader, 2):
            try:
                row_stream_id = _csv_uint(row["stream_id"], "stream_id")
                record_index = _csv_uint(row["record_frame_index"], "record_frame_index")
                pts_ns = _csv_uint(row["pts_ns"], "pts_ns")
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"invalid recording admission {path.name}:{line_number}"
                ) from exc
            if row_stream_id != stream_id:
                raise RuntimeError(
                    f"recording admission stream mismatch at {path.name}:{line_number}: "
                    f"expected {stream_id}, got {row_stream_id}"
                )
            if record_index != expected_index:
                raise RuntimeError(
                    "recording admission index is not contiguous at "
                    f"{path.name}:{line_number}: expected {expected_index}, got {record_index}"
                )
            if previous_pts is not None and pts_ns <= previous_pts:
                raise RuntimeError(
                    f"recording admission PTS is not strictly increasing at {path.name}:{line_number}"
                )
            previous_pts = pts_ns
            expected_index += 1
            yield pts_ns
    except (OSError, csv.Error) as exc:
        raise RuntimeError(
            f"could not read recording admission ledger {path.name}: {exc}"
        ) from exc


def recorded_payloads(
    run_dir: Path,
    stream_id: int,
    stats: StreamStats,
) -> Iterator[dict]:
    """Yield capture metadata only for frames admitted to recording."""

    capture_path = run_dir / f"capture_cam{stream_id}.jsonl"
    admission_path = run_dir / (
        "record_admission.csv"
        if stream_id == 0
        else f"record_admission_cam{stream_id}.csv"
    )
    admissions = iter_admission_pts(admission_path, stream_id)
    admission = next(admissions, None)
    for payload in iter_capture_payloads(capture_path, stream_id):
        stats.source_frames += 1
        pts_ns = int(payload.get("gst_pts_ns") or 0)
        while admission is not None and admission < pts_ns:
            stats.unmatched_admissions += 1
            admission = next(admissions, None)
        if admission == pts_ns:
            stats.recorded_frames += 1
            admission = next(admissions, None)
            yield payload
    while admission is not None:
        stats.unmatched_admissions += 1
        admission = next(admissions, None)


def payload_sort_key(payload: dict) -> tuple[int, int, int]:
    return (
        int(payload.get("host_received_monotonic_ns") or 0),
        int(payload.get("camera_index") or 0),
        int(payload.get("source_sequence_index") or 0),
    )
