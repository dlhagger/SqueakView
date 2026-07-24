#!/usr/bin/env python3
from __future__ import annotations

"""Align SqueakView run outputs onto the microcontroller TTL time base.

This script treats serial CAMERA_HIGH rows as the ground-truth trigger clock.
The FLIR frame counter is camera-owned and may begin at any value, so each run
derives its own epoch from the first recorded frame and first post-START trigger:

    camera_frame_id = CAMERA_HIGH count + camera_frame_id_offset

It writes analysis-ready CSVs under <run_dir>/analysis by default.
The main output is aligned_all.csv, a single long-format table with frames,
serial events, and detections sorted on the same microcontroller timeline.
The script intentionally uses only the Python standard library so it can run
inside the acquisition environment without adding pandas as a runtime dependency.
"""

import argparse
import bisect
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _latest_run_dir() -> Path | None:
    marker = _repo_root() / "runs" / ".latest_run"
    try:
        text = marker.read_text().strip()
    except FileNotFoundError:
        return None
    if not text:
        return None
    path = Path(text)
    return path if path.exists() else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fmt(value: Any) -> str:
    return "" if value is None else str(value)


def _fmt_float(value: float | None, digits: int = 9) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _normalize_serial_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    last_high_count: int | None = None
    last_low_count: int | None = None
    for index, row in enumerate(rows):
        out: dict[str, Any] = dict(row)
        out["_serial_index"] = index
        event_type = (row.get("eventType") or "").strip()
        out["eventType"] = event_type
        out["_unix_time_us"] = _to_int(row.get("unixTime"))
        out["_rp2040_time_us"] = _to_int(row.get("rp2040Time"))
        out["_count"] = _to_int(row.get("count"))
        out["_host_unix_ns"] = _to_int(row.get("hostUnixNs"))
        out["_host_monotonic_ns"] = _to_int(row.get("hostMonotonicNs"))

        if event_type == "CAMERA_HIGH" and out["_count"] is not None:
            last_high_count = int(out["_count"])
        if event_type == "CAMERA_LOW" and out["_count"] is not None:
            last_low_count = int(out["_count"])
        out["_last_high_count_before_or_at_row"] = last_high_count
        out["_last_low_count_before_or_at_row"] = last_low_count
        normalized.append(out)
    return normalized


def _ffprobe_video(path: Path) -> dict[str, Any]:
    if not path.exists() or not shutil.which("ffprobe"):
        return {}
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    except Exception:
        return {}
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()}
    stdout = proc.stdout.strip()
    json_start = stdout.find("{")
    if json_start > 0:
        stdout = stdout[json_start:]
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    streams = data.get("streams") or []
    return dict(streams[0]) if streams else {}


def _raw_video_info(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "raw.mp4"
    if not path.exists():
        return {}
    return {"file": str(path), "file_name": path.name, **_ffprobe_video(path)}


def _video_frame_map(frame_rows: list[dict[str, str]], run_dir: Path) -> dict[int, dict[str, Any]]:
    """Map authoritative frames.csv rows to their zero-based raw.mp4 frame indexes."""

    raw_path = run_dir / "raw.mp4"
    if not raw_path.exists():
        return {}

    ordered_rows: list[tuple[int, int, dict[str, str]]] = []
    for row_index, row in enumerate(frame_rows):
        stream_id = _to_int(row.get("stream_id"))
        if stream_id not in (None, 0):
            continue
        sort_key = _to_int(row.get("raw_frame_index"))
        if sort_key is None:
            sort_key = _to_int(row.get("camera_frame_id"))
        if sort_key is not None:
            ordered_rows.append((sort_key, row_index, row))
    ordered_rows.sort(key=lambda item: (item[0], item[1]))

    mapping: dict[int, dict[str, Any]] = {}
    for video_frame_index, (_sort_key, row_index, _row) in enumerate(ordered_rows):
        mapping[row_index] = {
            "raw_video_file": raw_path.name,
            "video_frame_index": video_frame_index,
            "video_mapping_source": "single_file_frames_csv",
        }
    return mapping


class CameraHighIndex:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = [
            row
            for row in rows
            if row.get("eventType") == "CAMERA_HIGH"
            and row.get("_count") is not None
            and row.get("_rp2040_time_us") is not None
        ]
        self.rows.sort(key=lambda row: (int(row["_rp2040_time_us"]), int(row["_count"])))
        self.by_count = {int(row["_count"]): row for row in self.rows}
        self.times = [int(row["_rp2040_time_us"]) for row in self.rows]
        self.counts = [int(row["_count"]) for row in self.rows]
        self.first_rp2040_us = self.times[0] if self.times else None

    def frame_time_s(self, rp2040_us: int | None) -> float | None:
        if rp2040_us is None or self.first_rp2040_us is None:
            return None
        return (int(rp2040_us) - self.first_rp2040_us) / 1_000_000.0

    def previous(self, rp2040_us: int | None) -> dict[str, Any] | None:
        if rp2040_us is None or not self.rows:
            return None
        idx = bisect.bisect_right(self.times, int(rp2040_us)) - 1
        return self.rows[idx] if idx >= 0 else None

    def nearest(self, rp2040_us: int | None) -> dict[str, Any] | None:
        if rp2040_us is None or not self.rows:
            return None
        idx = bisect.bisect_left(self.times, int(rp2040_us))
        candidates: list[dict[str, Any]] = []
        if idx > 0:
            candidates.append(self.rows[idx - 1])
        if idx < len(self.rows):
            candidates.append(self.rows[idx])
        if not candidates:
            return None
        return min(candidates, key=lambda row: abs(int(row["_rp2040_time_us"]) - int(rp2040_us)))


def _detect_frame_gaps(frame_ids: list[int]) -> list[dict[str, int]]:
    gaps: list[dict[str, int]] = []
    prev: int | None = None
    for frame_id in frame_ids:
        if prev is not None and frame_id != prev + 1:
            gaps.append({"expected": prev + 1, "actual": frame_id})
        prev = frame_id
    return gaps


def _marker_name(row: dict[str, Any]) -> str:
    return (row.get("reason") or row.get("context") or row.get("unixTime") or "").strip()


def _derive_frame_epoch(
    frame_rows: list[dict[str, str]],
    serial_rows: list[dict[str, Any]],
    camera_high: CameraHighIndex,
) -> dict[str, Any]:
    first_frame = next(
        (row for row in frame_rows if _to_int(row.get("camera_frame_id")) is not None),
        None,
    )
    if first_frame is None:
        raise RuntimeError("frames.csv contains no camera_frame_id values")

    start_marker_index = next(
        (
            int(row["_serial_index"])
            for row in serial_rows
            if row.get("eventType") == "MARKER" and _marker_name(row) == "START_SENT"
        ),
        None,
    )
    eligible_highs = (
        [row for row in camera_high.rows if int(row["_serial_index"]) > start_marker_index]
        if start_marker_index is not None
        else camera_high.rows
    )
    if not eligible_highs:
        raise RuntimeError("serial.csv has no CAMERA_HIGH row after START_SENT")

    first_high = eligible_highs[0]
    first_camera_frame_id = int(_to_int(first_frame.get("camera_frame_id")))
    first_ttl_count = int(first_high["_count"])
    return {
        "method": (
            "first_recorded_frame_to_first_camera_high_after_start_sent"
            if start_marker_index is not None
            else "first_recorded_frame_to_first_camera_high"
        ),
        "first_camera_frame_id": first_camera_frame_id,
        "first_raw_frame_index": _to_int(first_frame.get("raw_frame_index")),
        "first_ttl_count": first_ttl_count,
        "camera_frame_id_offset": first_camera_frame_id - first_ttl_count,
        "first_camera_timestamp_ns": _to_int(first_frame.get("camera_timestamp_ns")),
        "first_rp2040_time_us": int(first_high["_rp2040_time_us"]),
    }


def build_alignment(
    run_dir: Path,
    out_dir: Path,
    *,
    detections_path: Path | None = None,
) -> dict[str, Any]:
    frames_path = run_dir / "frames.csv"
    serial_path = run_dir / "serial.csv"
    detections_path = detections_path or (run_dir / "detections.csv")
    drop_events_path = run_dir / "drop_events.csv"

    if not frames_path.exists():
        raise FileNotFoundError(f"missing required run outputs in {run_dir}: {frames_path.name}")
    frame_rows = _read_csv(frames_path)
    raw_video_info = _raw_video_info(run_dir)
    video_frame_by_row = _video_frame_map(frame_rows, run_dir)
    if not serial_path.exists():
        raise FileNotFoundError(f"missing required run outputs in {run_dir}: {serial_path.name}")

    serial_rows = _normalize_serial_rows(_read_csv(serial_path))
    detection_rows = _read_csv(detections_path) if detections_path.exists() else []
    drop_rows = _read_csv(drop_events_path) if drop_events_path.exists() else []

    camera_high = CameraHighIndex(serial_rows)
    if not camera_high.rows:
        raise RuntimeError("serial.csv contains no CAMERA_HIGH rows with count and rp2040Time")

    frame_epoch = _derive_frame_epoch(frame_rows, serial_rows, camera_high)
    camera_frame_id_offset = int(frame_epoch["camera_frame_id_offset"])

    detection_count_by_raw_frame: dict[int, int] = {}
    for det in detection_rows:
        raw_frame_index = _to_int(det.get("raw_frame_num"))
        if raw_frame_index is None:
            continue
        detection_count_by_raw_frame[raw_frame_index] = (
            detection_count_by_raw_frame.get(raw_frame_index, 0) + 1
        )

    marker_counts: dict[str, int | None] = {}
    for row in serial_rows:
        if row.get("eventType") != "MARKER":
            continue
        marker_name = _marker_name(row)
        if marker_name:
            marker_counts[marker_name] = row.get("_last_high_count_before_or_at_row")

    aligned_frames: list[dict[str, Any]] = []
    frame_by_id: dict[int, dict[str, Any]] = {}
    frame_by_raw_index: dict[int, dict[str, Any]] = {}
    frame_ids: list[int] = []
    for row_index, row in enumerate(frame_rows):
        camera_frame_id = _to_int(row.get("camera_frame_id"))
        if camera_frame_id is None:
            continue
        video_frame = video_frame_by_row.get(row_index, {})
        raw_video_file = video_frame.get("raw_video_file", "")
        video_frame_index = video_frame.get("video_frame_index", "")
        video_mapping_source = video_frame.get("video_mapping_source", "")
        frame_ids.append(camera_frame_id)
        ttl_count = camera_frame_id - camera_frame_id_offset
        high = camera_high.by_count.get(ttl_count)
        frame_rp2040_us = _to_int(high.get("rp2040Time")) if high else None
        controller_unix_us = _to_int(high.get("unixTime")) if high else None
        pts_ns = _to_int(row.get("pts_ns"))
        duration_ns = _to_int(row.get("duration_ns"))
        raw_frame_index = _to_int(row.get("raw_frame_index"))
        detection_count = detection_count_by_raw_frame.get(raw_frame_index, 0)
        out = {
            "raw_frame_index": row.get("raw_frame_index", ""),
            "camera_frame_id": camera_frame_id,
            "ttl_count": ttl_count,
            "frame_rp2040_us": _fmt(frame_rp2040_us),
            "frame_time_s": _fmt_float(camera_high.frame_time_s(frame_rp2040_us)),
            "controller_unix_us": _fmt(controller_unix_us),
            "camera_timestamp_ns": row.get("camera_timestamp_ns", ""),
            "frame_pts_ns": _fmt(pts_ns),
            "frame_pts_s": _fmt_float((pts_ns / 1_000_000_000.0) if pts_ns is not None else None),
            "duration_ns": _fmt(duration_ns),
            "frame_host_unix_ns": row.get("host_unix_ns", ""),
            "frame_host_monotonic_ns": row.get("host_monotonic_ns", ""),
            "ttl_host_unix_ns": high.get("hostUnixNs", "") if high else "",
            "ttl_host_monotonic_ns": high.get("hostMonotonicNs", "") if high else "",
            "raw_video_file": raw_video_file,
            "video_frame_index": video_frame_index,
            "video_mapping_source": video_mapping_source,
            "status": row.get("status", ""),
            "has_ttl": "1" if high else "0",
            "has_detection": "1" if detection_count else "0",
            "detection_count": detection_count,
        }
        aligned_frames.append(out)
        frame_by_id[camera_frame_id] = out
        if raw_frame_index is not None:
            frame_by_raw_index[raw_frame_index] = out

    mapped_pair_count = sum(1 for row in aligned_frames if row["has_ttl"] == "1")
    missing_ttl_pair_count = len(aligned_frames) - mapped_pair_count
    source_sequence_mismatches: list[dict[str, int]] = []
    first_raw_frame_index = frame_epoch.get("first_raw_frame_index")
    if first_raw_frame_index is not None:
        for frame in aligned_frames:
            raw_frame_index = _to_int(frame.get("raw_frame_index"))
            if raw_frame_index is None:
                continue
            expected_camera_frame_id = int(frame_epoch["first_camera_frame_id"]) + (
                raw_frame_index - int(first_raw_frame_index)
            )
            actual_camera_frame_id = int(frame["camera_frame_id"])
            if actual_camera_frame_id != expected_camera_frame_id:
                source_sequence_mismatches.append(
                    {
                        "raw_frame_index": raw_frame_index,
                        "expected_camera_frame_id": expected_camera_frame_id,
                        "actual_camera_frame_id": actual_camera_frame_id,
                    }
                )

    clock_elapsed_errors_us: list[float] = []
    first_camera_timestamp_ns = frame_epoch.get("first_camera_timestamp_ns")
    first_rp2040_time_us = frame_epoch.get("first_rp2040_time_us")
    if first_camera_timestamp_ns is not None and first_rp2040_time_us is not None:
        for frame in aligned_frames:
            camera_timestamp_ns = _to_int(frame.get("camera_timestamp_ns"))
            frame_rp2040_us = _to_int(frame.get("frame_rp2040_us"))
            if camera_timestamp_ns is None or frame_rp2040_us is None:
                continue
            camera_elapsed_us = (camera_timestamp_ns - int(first_camera_timestamp_ns)) / 1_000.0
            rp2040_elapsed_us = frame_rp2040_us - int(first_rp2040_time_us)
            clock_elapsed_errors_us.append(camera_elapsed_us - rp2040_elapsed_us)

    ttl_intervals_us = [
        later - earlier
        for earlier, later in zip(camera_high.times, camera_high.times[1:])
        if later > earlier
    ]
    median_ttl_interval_us = (
        sorted(ttl_intervals_us)[len(ttl_intervals_us) // 2]
        if ttl_intervals_us
        else None
    )
    clock_tolerance_us = (
        max(1_000.0, median_ttl_interval_us / 2.0)
        if median_ttl_interval_us is not None
        else None
    )
    clock_error_max_abs_us = (
        max(abs(value) for value in clock_elapsed_errors_us)
        if clock_elapsed_errors_us
        else None
    )
    clock_within_tolerance = (
        clock_error_max_abs_us <= clock_tolerance_us
        if clock_error_max_abs_us is not None and clock_tolerance_us is not None
        else None
    )

    frame_epoch.update(
        {
            "validated_pairs": mapped_pair_count,
            "missing_ttl_pairs": missing_ttl_pair_count,
            "source_sequence_mismatch_count": len(source_sequence_mismatches),
            "source_sequence_mismatches_sample": source_sequence_mismatches[:50],
            "clock_validation_pairs": len(clock_elapsed_errors_us),
            "clock_elapsed_error_us_final": (
                clock_elapsed_errors_us[-1] if clock_elapsed_errors_us else None
            ),
            "clock_elapsed_error_us_max_abs": clock_error_max_abs_us,
            "clock_tolerance_us": clock_tolerance_us,
            "clock_within_tolerance": clock_within_tolerance,
            "validated": (
                missing_ttl_pair_count == 0
                and not source_sequence_mismatches
                and clock_within_tolerance is not False
            ),
        }
    )

    aligned_events: list[dict[str, Any]] = []
    for row in serial_rows:
        event_type = row.get("eventType", "")
        rp2040_us = row.get("_rp2040_time_us")
        count = row.get("_count")
        prev_high = camera_high.previous(rp2040_us)
        nearest_high = camera_high.nearest(rp2040_us)
        prev_count = int(prev_high["_count"]) if prev_high else row.get("_last_high_count_before_or_at_row")
        prev_rp = int(prev_high["_rp2040_time_us"]) if prev_high else None
        if prev_high is None and prev_count is not None:
            ordered_prev_high = camera_high.by_count.get(int(prev_count))
            if ordered_prev_high is not None:
                prev_rp = int(ordered_prev_high["_rp2040_time_us"])
        nearest_count = int(nearest_high["_count"]) if nearest_high else None
        nearest_rp = int(nearest_high["_rp2040_time_us"]) if nearest_high else None
        previous_frame_id = prev_count + camera_frame_id_offset if prev_count is not None else None
        nearest_frame_id = nearest_count + camera_frame_id_offset if nearest_count is not None else None

        if event_type == "CAMERA_HIGH" and count is not None:
            alignment_method = "camera_high_exact"
        elif rp2040_us is not None:
            alignment_method = "rp2040_previous_camera_high"
        else:
            alignment_method = "serial_order_previous_camera_high"

        marker_name = ""
        if event_type == "MARKER":
            marker_name = _marker_name(row)

        aligned_events.append(
            {
                "serial_index": row.get("_serial_index", ""),
                "eventType": event_type,
                "event_name": marker_name or event_type,
                "rp2040_time_us": _fmt(rp2040_us),
                "event_time_s": _fmt_float(camera_high.frame_time_s(rp2040_us)),
                "controller_unix_us": _fmt(row.get("_unix_time_us")),
                "side": row.get("side", ""),
                "count": _fmt(count),
                "duration": row.get("duration", ""),
                "latency": row.get("latency", ""),
                "value": row.get("value", ""),
                "context": row.get("context", ""),
                "reason": row.get("reason", ""),
                "previous_ttl_count": _fmt(prev_count),
                "previous_frame_id": _fmt(previous_frame_id),
                "previous_frame_rp2040_us": _fmt(prev_rp),
                "offset_from_previous_frame_ms": _fmt_float(
                    ((int(rp2040_us) - prev_rp) / 1_000.0) if rp2040_us is not None and prev_rp is not None else None,
                    digits=6,
                ),
                "nearest_ttl_count": _fmt(nearest_count),
                "nearest_frame_id": _fmt(nearest_frame_id),
                "nearest_frame_rp2040_us": _fmt(nearest_rp),
                "offset_from_nearest_frame_ms": _fmt_float(
                    ((int(rp2040_us) - nearest_rp) / 1_000.0)
                    if rp2040_us is not None and nearest_rp is not None
                    else None,
                    digits=6,
                ),
                "alignment_method": alignment_method,
                "hostUnixNs": row.get("hostUnixNs", ""),
                "hostMonotonicNs": row.get("hostMonotonicNs", ""),
                "rawLine": row.get("rawLine", ""),
            }
        )

    aligned_detections: list[dict[str, Any]] = []
    for det_index, det in enumerate(detection_rows):
        raw_frame_num = _to_int(det.get("raw_frame_num"))
        frame = frame_by_raw_index.get(raw_frame_num) if raw_frame_num is not None else None
        if frame is None and raw_frame_num is not None:
            # Compatibility with legacy detections that stored the camera frame ID.
            frame = frame_by_id.get(raw_frame_num)
        camera_frame_id = _to_int(frame.get("camera_frame_id")) if frame else None
        mapping_method = det.get("raw_frame_mapping_method", "") or "legacy_no_provenance"
        mapping_ok = det.get("raw_frame_mapping_ok", "")
        mapping_pts_ns = det.get("raw_frame_mapping_pts_ns", "")
        mapping_delta = det.get("raw_frame_mapping_delta", "")
        aligned_detections.append(
            {
                "detection_index": det_index,
                "raw_frame_index": frame.get("raw_frame_index", "") if frame else "",
                "camera_frame_id": _fmt(camera_frame_id),
                "ttl_count": frame.get("ttl_count", "") if frame else "",
                "detection_rp2040_us": frame.get("frame_rp2040_us", "") if frame else "",
                "detection_time_s": frame.get("frame_time_s", "") if frame else "",
                "frame_pts_ns": frame.get("frame_pts_ns", "") if frame else "",
                "frame_pts_s": frame.get("frame_pts_s", "") if frame else "",
                "raw_video_file": frame.get("raw_video_file", "") if frame else "",
                "video_frame_index": frame.get("video_frame_index", "") if frame else "",
                "video_mapping_source": frame.get("video_mapping_source", "") if frame else "",
                "raw_frame_mapping_method": mapping_method,
                "raw_frame_mapping_ok": mapping_ok,
                "raw_frame_mapping_pts_ns": mapping_pts_ns,
                "raw_frame_mapping_delta": mapping_delta,
                "source": det.get("source", ""),
                "class_label": det.get("class_label", ""),
                "conf": det.get("conf", ""),
                "x": det.get("x", ""),
                "y": det.get("y", ""),
                "w": det.get("w", ""),
                "h": det.get("h", ""),
                "original_frame": det.get("frame", ""),
                "original_ts_us": det.get("ts_us", ""),
                "pose_schema": det.get("pose_schema", ""),
                "kpt_count": det.get("kpt_count", ""),
                "kpt_names_json": det.get("kpt_names_json", ""),
                "kpt_values_json": det.get("kpt_values_json", ""),
            }
        )

    aligned_all: list[dict[str, Any]] = []

    def aligned_all_sort_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
        type_order = {"FRAME": 0, "SERIAL": 1, "DETECTION": 2}
        return (
            _to_int(row.get("time_rp2040_us")) or -1,
            type_order.get(str(row.get("record_type")), 9),
            _to_int(row.get("camera_frame_id")) or -1,
            _to_int(row.get("detection_index")) or 0,
        )

    for frame in aligned_frames:
        aligned_all.append(
            {
                "record_type": "FRAME",
                "time_rp2040_us": frame["frame_rp2040_us"],
                "time_s": frame["frame_time_s"],
                "time_source": "camera_high",
                "event_type": "CAMERA_FRAME",
                "event_name": "CAMERA_FRAME",
                "raw_event_rp2040_us": "",
                "raw_event_time_s": "",
                "trigger_event_type": "CAMERA_HIGH",
                "frame_trigger_rp2040_us": frame["frame_rp2040_us"],
                "frame_trigger_time_s": frame["frame_time_s"],
                "offset_from_frame_trigger_ms": "0.000000",
                "raw_frame_index": frame["raw_frame_index"],
                "camera_frame_id": frame["camera_frame_id"],
                "ttl_count": frame["ttl_count"],
                "source": "cam0",
                "frame_pts_ns": frame["frame_pts_ns"],
                "frame_pts_s": frame["frame_pts_s"],
                "duration_ns": frame["duration_ns"],
                "raw_video_file": frame["raw_video_file"],
                "video_frame_index": frame["video_frame_index"],
                "video_mapping_source": frame["video_mapping_source"],
                "frame_status": frame["status"],
                "has_detection": frame["has_detection"],
                "detection_count": frame["detection_count"],
                "frame_host_unix_ns": frame["frame_host_unix_ns"],
                "frame_host_monotonic_ns": frame["frame_host_monotonic_ns"],
                "ttl_host_unix_ns": frame["ttl_host_unix_ns"],
                "ttl_host_monotonic_ns": frame["ttl_host_monotonic_ns"],
            }
        )

    for event in aligned_events:
        event_time = event["rp2040_time_us"]
        time_source = "rp2040"
        if not event_time:
            event_time = event["previous_frame_rp2040_us"]
            time_source = "serial_order_previous_camera_high"
        aligned_all.append(
            {
                "record_type": "SERIAL",
                "time_rp2040_us": event_time,
                "time_s": _fmt_float(camera_high.frame_time_s(_to_int(event_time))),
                "time_source": time_source,
                "event_type": event["eventType"],
                "event_name": event["event_name"],
                "raw_event_rp2040_us": event["rp2040_time_us"],
                "raw_event_time_s": event["event_time_s"],
                "trigger_event_type": "CAMERA_HIGH" if event["previous_ttl_count"] else "",
                "frame_trigger_rp2040_us": event["previous_frame_rp2040_us"],
                "frame_trigger_time_s": _fmt_float(camera_high.frame_time_s(_to_int(event["previous_frame_rp2040_us"]))),
                "offset_from_frame_trigger_ms": event["offset_from_previous_frame_ms"],
                "camera_frame_id": event["previous_frame_id"],
                "ttl_count": event["previous_ttl_count"],
                "serial_index": event["serial_index"],
                "controller_unix_us": event["controller_unix_us"],
                "side": event["side"],
                "event_count": event["count"],
                "event_duration": event["duration"],
                "event_latency": event["latency"],
                "event_value": event["value"],
                "event_context": event["context"],
                "event_reason": event["reason"],
                "previous_ttl_count": event["previous_ttl_count"],
                "previous_frame_id": event["previous_frame_id"],
                "previous_frame_rp2040_us": event["previous_frame_rp2040_us"],
                "offset_from_previous_frame_ms": event["offset_from_previous_frame_ms"],
                "nearest_ttl_count": event["nearest_ttl_count"],
                "nearest_frame_id": event["nearest_frame_id"],
                "nearest_frame_rp2040_us": event["nearest_frame_rp2040_us"],
                "offset_from_nearest_frame_ms": event["offset_from_nearest_frame_ms"],
                "alignment_method": event["alignment_method"],
                "serial_host_unix_ns": event["hostUnixNs"],
                "serial_host_monotonic_ns": event["hostMonotonicNs"],
                "raw_line": event["rawLine"],
            }
        )

    for det in aligned_detections:
        aligned_all.append(
            {
                "record_type": "DETECTION",
                "time_rp2040_us": det["detection_rp2040_us"],
                "time_s": det["detection_time_s"],
                "time_source": "detection_frame",
                "event_type": "DETECTION",
                "event_name": det["class_label"],
                "raw_event_rp2040_us": "",
                "raw_event_time_s": "",
                "trigger_event_type": "CAMERA_HIGH",
                "frame_trigger_rp2040_us": det["detection_rp2040_us"],
                "frame_trigger_time_s": det["detection_time_s"],
                "offset_from_frame_trigger_ms": "0.000000",
                "raw_frame_index": det["raw_frame_index"],
                "camera_frame_id": det["camera_frame_id"],
                "ttl_count": det["ttl_count"],
                "source": det["source"],
                "frame_pts_ns": det["frame_pts_ns"],
                "frame_pts_s": det["frame_pts_s"],
                "raw_video_file": det["raw_video_file"],
                "video_frame_index": det["video_frame_index"],
                "video_mapping_source": det["video_mapping_source"],
                "raw_frame_mapping_method": det["raw_frame_mapping_method"],
                "raw_frame_mapping_ok": det["raw_frame_mapping_ok"],
                "raw_frame_mapping_pts_ns": det["raw_frame_mapping_pts_ns"],
                "raw_frame_mapping_delta": det["raw_frame_mapping_delta"],
                "detection_index": det["detection_index"],
                "class_label": det["class_label"],
                "conf": det["conf"],
                "x": det["x"],
                "y": det["y"],
                "w": det["w"],
                "h": det["h"],
                "original_frame": det["original_frame"],
                "original_ts_us": det["original_ts_us"],
                "pose_schema": det["pose_schema"],
                "kpt_count": det["kpt_count"],
                "kpt_names_json": det["kpt_names_json"],
                "kpt_values_json": det["kpt_values_json"],
            }
        )

    aligned_all.sort(key=aligned_all_sort_key)

    timeline: list[dict[str, Any]] = []
    for frame in aligned_frames:
        timeline.append(
            {
                "record_type": "FRAME",
                "time_rp2040_us": frame["frame_rp2040_us"],
                "time_s": frame["frame_time_s"],
                "time_source": "camera_high",
                "raw_frame_index": frame["raw_frame_index"],
                "camera_frame_id": frame["camera_frame_id"],
                "ttl_count": frame["ttl_count"],
                "event_type": "FRAME",
                "event_name": "FRAME",
                "detection_index": "",
                "conf": "",
                "x": "",
                "y": "",
                "w": "",
                "h": "",
                "context": "",
                "reason": "",
                "raw_line": "",
            }
        )
    for event in aligned_events:
        event_time = event["rp2040_time_us"]
        time_source = "rp2040"
        if not event_time:
            event_time = event["previous_frame_rp2040_us"]
            time_source = "serial_order_previous_camera_high"
        timeline.append(
            {
                "record_type": "SERIAL",
                "time_rp2040_us": event_time,
                "time_s": _fmt_float(camera_high.frame_time_s(_to_int(event_time))),
                "time_source": time_source,
                "raw_frame_index": "",
                "camera_frame_id": event["previous_frame_id"],
                "ttl_count": event["previous_ttl_count"],
                "event_type": event["eventType"],
                "event_name": event["event_name"],
                "detection_index": "",
                "conf": "",
                "x": "",
                "y": "",
                "w": "",
                "h": "",
                "context": event["context"],
                "reason": event["reason"],
                "raw_line": event["rawLine"],
            }
        )
    for det in aligned_detections:
        timeline.append(
            {
                "record_type": "DETECTION",
                "time_rp2040_us": det["detection_rp2040_us"],
                "time_s": det["detection_time_s"],
                "time_source": "detection_frame",
                "raw_frame_index": det["raw_frame_index"],
                "camera_frame_id": det["camera_frame_id"],
                "ttl_count": det["ttl_count"],
                "event_type": "DETECTION",
                "event_name": det["class_label"],
                "detection_index": det["detection_index"],
                "conf": det["conf"],
                "x": det["x"],
                "y": det["y"],
                "w": det["w"],
                "h": det["h"],
                "context": "",
                "reason": "",
                "raw_line": "",
            }
        )

    def timeline_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
        type_order = {"FRAME": 0, "SERIAL": 1, "DETECTION": 2}
        return (_to_int(row.get("time_rp2040_us")) or -1, type_order.get(str(row.get("record_type")), 9), _to_int(row.get("detection_index")) or 0)

    timeline.sort(key=timeline_sort_key)

    frame_fields = [
        "camera_frame_id",
        "raw_frame_index",
        "ttl_count",
        "frame_rp2040_us",
        "frame_time_s",
        "controller_unix_us",
        "camera_timestamp_ns",
        "frame_pts_ns",
        "frame_pts_s",
        "duration_ns",
        "frame_host_unix_ns",
        "frame_host_monotonic_ns",
        "ttl_host_unix_ns",
        "ttl_host_monotonic_ns",
        "raw_video_file",
        "video_frame_index",
        "video_mapping_source",
        "status",
        "has_ttl",
        "has_detection",
        "detection_count",
    ]
    event_fields = [
        "serial_index",
        "eventType",
        "event_name",
        "rp2040_time_us",
        "event_time_s",
        "controller_unix_us",
        "side",
        "count",
        "duration",
        "latency",
        "value",
        "context",
        "reason",
        "previous_ttl_count",
        "previous_frame_id",
        "previous_frame_rp2040_us",
        "offset_from_previous_frame_ms",
        "nearest_ttl_count",
        "nearest_frame_id",
        "nearest_frame_rp2040_us",
        "offset_from_nearest_frame_ms",
        "alignment_method",
        "hostUnixNs",
        "hostMonotonicNs",
        "rawLine",
    ]
    detection_fields = [
        "detection_index",
        "raw_frame_index",
        "camera_frame_id",
        "ttl_count",
        "detection_rp2040_us",
        "detection_time_s",
        "frame_pts_ns",
        "frame_pts_s",
        "raw_video_file",
        "video_frame_index",
        "video_mapping_source",
        "raw_frame_mapping_method",
        "raw_frame_mapping_ok",
        "raw_frame_mapping_pts_ns",
        "raw_frame_mapping_delta",
        "source",
        "class_label",
        "conf",
        "x",
        "y",
        "w",
        "h",
        "original_frame",
        "original_ts_us",
        "pose_schema",
        "kpt_count",
        "kpt_names_json",
        "kpt_values_json",
    ]
    timeline_fields = [
        "record_type",
        "time_rp2040_us",
        "time_s",
        "time_source",
        "camera_frame_id",
        "raw_frame_index",
        "ttl_count",
        "event_type",
        "event_name",
        "detection_index",
        "conf",
        "x",
        "y",
        "w",
        "h",
        "context",
        "reason",
        "raw_line",
    ]
    aligned_all_fields = [
        "record_type",
        "time_rp2040_us",
        "time_s",
        "time_source",
        "event_type",
        "event_name",
        "raw_event_rp2040_us",
        "raw_event_time_s",
        "trigger_event_type",
        "frame_trigger_rp2040_us",
        "frame_trigger_time_s",
        "offset_from_frame_trigger_ms",
        "camera_frame_id",
        "raw_frame_index",
        "ttl_count",
        "source",
        "frame_pts_ns",
        "frame_pts_s",
        "duration_ns",
        "raw_video_file",
        "video_frame_index",
        "video_mapping_source",
        "frame_status",
        "has_detection",
        "detection_count",
        "frame_host_unix_ns",
        "frame_host_monotonic_ns",
        "ttl_host_unix_ns",
        "ttl_host_monotonic_ns",
        "serial_index",
        "controller_unix_us",
        "side",
        "event_count",
        "event_duration",
        "event_latency",
        "event_value",
        "event_context",
        "event_reason",
        "previous_ttl_count",
        "previous_frame_id",
        "previous_frame_rp2040_us",
        "offset_from_previous_frame_ms",
        "nearest_ttl_count",
        "nearest_frame_id",
        "nearest_frame_rp2040_us",
        "offset_from_nearest_frame_ms",
        "alignment_method",
        "serial_host_unix_ns",
        "serial_host_monotonic_ns",
        "raw_line",
        "detection_index",
        "raw_frame_mapping_method",
        "raw_frame_mapping_ok",
        "raw_frame_mapping_pts_ns",
        "raw_frame_mapping_delta",
        "class_label",
        "conf",
        "x",
        "y",
        "w",
        "h",
        "original_frame",
        "original_ts_us",
        "pose_schema",
        "kpt_count",
        "kpt_names_json",
        "kpt_values_json",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "aligned_all.csv", aligned_all, aligned_all_fields)
    _write_csv(out_dir / "aligned_frames.csv", aligned_frames, frame_fields)
    _write_csv(out_dir / "aligned_events.csv", aligned_events, event_fields)
    _write_csv(out_dir / "aligned_detections.csv", aligned_detections, detection_fields)
    _write_csv(out_dir / "aligned_timeline.csv", timeline, timeline_fields)

    frame_gaps = _detect_frame_gaps(sorted(frame_ids))
    video_mapping_source_counts: dict[str, int] = {}
    for row in aligned_frames:
        source = str(row.get("video_mapping_source") or "")
        video_mapping_source_counts[source] = video_mapping_source_counts.get(source, 0) + 1
    video_nb_frames = _to_int(raw_video_info.get("nb_frames"))

    detection_mapping_method_counts: dict[str, int] = {}
    detection_mapping_failed_rows = 0
    detection_mapping_fallback_rows = 0
    detection_mapping_legacy_rows = 0
    detection_missing_frame_count = 0
    detection_ts_mismatch_count = 0
    detection_pts_mismatch_count = 0
    detection_missing_frames: list[dict[str, Any]] = []
    detection_ts_mismatches: list[dict[str, Any]] = []
    detection_pts_mismatches: list[dict[str, Any]] = []
    for det_index, det in enumerate(detection_rows):
        method = det.get("raw_frame_mapping_method", "") or "legacy_no_provenance"
        detection_mapping_method_counts[method] = detection_mapping_method_counts.get(method, 0) + 1
        if method.startswith("fallback"):
            detection_mapping_fallback_rows += 1
        if method == "legacy_no_provenance":
            detection_mapping_legacy_rows += 1

        mapping_ok = (det.get("raw_frame_mapping_ok", "") or "").strip()
        if mapping_ok and mapping_ok != "1":
            detection_mapping_failed_rows += 1

        raw_frame_num = _to_int(det.get("raw_frame_num"))
        frame = frame_by_raw_index.get(raw_frame_num) if raw_frame_num is not None else None
        if frame is None and raw_frame_num is not None:
            # Compatibility with legacy detections that stored the camera frame ID.
            frame = frame_by_id.get(raw_frame_num)
        camera_frame_id = _to_int(frame.get("camera_frame_id")) if frame else None
        if frame is None:
            detection_missing_frame_count += 1
            if len(detection_missing_frames) < 50:
                detection_missing_frames.append(
                    {
                        "detection_index": det_index,
                        "raw_frame_num": raw_frame_num,
                        "frame": det.get("frame", ""),
                        "ts_us": det.get("ts_us", ""),
                    }
                )
            continue

        frame_pts_ns = _to_int(frame.get("frame_pts_ns"))
        detection_ts_us = _to_int(det.get("ts_us"))
        if frame_pts_ns is not None and detection_ts_us is not None:
            expected_ts_us = frame_pts_ns // 1_000
            if abs(detection_ts_us - expected_ts_us) > 1:
                detection_ts_mismatch_count += 1
                if len(detection_ts_mismatches) < 50:
                    detection_ts_mismatches.append(
                        {
                            "detection_index": det_index,
                            "raw_frame_num": raw_frame_num,
                            "detection_ts_us": detection_ts_us,
                            "frame_pts_us": expected_ts_us,
                            "delta_us": detection_ts_us - expected_ts_us,
                        }
                    )

        mapping_pts_ns = _to_int(det.get("raw_frame_mapping_pts_ns"))
        if frame_pts_ns is not None and mapping_pts_ns is not None and mapping_pts_ns != frame_pts_ns:
            detection_pts_mismatch_count += 1
            if len(detection_pts_mismatches) < 50:
                detection_pts_mismatches.append(
                    {
                        "detection_index": det_index,
                        "raw_frame_num": raw_frame_num,
                        "mapping_pts_ns": mapping_pts_ns,
                        "frame_pts_ns": frame_pts_ns,
                        "delta_ns": mapping_pts_ns - frame_pts_ns,
                    }
                )

    summary = {
        "run_dir": str(run_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "time_base": "serial.csv CAMERA_HIGH rp2040Time",
        "frame_alignment_rule": "camera_frame_id = CAMERA_HIGH count + dynamic offset",
        "frame_alignment": frame_epoch,
        "outputs": {
            "aligned_all": str(out_dir / "aligned_all.csv"),
            "aligned_frames": str(out_dir / "aligned_frames.csv"),
            "aligned_events": str(out_dir / "aligned_events.csv"),
            "aligned_detections": str(out_dir / "aligned_detections.csv"),
            "aligned_timeline": str(out_dir / "aligned_timeline.csv"),
        },
        "counts": {
            "recorded_frames": len(aligned_frames),
            "camera_high_events": len(camera_high.rows),
            "serial_rows": len(serial_rows),
            "detections": len(aligned_detections),
            "aligned_all_rows": len(aligned_all),
            "drop_events": len(drop_rows),
            "frame_gaps_detected": len(frame_gaps),
            "frames_missing_ttl": sum(1 for row in aligned_frames if row["has_ttl"] != "1"),
        },
        "frame_range": {
            "first_camera_frame_id": min(frame_ids) if frame_ids else None,
            "last_camera_frame_id": max(frame_ids) if frame_ids else None,
        },
        "markers": marker_counts,
        "capture_stop_requested_ttl_count": marker_counts.get("CAPTURE_STOP_REQUESTED"),
        "capture_stop_done_ttl_count": marker_counts.get("CAPTURE_STOP_DONE"),
        "stop_sent_ttl_count": marker_counts.get("STOP_SENT"),
        "post_capture_ttl_tail": (
            (max(camera_high.counts) - int(marker_counts["CAPTURE_STOP_REQUESTED"]))
            if marker_counts.get("CAPTURE_STOP_REQUESTED") is not None and camera_high.counts
            else None
        ),
        "validation": {
            "video_total_nb_frames": video_nb_frames,
            "video_frame_count_matches_frames_csv": (
                video_nb_frames == len(aligned_frames) if video_nb_frames is not None else None
            ),
            "detections_missing_frame_count": detection_missing_frame_count,
            "detection_ts_mismatch_count": detection_ts_mismatch_count,
            "detection_pts_mismatch_count": detection_pts_mismatch_count,
            "detection_mapping_method_counts": detection_mapping_method_counts,
            "video_mapping_source_counts": video_mapping_source_counts,
            "detection_mapping_failed_rows": detection_mapping_failed_rows,
            "detection_mapping_fallback_rows": detection_mapping_fallback_rows,
            "detection_mapping_legacy_rows": detection_mapping_legacy_rows,
            "detection_missing_frames_sample": detection_missing_frames,
            "detection_ts_mismatches_sample": detection_ts_mismatches,
            "detection_pts_mismatches_sample": detection_pts_mismatches,
        },
        "frame_gaps": frame_gaps[:50],
        "raw_video_info": raw_video_info,
    }
    (out_dir / "alignment_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Run directory. Defaults to runs/.latest_run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_dir>/analysis.",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=None,
        help="Detection CSV to align. Defaults to <run_dir>/detections.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir or _latest_run_dir()
    if run_dir is None:
        raise SystemExit("No run_dir given and runs/.latest_run is missing")
    run_dir = run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "analysis")).resolve()
    summary = build_alignment(
        run_dir, out_dir,
        detections_path=args.detections.resolve() if args.detections else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
