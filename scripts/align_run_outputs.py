#!/usr/bin/env python3
from __future__ import annotations

"""Align SqueakView run outputs onto the microcontroller TTL time base.

This script treats serial CAMERA_HIGH rows as the ground-truth frame clock:

    camera_frame_id = CAMERA_HIGH count - 1

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


def _ffprobe_raw_segments(run_dir: Path) -> dict[str, Any]:
    paths = sorted(run_dir.glob("raw_*.mp4"))
    if not paths and (run_dir / "raw.mp4").exists():
        paths = [run_dir / "raw.mp4"]

    total_frames = 0
    missing_frame_counts = False
    segments: list[dict[str, Any]] = []
    sample_segments: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        info = _ffprobe_video(path)
        nb_frames = _to_int(info.get("nb_frames"))
        if nb_frames is None:
            missing_frame_counts = True
        else:
            total_frames += nb_frames

        segment = {
            "index": index,
            "file": str(path),
            "file_name": path.name,
            "nb_frames": nb_frames,
            "duration": info.get("duration"),
            "codec_name": info.get("codec_name"),
            "error": info.get("error"),
        }
        segments.append(segment)
        if index < 5 or index >= max(5, len(paths) - 5):
            sample_segments.append(segment)

    return {
        "segment_count": len(paths),
        "total_nb_frames": None if missing_frame_counts else total_frames,
        "all_segments_have_nb_frames": bool(paths) and not missing_frame_counts,
        "segments": segments,
        "sample_segments": sample_segments,
    }


def _video_segment_frame_map(frame_rows: list[dict[str, str]], run_dir: Path) -> dict[int, dict[str, Any]]:
    """Map frames.csv row indexes to authoritative video files.

    Single-file runs are unambiguous. Chunked runs require video_segments.csv,
    written by splitmuxsink's format-location-full signal. Runtime PTS estimates
    are intentionally not accepted as segment provenance.
    """
    ordered_rows: list[tuple[int, int, dict[str, str]]] = []
    for row_index, row in enumerate(frame_rows):
        stream_id = _to_int(row.get("stream_id"))
        if stream_id not in (None, 0):
            continue
        sort_key = _to_int(row.get("raw_frame_index"))
        if sort_key is None:
            sort_key = _to_int(row.get("camera_frame_id"))
        if sort_key is None:
            continue
        ordered_rows.append((sort_key, row_index, row))
    ordered_rows.sort(key=lambda item: (item[0], item[1]))

    chunked_paths = sorted(run_dir.glob("raw_*.mp4"))
    if not chunked_paths:
        raw_path = run_dir / "raw.mp4"
        if not raw_path.exists():
            return {}
        mapping: dict[int, dict[str, Any]] = {}
        if not ordered_rows:
            return mapping
        first_raw_frame_index = _to_int(ordered_rows[0][2].get("raw_frame_index"))
        if first_raw_frame_index is None:
            first_raw_frame_index = _to_int(ordered_rows[0][2].get("camera_frame_id")) or 0
        for _sort_key, row_index, row in ordered_rows:
            raw_frame_index = _to_int(row.get("raw_frame_index"))
            if raw_frame_index is None:
                raw_frame_index = _to_int(row.get("camera_frame_id"))
            mapping[row_index] = {
                "record_segment_index": 0,
                "record_segment_file": raw_path.name,
                "segment_local_frame_index": (
                    int(raw_frame_index) - int(first_raw_frame_index) if raw_frame_index is not None else ""
                ),
                "segment_start_raw_frame_index": first_raw_frame_index,
                "segment_mapping_source": "single_file",
            }
        return mapping

    ledger_path = run_dir / "video_segments.csv"
    if not ledger_path.exists():
        raise RuntimeError(
            "chunked MP4 files exist without writer-owned video_segments.csv; "
            "segment provenance is not authoritative"
        )
    ledger_rows = _read_csv(ledger_path)
    if not ledger_rows:
        raise RuntimeError("video_segments.csv is empty; segment provenance is not authoritative")

    segments_by_stream: dict[int, list[dict[str, Any]]] = {}
    for row in ledger_rows:
        stream_id = _to_int(row.get("stream_id"))
        if stream_id is None:
            stream_id = 0
        start_raw = _to_int(row.get("first_raw_frame_index"))
        if start_raw is None:
            raise RuntimeError("video_segments.csv has a segment without first_raw_frame_index")
        mapping_source = (row.get("mapping_source") or "").strip()
        if mapping_source != "splitmux_format_location_full_pts":
            raise RuntimeError(
                f"video_segments.csv segment {row.get('segment_index', '')} has non-authoritative "
                f"mapping_source={mapping_source!r}"
            )
        segments_by_stream.setdefault(stream_id, []).append(
            {
                "record_segment_index": _to_int(row.get("segment_index")),
                "record_segment_file": Path(row.get("file") or "").name,
                "segment_start_raw_frame_index": start_raw,
            }
        )

    for segments in segments_by_stream.values():
        segments.sort(key=lambda item: int(item["segment_start_raw_frame_index"]))

    mapping: dict[int, dict[str, Any]] = {}
    cursors: dict[int, int] = {}
    for _sort_key, row_index, row in ordered_rows:
        stream_id = _to_int(row.get("stream_id"))
        if stream_id is None:
            stream_id = 0
        segments = segments_by_stream.get(stream_id)
        if not segments:
            raise RuntimeError(f"video_segments.csv has no segment ledger for stream {stream_id}")
        raw_frame_index = _to_int(row.get("raw_frame_index"))
        if raw_frame_index is None:
            raw_frame_index = _to_int(row.get("camera_frame_id"))
        if raw_frame_index is None:
            continue
        cursor = cursors.get(stream_id, 0)
        while cursor + 1 < len(segments) and int(segments[cursor + 1]["segment_start_raw_frame_index"]) <= int(raw_frame_index):
            cursor += 1
        cursors[stream_id] = cursor
        segment = segments[cursor]
        start = int(segment["segment_start_raw_frame_index"])
        mapping[row_index] = {
            "record_segment_index": segment["record_segment_index"],
            "record_segment_file": segment["record_segment_file"],
            "segment_local_frame_index": int(raw_frame_index) - start,
            "segment_start_raw_frame_index": start,
            "segment_mapping_source": "video_segments_csv",
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


def build_alignment(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    frames_path = run_dir / "frames.csv"
    serial_path = run_dir / "serial.csv"
    detections_path = run_dir / "detections.csv"
    drop_events_path = run_dir / "drop_events.csv"

    if not frames_path.exists():
        raise FileNotFoundError(f"missing required run outputs in {run_dir}: {frames_path.name}")
    frame_rows = _read_csv(frames_path)
    raw_video_segments = _ffprobe_raw_segments(run_dir)
    video_segment_by_row = _video_segment_frame_map(frame_rows, run_dir)
    if not serial_path.exists():
        raise FileNotFoundError(f"missing required run outputs in {run_dir}: {serial_path.name}")

    serial_rows = _normalize_serial_rows(_read_csv(serial_path))
    detection_rows = _read_csv(detections_path) if detections_path.exists() else []
    drop_rows = _read_csv(drop_events_path) if drop_events_path.exists() else []

    camera_high = CameraHighIndex(serial_rows)
    if not camera_high.rows:
        raise RuntimeError("serial.csv contains no CAMERA_HIGH rows with count and rp2040Time")

    detection_count_by_frame: dict[int, int] = {}
    for det in detection_rows:
        frame_id = _to_int(det.get("raw_frame_num"))
        if frame_id is None:
            continue
        detection_count_by_frame[frame_id] = detection_count_by_frame.get(frame_id, 0) + 1

    marker_counts: dict[str, int | None] = {}
    for row in serial_rows:
        if row.get("eventType") != "MARKER":
            continue
        marker_name = (row.get("reason") or row.get("context") or row.get("unixTime") or "").strip()
        if marker_name:
            marker_counts[marker_name] = row.get("_last_high_count_before_or_at_row")

    aligned_frames: list[dict[str, Any]] = []
    frame_by_id: dict[int, dict[str, Any]] = {}
    frame_ids: list[int] = []
    for row_index, row in enumerate(frame_rows):
        camera_frame_id = _to_int(row.get("camera_frame_id"))
        if camera_frame_id is None:
            continue
        video_segment = video_segment_by_row.get(row_index)
        segment_mapping_source = "runtime_pts"
        record_segment_index = row.get("record_segment_index", "")
        record_segment_file = row.get("record_segment_file", "")
        segment_local_frame_index = row.get("segment_local_frame_index", "")
        segment_start_raw_frame_index = row.get("segment_start_raw_frame_index", "")
        if video_segment:
            segment_mapping_source = str(video_segment.get("segment_mapping_source") or "authoritative_video")
            record_segment_index = video_segment.get("record_segment_index", "")
            record_segment_file = video_segment.get("record_segment_file", "")
            segment_local_frame_index = video_segment.get("segment_local_frame_index", "")
            segment_start_raw_frame_index = video_segment.get("segment_start_raw_frame_index", "")
        frame_ids.append(camera_frame_id)
        ttl_count = camera_frame_id + 1
        high = camera_high.by_count.get(ttl_count)
        frame_rp2040_us = _to_int(high.get("rp2040Time")) if high else None
        controller_unix_us = _to_int(high.get("unixTime")) if high else None
        pts_ns = _to_int(row.get("pts_ns"))
        duration_ns = _to_int(row.get("duration_ns"))
        detection_count = detection_count_by_frame.get(camera_frame_id, 0)
        out = {
            "raw_frame_index": row.get("raw_frame_index", ""),
            "camera_frame_id": camera_frame_id,
            "ttl_count": ttl_count,
            "frame_rp2040_us": _fmt(frame_rp2040_us),
            "frame_time_s": _fmt_float(camera_high.frame_time_s(frame_rp2040_us)),
            "controller_unix_us": _fmt(controller_unix_us),
            "frame_pts_ns": _fmt(pts_ns),
            "frame_pts_s": _fmt_float((pts_ns / 1_000_000_000.0) if pts_ns is not None else None),
            "duration_ns": _fmt(duration_ns),
            "frame_host_unix_ns": row.get("host_unix_ns", ""),
            "frame_host_monotonic_ns": row.get("host_monotonic_ns", ""),
            "ttl_host_unix_ns": high.get("hostUnixNs", "") if high else "",
            "ttl_host_monotonic_ns": high.get("hostMonotonicNs", "") if high else "",
            "record_segment_index": record_segment_index,
            "record_segment_file": record_segment_file,
            "segment_local_frame_index": segment_local_frame_index,
            "segment_start_raw_frame_index": segment_start_raw_frame_index,
            "segment_mapping_source": segment_mapping_source,
            "status": row.get("status", ""),
            "has_ttl": "1" if high else "0",
            "has_detection": "1" if detection_count else "0",
            "detection_count": detection_count,
        }
        aligned_frames.append(out)
        frame_by_id[camera_frame_id] = out

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
        previous_frame_id = prev_count - 1 if prev_count is not None else None
        nearest_frame_id = nearest_count - 1 if nearest_count is not None else None

        if event_type == "CAMERA_HIGH" and count is not None:
            alignment_method = "camera_high_exact"
        elif rp2040_us is not None:
            alignment_method = "rp2040_previous_camera_high"
        else:
            alignment_method = "serial_order_previous_camera_high"

        marker_name = ""
        if event_type == "MARKER":
            marker_name = (row.get("reason") or row.get("context") or row.get("unixTime") or "").strip()

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
        camera_frame_id = _to_int(det.get("raw_frame_num"))
        frame = frame_by_id.get(camera_frame_id) if camera_frame_id is not None else None
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
                "record_segment_index": frame.get("record_segment_index", "") if frame else "",
                "record_segment_file": frame.get("record_segment_file", "") if frame else "",
                "segment_local_frame_index": frame.get("segment_local_frame_index", "") if frame else "",
                "segment_start_raw_frame_index": frame.get("segment_start_raw_frame_index", "") if frame else "",
                "segment_mapping_source": frame.get("segment_mapping_source", "") if frame else "",
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
                "record_segment_index": frame["record_segment_index"],
                "record_segment_file": frame["record_segment_file"],
                "segment_local_frame_index": frame["segment_local_frame_index"],
                "segment_start_raw_frame_index": frame["segment_start_raw_frame_index"],
                "segment_mapping_source": frame["segment_mapping_source"],
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
                "record_segment_index": det["record_segment_index"],
                "record_segment_file": det["record_segment_file"],
                "segment_local_frame_index": det["segment_local_frame_index"],
                "segment_start_raw_frame_index": det["segment_start_raw_frame_index"],
                "segment_mapping_source": det["segment_mapping_source"],
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
        "frame_pts_ns",
        "frame_pts_s",
        "duration_ns",
        "frame_host_unix_ns",
        "frame_host_monotonic_ns",
        "ttl_host_unix_ns",
        "ttl_host_monotonic_ns",
        "record_segment_index",
        "record_segment_file",
        "segment_local_frame_index",
        "segment_start_raw_frame_index",
        "segment_mapping_source",
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
        "record_segment_index",
        "record_segment_file",
        "segment_local_frame_index",
        "segment_start_raw_frame_index",
        "segment_mapping_source",
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
        "record_segment_index",
        "record_segment_file",
        "segment_local_frame_index",
        "segment_start_raw_frame_index",
        "segment_mapping_source",
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
    segment_mapping_source_counts: dict[str, int] = {}
    for row in aligned_frames:
        source = str(row.get("segment_mapping_source") or "")
        segment_mapping_source_counts[source] = segment_mapping_source_counts.get(source, 0) + 1
    raw_video = run_dir / "raw_000000.mp4"
    if not raw_video.exists():
        raw_video = run_dir / "raw.mp4"
    video_info = _ffprobe_video(raw_video)
    video_nb_frames = _to_int(raw_video_segments.get("total_nb_frames"))
    if video_nb_frames is None:
        video_nb_frames = _to_int(video_info.get("nb_frames"))

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

        camera_frame_id = _to_int(det.get("raw_frame_num"))
        frame = frame_by_id.get(camera_frame_id) if camera_frame_id is not None else None
        if frame is None:
            detection_missing_frame_count += 1
            if len(detection_missing_frames) < 50:
                detection_missing_frames.append(
                    {
                        "detection_index": det_index,
                        "raw_frame_num": camera_frame_id,
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
                            "raw_frame_num": camera_frame_id,
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
                        "raw_frame_num": camera_frame_id,
                        "mapping_pts_ns": mapping_pts_ns,
                        "frame_pts_ns": frame_pts_ns,
                        "delta_ns": mapping_pts_ns - frame_pts_ns,
                    }
                )

    summary = {
        "run_dir": str(run_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "time_base": "serial.csv CAMERA_HIGH rp2040Time",
        "frame_alignment_rule": "camera_frame_id = CAMERA_HIGH count - 1",
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
            "segment_mapping_source_counts": segment_mapping_source_counts,
            "detection_mapping_failed_rows": detection_mapping_failed_rows,
            "detection_mapping_fallback_rows": detection_mapping_fallback_rows,
            "detection_mapping_legacy_rows": detection_mapping_legacy_rows,
            "detection_missing_frames_sample": detection_missing_frames,
            "detection_ts_mismatches_sample": detection_ts_mismatches,
            "detection_pts_mismatches_sample": detection_pts_mismatches,
        },
        "frame_gaps": frame_gaps[:50],
        "video": video_info,
        "raw_video_segments": raw_video_segments,
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir or _latest_run_dir()
    if run_dir is None:
        raise SystemExit("No run_dir given and runs/.latest_run is missing")
    run_dir = run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "analysis")).resolve()
    summary = build_alignment(run_dir, out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
