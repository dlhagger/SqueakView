"""Read-only, bounded v2 session and sparse-camera timing verification."""
from __future__ import annotations

from bisect import bisect_left
import csv
from dataclasses import dataclass
import json
from pathlib import Path

from . import controller_protocol_v2 as protocol
from .bounded_csv import bounded_csv_lines
from .run_layout import artifact_path, is_primary_layout

MAX_ANCHORS = 65_536


@dataclass(frozen=True)
class V2Timing:
    boot_id: int
    session_id: int
    start_sequence: int
    stop_sequence: int
    start_us: int
    stop_us: int
    anchors: tuple[dict, ...]
    records: int


def _uint(value, label):
    if isinstance(value, bool) or str(value).strip() != str(value) or not str(value).isascii() or not str(value).isdecimal():
        raise ValueError(f"invalid {label}")
    return int(value)


def journal_records(path: Path):
    """Verify both encoded CRC bytes and the corresponding JSON projection."""
    previous = boot = None
    for line in bounded_csv_lines(path):
        row = json.loads(line)
        frame = protocol.decode_frame(protocol.cobs_encode(bytes.fromhex(row["decoded_hex"]))[:-1])
        for name in ("boot_id", "session_id", "sequence", "monotonic_us", "message_type", "flags"):
            if _uint(row[name], name) != getattr(frame, name):
                raise ValueError(f"journal {name} disagrees with encoded frame")
        if row["payload_utf8"] != frame.text:
            raise ValueError("journal payload disagrees with encoded frame")
        if not frame.reliable or frame.sequence == 0:
            raise ValueError("journal record is not a reliable positive sequence")
        if previous is not None and frame.sequence != previous + 1:
            raise ValueError("protocol-v2 journal contains a durable sequence gap")
        if boot is not None and frame.boot_id != boot:
            raise ValueError("protocol-v2 journal crosses a controller boot boundary")
        if frame.integrity_latched or frame.message_type == protocol.MessageType.INTEGRITY_FAULT:
            raise ValueError("controller integrity fault in journal")
        previous, boot = frame.sequence, frame.boot_id
        yield row


def read_v2_timing(run_dir: Path, *, triggered: bool) -> V2Timing:
    from . import run_context
    summary = run_context.read_json_required(artifact_path(run_dir, "diagnostics/controller_v2_summary.json"))
    counts = summary.get("counts", {})
    if (summary.get("protocol") != "mousehouse_v2" or summary.get("integrity_latched") or
        any(counts.get(key, 0) for key in ("boot_boundaries", "conflicting_duplicates", "crc_or_framing_errors"))):
        raise ValueError("controller transport summary failed integrity checks")
    anchors = []
    starts, stops = [], []
    sessions = set()
    first = boot = None
    records = 0
    journal_path = artifact_path(run_dir, "diagnostics/controller_v2.jsonl")
    before = journal_path.stat()
    for row in journal_records(journal_path):
        records += 1
        first = row["sequence"] if first is None else first
        boot = row["boot_id"]
        if row["session_id"]:
            sessions.add(row["session_id"])
            if len(sessions) > 1:
                raise ValueError("journal crosses controller session boundaries")
        payload = row["payload_utf8"]
        kind = payload.split(",", 1)[0]
        if kind in {"CAMERA_EPOCH", "CAMERA_CHECKPOINT", "CAMERA_STOP"}:
            expected_type = getattr(protocol.MessageType, kind)
            if row["message_type"] != expected_type:
                raise ValueError("camera payload has the wrong message type")
            parsed = protocol.parse_camera_payload(payload)
            anchor = {name: _uint(parsed[name], name) for name in ("count", "timestamp_us", "period_us", "pulse_us", "suppressed")}
            anchor.update(record=kind, sequence=row["sequence"], boot_id=boot, session_id=row["session_id"],
                          reason=parsed["reason"], health=int(parsed["health"], 0), queue=parsed["queue"],
                          host_unix_ns=row["host_unix_ns"], host_monotonic_ns=row["host_monotonic_ns"])
            if len(anchors) >= MAX_ANCHORS:
                raise ValueError("v2 anchor count exceeds supported bound")
            anchors.append(anchor)
        if not triggered and kind in {"ACK_START_BEHAVIOR", "ACK_STOP"}:
            (starts if kind == "ACK_START_BEHAVIOR" else stops).append(row)
            if len(starts) > 1 or len(stops) > 1:
                raise ValueError("multiple behavior session boundaries")
    after = journal_path.stat()
    if any(getattr(before, key) != getattr(after, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("controller journal changed during timing verification")
    if (not records or len(sessions) != 1 or summary.get("boot_id") != boot or
        summary.get("session_id") != next(iter(sessions)) or counts.get("frames_stored") != records or
        summary.get("journal_start_sequence", first) != first):
        raise ValueError("controller summary/journal identities or counts disagree")
    session = next(iter(sessions))
    if triggered:
        epochs = [row for row in anchors if row["record"] == "CAMERA_EPOCH"]
        ends = [row for row in anchors if row["record"] == "CAMERA_STOP"]
        if len(epochs) != 1 or len(ends) != 1 or anchors[0] != epochs[0] or anchors[-1] != ends[0]:
            raise ValueError("expected one ordered CAMERA_EPOCH and CAMERA_STOP")
        previous = None
        period = anchors[0]["period_us"]
        for anchor in anchors:
            if (anchor["session_id"] != session or not anchor["count"] or not period or
                anchor["period_us"] != period or not 0 < anchor["pulse_us"] <= period or anchor["suppressed"]):
                raise ValueError("inconsistent or suppressed controller camera timing")
            if previous and (anchor["count"] < previous["count"] or
                (anchor["count"] == previous["count"] and anchor["timestamp_us"] != previous["timestamp_us"]) or
                (anchor["count"] > previous["count"] and anchor["timestamp_us"] <= previous["timestamp_us"])):
                raise ValueError("conflicting or non-monotonic controller anchors")
            previous = anchor
        start, end = anchors[0], anchors[-1]
        return V2Timing(boot, session, start["sequence"], end["sequence"], start["timestamp_us"], end["timestamp_us"], tuple(anchors), records)
    if anchors or len(starts) != 1 or len(stops) != 1:
        raise ValueError("expected behavior-only start/stop boundaries without camera anchors")
    start, end = starts[0], stops[0]
    if start["session_id"] != session or end["session_id"] != session or start["sequence"] >= end["sequence"] or start["monotonic_us"] >= end["monotonic_us"]:
        raise ValueError("invalid behavior session boundaries")
    return V2Timing(boot, session, start["sequence"], end["sequence"], start["monotonic_us"], end["monotonic_us"], (), records)


def camera_time(anchors, count, *, counts=None):
    """Use the same integer nearest-microsecond rule as primary finalization."""
    if counts is None:
        counts = [row["count"] for row in anchors]
    i = bisect_left(counts, count)
    if i < len(counts) and counts[i] == count:
        return anchors[i]["timestamp_us"], "exact_anchor"
    if i == 0 or i == len(counts):
        raise ValueError("frame outside controller epoch")
    left, right = anchors[i - 1], anchors[i]
    span = right["count"] - left["count"]
    return left["timestamp_us"] + ((count-left["count"])*(right["timestamp_us"]-left["timestamp_us"])+span//2)//span, "piecewise_interpolated"


def verify_v2_frame_timing(run_dir: Path, *, triggered: bool, camera_count: int) -> dict:
    """Verify source sequence, native clocks, and any exported timing mappings."""
    from .run_outputs import ClockMapping
    if type(camera_count) is not int or not 1 <= camera_count <= 64:
        raise ValueError("invalid camera count")
    timing = read_v2_timing(run_dir, triggered=triggered)
    mapping = ClockMapping(run_dir)
    expected_count = timing.anchors[-1]["count"] - timing.anchors[0]["count"] + 1 if triggered else None
    cameras = []
    # Collapse repeated equal anchors (e.g. epoch/stop for a one-frame run).
    anchors = list({anchor["count"]: anchor for anchor in timing.anchors}.values())
    counts = [anchor["count"] for anchor in anchors]
    for stream in range(camera_count):
        path = artifact_path(run_dir, "frames.csv" if stream == 0 else f"frames_cam{stream}.csv")
        count = 0
        previous_camera = None
        for row in csv.DictReader(bounded_csv_lines(path)):
            if (_uint(row["stream_id"], "stream_id") != stream or
                _uint(row["source_sequence_index"], "source sequence") != count or
                _uint(row["raw_frame_index"], "raw frame index") != count):
                raise ValueError("frame identity is not contiguous")
            native = _uint(row["camera_timestamp_ns"], "camera timestamp")
            if previous_camera is not None and native <= previous_camera:
                raise ValueError("camera clock is not strictly increasing")
            previous_camera = native
            if is_primary_layout(run_dir) or "controller_time_method" in row:
                if triggered:
                    value, method = camera_time(anchors, anchors[0]["count"] + count, counts=counts)
                else:
                    expected = mapping.apply({"host_unix_ns": row["host_unix_ns"]})
                    value, method = expected["controller_time_us"], expected["controller_time_method"]
                if str(row.get("controller_time_us", "")) != str(value) or row.get("controller_time_method") != method:
                    raise ValueError("frame/controller timing mapping disagrees with evidence")
                if method != "unavailable" and (str(row.get("controller_boot_id")) != str(timing.boot_id) or str(row.get("controller_session_id")) != str(timing.session_id)):
                    raise ValueError("frame mapping crosses controller clocks")
            count += 1
        if not count or (expected_count is not None and count != expected_count):
            raise ValueError("controller/frame count mismatch")
        cameras.append({"stream_id": stream, "frames": count})
    return {"schema_version": "1.0", "passed": True, "method": "v2_sparse_anchors" if triggered else "v2_behavior_session",
            "boot_id": timing.boot_id, "session_id": timing.session_id, "cameras": cameras}
