"""Timestamped primary tables, with streaming rewrites and disk-backed joins.

Capture timestamps are preserved. Controller frame times between trigger anchors
are estimates, never presented as direct hardware measurements.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import threading

from .run_layout import artifact_path, is_primary_layout
from .bounded_csv import bounded_csv_lines

TIMING_FIELDS = ["host_unix_ns", "host_monotonic_ns", "controller_time_us",
                 "controller_time_method", "controller_boot_id", "controller_session_id"]
WEIGH_FIELDS = ["record_type", "record_id", "measurement_type", "mouse_id", "bottle_id", "side", "fluid", "phase",
                "weight_g", "entered_at_iso", "entered_at_unix_ns", "note",
                "event", "swap_id", "state", "selected_sides", "initial_weight_g", "final_weight_g",
                "old_weight_g", "start_count", "end_count", "started_monotonic_us", "ended_monotonic_us",
                "lick_count", "intake_g", "paused_monotonic_us", "resumed_monotonic_us",
                "pause_left_count", "pause_right_count", "resume_left_count", "resume_right_count",
                "left_count", "right_count", "error", "details_json"] + TIMING_FIELDS
_WEIGH_LOCK = threading.RLock()


def _json(path):
    if not path.exists():
        return {}
    from .run_context import read_json_required
    return read_json_required(path, max_bytes=64 * 1024 * 1024 if path.name == "bottle_history.json" else 16 * 1024 * 1024)


def _integer(value):
    return int(value) if value is not None and str(value).strip().isdigit() else None


@contextmanager
def _replacement(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _rewrite(path, transform, extra=TIMING_FIELDS):
    if not path.exists():
        return
    with _replacement(path) as target:
        reader = csv.DictReader(bounded_csv_lines(path))
        fields = list(dict.fromkeys(list(reader.fieldnames or []) + list(extra)))
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            writer.writerow(transform(row))


class ClockMapping:
    def __init__(self, run_dir):
        evidence = _json(artifact_path(run_dir, "diagnostics/clock_validation.json"))
        self.sample = None
        if evidence.get("result") == "PASS":
            samples = evidence.get("after_samples") or evidence.get("before_samples") or []
            if samples:
                self.sample = min(samples, key=lambda s: s["round_trip_ns"])
        summary = _json(artifact_path(run_dir, "diagnostics/controller_v2_summary.json"))
        self.boot = summary.get("boot_id", "")
        self.session = summary.get("session_id", "")
        self.valid_boot = not summary.get("integrity_latched") and not summary.get("counts", {}).get("boot_boundaries", 0)

    def apply(self, row):
        row["host_unix_ns"] = row.get("host_unix_ns") or row.get("hostUnixNs") or row.get("entered_at_unix_ns") or ""
        row["host_monotonic_ns"] = row.get("host_monotonic_ns") or row.get("hostMonotonicNs") or ""
        if _integer(row.get("controller_time_us")) is not None:
            return row
        host = _integer(row["host_unix_ns"])
        row.update(controller_time_us="", controller_time_method="unavailable",
                   controller_boot_id="", controller_session_id="")
        if self.sample is not None and host is not None and self.valid_boot:
            # TIME_SYNC provides an idle round-trip midpoint anchor, not an
            # exact event timestamp. Wall-clock extrapolation stays labelled.
            sample = self.sample
            controller = (int(sample["rp2040_receive_us"]) + int(sample["rp2040_send_us"])) // 2
            midpoint = (int(sample["jetson_send_ns"]) + int(sample["jetson_receive_ns"])) // 2
            estimate = controller + (host - midpoint) // 1000
            if estimate >= 0:
                row.update(controller_time_us=estimate, controller_time_method="host_clock_estimate",
                           controller_boot_id=self.boot, controller_session_id=self.session)
        return row


def append_weigh_records(run_dir, rows, *, deduplicate=False):
    """Commit one CSV transaction without losing existing measurements or history."""
    path = artifact_path(run_dir, "weigh_in.csv")
    with _WEIGH_LOCK:
        latest = {}
        def key(row):
            kind = row.get("record_type")
            if kind == "bottle_state":
                identity = row.get("bottle_id") or ""
            elif kind == "swap_gap":
                identity = row.get("swap_id") or ""
            elif kind == "swap_event":
                identity = row.get("record_id") or ""
            else:
                identity = ""
            return kind, identity
        if deduplicate and path.exists():
            for row in csv.DictReader(bounded_csv_lines(path)):
                if row.get("record_type") in ("bottle_state", "swap_gap", "swap_state", "swap_event"):
                    latest[key(row)] = row.get("details_json")
        additions = [row for row in rows if not deduplicate or latest.get(key(row)) != row.get("details_json")]
        if not additions:
            return
        with _replacement(path) as target:
            writer = csv.DictWriter(target, fieldnames=WEIGH_FIELDS)
            writer.writeheader()
            if path.exists():
                for row in csv.DictReader(bounded_csv_lines(path)):
                    writer.writerow(row)
            writer.writerows(additions)


def write_weigh_in(run_dir, bottles):
    from .measurements import validate_bottle_inputs
    validate_bottle_inputs(bottles)
    with _WEIGH_LOCK:
        _write_weigh_in(run_dir, bottles)


def _write_weigh_in(run_dir, bottles):
    """Append new/corrected measurements; unchanged values retain entry time."""
    path = artifact_path(run_dir, "weigh_in.csv")
    from .bottle_csv import load_swap_state
    history = load_swap_state(run_dir) or _json(artifact_path(run_dir, "bottle_history.json"))
    manifest = _json(artifact_path(run_dir, "run_manifest.json"))
    entries = history.get("history", []) + list(history.get("current", {}).values())
    if not entries:
        entries = [dict(values, side=side, bottle_id=f"initial_{side}")
                   for side, values in (bottles or {}).items() if side in ("left", "right")]
    desired = []
    for entry in entries:
        for phase in ("initial", "final"):
            weight = entry.get(f"{phase}_weight_g")
            if f"{phase}_weight_g" in entry:
                desired.append(dict(measurement_type="bottle", bottle_id=entry["bottle_id"],
                    side=entry["side"], fluid=entry.get("fluid", ""), phase=phase,
                    weight_g=weight, note=entry.get("note", "")))
    mouse_weight = (bottles or {}).get("mouse_weight_g")
    if "mouse_weight_g" in (bottles or {}):
        desired.append(dict(measurement_type="mouse", bottle_id="", side="", fluid="",
                            phase="weigh_in", weight_g=mouse_weight, note=""))
    latest = {}
    def key(row):
        return row["measurement_type"], row["bottle_id"], row["phase"]
    if path.exists():
        for row in csv.DictReader(bounded_csv_lines(path)):
            if row.get("record_type") in (None, "", "measurement"):
                latest[key(row)] = row
    mapping = ClockMapping(run_dir)
    now, monotonic = time.time_ns(), time.monotonic_ns()
    additions = []
    for row in desired:
        previous = latest.get(key(row))
        previous_weight = float(previous["weight_g"]) if previous and previous.get("weight_g") not in (None, "") else None
        new_weight = float(row["weight_g"]) if row["weight_g"] not in (None, "") else None
        # Blank weights are explicit clears only after a value existed. They
        # remain ordinary timestamped measurement rows for older CSV readers.
        if new_weight is None:
            if previous_weight is None:
                continue
            row["event"] = "measurement_cleared"
        if previous and all(str(previous.get(name, "")) == str(row.get(name, "")) for name in ("fluid", "note")) and previous_weight == new_weight:
            continue
        row.update(record_type="measurement", mouse_id=manifest.get("mouse_id") or _json(artifact_path(run_dir, "run_status.json")).get("mouse_id") or "", entered_at_unix_ns=now,
                   entered_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now / 1e9)),
                   host_unix_ns=now, host_monotonic_ns=monotonic)
        additions.append(mapping.apply(row))
    with _replacement(path) as target:
        writer = csv.DictWriter(target, fieldnames=WEIGH_FIELDS)
        writer.writeheader()
        if path.exists():
            for row in csv.DictReader(bounded_csv_lines(path)):
                writer.writerow(row)
        writer.writerows(additions)


def finalize_primary_tables(run_dir, camera_count=1):
    """Publish controller times on every primary row after capture closes.

    Native-only runs retain their original clocks and explicit unavailable
    controller mappings. The index is disposable and memory use stays bounded.
    """
    run_dir = Path(run_dir)
    if not is_primary_layout(run_dir):
        return
    mapping = ClockMapping(run_dir)
    diagnostics = artifact_path(run_dir, "diagnostics")
    diagnostics.mkdir(parents=True, exist_ok=True)
    from squeakview.common.serial import SERIAL_HEADER
    from squeakview.apps.inference.pose_persistence import OBJECT_HEADERS
    for name, header in (("sensors.csv", SERIAL_HEADER), ("detections.csv", OBJECT_HEADERS + TIMING_FIELDS), ("weigh_in.csv", WEIGH_FIELDS)):
        path = artifact_path(run_dir, name)
        if not path.exists():
            with _replacement(path) as handle:
                csv.writer(handle).writerow(list(dict.fromkeys(header)))
    with tempfile.TemporaryDirectory(prefix=".timestamps.", dir=artifact_path(run_dir, "diagnostics")) as directory:
        with sqlite3.connect(str(Path(directory) / "frames.sqlite3")) as index:
            index.execute("PRAGMA cache_size=-1024")
            index.execute("CREATE TABLE anchors (count INTEGER PRIMARY KEY, us INTEGER, boot TEXT, session TEXT, kind TEXT)")
            index.execute("CREATE TABLE frames (stream INTEGER, sequence INTEGER, timing TEXT, PRIMARY KEY(stream, sequence))")
            index.execute("CREATE TABLE boundaries (kind TEXT PRIMARY KEY, count INTEGER, boot TEXT, session TEXT)")
            journal = artifact_path(run_dir, "diagnostics/controller_v2.jsonl")
            if journal.exists() and mapping.valid_boot:
                for line in bounded_csv_lines(journal):
                    record = json.loads(line)
                    text = record["payload_utf8"]
                    kind = text.split(",", 1)[0]
                    if kind in ("CAMERA_EPOCH", "CAMERA_CHECKPOINT", "CAMERA_STOP"):
                        fields = dict(item.split("=", 1) for item in text.split(",")[1:])
                        values = (int(fields["count"]), int(fields["timestamp_us"]), str(record["boot_id"]), str(record["session_id"]), kind)
                        previous = index.execute("SELECT us,boot,session FROM anchors WHERE count=?", values[:1]).fetchone()
                        if previous is not None and previous != values[1:4]:
                            raise ValueError("conflicting controller camera timing anchors")
                        index.execute("INSERT OR REPLACE INTO anchors VALUES (?,?,?,?,?)", values)
                        if kind in ("CAMERA_EPOCH", "CAMERA_STOP"):
                            boundary = (kind, values[0], values[2], values[3])
                            old = index.execute("SELECT kind,count,boot,session FROM boundaries WHERE kind=?", (kind,)).fetchone()
                            if old is not None and old != boundary:
                                raise ValueError("multiple controller camera epochs/stops")
                            index.execute("INSERT OR REPLACE INTO boundaries VALUES (?,?,?,?)", boundary)
            first = index.execute("SELECT count,boot,session FROM boundaries WHERE kind='CAMERA_EPOCH'").fetchall()
            last = index.execute("SELECT count FROM boundaries WHERE kind='CAMERA_STOP'").fetchone()
            previous = None
            for anchor in index.execute("SELECT count,us,boot,session FROM anchors ORDER BY count"):
                if not first or last is None:
                    raise ValueError("controller camera timing is missing epoch/stop anchors")
                if anchor[2:] != first[0][1:] or (previous is not None and anchor[1] <= previous[1]):
                    raise ValueError("controller camera timing anchors cross clocks or are not monotonic")
                previous = anchor
            frame_count = 0
            for stream in range(camera_count):
                path = artifact_path(run_dir, "frames.csv" if stream == 0 else f"frames_cam{stream}.csv")
                if not path.is_file():
                    raise ValueError(f"recorded frame ledger is missing: {path.name}")
                def frame_time(row):
                    nonlocal frame_count
                    row = mapping.apply(row)
                    if _integer(row.get("camera_timestamp_ns")) is None or _integer(row.get("host_unix_ns")) is None:
                        raise ValueError("recorded frame is missing its native timestamp")
                    sequence = int(row["source_sequence_index"])
                    raw = int(row["raw_frame_index"])
                    if len(first) == 1 and last is not None:
                        count = first[0][0] + raw
                        left = index.execute("SELECT count,us,boot,session FROM anchors WHERE count<=? ORDER BY count DESC LIMIT 1", (count,)).fetchone()
                        right = index.execute("SELECT count,us,boot,session FROM anchors WHERE count>=? ORDER BY count LIMIT 1", (count,)).fetchone()
                        if left and right and left[2:] == right[2:] and left[2:] == first[0][1:] and right[1] >= left[1]:
                            value = left[1] if left[0] == right[0] else left[1] + ((count-left[0])*(right[1]-left[1]) + (right[0]-left[0])//2)//(right[0]-left[0])
                            row.update(controller_time_us=value, controller_time_method="exact_anchor" if left[0]==right[0] else "piecewise_interpolated", controller_boot_id=left[2], controller_session_id=left[3])
                    timing = {name: row.get(name, "") for name in TIMING_FIELDS + ["camera_timestamp_ns", "gst_pts_ns"]}
                    index.execute("INSERT INTO frames VALUES (?,?,?)", (stream, sequence, json.dumps(timing)))
                    if stream == 0:
                        frame_count += 1
                    return row
                _rewrite(path, frame_time)
            if len(first) == 1 and last is not None and last[0] - first[0][0] + 1 != frame_count:
                raise ValueError("controller/frame timing counts disagree")
            def detection_time(row):
                match = index.execute("SELECT timing FROM frames WHERE stream=? AND sequence=?", (int(row["stream_id"]), int(row["source_sequence_index"]))).fetchone()
                if match is None:
                    raise ValueError("detection has no recorded frame timestamp")
                timing = json.loads(match[0])
                for name in ("camera_timestamp_ns", "gst_pts_ns"):
                    original = _integer(row.get(name))
                    if original is not None and original != _integer(timing.get(name)):
                        raise ValueError(f"detection/frame {name} mismatch")
                row.update(timing)
                return row
            _rewrite(artifact_path(run_dir, "detections.csv"), detection_time,
                     TIMING_FIELDS + ["camera_timestamp_ns", "gst_pts_ns"])
    def entry_time(row):
        row = mapping.apply(row)
        if _integer(row.get("host_unix_ns")) is None:
            raise ValueError("primary entry is missing its native host timestamp")
        return row
    _rewrite(artifact_path(run_dir, "sensors.csv"), entry_time)
    _rewrite(artifact_path(run_dir, "weigh_in.csv"), entry_time)
