"""Disk-backed, bounded-memory alignment for long SqueakView runs."""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

from squeakview.apps.inference.video_probe import probe_video_frames
from squeakview.common.bounded_csv import BoundedCsvError, MAX_CSV_RECORD_BYTES
from squeakview.common.run_context import atomic_write_json


MAX_SQLITE_INTEGER = (1 << 63) - 1
_MARKERS = frozenset(
    {"START_SENT", "CAPTURE_STOP_REQUESTED", "CAPTURE_STOP_DONE", "STOP_SENT"}
)
_REQUIRED_EPOCH_MARKERS = (
    "START_SENT", "CAPTURE_STOP_REQUESTED", "STOP_SENT", "CAPTURE_STOP_DONE"
)


def _to_int(value: Any) -> int | None:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "n/a"}:
        return None
    if not text.isascii() or not text.isdecimal():
        raise ValueError(f"expected an unsigned decimal integer, got {text!r}")
    number = int(text)
    if number > MAX_SQLITE_INTEGER:
        raise ValueError("integer exceeds the signed 64-bit scientific ledger bound")
    return number


def _required_uint(row: dict[str, str], field: str, *, source: str) -> int:
    value = _to_int(row.get(field))
    if value is None:
        raise ValueError(f"{source} row is missing required integer field {field}")
    return value


def _fmt(value: Any) -> str:
    return "" if value is None else str(value)


def _float(value: float | None, digits: int = 9) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _marker(row: dict[str, str]) -> str:
    return (row.get("reason") or row.get("context") or row.get("unixTime") or "").strip()


def _stable_physical_lines(path: Path) -> Iterator[str]:
    path = Path(path).absolute()
    try:
        initial = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"scientific alignment input could not be read: {path}: {exc}") from exc
    if stat.S_ISLNK(initial.st_mode):
        raise ValueError(f"scientific alignment input must not be a symlink: {path}")
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError(f"scientific alignment input is not a regular file: {path}")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"scientific alignment input is not a regular file: {path}")
            line_number = 0
            while True:
                raw = handle.readline(MAX_CSV_RECORD_BYTES + 1)
                if not raw:
                    break
                line_number += 1
                if len(raw) > MAX_CSV_RECORD_BYTES:
                    raise BoundedCsvError(
                        f"CSV record exceeds {MAX_CSV_RECORD_BYTES} byte limit at "
                        f"{path.name}:{line_number}"
                    )
                try:
                    yield raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raise BoundedCsvError(
                        f"CSV input is not strict UTF-8 at {path.name}:{line_number}"
                    ) from exc
            after = os.fstat(handle.fileno())
        current = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"scientific alignment input could not be read: {path}: {exc}") from exc
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or any(
            getattr(before, field) != getattr(after, field)
            or getattr(before, field) != getattr(current, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        )
    ):
        raise ValueError(f"scientific alignment input changed while reading: {path}")


def _parse_physical_record(line: str, *, path: Path, line_number: int) -> list[str]:
    try:
        rows = list(csv.reader([line], strict=True))
    except csv.Error as exc:
        raise ValueError(
            f"CSV record is malformed or spans physical lines at {path.name}:{line_number}: {exc}"
        ) from exc
    if len(rows) != 1:
        raise ValueError(f"CSV physical record is invalid at {path.name}:{line_number}")
    return rows[0]


def _reader(
    path: Path, *, required_columns: frozenset[str] = frozenset()
) -> Iterator[dict[str, str]]:
    lines = _stable_physical_lines(path)
    try:
        header_line = next(lines)
    except StopIteration as exc:
        raise ValueError(f"CSV input is empty: {path}") from exc
    header = _parse_physical_record(header_line, path=path, line_number=1)
    if not header or any(not name for name in header) or len(header) != len(set(header)):
        raise ValueError(f"CSV header is empty, duplicated, or invalid: {path}")
    missing = required_columns.difference(header)
    if missing:
        raise ValueError(
            f"CSV input {path.name} is missing required columns: "
            + ", ".join(sorted(missing))
        )
    for line_number, line in enumerate(lines, start=2):
        values = _parse_physical_record(line, path=path, line_number=line_number)
        if len(values) != len(header):
            raise ValueError(
                f"CSV row has {len(values)} fields; expected {len(header)} at "
                f"{path.name}:{line_number}"
            )
        yield dict(zip(header, values, strict=True))


def _video_validation(
    path: Path, supplied: Mapping[str, object] | None
) -> dict[str, object]:
    """Get one authoritative decoded-frame count for alignment.

    The post-run coordinator supplies the result it just produced, avoiding a
    second full read of long recordings. Standalone alignment performs the
    same explicit Jetson full decode itself.
    """

    result = dict(supplied) if supplied is not None else probe_video_frames(path)
    count = result.get("count")
    if type(count) is not int or count < 0:
        return {
            "count": None,
            "method": result.get("method"),
            "error": result.get("error")
            or "validated video frame count is unavailable",
        }
    return {
        "count": count,
        "method": result.get("method"),
        "error": result.get("error"),
    }


def _open_index(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute(
        "CREATE TABLE highs(count INTEGER PRIMARY KEY, rp INTEGER, unix_us INTEGER, "
        "host_unix TEXT, host_mono TEXT, serial_index INTEGER)"
    )
    db.execute("CREATE INDEX highs_rp ON highs(rp)")
    db.execute(
        "CREATE TABLE detection_counts(raw_index INTEGER PRIMARY KEY, count INTEGER)"
    )
    db.execute(
        "CREATE TABLE frame_lookup(raw_index INTEGER PRIMARY KEY, camera_id INTEGER UNIQUE, "
        "ttl INTEGER, rp TEXT, time_s TEXT, pts TEXT, pts_s TEXT, raw_video TEXT, "
        "video_index TEXT, video_source TEXT)"
    )
    return db


def _input_snapshot(paths: dict[str, Path], *, required: frozenset[str]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name, unresolved in paths.items():
        path = Path(unresolved).absolute()
        try:
            metadata = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            metadata = None
        except OSError as exc:
            raise ValueError(f"scientific alignment input could not be inspected: {path}: {exc}") from exc
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"scientific alignment input must not be a symlink: {path}")
        if metadata is not None and not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"scientific alignment input is not a regular file: {path}")
        available = metadata is not None
        if name in required and not available:
            raise ValueError(
                f"required scientific alignment input is unavailable: {path}"
            )
        snapshot[name] = {
            "path": str(path),
            "available": available,
            "device": int(metadata.st_dev) if metadata is not None else None,
            "inode": int(metadata.st_ino) if metadata is not None else None,
            "size_bytes": int(metadata.st_size) if metadata is not None else None,
            "mtime_ns": int(metadata.st_mtime_ns) if metadata is not None else None,
            "ctime_ns": int(metadata.st_ctime_ns) if metadata is not None else None,
        }
    return snapshot


def _index_serial(db: sqlite3.Connection, serial_path: Path) -> dict[str, Any]:
    serial_count = 0
    camera_high_count = 0
    last_high_count: int | None = None
    last_high_rp: int | None = None
    last_epoch_high_rp: int | None = None
    first_high: tuple[int, int, int] | None = None
    first_after_start: tuple[int, int, int] | None = None
    start_index: int | None = None
    markers: dict[str, int | None] = {}
    marker_indices: dict[str, int] = {}
    interval_sample: list[int] = []
    batch: list[tuple[int, int, int | None, str, str, int]] = []
    for serial_index, row in enumerate(
        _reader(
            serial_path,
            required_columns=frozenset({"eventType", "count", "rp2040Time"}),
        )
    ):
        serial_count += 1
        event = (row.get("eventType") or "").strip()
        if event == "MARKER":
            name = _marker(row)
            if name in _MARKERS:
                if name in markers:
                    raise RuntimeError(f"serial.csv contains duplicate marker {name}")
                markers[name] = last_high_count
                marker_indices[name] = serial_index
            if name == "START_SENT" and start_index is None:
                start_index = serial_index
        if event != "CAMERA_HIGH":
            continue
        count = _required_uint(row, "count", source="serial.csv CAMERA_HIGH")
        rp = _required_uint(row, "rp2040Time", source="serial.csv CAMERA_HIGH")
        camera_high_count += 1
        current = (count, rp, serial_index)
        if first_high is None:
            first_high = current
        if start_index is not None and serial_index > start_index and first_after_start is None:
            first_after_start = current
        if (
            start_index is not None
            and serial_index > start_index
            and last_epoch_high_rp is not None
            and rp > last_epoch_high_rp
            and len(interval_sample) < 10001
        ):
            interval_sample.append(rp - last_epoch_high_rp)
        if last_high_count is not None and count <= last_high_count:
            raise RuntimeError("serial CAMERA_HIGH count is duplicate or out of order")
        if last_high_rp is not None and rp <= last_high_rp:
            raise RuntimeError("serial CAMERA_HIGH clock is duplicate or out of order")
        last_high_count, last_high_rp = count, rp
        if start_index is not None and serial_index > start_index:
            last_epoch_high_rp = rp
        batch.append(
            (count, rp, _to_int(row.get("unixTime")), row.get("hostUnixNs", ""),
             row.get("hostMonotonicNs", ""), serial_index)
        )
        if len(batch) >= 10000:
            with db:
                db.executemany("INSERT INTO highs VALUES (?, ?, ?, ?, ?, ?)", batch)
            batch.clear()
    if batch:
        with db:
            db.executemany("INSERT INTO highs VALUES (?, ?, ?, ?, ?, ?)", batch)
    missing_markers = [
        name for name in _REQUIRED_EPOCH_MARKERS if name not in marker_indices
    ]
    if missing_markers:
        raise RuntimeError(
            "serial.csv cannot define the scientific trigger epoch; missing marker(s): "
            + ", ".join(missing_markers)
        )
    ordered_indices = [marker_indices[name] for name in _REQUIRED_EPOCH_MARKERS]
    if ordered_indices != sorted(ordered_indices) or len(set(ordered_indices)) != len(
        ordered_indices
    ):
        raise RuntimeError(
            "serial.csv scientific trigger epoch markers are out of order"
        )
    chosen = first_after_start
    if chosen is None:
        raise RuntimeError(
            "serial.csv contains no CAMERA_HIGH rows inside the START_SENT epoch"
        )
    interval_sample.sort()
    median_interval = interval_sample[len(interval_sample) // 2] if interval_sample else None
    return {
        "serial_rows": serial_count, "camera_high_events": camera_high_count,
        "last_high_count": last_high_count, "first_high": chosen,
        "start_marker_seen": True, "markers": markers,
        "marker_indices": marker_indices,
        "median_interval_us": median_interval,
    }


def _index_detection_counts(db: sqlite3.Connection, path: Path) -> None:
    if not path.exists():
        return
    previous: int | None = None
    count = 0
    batch: list[tuple[int, int]] = []
    for row in _reader(
        path, required_columns=frozenset({"source_sequence_index"})
    ):
        raw = _to_int(row.get("source_sequence_index"))
        if raw is None:
            continue
        if previous is not None and raw < previous:
            raise RuntimeError("objects.csv is not ordered by source_sequence_index")
        if previous is None:
            previous, count = raw, 1
        elif raw == previous:
            count += 1
        else:
            batch.append((previous, count))
            previous, count = raw, 1
        if len(batch) >= 10000:
            with db:
                db.executemany("INSERT INTO detection_counts VALUES (?, ?)", batch)
            batch.clear()
    if previous is not None:
        batch.append((previous, count))
    if batch:
        with db:
            db.executemany("INSERT INTO detection_counts VALUES (?, ?)", batch)


def _first_frame(path: Path) -> dict[str, str]:
    for row in _reader(
        path,
        required_columns=frozenset(
            {"camera_frame_id", "raw_frame_index", "camera_timestamp_ns", "pts_ns"}
        ),
    ):
        _required_uint(row, "camera_frame_id", source="frames.csv")
        _required_uint(row, "raw_frame_index", source="frames.csv")
        _required_uint(row, "camera_timestamp_ns", source="frames.csv")
        _required_uint(row, "pts_ns", source="frames.csv")
        return row
    raise RuntimeError("frames.csv contains no camera_frame_id values")


def _write_frames(
    db: sqlite3.Connection, frames_path: Path, epoch: dict[str, Any],
) -> dict[str, Any]:
    count = missing_ttl = mismatch_count = clock_pairs = gap_count = 0
    missing_camera_frames = 0
    mismatch_sample: list[dict[str, int]] = []
    gap_sample: list[dict[str, int]] = []
    previous_camera_id: int | None = None
    previous_raw_index: int | None = None
    clock_max: float | None = None
    clock_final: float | None = None
    sum_x = sum_y = sum_xx = sum_xy = 0.0
    first_camera = epoch["first_camera_frame_id"]
    first_camera_ts = epoch["first_camera_timestamp_ns"]
    first_rp = epoch["first_rp2040_time_us"]
    insert_batch: list[tuple] = []
    try:
        for row in _reader(
            frames_path,
            required_columns=frozenset(
                {"camera_frame_id", "raw_frame_index", "camera_timestamp_ns", "pts_ns"}
            ),
        ):
            camera_id = _required_uint(row, "camera_frame_id", source="frames.csv")
            raw_index = _required_uint(row, "raw_frame_index", source="frames.csv")
            ttl = camera_id - int(epoch["camera_frame_id_offset"])
            high = db.execute(
                "SELECT rp, unix_us, host_unix, host_mono FROM highs "
                "WHERE count=? AND serial_index>?",
                (ttl, int(epoch["start_marker_index"])),
            ).fetchone()
            rp = int(high[0]) if high else None
            pts = _required_uint(row, "pts_ns", source="frames.csv")
            det_row = db.execute(
                "SELECT count FROM detection_counts WHERE raw_index=?", (raw_index,)
            ).fetchone() if raw_index is not None else None
            detections = int(det_row[0]) if det_row else 0
            out = {
                "camera_frame_id": camera_id, "raw_frame_index": _fmt(raw_index),
                "ttl_count": ttl, "frame_rp2040_us": _fmt(rp),
                "frame_time_s": _float((rp - first_rp) / 1_000_000.0 if rp is not None else None),
                "controller_unix_us": _fmt(high[1] if high else None),
                "camera_timestamp_ns": row.get("camera_timestamp_ns", ""),
                "frame_pts_ns": _fmt(pts),
                "frame_pts_s": _float(pts / 1_000_000_000.0 if pts is not None else None),
                "duration_ns": row.get("duration_ns", ""),
                "frame_host_unix_ns": row.get("host_unix_ns", ""),
                "frame_host_monotonic_ns": row.get("host_monotonic_ns", ""),
                "ttl_host_unix_ns": high[2] if high else "",
                "ttl_host_monotonic_ns": high[3] if high else "",
                "raw_video_file": "raw.mp4", "video_frame_index": count,
                "video_mapping_source": "single_file_frames_csv", "status": row.get("status", ""),
                "has_ttl": "1" if high else "0", "has_detection": "1" if detections else "0",
                "detection_count": detections,
            }
            if raw_index is not None:
                insert_batch.append((raw_index, camera_id, ttl, _fmt(rp), out["frame_time_s"],
                                     out["frame_pts_ns"], out["frame_pts_s"], "raw.mp4", str(count),
                                     "single_file_frames_csv"))
            if len(insert_batch) >= 10000:
                with db:
                    db.executemany("INSERT INTO frame_lookup VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", insert_batch)
                insert_batch.clear()
            count += 1
            if high is None:
                missing_ttl += 1
            if previous_camera_id is not None and camera_id != previous_camera_id + 1:
                gap_count += 1
                missing_camera_frames += max(0, camera_id - previous_camera_id - 1)
                if len(gap_sample) < 50:
                    gap_sample.append({
                        "raw_frame_index": raw_index,
                        "expected": previous_camera_id + 1,
                        "actual": camera_id,
                        "missing_frames": max(0, camera_id - previous_camera_id - 1),
                    })
            previous_camera_id = camera_id
            if raw_index is not None:
                if previous_raw_index is not None and raw_index != previous_raw_index + 1:
                    mismatch_count += 1
                    if len(mismatch_sample) < 50:
                        mismatch_sample.append({
                            "expected_raw_frame_index": previous_raw_index + 1,
                            "actual_raw_frame_index": raw_index,
                        })
                previous_raw_index = raw_index
            camera_ts = _required_uint(
                row, "camera_timestamp_ns", source="frames.csv"
            )
            if first_camera_ts is not None and rp is not None:
                error = (camera_ts - first_camera_ts) / 1000.0 - (rp - first_rp)
                elapsed = float(rp - first_rp)
                clock_pairs += 1
                clock_final = error
                clock_max = max(abs(error), clock_max or 0.0)
                sum_x += elapsed
                sum_y += error
                sum_xx += elapsed * elapsed
                sum_xy += elapsed * error
        if insert_batch:
            with db:
                db.executemany("INSERT INTO frame_lookup VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", insert_batch)
    finally:
        pass
    clock_slope = clock_intercept = clock_residual_max = None
    denominator = clock_pairs * sum_xx - sum_x * sum_x
    if clock_pairs and denominator:
        clock_slope = (clock_pairs * sum_xy - sum_x * sum_y) / denominator
        clock_intercept = (sum_y - clock_slope * sum_x) / clock_pairs
        clock_residual_max = 0.0
        for row in _reader(
            frames_path,
            required_columns=frozenset(
                {"camera_frame_id", "camera_timestamp_ns"}
            ),
        ):
            camera_id = _required_uint(row, "camera_frame_id", source="frames.csv")
            camera_ts = _required_uint(
                row, "camera_timestamp_ns", source="frames.csv"
            )
            if first_camera_ts is None:
                continue
            ttl = camera_id - int(epoch["camera_frame_id_offset"])
            high = db.execute(
                "SELECT rp FROM highs WHERE count=? AND serial_index>?",
                (ttl, int(epoch["start_marker_index"])),
            ).fetchone()
            if high is None:
                continue
            elapsed = float(int(high[0]) - first_rp)
            error = (camera_ts - first_camera_ts) / 1000.0 - elapsed
            residual = error - (clock_intercept + clock_slope * elapsed)
            clock_residual_max = max(clock_residual_max, abs(residual))
    elif clock_pairs == 1:
        clock_slope, clock_intercept, clock_residual_max = 0.0, clock_final, 0.0
    return {"count": count, "missing_ttl": missing_ttl, "mismatch_count": mismatch_count,
            "mismatch_sample": mismatch_sample, "clock_pairs": clock_pairs,
            "clock_final": clock_final, "clock_max": clock_max,
            "clock_drift_ppm": clock_slope * 1_000_000.0 if clock_slope is not None else None,
            "clock_fit_intercept_us": clock_intercept,
            "clock_residual_max": clock_residual_max, "gaps": gap_sample,
            "gap_count": gap_count, "missing_camera_frames": missing_camera_frames,
            "first_camera_id": first_camera, "last_camera_id": previous_camera_id}


def _write_detections(
    db: sqlite3.Connection, path: Path,
) -> dict[str, Any]:
    stats = {"count": 0, "missing": 0, "ts_mismatch": 0, "pts_mismatch": 0,
             "failed": 0, "fallback": 0, "methods": {}}
    samples = {"missing": [], "ts": [], "pts": []}
    if not path.exists():
        stats["samples"] = samples
        return stats
    cached_key: int | None = None
    cached_frame = None
    try:
        for det_index, det in enumerate(
            _reader(
                path,
                required_columns=frozenset({"source_sequence_index", "gst_pts_ns"}),
            )
        ):
            raw = _to_int(det.get("source_sequence_index"))
            object_pts = _required_uint(det, "gst_pts_ns", source="objects.csv")
            if raw != cached_key:
                cached_key = raw
                cached_frame = db.execute(
                    "SELECT raw_index,camera_id,ttl,rp,time_s,pts,pts_s,raw_video,video_index,video_source "
                    "FROM frame_lookup WHERE raw_index=?", (raw,)
                ).fetchone() if raw is not None else None
            frame = cached_frame
            method = "flir_user_meta" if raw is not None else "unmapped"
            stats["methods"][method] = stats["methods"].get(method, 0) + 1
            stats["fallback"] += int(method.startswith("fallback"))
            mapping_ok = "1" if raw is not None and frame is not None else "0"
            stats["failed"] += int(mapping_ok != "1")
            if frame is None:
                stats["missing"] += 1
                if len(samples["missing"]) < 50:
                    samples["missing"].append({"detection_index": det_index, "raw_frame_num": raw})
            frame_pts = _to_int(frame[5]) if frame else None
            det_ts = object_pts // 1000 if object_pts is not None else None
            if frame_pts is not None and det_ts is not None and abs(det_ts - frame_pts // 1000) > 1:
                stats["ts_mismatch"] += 1
                if len(samples["ts"]) < 50:
                    samples["ts"].append({"detection_index": det_index, "raw_frame_num": raw,
                                          "detection_ts_us": det_ts, "frame_pts_us": frame_pts // 1000,
                                          "delta_us": det_ts - frame_pts // 1000})
            mapping_pts = object_pts
            if frame_pts is not None and mapping_pts is not None and frame_pts != mapping_pts:
                stats["pts_mismatch"] += 1
                if len(samples["pts"]) < 50:
                    samples["pts"].append({"detection_index": det_index, "raw_frame_num": raw,
                                           "mapping_pts_ns": mapping_pts, "frame_pts_ns": frame_pts,
                                           "delta_ns": mapping_pts - frame_pts})
            stats["count"] += 1
    finally:
        pass
    stats["samples"] = samples
    return stats


def _data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in _reader(path))


def build_alignment(
    run_dir: Path,
    out_dir: Path,
    *,
    objects_path: Path | None = None,
    video_validation: Mapping[str, object] | None = None,
    include_objects: bool = True,
) -> dict[str, Any]:
    raw_run_dir = Path(run_dir).absolute()
    raw_out_dir = Path(out_dir).absolute()
    if raw_run_dir.is_symlink() or not raw_run_dir.is_dir():
        raise ValueError(f"run directory must be a real directory: {raw_run_dir}")
    if raw_out_dir.is_symlink():
        raise ValueError(f"alignment output directory must not be a symlink: {raw_out_dir}")
    run_dir = raw_run_dir.resolve()
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = raw_out_dir.resolve()
    if not out_dir.is_dir():
        raise ValueError(f"alignment output is not a directory: {out_dir}")
    frames_path, serial_path = run_dir / "frames.csv", run_dir / "serial.csv"
    objects_path = Path(objects_path).absolute() if objects_path else run_dir / "objects.csv"
    evidence_paths = {
        "frames": frames_path,
        "serial": serial_path,
        "errors": run_dir / "diagnostics" / "errors.csv",
        "video": run_dir / "raw.mp4",
    }
    if include_objects:
        evidence_paths["objects"] = objects_path
    source_evidence = _input_snapshot(
        evidence_paths, required=frozenset({"frames", "serial"})
    )
    summary_path = out_dir / "alignment_summary.json"
    if summary_path.is_symlink():
        raise ValueError("alignment summary output must not be a symlink")
    descriptor, raw_index_path = tempfile.mkstemp(
        prefix=".alignment_index.", suffix=".sqlite", dir=out_dir
    )
    os.close(descriptor)
    index_path = Path(raw_index_path)
    db: sqlite3.Connection | None = None
    try:
        db = _open_index(index_path)
        serial = _index_serial(db, serial_path)
        if include_objects:
            _index_detection_counts(db, objects_path)
        first_frame = _first_frame(frames_path)
        first_high_count, first_high_rp, _ = serial["first_high"]
        first_camera_id = int(_to_int(first_frame.get("camera_frame_id")))
        epoch = {
            "method": "first_recorded_frame_to_first_camera_high_after_start_sent",
            "marker_indices": dict(serial["marker_indices"]),
            "start_marker_index": serial["marker_indices"]["START_SENT"],
            "shutdown_marker_index": serial["marker_indices"]["CAPTURE_STOP_REQUESTED"],
            "first_camera_frame_id": first_camera_id,
            "first_raw_frame_index": _to_int(first_frame.get("raw_frame_index")),
            "first_ttl_count": first_high_count,
            "camera_frame_id_offset": first_camera_id - first_high_count,
            "first_camera_timestamp_ns": _to_int(first_frame.get("camera_timestamp_ns")),
            "first_rp2040_time_us": first_high_rp,
        }
        frames = _write_frames(db, frames_path, epoch)
        epoch_highs = int(
            db.execute(
                "SELECT COUNT(*) FROM highs WHERE serial_index>?",
                (epoch["start_marker_index"],),
            ).fetchone()[0]
        )
        unmatched_epoch_highs = int(
            db.execute(
                "SELECT COUNT(*) FROM highs h LEFT JOIN frame_lookup f ON f.ttl=h.count "
                "WHERE h.serial_index>? AND f.ttl IS NULL",
                (epoch["start_marker_index"],),
            ).fetchone()[0]
        )
        shutdown_tail_highs = int(
            db.execute(
                "SELECT COUNT(*) FROM highs WHERE serial_index>?",
                (epoch["shutdown_marker_index"],),
            ).fetchone()[0]
        )
        unmatched_shutdown_tail_highs = int(
            db.execute(
                "SELECT COUNT(*) FROM highs h LEFT JOIN frame_lookup f ON f.ttl=h.count "
                "WHERE h.serial_index>? AND f.ttl IS NULL",
                (epoch["shutdown_marker_index"],),
            ).fetchone()[0]
        )
        observations = (
            _write_detections(db, objects_path)
            if include_objects
            else {
                "count": 0,
                "missing": 0,
                "ts_mismatch": 0,
                "pts_mismatch": 0,
                "failed": 0,
                "fallback": 0,
                "methods": {},
                "samples": {"missing": [], "ts": [], "pts": []},
            }
        )
        median = serial["median_interval_us"]
        tolerance = max(1000.0, median / 2.0) if median is not None else None
        within = (
            frames["clock_residual_max"] <= tolerance
            if frames["clock_residual_max"] is not None and tolerance is not None
            else None
        )
        epoch.update({
            "validated_pairs": frames["count"] - frames["missing_ttl"],
            "missing_ttl_pairs": frames["missing_ttl"],
            "source_sequence_mismatch_count": frames["mismatch_count"],
            "source_sequence_mismatches_sample": frames["mismatch_sample"],
            "clock_validation_pairs": frames["clock_pairs"],
            "clock_elapsed_error_us_final": frames["clock_final"],
            "clock_elapsed_error_us_max_abs": frames["clock_max"],
            "clock_drift_ppm": frames["clock_drift_ppm"],
            "clock_fit_intercept_us": frames["clock_fit_intercept_us"],
            "clock_detrended_residual_us_max_abs": frames["clock_residual_max"],
            "clock_tolerance_us": tolerance, "clock_within_tolerance": within,
            "controller_high_events_in_epoch": epoch_highs,
            "controller_high_events_unmatched": unmatched_epoch_highs,
            "shutdown_tail_high_events": shutdown_tail_highs,
            "shutdown_tail_high_events_unmatched": unmatched_shutdown_tail_highs,
            "boundary_tail_policy": (
                "every CAMERA_HIGH after START_SENT through end-of-serial, including "
                "events after CAPTURE_STOP_REQUESTED, must map to one recorded camera frame"
            ),
            "epoch_markers_complete": True,
            "validated": (
                frames["missing_ttl"] == 0
                and frames["mismatch_count"] == 0
                and frames["gap_count"] == 0
                and frames["count"] == epoch_highs
                and unmatched_epoch_highs == 0
                and unmatched_shutdown_tail_highs == 0
                and within is not False
            ),
        })
        decoded = _video_validation(run_dir / "raw.mp4", video_validation)
        video_frames = decoded["count"]
        video_info = {
            "file": str(run_dir / "raw.mp4"),
            "file_name": "raw.mp4",
            "decoded_frames": video_frames,
            "frame_count_method": decoded.get("method"),
            "frame_count_error": decoded.get("error"),
        }
        summary = {
            "schema_version": "2.0",
            "run_dir": str(run_dir), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "time_base": "serial.csv CAMERA_HIGH rp2040Time",
            "frame_alignment_rule": "camera_frame_id = CAMERA_HIGH count + dynamic offset",
            "frame_alignment": epoch,
            "start_marker_seen": serial["start_marker_seen"],
            "counts": {"recorded_frames": frames["count"], "camera_high_events": serial["camera_high_events"],
                       "serial_rows": serial["serial_rows"],
                       "object_observations": observations["count"] if include_objects else None,
                       "drop_events": _data_rows(run_dir / "diagnostics" / "errors.csv"),
                       "frame_gaps_detected": frames["gap_count"],
                       "camera_frames_missing": frames["missing_camera_frames"],
                       "frames_missing_ttl": frames["missing_ttl"]},
            "frame_range": {"first_camera_frame_id": frames["first_camera_id"], "last_camera_frame_id": frames["last_camera_id"]},
            "markers": serial["markers"],
            "capture_stop_requested_ttl_count": serial["markers"].get("CAPTURE_STOP_REQUESTED"),
            "capture_stop_done_ttl_count": serial["markers"].get("CAPTURE_STOP_DONE"),
            "stop_sent_ttl_count": serial["markers"].get("STOP_SENT"),
            "post_capture_ttl_tail": ((serial["last_high_count"] - serial["markers"]["CAPTURE_STOP_REQUESTED"])
                                      if serial["last_high_count"] is not None and serial["markers"].get("CAPTURE_STOP_REQUESTED") is not None else None),
            "validation": {
                "video_total_nb_frames": video_frames,
                "video_frame_count_matches_frames_csv": video_frames == frames["count"] if video_frames is not None else None,
                "objects_missing_frame_count": observations["missing"] if include_objects else None,
                "object_ts_mismatch_count": observations["ts_mismatch"] if include_objects else None,
                "object_pts_mismatch_count": observations["pts_mismatch"] if include_objects else None,
                "object_mapping_method_counts": observations["methods"] if include_objects else None,
                "video_mapping_source_counts": {"single_file_frames_csv": frames["count"]},
                "object_mapping_failed_rows": observations["failed"] if include_objects else None,
                "object_mapping_fallback_rows": observations["fallback"] if include_objects else None,
                "object_missing_frames_sample": observations["samples"]["missing"] if include_objects else None,
                "object_ts_mismatches_sample": observations["samples"]["ts"] if include_objects else None,
                "object_pts_mismatches_sample": observations["samples"]["pts"] if include_objects else None,
            },
            "frame_gaps": frames["gaps"], "raw_video_info": video_info,
            "processing": {
                "mode": "streaming_disk_backed",
                "index": "temporary_sqlite",
                "object_validation_deferred": not include_objects,
                "source_evidence": source_evidence,
            },
        }
        final_evidence = _input_snapshot(
            evidence_paths, required=frozenset({"frames", "serial"})
        )
        if final_evidence != source_evidence:
            changed = sorted(
                name
                for name in source_evidence
                if source_evidence[name] != final_evidence[name]
            )
            raise RuntimeError(
                "scientific alignment inputs changed during processing: "
                + ", ".join(changed)
            )
        atomic_write_json(summary_path, summary)
        return summary
    finally:
        if db is not None:
            db.close()
        index_path.unlink(missing_ok=True)
