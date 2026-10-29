"""Bounded, fail-closed CSV persistence for tracked pose observations."""

from __future__ import annotations

import csv
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from squeakview.common.run_layout import artifact_path, is_primary_layout


STREAM_LEDGER_BUFFER_BYTES = 1024 * 1024

OBJECT_HEADERS = [
    "observation_id", "stream_id", "deepstream_frame_number", "source_sequence_index",
    "camera_frame_id", "camera_timestamp_ns", "gst_pts_ns", "class_id", "class_label",
    "track_id", "detected_this_frame", "tracker_predicted", "detector_confidence",
    "tracker_confidence", "detector_x", "detector_y", "detector_w", "detector_h",
    "track_x", "track_y", "track_w", "track_h", "pose_available", "schema_version",
]
KEYPOINT_HEADERS = [
    "observation_id", "stream_id", "deepstream_frame_number", "source_sequence_index",
    "camera_frame_id", "track_id", "class_id", "class_label", "keypoint_index",
    "keypoint_name", "x_px", "y_px", "x_norm", "y_norm", "confidence", "visible",
    "coordinate_space", "source",
]


class PoseCsvWriter:
    """Write unified detections for new runs, preserving historical pose schemas.

    All write and close errors propagate to the streaming lifecycle owner. No
    row is silently discarded or evicted.
    """

    def __init__(self, run_dir: Path, keypoint_names: Sequence[str] = (), *, require_native_timing=False) -> None:
        run_dir = Path(run_dir)
        self._lock = threading.Lock()
        self._closed = False
        self._files: dict[str, Any] = {}
        self._writers: dict[str, csv.writer] = {}
        self._primary = is_primary_layout(run_dir)
        self._require_native_timing = bool(require_native_timing and self._primary)
        self._last_timing = {}
        self._pending: dict[str, object] | None = None
        self._keypoint_names = tuple(keypoint_names)
        self._detection_headers = OBJECT_HEADERS + ["host_unix_ns", "host_monotonic_ns"] + [
            f"{name}_{suffix}" for name in self._keypoint_names
            for suffix in ("x_px", "y_px", "confidence", "visible")
        ]
        if len(set(self._detection_headers)) != len(self._detection_headers):
            raise ValueError("keypoint names produce duplicate detection columns")
        try:
            if self._primary:
                handle = artifact_path(run_dir, "detections.csv").open(
                    "w", newline="", buffering=STREAM_LEDGER_BUFFER_BYTES)
                self._files["detections"] = handle
                self._writers["detections"] = csv.writer(handle)
                self._writers["detections"].writerow(self._detection_headers)
                return
            for name, headers in (
                ("objects", OBJECT_HEADERS),
                ("keypoints", KEYPOINT_HEADERS),
            ):
                handle = (artifact_path(run_dir, f"{name}.csv")).open(
                    "w", newline="", buffering=STREAM_LEDGER_BUFFER_BYTES
                )
                self._files[name] = handle
                writer = csv.writer(handle)
                self._writers[name] = writer
                writer.writerow(headers)
        except Exception:
            for handle in self._files.values():
                try:
                    handle.close()
                except Exception:
                    pass
            raise

    def write_object(self, row: Sequence[object], *, host_unix_ns: object = "",
                     host_monotonic_ns: object = "") -> None:
        if not self._primary:
            self._write("objects", row, expected_columns=len(OBJECT_HEADERS))
            return
        if len(row) != len(OBJECT_HEADERS):
            raise ValueError(f"objects pose row has {len(row)} columns; expected {len(OBJECT_HEADERS)}")
        with self._lock:
            if self._closed:
                raise RuntimeError("pose CSV writer is closed")
            values = dict(zip(OBJECT_HEADERS, row))
            if self._require_native_timing:
                times = dict(values, host_unix_ns=host_unix_ns, host_monotonic_ns=host_monotonic_ns)
                for name in ('stream_id', 'source_sequence_index', 'camera_timestamp_ns', 'gst_pts_ns', 'host_unix_ns', 'host_monotonic_ns'):
                    value = times.get(name)
                    if isinstance(value, bool) or not str(value).isascii() or not str(value).isdecimal():
                        raise ValueError(f'detection is missing a valid native {name}')
                stream = int(values['stream_id'])
                timing = (int(values['source_sequence_index']), int(values['camera_timestamp_ns']), int(host_unix_ns), int(host_monotonic_ns))
                previous = self._last_timing.get(stream)
                if stream >= 64 or (previous is not None and (timing[0] < previous[0] or
                        (timing[0] == previous[0] and timing != previous) or
                        (timing[0] > previous[0] and timing[1] <= previous[1]))):
                    raise ValueError('detection frame identity or native clock regressed')
                self._last_timing[stream] = timing
            self._flush_detection()
            self._pending = values
            self._pending.update(host_unix_ns=host_unix_ns, host_monotonic_ns=host_monotonic_ns)

    def _flush_detection(self) -> None:
        if self._pending is not None:
            self._writers["detections"].writerow(
                [self._pending.get(name, "") for name in self._detection_headers])
            self._pending = None

    def write_keypoints(self, rows: Iterable[Sequence[object]]) -> None:
        for row in rows:
            if not self._primary:
                self._write("keypoints", row, expected_columns=len(KEYPOINT_HEADERS))
                continue
            if len(row) != len(KEYPOINT_HEADERS):
                raise ValueError(f"keypoints pose row has {len(row)} columns; expected {len(KEYPOINT_HEADERS)}")
            point = dict(zip(KEYPOINT_HEADERS, row))
            with self._lock:
                if self._closed:
                    raise RuntimeError("pose CSV writer is closed")
                if self._pending is None or point["observation_id"] != self._pending["observation_id"]:
                    raise ValueError("keypoint does not belong to the current observation")
                name = str(point["keypoint_name"])
                if name not in self._keypoint_names:
                    raise ValueError(f"unknown keypoint: {name}")
                for suffix in ("x_px", "y_px", "confidence", "visible"):
                    self._pending[f"{name}_{suffix}"] = point[suffix]

    def _write(self, name: str, row: Sequence[object], *, expected_columns: int) -> None:
        if len(row) != expected_columns:
            raise ValueError(
                f"{name} pose row has {len(row)} columns; expected {expected_columns}"
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("pose CSV writer is closed")
            self._writers[name].writerow(row)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            errors: list[BaseException] = []
            if self._primary:
                try:
                    self._flush_detection()
                except BaseException as exc:
                    errors.append(exc)
            for handle in self._files.values():
                try:
                    handle.flush()
                except BaseException as exc:
                    errors.append(exc)
                try:
                    handle.close()
                except BaseException as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]
