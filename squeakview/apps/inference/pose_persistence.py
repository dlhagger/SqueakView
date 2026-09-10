"""Bounded, fail-closed CSV persistence for tracked pose observations."""

from __future__ import annotations

import csv
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


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
    """Own the two append-only pose ledgers and their bounded buffers.

    All write and close errors propagate to the streaming lifecycle owner. No
    row is silently discarded or evicted.
    """

    def __init__(self, run_dir: Path) -> None:
        run_dir = Path(run_dir)
        self._lock = threading.Lock()
        self._closed = False
        self._files: dict[str, Any] = {}
        self._writers: dict[str, csv.writer] = {}
        try:
            for name, headers in (
                ("objects", OBJECT_HEADERS),
                ("keypoints", KEYPOINT_HEADERS),
            ):
                handle = (run_dir / f"{name}.csv").open(
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

    def write_object(self, row: Sequence[object]) -> None:
        self._write("objects", row, expected_columns=len(OBJECT_HEADERS))

    def write_keypoints(self, rows: Iterable[Sequence[object]]) -> None:
        for row in rows:
            self._write("keypoints", row, expected_columns=len(KEYPOINT_HEADERS))

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
