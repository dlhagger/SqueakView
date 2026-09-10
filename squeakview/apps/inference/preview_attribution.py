"""Bounded preview-boundary ledgers and post-run shedding reconciliation."""

from __future__ import annotations

import atexit
import csv
import json
import os
import stat
import threading
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterator

from pyservicemaker import BatchMetadataOperator

from .capture_reconciliation import MAX_LEDGER_LINE_BYTES, _bounded_text_lines
from squeakview.common.diagnostics.evidence_identity import stable_file_identity
from squeakview.common.recording_evidence import MAX_RECORDING_CAMERAS


HEADERS = (
    "boundary", "stream_id", "source_sequence_index", "camera_frame_id",
    "pts_ns", "observer_monotonic_ns",
)


def boundary_path(run_dir: Path, stream_id: int, boundary: str) -> Path:
    suffix = "" if stream_id == 0 else f"_cam{stream_id}"
    return Path(run_dir) / "diagnostics" / f"preview_{boundary}{suffix}.csv"


class PreviewBoundaryOperator(BatchMetadataOperator):
    """Persist source identity before or after the leaky preview queue."""

    def __init__(self, path: Path, stream_id: int, boundary: str, meta_type: int):
        super().__init__()
        if boundary not in {"admission", "delivery"}:
            raise ValueError("preview boundary must be admission or delivery")
        self.path = Path(path)
        self.stream_id = int(stream_id)
        self.boundary = boundary
        self.meta_type = int(meta_type)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", buffering=1024 * 1024)
        self._writer = csv.writer(self._file)
        self._writer.writerow(HEADERS)
        self._lock = threading.Lock()
        self._closed = False
        self.error: str | None = None
        atexit.register(self.close)

    def _identity(self, frame_meta) -> tuple[object, object]:
        try:
            items = tuple(islice(frame_meta.user_meta_items(self.meta_type), 2))
            if len(items) != 1:
                return "", ""
            payload = items[0].get_user_data_json()
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                return "", ""
            sequence = payload.get("source_sequence_index")
            camera_frame_id = payload.get("camera_frame_id")
            if type(sequence) is not int or sequence < 0:
                sequence = ""
            if type(camera_frame_id) is not int or camera_frame_id < 0:
                camera_frame_id = ""
            return sequence, camera_frame_id
        except Exception:
            return "", ""

    def handle_metadata(self, batch_meta) -> None:
        with self._lock:
            if self._closed or self.error is not None:
                return
            try:
                for frame_meta in islice(batch_meta.frame_items, 2):
                    sequence, camera_frame_id = self._identity(frame_meta)
                    stream_id = int(
                        getattr(
                            frame_meta,
                            "source_id",
                            getattr(frame_meta, "pad_index", -1),
                        )
                    )
                    pts_ns = int(
                        getattr(frame_meta, "buffer_pts", None)
                        or getattr(frame_meta, "buf_pts", None)
                        or 0
                    )
                    self._writer.writerow(
                        (
                            self.boundary,
                            stream_id,
                            sequence,
                            camera_frame_id,
                            pts_ns,
                            time.monotonic_ns(),
                        )
                    )
            except Exception as exc:
                # Preview evidence is optional to acquisition. Its loss makes
                # preview qualification fail, never the recording pipeline.
                self.error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                try:
                    self._file.close()
                except Exception:
                    pass
            finally:
                atexit.unregister(self.close)


@dataclass(frozen=True, slots=True)
class BoundaryIdentity:
    sequence: int
    camera_frame_id: int
    pts_ns: int


def _uint(value: object, name: str) -> int:
    raw = value if isinstance(value, str) else ""
    if not raw or not raw.isascii() or not raw.isdigit():
        raise ValueError(f"{name} is not an unsigned decimal integer")
    return int(raw)


def _iter_boundary(path: Path, stream_id: int, boundary: str) -> Iterator[BoundaryIdentity]:
    try:
        metadata = path.lstat()
    except OSError:
        metadata = None
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"preview {boundary} ledger is missing: {path.name}")
    reader = csv.DictReader(_bounded_text_lines(path))
    if tuple(reader.fieldnames or ()) != HEADERS:
        raise RuntimeError(f"preview {boundary} ledger schema is invalid: {path.name}")
    previous: BoundaryIdentity | None = None
    for line_number, row in enumerate(reader, 2):
        try:
            if row.get("boundary") != boundary:
                raise ValueError("boundary label mismatch")
            if _uint(row.get("stream_id"), "stream_id") != stream_id:
                raise ValueError("stream ID mismatch")
            current = BoundaryIdentity(
                _uint(row.get("source_sequence_index"), "source_sequence_index"),
                _uint(row.get("camera_frame_id"), "camera_frame_id"),
                _uint(row.get("pts_ns"), "pts_ns"),
            )
        except ValueError as exc:
            raise RuntimeError(
                f"invalid preview {boundary} row {path.name}:{line_number}: {exc}"
            ) from exc
        if previous is not None and current.sequence <= previous.sequence:
            raise RuntimeError(
                f"preview {boundary} identities are not strictly ordered at "
                f"{path.name}:{line_number}"
            )
        previous = current
        yield current


def _camera_summary(run_dir: Path, stream_id: int) -> dict[str, object]:
    admission_path = boundary_path(run_dir, stream_id, "admission")
    delivery_path = boundary_path(run_dir, stream_id, "delivery")
    before = {
        "admission": stable_file_identity(admission_path),
        "delivery": stable_file_identity(delivery_path),
    }
    metadata_before = {}
    for name, path in (("admission", admission_path), ("delivery", delivery_path)):
        try:
            value = path.lstat()
            metadata_before[name] = (
                value.st_dev, value.st_ino, value.st_size,
                value.st_mtime_ns, value.st_ctime_ns,
            )
        except OSError as exc:
            raise RuntimeError(f"preview {name} ledger is unavailable: {exc}") from exc
    admissions = _iter_boundary(admission_path, stream_id, "admission")
    deliveries = _iter_boundary(delivery_path, stream_id, "delivery")
    admitted = delivered = shed = unmatched = 0
    admission = next(admissions, None)
    for delivery in deliveries:
        while admission is not None and admission.sequence < delivery.sequence:
            admitted += 1
            shed += 1
            admission = next(admissions, None)
        if admission is None or admission.sequence > delivery.sequence:
            unmatched += 1
            continue
        admitted += 1
        if admission != delivery:
            unmatched += 1
        else:
            delivered += 1
        admission = next(admissions, None)
    while admission is not None:
        admitted += 1
        shed += 1
        admission = next(admissions, None)
    for name, path in (("admission", admission_path), ("delivery", delivery_path)):
        try:
            value = path.lstat()
            metadata_after = (
                value.st_dev, value.st_ino, value.st_size,
                value.st_mtime_ns, value.st_ctime_ns,
            )
        except OSError as exc:
            raise RuntimeError(f"preview {name} ledger disappeared: {exc}") from exc
        if metadata_after != metadata_before[name]:
            raise RuntimeError(f"preview {name} ledger changed during reconciliation")
    admission_identity = before["admission"]
    delivery_identity = before["delivery"]
    return {
        "stream_id": stream_id,
        "admitted_frames": admitted,
        "delivered_frames": delivered,
        "shed_frames": shed,
        "unmatched_delivery_frames": unmatched,
        "admission_ledger": str(admission_path.relative_to(run_dir)),
        "delivery_ledger": str(delivery_path.relative_to(run_dir)),
        "admission_identity": admission_identity,
        "delivery_identity": delivery_identity,
        "passed": (
            admitted > 0
            and unmatched == 0
            and delivered + shed == admitted
            and admission_identity.get("available") is True
            and delivery_identity.get("available") is True
        ),
    }


def reconcile_preview(run_dir: Path, camera_count: int, *, required: bool) -> dict[str, object]:
    """Summarize intentional preview shedding without affecting recording validity."""

    if not required:
        return {
            "schema_version": "1.0",
            "required": False,
            "policy": "preview_shedding_is_observed_but_never_invalidates_recording",
            "cameras": [],
            "passed": True,
        }
    if type(camera_count) is not int or not 1 <= camera_count <= MAX_RECORDING_CAMERAS:
        return {
            "schema_version": "1.0",
            "required": True,
            "policy": "preview_shedding_is_observed_but_never_invalidates_recording",
            "cameras": [],
            "errors": [
                f"camera_count must be an integer from 1 through {MAX_RECORDING_CAMERAS}"
            ],
            "passed": False,
        }
    cameras: list[dict[str, object]] = []
    errors: list[str] = []
    expected_names = {
        boundary_path(Path(run_dir), stream_id, boundary).name
        for stream_id in range(camera_count)
        for boundary in ("admission", "delivery")
    }
    observed_names: set[str] = set()
    overflow = False
    try:
        with os.scandir(Path(run_dir) / "diagnostics") as entries:
            for entry in entries:
                if not Path(entry.name).match("preview_*.csv"):
                    continue
                if len(observed_names) >= len(expected_names) + 1:
                    overflow = True
                    break
                observed_names.add(entry.name)
    except OSError as exc:
        errors.append(f"preview ledger enumeration failed: {exc}")
    if overflow or observed_names != expected_names:
        errors.append(
            "preview ledger set is not exact: expected "
            f"{sorted(expected_names)}, observed {sorted(observed_names)}"
            + (" (overflow)" if overflow else "")
        )
    for stream_id in range(camera_count):
        try:
            cameras.append(_camera_summary(Path(run_dir), stream_id))
        except Exception as exc:
            errors.append(f"stream {stream_id}: {exc}")
    return {
        "schema_version": "1.0",
        "required": True,
        "policy": "preview_shedding_is_observed_but_never_invalidates_recording",
        "cameras": cameras,
        "errors": errors,
        "passed": not errors and len(cameras) == camera_count and all(
            camera.get("passed") is True for camera in cameras
        ),
    }


__all__ = [
    "HEADERS",
    "PreviewBoundaryOperator",
    "boundary_path",
    "reconcile_preview",
]
