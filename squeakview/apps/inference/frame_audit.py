"""Scientific FLIR frame metadata and audit-sidecar operator."""

from __future__ import annotations

import atexit
import csv
import ctypes
from itertools import islice
import json
import threading
import time
from pathlib import Path

from pyservicemaker import BatchMetadataOperator

from squeakview.common import run_context


FLIR_FRAME_META_DESCRIPTOR = b"SQUEAKVIEW.FLIR.FRAME_META.v1"
STREAM_LEDGER_BUFFER_BYTES = 1024 * 1024


def _user_meta_type(descriptor: bytes) -> int:
    """Resolve a process-local NvDs user-meta type without depending on pyds."""

    candidates = (
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_meta.so",
        "libnvds_meta.so",
    )
    last_error: OSError | None = None
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            function = library.nvds_get_user_meta_type
            function.argtypes = [ctypes.c_char_p]
            function.restype = ctypes.c_int
            return int(function(descriptor))
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"DeepStream metadata library is unavailable: {last_error}")


def _flir_frame_meta_type() -> int:
    return _user_meta_type(FLIR_FRAME_META_DESCRIPTOR)


class FrameCsvOperator(BatchMetadataOperator):
    """Write scientific frame metadata from a DeepStream batch or source ledger."""

    HEADERS = [
        "stream_id", "camera_serial", "deepstream_frame_number",
        "source_sequence_index", "source", "raw_frame_index", "pts_ns",
        "dts_ns", "duration_ns", "host_monotonic_ns", "host_unix_ns",
        "status", "camera_frame_id", "camera_frame_id_available",
        "stream_frame_id", "chunk_frame_id", "frame_id_delta_consistent",
        "missing_frames_before", "pipeline_missing_frames_before",
        "camera_timestamp_ns", "chunk_timestamp_raw", "timestamp_increment_ns",
        "gst_pts_ns", "timestamp_origin", "host_received_monotonic_ns",
        "host_received_unix_ns", "copy_complete_monotonic_ns",
        "observer_monotonic_ns", "exposure_us", "gain_db", "black_level",
        "payload_crc_valid", "image_status", "source_width", "source_height",
        "source_pixel_format", "metadata_status", "inference_admitted",
    ]
    TELEMETRY_HEADERS = [
        "host_unix_ns", "host_monotonic_ns", "stream_id", "camera_serial",
        "source_sequence_index", "camera_frame_id", "sensor_temperature_c",
        "mainboard_temperature_c", "stream_started_frames", "stream_delivered_frames",
        "stream_incomplete_frames", "stream_lost_frames", "stream_dropped_frames",
        "stream_input_buffers", "stream_output_buffers",
    ]
    EVENT_HEADERS = [
        "host_unix_ns", "host_monotonic_ns", "event_type", "stream_id",
        "expected_frame_id", "actual_frame_id", "details",
    ]

    def __init__(
        self,
        path: Path,
        *,
        meta_type: int | None = None,
        audit_dir: Path | None = None,
        write_audit_sidecars: bool = True,
        max_cameras: int = 16,
    ):
        super().__init__()
        if int(max_cameras) < 1:
            raise ValueError("max_cameras must be positive")
        self.path = path
        self.meta_type = _flir_frame_meta_type() if meta_type is None else int(meta_type)
        self.max_cameras = int(max_cameras)
        path.parent.mkdir(parents=True, exist_ok=True)
        # These writers execute on a DeepStream streaming thread.  A bounded
        # userspace buffer avoids a flush syscall for every scientific row;
        # normal and fail-closed teardown flush and close the handles.
        self._file = path.open(
            "w", newline="", buffering=STREAM_LEDGER_BUFFER_BYTES
        )
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADERS)
        self._lock = threading.Lock()
        self._closed = False
        self._audit_dir = Path(audit_dir) if audit_dir is not None else path.parent
        self._write_audit_sidecars = bool(write_audit_sidecars)
        if self._write_audit_sidecars:
            self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._runtime_path = (
            self._audit_dir / "camera_runtime.json"
            if self._write_audit_sidecars
            else None
        )
        self._camera_runtime: dict[str, dict] = {}
        self._last_source_sequence: dict[int, int] = {}
        self._telemetry_file = None
        self._telemetry_writer = None
        self._events_file = None
        self._events_writer = None
        if self._write_audit_sidecars:
            self._telemetry_file = (self._audit_dir / "camera.csv").open(
                "w", newline="", buffering=STREAM_LEDGER_BUFFER_BYTES
            )
            self._telemetry_writer = csv.writer(self._telemetry_file)
            self._telemetry_writer.writerow(self.TELEMETRY_HEADERS)
            self._events_file = (self._audit_dir / "errors.csv").open(
                "w", newline="", buffering=STREAM_LEDGER_BUFFER_BYTES
            )
            self._events_writer = csv.writer(self._events_file)
            self._events_writer.writerow(self.EVENT_HEADERS)
        atexit.register(self.close)

    @staticmethod
    def _frame_pts(frame_meta) -> int:
        return int(
            getattr(frame_meta, "buffer_pts", None)
            or getattr(frame_meta, "buf_pts", None)
            or 0
        )

    def _metadata(self, frame_meta) -> tuple[dict, str]:
        try:
            # Only the first two matching entries affect the result.  Avoid
            # materializing a potentially corrupt/unbounded native iterator on
            # the streaming thread merely to distinguish one from many.
            items = tuple(islice(frame_meta.user_meta_items(self.meta_type), 2))
        except Exception as exc:
            return {}, f"user_meta_error:{type(exc).__name__}"
        if not items:
            return {}, "missing"
        for item in items:
            try:
                payload = item.get_user_data_json()
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if isinstance(payload, dict):
                    return payload, "ok" if len(items) == 1 else "ok_multiple"
            except Exception:
                continue
        return {}, "invalid_json"

    @staticmethod
    def _value(payload: dict, name: str):
        value = payload.get(name)
        return "" if value is None else value

    def _remember_camera(self, payload: dict) -> None:
        if self._runtime_path is None or not payload:
            return
        serial = str(payload.get("camera_serial") or "")
        camera_index = payload.get("camera_index")
        identity = (
            f"index:{camera_index}"
            if camera_index not in (None, "")
            else f"serial:{serial}"
        )
        existing = self._camera_runtime.get(identity)
        if existing is not None:
            existing_serial = str(existing.get("camera_serial") or "")
            if serial and existing_serial and serial != existing_serial:
                raise RuntimeError(
                    "camera runtime identity changed for "
                    f"{identity}: {existing_serial!r} -> {serial!r}"
                )
            return
        if len(self._camera_runtime) >= self.max_cameras:
            raise RuntimeError(
                "camera runtime metadata exceeded its bounded capacity of "
                f"{self.max_cameras} cameras"
            )
        self._camera_runtime[identity] = {
            "camera_index": payload.get("camera_index"),
            "camera_serial": payload.get("camera_serial"),
            "device_model": payload.get("device_model"),
            "firmware_version": payload.get("firmware_version"),
            "source_width": payload.get("source_width"),
            "source_height": payload.get("source_height"),
            "source_pixel_format": payload.get("source_pixel_format"),
            "actual_fps": payload.get("actual_fps"),
            "configured_exposure_us": payload.get("configured_exposure_us"),
            "configured_gain_db": payload.get("configured_gain_db"),
            "configured_stream_buffer_count": payload.get("configured_stream_buffer_count"),
            "timestamp_increment_ns": payload.get("timestamp_increment_ns"),
            "timestamp_latch_available": payload.get("timestamp_latch_available"),
            "timestamp_latch_raw": payload.get("timestamp_latch_raw"),
            "timestamp_latch_host_monotonic_before_ns": payload.get("timestamp_latch_host_monotonic_before_ns"),
            "timestamp_latch_host_monotonic_after_ns": payload.get("timestamp_latch_host_monotonic_after_ns"),
            "timestamp_latch_host_unix_before_ns": payload.get("timestamp_latch_host_unix_before_ns"),
            "timestamp_latch_host_unix_after_ns": payload.get("timestamp_latch_host_unix_after_ns"),
            "enabled_chunks": payload.get("enabled_chunks"),
        }

    def _persist_camera_runtime(self) -> None:
        if self._runtime_path is None or not self._camera_runtime:
            return
        run_context.atomic_write_json(
            self._runtime_path,
            {
                "schema_version": "1.0",
                "metadata_type": FLIR_FRAME_META_DESCRIPTOR.decode(),
                "cameras": list(self._camera_runtime.values()),
            },
        )

    def _event(
        self,
        event_type: str,
        *,
        stream_id: int,
        expected_frame_id="",
        actual_frame_id="",
        details: dict | None = None,
        host_unix_ns: int | None = None,
        host_monotonic_ns: int | None = None,
    ) -> None:
        if self._events_writer is None:
            return
        self._events_writer.writerow(
            [
                host_unix_ns if host_unix_ns is not None else time.time_ns(),
                host_monotonic_ns if host_monotonic_ns is not None else time.monotonic_ns(),
                event_type, stream_id, expected_frame_id, actual_frame_id,
                json.dumps(details or {}, sort_keys=True, separators=(",", ":")),
            ]
        )

    def _write_audit_rows(
        self, payload: dict, status: str, stream_id: int, pipeline_missing
    ) -> None:
        event_clocks = {
            "host_unix_ns": payload.get("host_received_unix_ns"),
            "host_monotonic_ns": payload.get("host_received_monotonic_ns"),
        }
        if payload.get("telemetry_sample") and self._telemetry_writer is not None:
            self._telemetry_writer.writerow(
                [
                    self._value(payload, "host_received_unix_ns"),
                    self._value(payload, "host_received_monotonic_ns"), stream_id,
                    self._value(payload, "camera_serial"),
                    self._value(payload, "source_sequence_index"),
                    self._value(payload, "camera_frame_id"),
                    self._value(payload, "sensor_temperature_c"),
                    self._value(payload, "mainboard_temperature_c"),
                    self._value(payload, "stream_started_frames"),
                    self._value(payload, "stream_delivered_frames"),
                    self._value(payload, "stream_incomplete_frames"),
                    self._value(payload, "stream_lost_frames"),
                    self._value(payload, "stream_dropped_frames"),
                    self._value(payload, "stream_input_buffers"),
                    self._value(payload, "stream_output_buffers"),
                ]
            )
        camera_missing = payload.get("missing_frames_before")
        camera_frame_id = payload.get("camera_frame_id")
        if isinstance(camera_missing, int) and camera_missing > 0 and isinstance(camera_frame_id, int):
            self._event(
                "camera_frame_gap", stream_id=stream_id,
                expected_frame_id=camera_frame_id - camera_missing,
                actual_frame_id=camera_frame_id,
                details={"missing_frames": camera_missing}, **event_clocks,
            )
        if isinstance(pipeline_missing, int) and pipeline_missing > 0:
            source_sequence = payload.get("source_sequence_index")
            self._event(
                "pipeline_frame_gap", stream_id=stream_id,
                expected_frame_id=(source_sequence - pipeline_missing) if isinstance(source_sequence, int) else "",
                actual_frame_id=source_sequence if isinstance(source_sequence, int) else "",
                details={"missing_frames": pipeline_missing}, **event_clocks,
            )
        if payload.get("crc_valid") is False:
            self._event(
                "payload_crc_failure", stream_id=stream_id,
                actual_frame_id=camera_frame_id if isinstance(camera_frame_id, int) else "",
                **event_clocks,
            )
        if not status.startswith("ok"):
            self._event("frame_metadata_" + status, stream_id=stream_id, **event_clocks)

    def handle_metadata(self, batch_meta) -> None:
        with self._lock:
            # Fail-fast teardown can close the Python-owned ledgers while the
            # native pipeline is still producing callbacks.  Ignore those
            # late callbacks instead of writing through a closed CSV handle;
            # the run status is already marked invalid before process exit.
            if self._closed:
                return
            for frame_meta in batch_meta.frame_items:
                payload, status = self._metadata(frame_meta)
                self._remember_camera(payload)
                stream_id = int(getattr(frame_meta, "source_id", frame_meta.pad_index))
                pts_ns = self._frame_pts(frame_meta)
                camera_timestamp = payload.get("transport_timestamp_ns")
                camera_frame_id = payload.get("camera_frame_id")
                source_sequence = payload.get("source_sequence_index")
                pipeline_missing = ""
                if isinstance(source_sequence, int):
                    previous_sequence = self._last_source_sequence.get(stream_id)
                    pipeline_missing = (
                        max(0, source_sequence - previous_sequence - 1)
                        if previous_sequence is not None else 0
                    )
                    self._last_source_sequence[stream_id] = source_sequence
                self._write_audit_rows(payload, status, stream_id, pipeline_missing)
                self._writer.writerow(
                    [
                        stream_id, self._value(payload, "camera_serial"),
                        int(frame_meta.frame_number),
                        self._value(payload, "source_sequence_index"),
                        f"flirspinsrc:{stream_id}",
                        self._value(payload, "source_sequence_index"),
                        pts_ns or self._value(payload, "gst_pts_ns"),
                        pts_ns or self._value(payload, "gst_pts_ns"),
                        int(round(1_000_000_000 / float(payload["actual_fps"])))
                        if payload.get("actual_fps") else "",
                        self._value(payload, "host_received_monotonic_ns"),
                        self._value(payload, "host_received_unix_ns"),
                        "ok" if status.startswith("ok") else status,
                        "" if camera_frame_id is None else camera_frame_id,
                        int(camera_frame_id is not None),
                        self._value(payload, "stream_frame_id"),
                        self._value(payload, "chunk_frame_id"),
                        self._value(payload, "frame_id_delta_consistent"),
                        self._value(payload, "missing_frames_before"), pipeline_missing,
                        "" if camera_timestamp is None else camera_timestamp,
                        self._value(payload, "chunk_timestamp_raw"),
                        self._value(payload, "timestamp_increment_ns"),
                        pts_ns or self._value(payload, "gst_pts_ns"),
                        self._value(payload, "timestamp_origin"),
                        self._value(payload, "host_received_monotonic_ns"),
                        self._value(payload, "host_received_unix_ns"),
                        self._value(payload, "copy_complete_monotonic_ns"),
                        time.monotonic_ns(),
                        self._value(payload, "chunk_exposure_us"),
                        self._value(payload, "chunk_gain_db"),
                        self._value(payload, "chunk_black_level"),
                        self._value(payload, "crc_valid"),
                        self._value(payload, "image_status"),
                        self._value(payload, "source_width"),
                        self._value(payload, "source_height"),
                        self._value(payload, "source_pixel_format"), status,
                        self._value(payload, "inference_admitted"),
                    ]
                )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            failures: list[str] = []

            def finish(name: str, action) -> None:
                try:
                    action()
                except Exception as exc:
                    failures.append(f"{name}: {type(exc).__name__}: {exc}")

            for name, handle in (
                ("frame ledger", self._file),
                ("camera telemetry", self._telemetry_file),
                ("camera error ledger", self._events_file),
            ):
                if handle is None:
                    continue
                finish(f"{name} flush", handle.flush)
                finish(f"{name} close", handle.close)
            # Runtime identity is needed only after capture closes. Persist it
            # here so atomic write/fsync work never blocks a streaming callback.
            finish("camera runtime persistence", self._persist_camera_runtime)
            if failures:
                raise RuntimeError("; ".join(failures))
