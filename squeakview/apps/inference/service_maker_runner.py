"""DeepStream 9.1 inference runner built on the PyServiceMaker Pipeline API."""
from __future__ import annotations

import atexit
from collections import OrderedDict
import csv
import ctypes
import json
import signal
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pyservicemaker import (
    BufferOperator,
    BatchMetadataOperator,
    EOSMessage,
    Pipeline,
    PipelineState,
    Probe,
    StateTransitionMessage,
)

from squeakview.common import run_context
from .pose_pipeline import (
    FramePoseStore,
    ObservationOperator,
    Yolo26PoseTensorOperator,
    load_pose_schema,
)


FLIR_FRAME_META_DESCRIPTOR = b"SQUEAKVIEW.FLIR.FRAME_META.v1"
RECORD_QUEUE_SECONDS = 4
RECORD_BACKPRESSURE_WARNING_SECONDS = 1
RECORD_BACKPRESSURE_FATAL_SECONDS = 3


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


def ts() -> str:
    return time.strftime("%H:%M:%S")


@dataclass(slots=True)
class InferenceConfig:
    """CLI-compatible configuration shared with the operator subprocess."""

    cfg_path: Path | None = field(default_factory=lambda: Path.cwd() / "config_infer_primary_11m.txt")
    capture_backend: str = "flir_direct"
    num_cameras: int = 1
    camera_serials: tuple[str, ...] = ()
    pixel_format: str = "Mono8"
    trigger_on: bool = False
    trigger_activation: str = "rising"
    exposure_us: float | None = 10000.0
    gain: float | None = -1.0
    width: int = 1280
    height: int = 720
    fps: int = 30
    bitrate: int = 4000
    preview_sockets: tuple[Path, ...] = ()
    enable_infer: bool = True
    run_dir: Path | None = None


def _flir_pixel_format(value: str | None) -> str:
    pixel_format = str(value or "Mono8").strip()
    return "Mono8" if pixel_format.upper() == "GRAY8" else pixel_format or "Mono8"


def _read_config_value(config_path: Path | None, key: str) -> str | None:
    if config_path is None:
        return None
    try:
        lines = config_path.read_text().splitlines()
    except OSError:
        return None
    prefix = key.lower()
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        name, value = raw.split("=", 1)
        if name.strip().lower() == prefix:
            return value.strip().strip('"')
    return None


def _load_class_names(config_path: Path | None) -> list[str]:
    raw_path = _read_config_value(config_path, "labelfile-path")
    if not raw_path or config_path is None:
        return []
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    try:
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    except OSError:
        return []


def _validate_config(config: InferenceConfig) -> None:
    backend = str(config.capture_backend or "").lower().strip()
    if backend != "flir_direct":
        raise ValueError(f"SqueakView only supports capture_backend='flir_direct' (got {backend!r})")
    if config.num_cameras < 1:
        raise ValueError("num_cameras must be at least 1")
    if config.camera_serials and len(config.camera_serials) != config.num_cameras:
        raise ValueError("camera serial count must match camera count")
    if len(set(config.camera_serials)) != len(config.camera_serials):
        raise ValueError("camera serials must be unique")
    for name in ("width", "height", "fps", "bitrate"):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if config.enable_infer:
        if config.cfg_path is None:
            raise ValueError("DeepStream config (--cfg) is required when inference is enabled")
        if not Path(config.cfg_path).is_file():
            raise FileNotFoundError(f"DeepStream config does not exist: {config.cfg_path}")
        batch_size = _read_config_value(Path(config.cfg_path), "batch-size")
        if batch_size is not None and int(batch_size) != int(config.num_cameras):
            raise ValueError(
                f"nvinfer config batch-size ({batch_size}) does not match camera count "
                f"({config.num_cameras})"
            )
    if config.preview_sockets and len(config.preview_sockets) != config.num_cameras:
        raise ValueError(
            "preview socket count "
            f"({len(config.preview_sockets)}) does not match camera count ({config.num_cameras})"
        )


class RecordingAdmissionOperator(BufferOperator):
    """Durably record buffers that enter the non-leaky recording branch."""

    HEADERS = ["stream_id", "record_frame_index", "pts_ns", "observer_monotonic_ns"]

    def __init__(
        self,
        path: Path,
        stream_id: int,
        telemetry: "RecordingPathTelemetry | None" = None,
    ):
        super().__init__()
        self.path = path
        self.stream_id = int(stream_id)
        self._file = path.open("w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADERS)
        self._count = 0
        self._lock = threading.Lock()
        self._closed = False
        self._telemetry = telemetry
        atexit.register(self.close)

    def handle_buffer(self, buffer) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._writer.writerow(
                [self.stream_id, self._count, int(buffer.timestamp), time.monotonic_ns()]
            )
            if self._telemetry is not None:
                self._telemetry.admit(int(buffer.timestamp))
            self._count += 1
        return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._file.flush()
            self._file.close()


class RecordingPathTelemetry:
    """Sample recording backlog and encoder latency with bounded state."""

    HEADERS = [
        "host_unix_ns",
        "host_monotonic_ns",
        "stream_id",
        "event",
        "pts_ns",
        "egress_timestamp_ns",
        "encoder_correlation",
        "queue_wait_ms",
        "encoder_latency_ms",
        "waiting_for_record_admission",
        "encoder_in_flight",
        "max_waiting_since_sample",
        "max_encoder_in_flight_since_sample",
        "pending_evictions",
    ]

    def __init__(
        self,
        path: Path,
        stream_id: int,
        *,
        sample_interval_s: float = 1.0,
        warning_depth: int = 24,
        fatal_depth: int | None = None,
        max_pending: int = 4096,
        on_fatal: Callable[[str], None] | None = None,
    ):
        self.path = path
        self.stream_id = int(stream_id)
        self.sample_interval_ns = max(1, int(sample_interval_s * 1_000_000_000))
        self.warning_depth = max(1, int(warning_depth))
        self.fatal_depth = (
            max(self.warning_depth + 1, int(fatal_depth))
            if fatal_depth is not None
            else None
        )
        self.max_pending = max(self.warning_depth, int(max_pending))
        self.on_fatal = on_fatal
        self._file = path.open("w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADERS)
        self._source_pending: OrderedDict[int, int] = OrderedDict()
        self._pending: OrderedDict[int, int] = OrderedDict()
        self._lock = threading.Lock()
        self._last_sample_ns = time.monotonic_ns()
        self._max_source_depth = 0
        self._max_encoder_depth = 0
        self._evictions = 0
        self._warning_active = False
        self._fatal_reported = False
        self._closed = False
        atexit.register(self.close)

    def _write(
        self,
        event: str,
        now_ns: int,
        pts_ns: int | str,
        egress_timestamp_ns="",
        encoder_correlation="",
        queue_wait_ms="",
        encoder_latency_ms="",
    ) -> None:
        self._writer.writerow(
            [
                time.time_ns(),
                now_ns,
                self.stream_id,
                event,
                pts_ns,
                egress_timestamp_ns,
                encoder_correlation,
                queue_wait_ms,
                encoder_latency_ms,
                len(self._source_pending),
                len(self._pending),
                self._max_source_depth,
                self._max_encoder_depth,
                self._evictions,
            ]
        )

    def _sample_if_due(
        self,
        now_ns: int,
        pts_ns: int,
        queue_wait_ms="",
        encoder_latency_ms="",
        egress_timestamp_ns="",
        encoder_correlation="",
    ) -> str | None:
        source_depth = len(self._source_pending)
        encoder_depth = len(self._pending)
        self._max_source_depth = max(self._max_source_depth, source_depth)
        self._max_encoder_depth = max(self._max_encoder_depth, encoder_depth)
        warning = max(source_depth, encoder_depth) >= self.warning_depth
        if warning != self._warning_active:
            self._warning_active = warning
            self._write(
                "backpressure_enter" if warning else "backpressure_exit",
                now_ns,
                pts_ns,
                egress_timestamp_ns,
                encoder_correlation,
                queue_wait_ms,
                encoder_latency_ms,
            )
        fatal_message = None
        if (
            self.fatal_depth is not None
            and source_depth >= self.fatal_depth
            and not self._fatal_reported
        ):
            self._fatal_reported = True
            fatal_message = (
                f"recording queue backlog reached {source_depth} frames "
                f"(fatal threshold {self.fatal_depth}) on stream {self.stream_id}"
            )
            self._write("backpressure_fatal", now_ns, pts_ns)
        if now_ns - self._last_sample_ns >= self.sample_interval_ns:
            self._write(
                "sample",
                now_ns,
                pts_ns,
                egress_timestamp_ns,
                encoder_correlation,
                queue_wait_ms,
                encoder_latency_ms,
            )
            self._last_sample_ns = now_ns
            self._max_source_depth = source_depth
            self._max_encoder_depth = encoder_depth
        return fatal_message

    def source(self, pts_ns: int) -> None:
        now_ns = time.monotonic_ns()
        fatal_message = None
        with self._lock:
            if self._closed:
                return
            self._source_pending[int(pts_ns)] = now_ns
            while len(self._source_pending) > self.max_pending:
                self._source_pending.popitem(last=False)
                self._evictions += 1
            fatal_message = self._sample_if_due(now_ns, int(pts_ns))
        if fatal_message is not None and self.on_fatal is not None:
            self.on_fatal(fatal_message)

    def admit(self, pts_ns: int) -> None:
        now_ns = time.monotonic_ns()
        with self._lock:
            if self._closed:
                return
            source_ns = self._source_pending.pop(int(pts_ns), None)
            queue_wait_ms = (
                f"{(now_ns - source_ns) / 1_000_000.0:.6f}"
                if source_ns is not None
                else ""
            )
            self._pending[int(pts_ns)] = now_ns
            while len(self._pending) > self.max_pending:
                self._pending.popitem(last=False)
                self._evictions += 1
            self._sample_if_due(now_ns, int(pts_ns), queue_wait_ms=queue_wait_ms)

    def egress(self, egress_timestamp_ns: int) -> None:
        now_ns = time.monotonic_ns()
        with self._lock:
            if self._closed:
                return
            output_timestamp = int(egress_timestamp_ns)
            input_pts_ns = output_timestamp
            admitted_ns = self._pending.pop(output_timestamp, None)
            correlation = "pts" if admitted_ns is not None else ""
            if admitted_ns is None and self._pending:
                # x264 is configured with bframes=0 and both lookaheads at 0,
                # so encoded access units retain input order even when the
                # parser exposes a rewritten timestamp through ServiceMaker.
                input_pts_ns, admitted_ns = self._pending.popitem(last=False)
                correlation = "fifo"
            elif admitted_ns is None:
                correlation = "unmatched"
            encoder_latency_ms = (
                f"{(now_ns - admitted_ns) / 1_000_000.0:.6f}"
                if admitted_ns is not None
                else ""
            )
            self._sample_if_due(
                now_ns,
                input_pts_ns,
                encoder_latency_ms=encoder_latency_ms,
                egress_timestamp_ns=output_timestamp,
                encoder_correlation=correlation,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._write("closed", time.monotonic_ns(), "")
            self._file.flush()
            self._file.close()


class RecordingEgressOperator(BufferOperator):
    def __init__(self, telemetry: RecordingPathTelemetry):
        super().__init__()
        self.telemetry = telemetry

    def handle_buffer(self, buffer) -> bool:
        self.telemetry.egress(int(buffer.timestamp))
        return True


class RecordingIngressOperator(BufferOperator):
    def __init__(self, telemetry: RecordingPathTelemetry):
        super().__init__()
        self.telemetry = telemetry

    def handle_buffer(self, buffer) -> bool:
        self.telemetry.source(int(buffer.timestamp))
        return True


class FrameCsvOperator(BatchMetadataOperator):
    """Write scientific frame metadata from a DeepStream batch or reconciled source ledger."""

    HEADERS = [
        "stream_id",
        "camera_serial",
        "deepstream_frame_number",
        "source_sequence_index",
        "source",
        "raw_frame_index",
        "pts_ns",
        "dts_ns",
        "duration_ns",
        "host_monotonic_ns",
        "host_unix_ns",
        "status",
        "camera_frame_id",
        "camera_frame_id_available",
        "stream_frame_id",
        "chunk_frame_id",
        "frame_id_delta_consistent",
        "missing_frames_before",
        "pipeline_missing_frames_before",
        "camera_timestamp_ns",
        "chunk_timestamp_raw",
        "timestamp_increment_ns",
        "gst_pts_ns",
        "timestamp_origin",
        "host_received_monotonic_ns",
        "host_received_unix_ns",
        "copy_complete_monotonic_ns",
        "observer_monotonic_ns",
        "exposure_us",
        "gain_db",
        "black_level",
        "payload_crc_valid",
        "image_status",
        "source_width",
        "source_height",
        "source_pixel_format",
        "metadata_status",
        "inference_admitted",
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
    ):
        super().__init__()
        self.path = path
        self.meta_type = _flir_frame_meta_type() if meta_type is None else int(meta_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="", buffering=1)
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
                "w", newline="", buffering=1
            )
            self._telemetry_writer = csv.writer(self._telemetry_file)
            self._telemetry_writer.writerow(self.TELEMETRY_HEADERS)
            self._events_file = (self._audit_dir / "errors.csv").open(
                "w", newline="", buffering=1
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
            items = list(frame_meta.user_meta_items(self.meta_type))
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
        serial = str(payload.get("camera_serial") or f"index:{payload.get('camera_index', '')}")
        if self._runtime_path is None or not payload or serial in self._camera_runtime:
            return
        self._camera_runtime[serial] = {
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
                event_type, stream_id,
                expected_frame_id, actual_frame_id,
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
                    self._value(payload, "host_received_monotonic_ns"),
                    stream_id,
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
                "camera_frame_gap",
                stream_id=stream_id,
                expected_frame_id=camera_frame_id - camera_missing,
                actual_frame_id=camera_frame_id,
                details={"missing_frames": camera_missing},
                **event_clocks,
            )
        if isinstance(pipeline_missing, int) and pipeline_missing > 0:
            source_sequence = payload.get("source_sequence_index")
            self._event(
                "pipeline_frame_gap",
                stream_id=stream_id,
                expected_frame_id=(source_sequence - pipeline_missing) if isinstance(source_sequence, int) else "",
                actual_frame_id=source_sequence if isinstance(source_sequence, int) else "",
                details={"missing_frames": pipeline_missing},
                **event_clocks,
            )
        if payload.get("crc_valid") is False:
            self._event(
                "payload_crc_failure",
                stream_id=stream_id,
                actual_frame_id=camera_frame_id if isinstance(camera_frame_id, int) else "",
                **event_clocks,
            )
        if not status.startswith("ok"):
            self._event("frame_metadata_" + status, stream_id=stream_id, **event_clocks)

    def handle_metadata(self, batch_meta) -> None:
        with self._lock:
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
                        stream_id,
                        self._value(payload, "camera_serial"),
                        int(frame_meta.frame_number),
                        self._value(payload, "source_sequence_index"),
                        f"flirspinsrc:{stream_id}",
                        self._value(payload, "source_sequence_index"),
                        pts_ns or self._value(payload, "gst_pts_ns"),
                        pts_ns or self._value(payload, "gst_pts_ns"),
                        (
                            int(round(1_000_000_000 / float(payload["actual_fps"])))
                            if payload.get("actual_fps") else ""
                        ),
                        self._value(payload, "host_received_monotonic_ns"),
                        self._value(payload, "host_received_unix_ns"),
                        "ok" if status.startswith("ok") else status,
                        "" if camera_frame_id is None else camera_frame_id,
                        int(camera_frame_id is not None),
                        self._value(payload, "stream_frame_id"),
                        self._value(payload, "chunk_frame_id"),
                        self._value(payload, "frame_id_delta_consistent"),
                        self._value(payload, "missing_frames_before"),
                        pipeline_missing,
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
                        self._value(payload, "source_pixel_format"),
                        status,
                        self._value(payload, "inference_admitted"),
                    ]
                )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._file.flush()
            self._file.close()
            if self._telemetry_file is not None:
                self._telemetry_file.flush()
                self._telemetry_file.close()
            if self._events_file is not None:
                self._events_file.flush()
                self._events_file.close()


class ServiceMakerApp:
    """Own and run the PyServiceMaker SqueakView pipeline."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        pipeline_factory: Callable[[str], Pipeline] = Pipeline,
        probe_factory: Callable[[str, object], Probe] = Probe,
    ):
        _validate_config(config)
        self.config = config
        self.pipeline_factory = pipeline_factory
        self.probe_factory = probe_factory
        if config.run_dir is None:
            self.run_dir = run_context.timestamped_run_dir("ds")
        else:
            self.run_dir = Path(config.run_dir).expanduser()
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = run_context.run_artifacts(self.run_dir)
        self.pipeline: Pipeline | None = None
        self.observations: ObservationOperator | None = None
        self.frames: FrameCsvOperator | None = None
        self.record_admissions: list[RecordingAdmissionOperator] = []
        self.record_telemetry: list[RecordingPathTelemetry] = []
        self._ready = False
        self._stopped = False
        self.exit_code = 0
        self._stop_event = threading.Event()

    def _recording_fault(self, message: str) -> None:
        if self.exit_code == 0:
            self.exit_code = 4
            print(f"[{ts()}] [RECORD] FATAL: {message}", flush=True)
        self._stop_event.set()

    def _prewarm_cuda(self) -> None:
        if not self.config.enable_infer:
            return
        started = time.monotonic()
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA is unavailable")
            probe = torch.arange(16, device="cuda", dtype=torch.float32)
            _ = probe.to(device="cpu")
            torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(f"CUDA prewarm failed before camera acquisition: {exc}") from exc
        elapsed = time.monotonic() - started
        print(
            f"[{ts()}] [CUDA] prewarm complete in {elapsed:.2f}s before acquisition",
            flush=True,
        )

    @staticmethod
    def _video_frame_probe(path: Path) -> dict[str, object]:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None or not path.is_file():
            reason = "ffprobe is unavailable" if ffprobe is None else "video file is missing"
            return {"count": None, "method": None, "error": reason}

        attempts = (
            (
                "container_nb_frames",
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_frames",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                10,
            ),
            (
                "decoded_nb_read_frames",
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                60,
            ),
        )
        errors: list[str] = []
        for method, command, timeout_s in attempts:
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                errors.append(f"{method} timed out after {timeout_s}s")
                continue
            output = result.stdout.strip()
            if result.returncode == 0:
                try:
                    return {"count": int(output), "method": method, "error": None}
                except ValueError:
                    errors.append(f"{method} returned {output or 'no frame count'}")
                    continue
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            errors.append(f"{method} failed: {detail}")
        return {"count": None, "method": None, "error": "; ".join(errors)}

    @classmethod
    def _video_frame_count(cls, path: Path) -> int | None:
        count = cls._video_frame_probe(path).get("count")
        return int(count) if count is not None else None

    def _camera_properties(self, index: int) -> dict[str, object]:
        cfg = self.config
        properties: dict[str, object] = {
            "camera-index": index,
            "width": int(cfg.width),
            "height": int(cfg.height),
            "fps": int(cfg.fps),
            "pixel-format": _flir_pixel_format(cfg.pixel_format),
            "trigger": bool(cfg.trigger_on),
            "trigger-activation": (
                "falling" if str(cfg.trigger_activation).lower().startswith("fall") else "rising"
            ),
            "exposure-us": -1.0 if cfg.exposure_us is None else float(cfg.exposure_us),
            "gain": -1.0 if cfg.gain is None else float(cfg.gain),
            "drop-incomplete": False,
            "buffer-handling": "OldestFirst",
            "stream-buffer-count": max(64, int(cfg.fps) * 2),
            "capture-log-path": str(self.run_dir / f"capture_cam{index}.jsonl"),
            "metadata-profile": "scientific",
            "max-consecutive-timeouts": 0 if cfg.trigger_on else 10,
        }
        if cfg.camera_serials:
            properties["camera-serial"] = cfg.camera_serials[index]
        return properties

    def _add_camera(self, pipeline: Pipeline, index: int) -> None:
        cfg = self.config
        source = f"flirsrc{index}"
        source_caps = f"source_caps{index}"
        tee = f"camera_tee{index}"
        record_queue = f"record_queue{index}"
        infer_queue = f"infer_queue{index}"
        infer_caps = f"infer_caps{index}"
        raw_path = self.artifacts.raw_video if index == 0 else self.run_dir / f"raw_cam{index}.mp4"

        pipeline.add("flirspinsrc", source, self._camera_properties(index))
        pipeline.add(
            "capsfilter",
            source_caps,
            {
                "caps": (
                    f"video/x-raw,format=GRAY8,width={cfg.width},height={cfg.height},"
                    f"framerate={cfg.fps}/1"
                )
            },
        )
        pipeline.add("tee", tee).link(source, source_caps, tee)

        pipeline.add(
            "queue",
            record_queue,
            {
                "max-size-buffers": max(120, int(cfg.fps) * RECORD_QUEUE_SECONDS),
                "max-size-bytes": 0,
                "max-size-time": 0,
            },
        )
        admission_path = (
            self.run_dir / "record_admission.csv"
            if index == 0 else self.run_dir / f"record_admission_cam{index}.csv"
        )
        diagnostics_dir = self.run_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = diagnostics_dir / (
            "recording.csv" if index == 0 else f"recording_cam{index}.csv"
        )
        telemetry = RecordingPathTelemetry(
            telemetry_path,
            index,
            warning_depth=max(24, int(cfg.fps) * RECORD_BACKPRESSURE_WARNING_SECONDS),
            fatal_depth=max(90, int(cfg.fps) * RECORD_BACKPRESSURE_FATAL_SECONDS),
            on_fatal=self._recording_fault,
        )
        self.record_telemetry.append(telemetry)
        pipeline.attach(
            source_caps,
            self.probe_factory(
                f"record_ingress{index}", RecordingIngressOperator(telemetry)
            ),
        )
        admission = RecordingAdmissionOperator(admission_path, index, telemetry)
        self.record_admissions.append(admission)
        pipeline.attach(record_queue, self.probe_factory(f"record_admission{index}", admission))
        pipeline.add(
            "x264enc",
            f"record_encoder{index}",
            {
                "tune": 4,  # GstX264EncTune.ZEROLATENCY
                "speed-preset": 1,  # GstX264EncPreset.ULTRAFAST
                "bitrate": int(cfg.bitrate),
                "key-int-max": int(cfg.fps),
                "bframes": 0,
                "rc-lookahead": 0,
                "sync-lookahead": 0,
                "sliced-threads": False,
                "vbv-buf-capacity": 100,
            },
        )
        pipeline.add("h264parse", f"record_parser{index}")
        pipeline.attach(
            f"record_parser{index}",
            self.probe_factory(
                f"record_egress{index}", RecordingEgressOperator(telemetry)
            ),
        )
        pipeline.add("mp4mux", f"record_muxer{index}")
        pipeline.add("filesink", f"record_sink{index}", {"location": str(raw_path)})
        pipeline.link(
            tee,
            record_queue,
            f"record_encoder{index}",
            f"record_parser{index}",
            f"record_muxer{index}",
            f"record_sink{index}",
        )

        pipeline.add(
            "queue",
            infer_queue,
            {
                "max-size-buffers": 32,
                "max-size-bytes": 0,
                "max-size-time": 0,
                "leaky": 2,
            },
        )
        pipeline.add("nvvideoconvert", f"infer_convert{index}", {"compute-hw": 2, "copy-hw": 2})
        pipeline.add(
            "capsfilter",
            infer_caps,
            {
                "caps": (
                    f"video/x-raw(memory:NVMM),format=NV12,width={cfg.width},"
                    f"height={cfg.height}"
                )
            },
        )
        pipeline.link(tee, infer_queue, f"infer_convert{index}", infer_caps)
        pipeline.link((infer_caps, "mux"), ("", "sink_%u"))

    def build(self) -> Pipeline:
        cfg = self.config
        pipeline = self.pipeline_factory("squeakview")
        pipeline.add(
            "nvstreammux",
            "mux",
            {
                "batch-size": int(cfg.num_cameras),
                "width": int(cfg.width),
                "height": int(cfg.height),
                "live-source": True,
                "batched-push-timeout": max(10_000, int(1_000_000 / int(cfg.fps))),
                "sync-inputs": bool(cfg.num_cameras > 1),
            },
        )
        for index in range(cfg.num_cameras):
            self._add_camera(pipeline, index)

        inference_dir = self.run_dir / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)
        self.frames = FrameCsvOperator(
            inference_dir / "frames.csv",
            write_audit_sidecars=False,
        )
        pipeline.attach("mux", self.probe_factory("frames", self.frames))

        tail = ["mux"]
        if cfg.enable_infer:
            class_names = _load_class_names(Path(cfg.cfg_path))
            pose_schema = load_pose_schema(Path(cfg.cfg_path), class_names)
            pose_store = FramePoseStore()
            pipeline.add(
                "nvinfer",
                "infer",
                {
                    "config-file-path": str(Path(cfg.cfg_path).resolve()),
                    "batch-size": cfg.num_cameras,
                    "filter-out-class-ids": ";".join(
                        str(item.class_id) for item in pose_schema.classes
                    ),
                },
            )
            tail.append("infer")
            tensor_operator = Yolo26PoseTensorOperator(pose_schema, pose_store)
            pipeline.attach("infer", self.probe_factory("yolo26_pose", tensor_operator))

            tracker_config = Path(__file__).resolve().parents[3] / "configs" / "tracker_mouse_nvdcf.yml"
            pipeline.add(
                "nvtracker",
                "tracker",
                {
                    "tracker-width": 640,
                    "tracker-height": 480,
                    "ll-lib-file": (
                        "/opt/nvidia/deepstream/deepstream/lib/"
                        "libnvds_nvmultiobjecttracker.so"
                    ),
                    "ll-config-file": str(tracker_config),
                    "operate-on-class-ids": ";".join(
                        str(item.class_id) for item in pose_schema.classes if item.track
                    ),
                    "display-tracking-id": False,
                    "tracking-id-reset-mode": 3,
                },
            )
            tail.append("tracker")
            self.observations = ObservationOperator(
                self.run_dir,
                pose_schema,
                store=pose_store,
                flir_meta_type=self.frames.meta_type,
            )
            pipeline.attach("tracker", self.probe_factory("observations", self.observations))
        if not cfg.preview_sockets:
            pipeline.add("fakesink", "sink", {"sync": False})
            tail.append("sink")
            pipeline.link(*tail)
            self._ready_origin = "sink"
        else:
            pipeline.add("nvosdbin", "osd")
            tail.append("osd")
            pipeline.add("nvstreamdemux", "preview_demux")
            tail.append("preview_demux")
            pipeline.link(*tail)
            for index, socket_path in enumerate(cfg.preview_sockets):
                queue_name = f"preview_queue{index}"
                sink_name = f"preview_sink{index}"
                pipeline.add(
                    "queue",
                    queue_name,
                    {
                        "leaky": 2,
                        "max-size-buffers": 1,
                        "max-size-bytes": 0,
                        "max-size-time": 0,
                    },
                )
                pipeline.add(
                    "nvunixfdsink",
                    sink_name,
                    {
                        "socket-path": str(socket_path),
                        "sync": False,
                        "async": False,
                        "buffer-timestamp-copy": True,
                        "qos": False,
                    },
                )
                pipeline.link(("preview_demux", queue_name), (f"src_{index}", ""))
                pipeline.link(queue_name, sink_name)
            self._ready_origin = "preview_sink0"
        self.pipeline = pipeline
        print(
            f"[{ts()}] [INFO] PyServiceMaker pipeline built: cameras={cfg.num_cameras} "
            f"inference={'on' if cfg.enable_infer else 'off'} run_dir={self.run_dir}",
            flush=True,
        )
        return pipeline

    def _on_message(self, message) -> None:
        if (
            isinstance(message, StateTransitionMessage)
            and message.new_state == PipelineState.PLAYING
            and message.origin == self._ready_origin
            and not self._ready
        ):
            self._ready = True
            print(f"[{ts()}] [READY] inference playing", flush=True)
        elif isinstance(message, EOSMessage):
            self._stop_event.set()

    def run(self) -> int:
        if self.pipeline is None:
            self.build()
        assert self.pipeline is not None
        try:
            self._prewarm_cuda()
            self.pipeline.start(self._on_message)
            while not self._stop_event.wait(0.2):
                pass
        except KeyboardInterrupt:
            print(f"[{ts()}] [INFO] Ctrl-C; stopping PyServiceMaker pipeline", flush=True)
        except Exception as exc:
            self.exit_code = 1
            print(f"[{ts()}] [FATAL] PyServiceMaker pipeline failed: {exc}", flush=True)
        finally:
            self.stop()
        return self.exit_code

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                self.pipeline.wait()
            except Exception:
                self.exit_code = 1
        if self.observations is not None:
            self.observations.close()
        for admission in self.record_admissions:
            admission.close()
        for telemetry in self.record_telemetry:
            telemetry.close()
        if self.frames is not None:
            self.frames.close()
        try:
            run_context.write_status(
                self.run_dir,
                "capture_closed",
                capture_exit_code=self.exit_code,
            )
        except Exception as exc:
            self.exit_code = 1
            print(f"[{ts()}] [RECORD] ERROR marking capture closed: {exc}", flush=True)

        print(
            f"[{ts()}] [CAPTURE] closed; post-run audit is handled independently",
            flush=True,
        )
        print(f"[{ts()}] [INFO] done. Files in: {self.run_dir}", flush=True)

def run(config: InferenceConfig) -> int:
    app = ServiceMakerApp(config)

    def _handle_signal(sig_num, _frame) -> None:
        print(f"[{ts()}] [SIG] {signal.Signals(sig_num).name}; stopping", flush=True)
        app.request_stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    app.build()
    return app.run()
