"""DeepStream 9.1 inference runner built on the PyServiceMaker Pipeline API."""
from __future__ import annotations

import atexit
from collections import Counter
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
from types import SimpleNamespace
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

    def __init__(self, path: Path, stream_id: int):
        super().__init__()
        self.path = path
        self.stream_id = int(stream_id)
        self._file = path.open("w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADERS)
        self._count = 0
        self._lock = threading.Lock()
        self._closed = False
        atexit.register(self.close)

    def handle_buffer(self, buffer) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._writer.writerow(
                [self.stream_id, self._count, int(buffer.timestamp), time.monotonic_ns()]
            )
            self._count += 1
        return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._file.flush()
            self._file.close()


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
        "metadata_json",
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

    def __init__(self, path: Path, *, meta_type: int | None = None):
        super().__init__()
        self.path = path
        self.meta_type = _flir_frame_meta_type() if meta_type is None else int(meta_type)
        self._file = path.open("w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADERS)
        self._lock = threading.Lock()
        self._closed = False
        self._runtime_path = path.with_name("camera_runtime.json")
        self._camera_runtime: dict[str, dict] = {}
        self._last_source_sequence: dict[int, int] = {}
        self._telemetry_file = path.with_name("camera_telemetry.csv").open(
            "w", newline="", buffering=1
        )
        self._telemetry_writer = csv.writer(self._telemetry_file)
        self._telemetry_writer.writerow(self.TELEMETRY_HEADERS)
        self._events_file = path.with_name("drop_events.csv").open("w", newline="", buffering=1)
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
        if not payload or serial in self._camera_runtime:
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
    ) -> None:
        self._events_writer.writerow(
            [
                time.time_ns(), time.monotonic_ns(), event_type, stream_id,
                expected_frame_id, actual_frame_id,
                json.dumps(details or {}, sort_keys=True, separators=(",", ":")),
            ]
        )

    def _write_audit_rows(
        self, payload: dict, status: str, stream_id: int, pipeline_missing
    ) -> None:
        if payload.get("telemetry_sample"):
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
            )
        if isinstance(pipeline_missing, int) and pipeline_missing > 0:
            source_sequence = payload.get("source_sequence_index")
            self._event(
                "pipeline_frame_gap",
                stream_id=stream_id,
                expected_frame_id=(source_sequence - pipeline_missing) if isinstance(source_sequence, int) else "",
                actual_frame_id=source_sequence if isinstance(source_sequence, int) else "",
                details={"missing_frames": pipeline_missing},
            )
        if payload.get("crc_valid") is False:
            self._event(
                "payload_crc_failure",
                stream_id=stream_id,
                actual_frame_id=camera_frame_id if isinstance(camera_frame_id, int) else "",
            )
        if not status.startswith("ok"):
            self._event("frame_metadata_" + status, stream_id=stream_id)

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
                        json.dumps(payload, sort_keys=True, separators=(",", ":")) if payload else "",
                    ]
                )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._file.flush()
            self._file.close()
            self._telemetry_file.flush()
            self._telemetry_file.close()
            self._events_file.flush()
            self._events_file.close()


def _load_capture_payloads(run_dir: Path, camera_count: int) -> list[dict]:
    payloads: list[dict] = []
    for index in range(camera_count):
        path = run_dir / f"capture_cam{index}.jsonl"
        if not path.exists():
            continue
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid capture ledger {path.name}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise RuntimeError(f"invalid capture ledger object in {path.name}:{line_number}")
                payload.setdefault("camera_index", index)
                payloads.append(payload)
    payloads.sort(
        key=lambda item: (int(item.get("host_received_monotonic_ns") or 0), int(item.get("camera_index") or 0))
    )
    return payloads

def _load_record_admissions(run_dir: Path, camera_count: int) -> tuple[Counter, dict[int, int]]:
    admissions: Counter = Counter()
    counts: dict[int, int] = {}
    for index in range(camera_count):
        path = run_dir / ("record_admission.csv" if index == 0 else f"record_admission_cam{index}.csv")
        counts[index] = 0
        if not path.exists():
            raise RuntimeError(f"recording admission ledger is missing: {path.name}")
        with path.open(newline="") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), 2):
                try:
                    pts_ns = int(row["pts_ns"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"invalid recording admission {path.name}:{line_number}") from exc
                admissions[(index, pts_ns)] += 1
                counts[index] += 1
    return admissions, counts


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
        self._ready = False
        self._stopped = False
        self.exit_code = 0
        self._stop_event = threading.Event()

    def _finalize_capture_manifest(self) -> list[dict]:
        payloads = _load_capture_payloads(self.run_dir, self.config.num_cameras)
        if not payloads:
            print(f"[{ts()}] [WARN] no source capture ledger was produced", flush=True)
            return []
        admissions, admission_counts = _load_record_admissions(
            self.run_dir, self.config.num_cameras
        )
        source_counts = {stream_id: 0 for stream_id in range(self.config.num_cameras)}
        for payload in payloads:
            stream_id = int(payload.get("camera_index") or 0)
            source_counts[stream_id] = source_counts.get(stream_id, 0) + 1
        recorded: list[dict] = []
        for payload in payloads:
            stream_id = int(payload.get("camera_index") or 0)
            pts_ns = int(payload.get("gst_pts_ns") or 0)
            key = (stream_id, pts_ns)
            if admissions[key] <= 0:
                continue
            admissions[key] -= 1
            recorded.append(payload)
        unmatched = sum(admissions.values())
        if unmatched:
            raise RuntimeError(f"{unmatched} recording admissions have no matching source metadata")
        reconciliation = {
            "schema_version": "1.0",
            "source_frames": source_counts,
            "record_admitted_frames": admission_counts,
            "source_not_recorded_frames": {
                stream_id: source_counts[stream_id] - admission_counts.get(stream_id, 0)
                for stream_id in source_counts
            },
            "policy": "frames.csv contains only buffers admitted to the non-leaky recording branch",
        }
        run_context.atomic_write_json(self.run_dir / "capture_reconciliation.json", reconciliation)
        payloads = recorded
        operator = FrameCsvOperator(self.artifacts.frames_csv, meta_type=0)
        try:
            for payload in payloads:
                stream_id = int(payload.get("camera_index") or 0)
                sequence = int(payload.get("source_sequence_index") or 0)
                pts_ns = int(payload.get("gst_pts_ns") or 0)
                user_meta = SimpleNamespace(
                    get_user_data_json=lambda payload=payload: payload
                )
                frame_meta = SimpleNamespace(
                    frame_number=sequence,
                    source_id=stream_id,
                    pad_index=stream_id,
                    buffer_pts=pts_ns,
                    user_meta_items=lambda _meta_type, user_meta=user_meta: iter([user_meta]),
                )
                operator.handle_metadata(SimpleNamespace(frame_items=[frame_meta]))
        finally:
            operator.close()
        print(
            f"[{ts()}] [CAPTURE] finalized {len(payloads)} record-admitted frames",
            flush=True,
        )
        return payloads

    def _write_inference_admission(self, payloads: list[dict]) -> None:
        if not self.config.enable_infer or not payloads:
            return
        ledger_path = self.run_dir / "inference" / "frames.csv"
        admitted: set[tuple[int, int]] = set()
        if ledger_path.exists():
            with ledger_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        admitted.add((int(row["stream_id"]), int(row["source_sequence_index"])))
                    except (KeyError, TypeError, ValueError):
                        continue
        output_path = self.run_dir / "inference_admission.csv"
        with output_path.open("w", newline="") as handle:
            fields = ["stream_id", "source_sequence_index", "camera_frame_id", "admitted"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for payload in payloads:
                stream_id = int(payload.get("camera_index") or 0)
                sequence = int(payload.get("source_sequence_index") or 0)
                writer.writerow(
                    {
                        "stream_id": stream_id,
                        "source_sequence_index": sequence,
                        "camera_frame_id": payload.get("camera_frame_id"),
                        "admitted": int((stream_id, sequence) in admitted),
                    }
                )
        capture_counts = {stream_id: 0 for stream_id in range(self.config.num_cameras)}
        for payload in payloads:
            stream_id = int(payload.get("camera_index") or 0)
            capture_counts[stream_id] = capture_counts.get(stream_id, 0) + 1
        admitted_counts = {stream_id: 0 for stream_id in capture_counts}
        for stream_id, _sequence in admitted:
            admitted_counts[stream_id] = admitted_counts.get(stream_id, 0) + 1
        summary = {
            "schema_version": "1.0",
            "policy": "capture_non_leaky_inference_leaky_downstream",
            "captured_frames": capture_counts,
            "inference_admitted_frames": admitted_counts,
            "inference_skipped_frames": {
                stream_id: capture_counts[stream_id] - admitted_counts.get(stream_id, 0)
                for stream_id in capture_counts
            },
        }
        run_context.atomic_write_json(self.run_dir / "inference_admission.json", summary)
        print(
            f"[{ts()}] [INFER] admission audit: {summary['inference_skipped_frames']}",
            flush=True,
        )

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
    def _video_frame_count(path: Path) -> int | None:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None or not path.is_file():
            return None
        result = subprocess.run(
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
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if result.returncode != 0:
            return None
        try:
            return int(result.stdout.strip())
        except ValueError:
            return None

    def _validate_recordings(self, payloads: list[dict]) -> None:
        if not payloads:
            return
        captured = {stream_id: 0 for stream_id in range(self.config.num_cameras)}
        for payload in payloads:
            stream_id = int(payload.get("camera_index") or 0)
            captured[stream_id] = captured.get(stream_id, 0) + 1
        cameras = []
        failed = False
        for stream_id in range(self.config.num_cameras):
            video_path = self.artifacts.raw_video if stream_id == 0 else self.run_dir / f"raw_cam{stream_id}.mp4"
            source_frames = captured.get(stream_id, 0)
            video_frames = self._video_frame_count(video_path)
            exists = video_path.is_file() and video_path.stat().st_size > 0
            matches = video_frames == source_frames if video_frames is not None else None
            if not exists or matches is False:
                failed = True
            cameras.append(
                {
                    "stream_id": stream_id,
                    "video": video_path.name,
                    "exists": exists,
                    "source_frames": source_frames,
                    "record_admitted_frames": source_frames,
                    "video_frames": video_frames,
                    "frame_count_matches": matches,
                }
            )
        report = {
            "schema_version": "1.0",
            "policy": "every_record_admitted_frame_must_be_present_in_ground_truth_video",
            "cameras": cameras,
            "passed": not failed and all(item["frame_count_matches"] is True for item in cameras),
        }
        run_context.atomic_write_json(self.run_dir / "recording_validation.json", report)
        level = "ERROR" if failed else ("PASS" if report["passed"] else "WARN")
        print(f"[{ts()}] [RECORD] {level} validation: {cameras}", flush=True)
        if failed:
            self.exit_code = 1

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
            {"max-size-buffers": 30, "max-size-bytes": 0, "max-size-time": 0},
        )
        admission_path = (
            self.run_dir / "record_admission.csv"
            if index == 0 else self.run_dir / f"record_admission_cam{index}.csv"
        )
        admission = RecordingAdmissionOperator(admission_path, index)
        self.record_admissions.append(admission)
        pipeline.attach(record_queue, self.probe_factory(f"record_admission{index}", admission))
        pipeline.add("videoconvert", f"record_convert{index}")
        pipeline.add(
            "capsfilter",
            f"record_caps{index}",
            {"caps": f"video/x-raw,format=I420,width={cfg.width},height={cfg.height}"},
        )
        pipeline.add(
            "x264enc",
            f"record_encoder{index}",
            {
                "tune": 4,  # GstX264EncTune.ZEROLATENCY
                "speed-preset": 1,  # GstX264EncPreset.ULTRAFAST
                "bitrate": int(cfg.bitrate),
                "key-int-max": int(cfg.fps),
            },
        )
        pipeline.add("h264parse", f"record_parser{index}")
        pipeline.add("mp4mux", f"record_muxer{index}")
        pipeline.add("filesink", f"record_sink{index}", {"location": str(raw_path)})
        pipeline.link(
            tee,
            record_queue,
            f"record_convert{index}",
            f"record_caps{index}",
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
        pipeline.add("nvvideoconvert", f"infer_convert{index}", {"compute-hw": 1, "copy-hw": 2})
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
        self.frames = FrameCsvOperator(inference_dir / "frames.csv")
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
        pipeline.add("nvosdbin", "osd")
        tail.append("osd")
        if not cfg.preview_sockets:
            pipeline.add("fakesink", "sink", {"sync": False})
            tail.append("sink")
            pipeline.link(*tail)
            self._ready_origin = "sink"
        else:
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
        if self.frames is not None:
            self.frames.close()
        try:
            payloads = self._finalize_capture_manifest()
            self._write_inference_admission(payloads)
            self._validate_recordings(payloads)
        except Exception as exc:
            self.exit_code = 1
            print(f"[{ts()}] [RECORD] ERROR finalizing capture audit: {exc}", flush=True)

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
