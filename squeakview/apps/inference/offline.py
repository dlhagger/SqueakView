"""Offline DeepStream replay of an immutable SqueakView ground-truth recording."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pyservicemaker import BatchMetadataOperator, EOSMessage, Pipeline, Probe

from squeakview.apps.inference.pose_pipeline import (
    FramePoseStore,
    ObservationOperator,
    Yolo26PoseTensorOperator,
    load_pose_schema,
)
from squeakview.apps.inference.contracts import load_class_names
from squeakview.model_package import validate_model_package
from squeakview.common.bounded_input import read_json_object
from squeakview.common.bounded_csv import bounded_csv_lines
from squeakview.common.diagnostics.evidence_identity import stable_file_identity


MAX_OFFLINE_METADATA_BYTES = 16 * 1024 * 1024
MAX_OFFLINE_LEDGER_LINE_BYTES = 64 * 1024
OFFLINE_SHUTDOWN_TIMEOUT_S = 45.0


def _sha256(path: Path) -> str:
    identity = stable_file_identity(path)
    digest = identity.get("sha256")
    if identity.get("available") is not True or not isinstance(digest, str):
        raise RuntimeError(
            f"could not identify stable input {path}: {identity.get('error')}"
        )
    return digest


def _read_json(path: Path) -> dict[str, Any]:
    return read_json_object(
        path, max_bytes=MAX_OFFLINE_METADATA_BYTES, label=path.name
    )


def _assert_direct_run_file(run_dir: Path, path: Path) -> None:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"offline replay input is unavailable: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode) or not resolved.is_relative_to(run_dir):
        raise ValueError(
            f"offline replay input must be a direct run-local regular file: {path}"
        )


def _int_value(row: dict[str, str], name: str) -> int | None:
    raw = str(row.get(name, "")).strip()
    if not raw:
        return None
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"frames.csv field {name} must be an unsigned integer")
    return int(raw)


def _terminate_process_group(worker: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(worker.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        worker.wait(timeout=2.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(worker.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        worker.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def _bounded_csv_lines(path: Path) -> Iterator[str]:
    with path.open("rb") as handle:
        while True:
            raw = handle.readline(MAX_OFFLINE_LEDGER_LINE_BYTES + 1)
            if not raw:
                return
            if len(raw) > MAX_OFFLINE_LEDGER_LINE_BYTES:
                raise ValueError(
                    f"frame ledger record exceeds {MAX_OFFLINE_LEDGER_LINE_BYTES} bytes"
                )
            try:
                yield raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ValueError("frame ledger is not valid UTF-8") from exc


class OfflineFrameLedger(Mapping[int, dict[str, int | None]]):
    """Disk-backed ordinal lookup for multi-day offline replay ledgers."""

    def __init__(self) -> None:
        # An empty filename gives SQLite a private temporary on-disk database.
        self._connection = sqlite3.connect("", check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.execute("PRAGMA temp_store=FILE")
        self._connection.execute(
            "CREATE TABLE frames("
            "ordinal INTEGER PRIMARY KEY, source_sequence_index INTEGER, "
            "raw_frame_index INTEGER, camera_frame_id INTEGER, "
            "camera_timestamp_ns INTEGER, pts_ns INTEGER)"
        )
        self._count = 0
        self._lock = threading.Lock()
        self._closed = False

    def insert_many(
        self,
        rows: list[
            tuple[int, int | None, int | None, int | None, int | None, int | None]
        ],
    ) -> None:
        with self._connection:
            self._connection.executemany(
                "INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?)", rows
            )
        self._count += len(rows)

    def __getitem__(self, ordinal: int) -> dict[str, int | None]:
        with self._lock:
            row = self._connection.execute(
                "SELECT source_sequence_index, raw_frame_index, camera_frame_id, "
                "camera_timestamp_ns, pts_ns FROM frames WHERE ordinal=?",
                (int(ordinal),),
            ).fetchone()
        if row is None:
            raise KeyError(ordinal)
        return dict(
            zip(
                ("source_sequence_index", "raw_frame_index", "camera_frame_id", "camera_timestamp_ns", "pts_ns"),
                row,
                strict=True,
            )
        )

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._count))

    def __len__(self) -> int:
        return self._count

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True


def _load_frame_ledger(path: Path) -> tuple[OfflineFrameLedger, int, int]:
    ledger = OfflineFrameLedger()
    width: int | None = None
    height: int | None = None
    batch: list[
        tuple[int, int | None, int | None, int | None, int | None, int | None]
    ] = []
    previous_sequence: int | None = None
    previous_camera_frame_id: int | None = None
    previous_camera_timestamp: int | None = None
    previous_pts: int | None = None
    try:
        reader = csv.DictReader(_bounded_csv_lines(path))
        required = {
            "stream_id", "source_sequence_index", "raw_frame_index",
            "camera_frame_id", "camera_timestamp_ns", "pts_ns",
            "source_width", "source_height",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "frames.csv is missing required columns: " + ", ".join(sorted(missing))
            )
        if len(reader.fieldnames or ()) != len(set(reader.fieldnames or ())):
            raise ValueError("frames.csv contains duplicate column names")
        for ordinal, row in enumerate(reader):
            if None in row:
                raise ValueError("frames.csv row contains unexpected extra columns")
            if _int_value(row, "stream_id") != 0:
                raise ValueError(
                    "offline replay currently requires a single-camera frame ledger"
                )
            row_width = _int_value(row, "source_width")
            row_height = _int_value(row, "source_height")
            if not row_width or not row_height:
                raise ValueError(
                    "frames.csv must contain source_width and source_height for offline replay"
                )
            if width is None:
                width, height = row_width, row_height
            elif (row_width, row_height) != (width, height):
                raise ValueError("frames.csv source dimensions change during the run")
            source_sequence = _int_value(row, "source_sequence_index")
            raw_index = _int_value(row, "raw_frame_index")
            if source_sequence is None:
                raise ValueError("frames.csv row has no source frame identity")
            identity = source_sequence
            expected_sequence = 0 if previous_sequence is None else previous_sequence + 1
            if identity != expected_sequence:
                raise ValueError(
                    "frames.csv source sequence is not contiguous: expected "
                    f"{expected_sequence}, got {identity}"
                )
            if raw_index != ordinal:
                raise ValueError(
                    f"frames.csv raw frame index is not contiguous: expected "
                    f"{ordinal}, got {raw_index}"
                )
            camera_frame_id = _int_value(row, "camera_frame_id")
            camera_timestamp = _int_value(row, "camera_timestamp_ns")
            pts_ns = _int_value(row, "pts_ns")
            if camera_frame_id is None or camera_timestamp is None or pts_ns is None:
                raise ValueError(
                    "frames.csv row is missing camera FrameID, camera timestamp, or PTS"
                )
            if (
                previous_camera_frame_id is not None
                and camera_frame_id != previous_camera_frame_id + 1
            ):
                raise ValueError("frames.csv camera FrameIDs are not contiguous")
            if previous_camera_timestamp is not None and camera_timestamp <= previous_camera_timestamp:
                raise ValueError("frames.csv camera timestamps are not strictly increasing")
            if previous_pts is not None and pts_ns <= previous_pts:
                raise ValueError("frames.csv PTS values are not strictly increasing")
            previous_sequence = identity
            previous_camera_frame_id = camera_frame_id
            previous_camera_timestamp = camera_timestamp
            previous_pts = pts_ns
            batch.append(
                (
                    ordinal, identity, raw_index,
                    camera_frame_id, camera_timestamp, pts_ns,
                )
            )
            if len(batch) >= 10_000:
                ledger.insert_many(batch)
                batch.clear()
        if batch:
            ledger.insert_many(batch)
        if not len(ledger) or width is None or height is None:
            raise ValueError(f"frame ledger is empty: {path}")
        return ledger, width, height
    except Exception:
        ledger.close()
        raise


def _resolve_default_config(run_dir: Path) -> Path:
    manifest = _read_json(run_dir / "run_manifest.json")
    try:
        raw = manifest["inference"]["model_package"]["config"]
    except (KeyError, TypeError) as exc:
        raise ValueError("run manifest has no inference.model_package.config; pass --cfg") from exc
    return Path(str(raw)).expanduser().resolve()


class FrameAuditOperator(BatchMetadataOperator):
    """Assert that decoded frame ordinals are contiguous and count every replayed frame."""

    def __init__(self, expected: int):
        super().__init__()
        self.expected = int(expected)
        self.count = 0
        self._lock = threading.Lock()

    def handle_metadata(self, batch_meta) -> None:
        with self._lock:
            for frame_meta in batch_meta.frame_items:
                actual = int(frame_meta.frame_number)
                if actual != self.count:
                    raise RuntimeError(
                        f"offline decoded-frame sequence is not contiguous: expected {self.count}, got {actual}"
                    )
                if actual >= self.expected:
                    raise RuntimeError(
                        f"offline decoder produced frame {actual} beyond the {self.expected}-row ledger"
                    )
                self.count += 1


@dataclass(slots=True)
class OfflineConfig:
    run_dir: Path
    cfg_path: Path | None = None
    out_dir: Path | None = None


class OfflineInferenceApp:
    """Replay raw.mp4 through the live TensorRT decoder and NvDCF tracker."""

    def __init__(
        self,
        config: OfflineConfig,
        *,
        pipeline_factory=Pipeline,
        probe_factory=Probe,
    ):
        self.config = config
        self.run_dir = Path(config.run_dir).expanduser().resolve()
        self.video_path = self.run_dir / "raw.mp4"
        self.frames_path = self.run_dir / "frames.csv"
        self.serial_path = self.run_dir / "serial.csv"
        self.errors_path = self.run_dir / "diagnostics" / "errors.csv"
        for path in (self.video_path, self.frames_path, self.serial_path):
            _assert_direct_run_file(self.run_dir, path)
        if self.errors_path.exists():
            _assert_direct_run_file(self.run_dir, self.errors_path)
        self.cfg_path = (
            Path(config.cfg_path).expanduser().resolve()
            if config.cfg_path is not None
            else _resolve_default_config(self.run_dir)
        )
        self.model = validate_model_package(self.cfg_path)
        self.tracker_path = Path(__file__).resolve().parents[3] / "configs" / "tracker_mouse_nvdcf.yml"
        self.source_hashes = self._current_source_hashes()
        self.ledger, self.width, self.height = _load_frame_ledger(self.frames_path)
        try:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.out_dir = (
                Path(config.out_dir).expanduser().resolve()
                if config.out_dir is not None
                else self.run_dir / "offline_inference" / f"{stamp}_{self.model.name}"
            )
            if self.out_dir.exists() and any(self.out_dir.iterdir()):
                raise FileExistsError(f"refusing to overwrite non-empty output directory: {self.out_dir}")
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.pipeline_factory = pipeline_factory
            self.probe_factory = probe_factory
            self.pipeline = None
            self.observations = None
            self.audit = FrameAuditOperator(len(self.ledger))
            self.stop_event = threading.Event()
            self.eos_received = False
            self.exit_code = 0
            self.started_at = datetime.now().astimezone().isoformat()
            self._write_manifest("starting")
        except BaseException:
            self.ledger.close()
            raise

    def _write_manifest(self, status: str, error: str | None = None) -> None:
        outputs = {}
        for name in ("objects.csv", "keypoints.csv"):
            path = self.out_dir / name
            if path.exists():
                count = max(0, sum(1 for _ in bounded_csv_lines(path)) - 1)
                outputs[name] = {"rows": count, "sha256": _sha256(path)}
        payload = {
            "schema_version": 1,
            "status": status,
            "started_at": self.started_at,
            "finished_at": datetime.now().astimezone().isoformat() if status != "starting" else None,
            "source": {
                "run_dir": str(self.run_dir),
                "video": str(self.video_path),
                "video_sha256": self.source_hashes["video_sha256"],
                "frames": str(self.frames_path),
                "frames_sha256": self.source_hashes["frames_sha256"],
                "serial": str(self.serial_path),
                "serial_sha256": self.source_hashes["serial_sha256"],
                "errors": str(self.errors_path),
                "errors_sha256": self.source_hashes["errors_sha256"],
                "expected_frames": len(self.ledger),
            },
            "model_package": self.model.manifest_snapshot(),
            "runtime_artifacts": {
                "parser_library": str(self.model.parser_library),
                "parser_sha256": self.source_hashes["parser_sha256"],
                "tracker_config": str(self.tracker_path),
                "tracker_sha256": self.source_hashes["tracker_sha256"],
                "model_artifact_sha256": {
                    name.removeprefix("model_").removesuffix("_sha256"): digest
                    for name, digest in self.source_hashes.items()
                    if name.startswith("model_") and name.endswith("_sha256")
                },
            },
            "mapping": "decoded frame ordinal joined to authoritative frames.csv row ordinal",
            "decoded_frames": self.audit.count,
            "outputs": outputs,
            "error": error,
        }
        target = self.out_dir / "offline_manifest.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(target)

    def _current_source_hashes(self) -> dict[str, str | None]:
        hashes: dict[str, str | None] = {
            "video_sha256": _sha256(self.video_path),
            "frames_sha256": _sha256(self.frames_path),
            "serial_sha256": _sha256(self.serial_path),
            "errors_sha256": (
                _sha256(self.errors_path) if self.errors_path.is_file() else None
            ),
            "parser_sha256": _sha256(self.model.parser_library),
            "tracker_sha256": _sha256(self.tracker_path),
        }
        for name in (
            "config", "manifest", "pose_sidecar", "onnx", "engine", "classes",
            "keypoint_labels", "import_report",
        ):
            path = getattr(self.model, name, None)
            if isinstance(path, Path):
                hashes[f"model_{name}_sha256"] = _sha256(path)
        return hashes

    def build(self):
        pipeline = self.pipeline_factory("squeakview-offline")
        pipeline.add(
            "nvurisrcbin", "source",
            {"uri": self.video_path.as_uri(), "disable-audio": True, "file-loop": False,
             "dec-skip-frames": 0, "drop-frame-interval": 0, "leaky": 0,
             "drop-on-latency": False},
        )
        pipeline.add(
            "nvstreammux", "mux",
            {
                "batch-size": 1,
                "batched-push-timeout": 33_333,
                "sync-inputs": False,
                "max-latency": 0,
            },
        )
        # New nvstreammux intentionally performs no scaling or color
        # conversion.  Normalize replay inputs explicitly, matching the live
        # path, so offline inference sees the same geometry and NVMM format.
        pipeline.add(
            "nvvideoconvert",
            "offline_convert",
            {"compute-hw": 2, "copy-hw": 2, "nvbuf-memory-type": 4},
        )
        pipeline.add(
            "capsfilter",
            "offline_caps",
            {
                "caps": (
                    "video/x-raw(memory:NVMM),format=NV12,"
                    f"width={self.width},height={self.height}"
                )
            },
        )
        pipeline.link("source", "offline_convert", "offline_caps")
        pipeline.link(("offline_caps", "mux"), ("", "sink_%u"))
        pipeline.attach("mux", self.probe_factory("offline_frame_audit", self.audit))

        class_names = load_class_names(self.cfg_path)
        schema = load_pose_schema(self.cfg_path, class_names)
        store = FramePoseStore()
        pipeline.add(
            "nvinfer", "infer",
            {
                "config-file-path": str(self.cfg_path), "batch-size": 1,
                "filter-out-class-ids": ";".join(str(item.class_id) for item in schema.classes),
            },
        )
        pipeline.attach("infer", self.probe_factory("yolo26_pose", Yolo26PoseTensorOperator(schema, store)))
        pipeline.add(
            "nvtracker", "tracker",
            {
                "tracker-width": 640, "tracker-height": 480,
                "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
                "ll-config-file": str(self.tracker_path),
                "operate-on-class-ids": ";".join(
                    str(item.class_id) for item in schema.classes if item.track
                ),
                "display-tracking-id": False, "tracking-id-reset-mode": 3,
            },
        )
        self.observations = ObservationOperator(
            self.out_dir, schema, store=store, flir_meta_type=None,
            frame_ledger=self.ledger,
            mapping_method="offline_video_ledger", source_name="offline_raw_mp4",
        )
        pipeline.attach("tracker", self.probe_factory("observations", self.observations))
        pipeline.add("fakesink", "sink", {"sync": False, "async": False})
        pipeline.link("mux", "infer", "tracker", "sink")
        self.pipeline = pipeline
        return pipeline

    def _on_message(self, message) -> None:
        if isinstance(message, EOSMessage):
            self.eos_received = True
            self.stop_event.set()

    def request_stop(self) -> None:
        self.stop_event.set()

    def _run_impl(self) -> int:
        if self.pipeline is None:
            self.build()
        try:
            self.pipeline.start(self._on_message)
            while not self.stop_event.wait(0.2):
                pass
        except KeyboardInterrupt:
            self.exit_code = 130
            self._write_manifest("failed", "offline replay interrupted")
        except Exception as exc:
            self.exit_code = 1
            self._write_manifest("failed", str(exc))
            print(f"[OFFLINE] failed: {exc}", flush=True)
        finally:
            if self.pipeline is not None:
                try:
                    if self.eos_received:
                        self.pipeline.wait()
                    else:
                        self.pipeline.stop()
                        self.pipeline.wait()
                except Exception as exc:
                    self.exit_code = self.exit_code or 1
                    self._write_manifest("failed", str(exc))
            if self.observations is not None:
                try:
                    self.observations.close()
                except Exception as exc:
                    self.exit_code = self.exit_code or 1
                    self._write_manifest(
                        "failed", f"offline observation close failed: {exc}"
                    )

        if self.exit_code == 0 and not self.eos_received:
            self.exit_code = 1
            error = "offline replay stopped before decoder EOS"
            self._write_manifest("failed", error)
            print(f"[OFFLINE] {error}", flush=True)
        elif self.exit_code == 0 and self.audit.count != len(self.ledger):
            self.exit_code = 1
            error = f"decoded {self.audit.count} frames; expected {len(self.ledger)}"
            self._write_manifest("failed", error)
            print(f"[OFFLINE] {error}", flush=True)
        elif self.exit_code == 0:
            try:
                current_hashes = self._current_source_hashes()
                changed = sorted(
                    name
                    for name, digest in current_hashes.items()
                    if digest != self.source_hashes[name]
                )
                if changed:
                    raise RuntimeError(
                        "offline replay inputs changed during processing: "
                        + ", ".join(changed)
                    )
                from scripts.align_run_outputs_streaming import build_alignment
                build_alignment(
                    self.run_dir, self.out_dir,
                    objects_path=self.out_dir / "objects.csv",
                    video_validation={
                        "count": self.audit.count,
                        "method": "offline_pipeline_full_decode",
                        "error": None,
                    },
                )
                changed_after_alignment = sorted(
                    name
                    for name, digest in self._current_source_hashes().items()
                    if digest != self.source_hashes[name]
                )
                if changed_after_alignment:
                    raise RuntimeError(
                        "offline replay inputs changed during alignment: "
                        + ", ".join(changed_after_alignment)
                    )
            except Exception as exc:
                self.exit_code = 1
                self._write_manifest("failed", f"alignment failed: {exc}")
                print(f"[OFFLINE] alignment failed: {exc}", flush=True)
            else:
                self._write_manifest("complete")
                print(f"[OFFLINE] complete: {self.out_dir}", flush=True)
        return self.exit_code

    def run(self) -> int:
        """Run replay while guaranteeing release of the disk-backed ledger."""

        try:
            return self._run_impl()
        finally:
            self.ledger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a SqueakView raw.mp4 through the live DeepStream model and tracker"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--cfg", type=Path, default=None, help="Model config; defaults to run manifest model")
    parser.add_argument("--out-dir", type=Path, default=None, help="New, empty derived-output directory")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _worker_command(
    config: OfflineConfig, *, expected_parent_pid: int | None = None
) -> list[str]:
    parent_pid = os.getpid() if expected_parent_pid is None else expected_parent_pid
    command = [
        sys.executable,
        "-m",
        "squeakview.apps.operator.backend.parent_death_exec",
        "--expected-parent-pid",
        str(parent_pid),
        "--signal",
        str(int(signal.SIGKILL)),
        "--",
        sys.executable,
        "-m",
        "squeakview.apps.inference.offline",
        str(config.run_dir),
        "--worker",
    ]
    if config.cfg_path is not None:
        command.extend(("--cfg", str(config.cfg_path)))
    if config.out_dir is not None:
        command.extend(("--out-dir", str(config.out_dir)))
    return command


def _normalized_exit_code(returncode: int | None) -> int:
    if returncode is None:
        return 1
    return int(returncode) if returncode >= 0 else 128 + abs(int(returncode))


def _supervise_worker(
    config: OfflineConfig,
    *,
    shutdown_timeout_s: float = OFFLINE_SHUTDOWN_TIMEOUT_S,
    popen_factory=subprocess.Popen,
) -> int:
    """Own a hard process boundary around potentially blocking native teardown."""

    if (
        isinstance(shutdown_timeout_s, bool)
        or not isinstance(shutdown_timeout_s, (int, float))
        or not 1.0 <= float(shutdown_timeout_s) <= 300.0
    ):
        raise ValueError("offline shutdown timeout must be between 1 and 300 seconds")
    worker = popen_factory(_worker_command(config), start_new_session=True)
    shutdown_deadline: float | None = None

    def forward(signum, _frame) -> None:
        nonlocal shutdown_deadline
        if worker.poll() is not None:
            return
        if shutdown_deadline is None:
            shutdown_deadline = time.monotonic() + float(shutdown_timeout_s)
            forwarded = signal.SIGINT if signum == signal.SIGINT else signal.SIGTERM
        else:
            forwarded = signal.SIGKILL
        try:
            os.killpg(worker.pid, forwarded)
        except ProcessLookupError:
            pass

    previous = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous:
        signal.signal(signum, forward)
    try:
        while True:
            try:
                return _normalized_exit_code(worker.wait(timeout=0.25))
            except subprocess.TimeoutExpired:
                if (
                    shutdown_deadline is not None
                    and time.monotonic() >= shutdown_deadline
                ):
                    print(
                        "[OFFLINE] worker did not finish native teardown within "
                        f"{shutdown_timeout_s:.1f}s; terminating its process group",
                        flush=True,
                    )
                    _terminate_process_group(worker)
                    return _normalized_exit_code(worker.returncode)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if worker.poll() is None:
            _terminate_process_group(worker)


def _run_worker(config: OfflineConfig) -> int:
    app = OfflineInferenceApp(config)

    def handle_signal(_signal, _frame) -> None:
        app.request_stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    return app.run()


def main() -> int:
    args = parse_args()
    config = OfflineConfig(
        run_dir=args.run_dir, cfg_path=args.cfg, out_dir=args.out_dir,
    )
    return _run_worker(config) if args.worker else _supervise_worker(config)


if __name__ == "__main__":
    raise SystemExit(main())
