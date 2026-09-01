from __future__ import annotations

"""Backend orchestrator for the operator GUI."""

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from squeakview.apps.operator.backend import process
from squeakview import config as squeakview_config
from squeakview import model_package
from squeakview.common import run_context
from squeakview.common.capture_policy import capture_buffer_policy
from squeakview.common.device_context import device_context_snapshot, file_identity
from squeakview.common import serial as serial_util


def _now() -> str:
    return time.strftime("%H:%M:%S")



@dataclass(slots=True)
class RunState:
    inference: Optional[process.ProcessHandle] = None
    serial: Optional[serial_util.SerialHandle] = None
    run_dir: Optional[Path] = None

    def any_running(self) -> bool:
        return bool(self.inference and self.inference.is_running())


class OperatorBackend:
    def __init__(
        self,
        emit_log: Callable[[str], None],
        ingest_dashboard: Optional[Callable[[str], None]] = None,
        on_run_started: Callable[[], None] | None = None,
        on_run_failed: Callable[[str], None] | None = None,
    ):
        self.emit = emit_log
        self.ingest = ingest_dashboard
        self.on_run_started = on_run_started
        self.on_run_failed = on_run_failed
        self.state = RunState()
        self.launch_cfg = process.LaunchConfig()
        self._metadata_written = False
        self._run_storage_info: dict[str, Any] = {}
        self._run_started_at: str | None = None
        self._run_dir_watch_thread: threading.Thread | None = None
        self._inference_ready = threading.Event()
        self._stop_requested = threading.Event()
        self._finalize_lock = threading.Lock()
        self._run_finalized = True
        self._recording_started = False
        self._controller_started = False
        self._model_snapshot: dict[str, str] | None = None
        self._device_context: dict[str, object] | None = None

    def _log(self, message: str) -> None:
        self.emit(f"[{_now()}] {message}")

    @property
    def finalization_in_progress(self) -> bool:
        """True while a stop/failure worker owns the run-finalization path."""

        return self._finalize_lock.locked() and not self._run_finalized

    def _serial_emit(self, message: str) -> None:
        self.emit(message)
        if self.ingest and "【SER】" in message:
            raw = message.split("【SER】", 1)[1].strip()
            self.ingest(raw)

    def _capture_emit(self, message: str) -> None:
        self.emit(message)

    def _inference_emit(self, message: str) -> None:
        self.emit(message)
        lower = message.lower()
        if "[ready] inference playing" in lower:
            self._mark_inference_ready()
        marker = "run dir:"
        if marker in lower:
            idx = lower.index(marker) + len(marker)
            path = message[idx:].strip()
            if os.path.isdir(path):
                run_path = Path(path)
                self.state.run_dir = run_path
                if self.state.serial:
                    self.state.serial.set_csv_path(run_path)
                self._ensure_metadata(run_path)
        else:
            self._maybe_set_run_dir_from_marker()

    def _mark_inference_ready(self) -> None:
        if self._recording_started or self._run_finalized or self._stop_requested.is_set():
            return
        self._recording_started = True
        run_dir = self.state.run_dir
        if run_dir is not None:
            try:
                run_context.write_status(run_dir, "recording")
                self._write_run_manifest(run_dir)
            except Exception as exc:
                self._log(f"[SAVE] failed to mark run recording: {exc}")
        self._inference_ready.set()
        self._log("[BACKEND] inference ready; run is recording")
        if self.on_run_started is not None:
            try:
                self.on_run_started()
            except Exception as exc:
                self._log(f"[BACKEND] run-start callback failed: {exc}")

    def _on_inference_exit(self, returncode: int) -> None:
        if self._stop_requested.is_set() or self._run_finalized:
            return
        phase = "after readiness" if self._inference_ready.is_set() else "before readiness"
        error = f"inference process exited unexpectedly {phase} (exit code {returncode})"
        self._log(f"[DS] {error}")
        self._finalize_run(
            final_state="failed",
            error=error,
            terminate_inference=False,
            known_inference_returncode=returncode,
        )

    @staticmethod
    def _serial_open_failure_message(handle: serial_util.SerialHandle, port: str, baud: int) -> str:
        detail = str(getattr(handle, "last_error", "") or "unknown serial error").strip()
        message = f"Could not open serial port {port} at {baud} baud.\n\nSystem error: {detail}"
        permission_terms = ("permission denied", "access denied", "operation not permitted")
        if any(term in detail.lower() for term in permission_terms):
            message += (
                "\n\nSerial access was denied. Run `sudo usermod -aG dialout $USER`, then "
                "sign out and back in (or reboot) before retrying."
            )
        elif "no such file" in detail.lower() or "cannot find" in detail.lower():
            message += "\n\nCheck that the controller is connected and that the selected serial port is correct."
        return message

    def _maybe_set_run_dir_from_marker(self) -> Path | None:
        existing = self.state.run_dir
        if existing and existing.exists():
            return existing
        candidate = run_context.latest_run_dir()
        if candidate and candidate.exists():
            self.state.run_dir = candidate
            if self.state.serial:
                self.state.serial.set_csv_path(candidate)
            self._ensure_metadata(candidate)
            return candidate
        return None

    def _start_run_dir_watch(self) -> None:
        def watcher() -> None:
            for _ in range(80):
                if self._maybe_set_run_dir_from_marker():
                    return
                time.sleep(0.1)
        thread = threading.Thread(target=watcher, daemon=True)
        self._run_dir_watch_thread = thread
        thread.start()

    def _set_fan_max(self) -> None:
        """Best-effort fan control, opt-in because jetson_clocks usually needs privileges."""
        fan_flag = os.environ.get("SQUEAKVIEW_SET_FAN", "0").lower()
        if fan_flag not in {"1", "true", "yes", "on"}:
            return
        try:
            result = subprocess.run(
                ["jetson_clocks", "--fan"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=4,
            )
            if result.returncode == 0:
                self._log("[SYS] jetson_clocks --fan applied")
            else:
                self._log(f"[SYS] jetson_clocks --fan failed (rc={result.returncode}): {result.stderr.strip()}")
        except FileNotFoundError:
            self._log("[SYS] jetson_clocks not found; fan not adjusted")
        except Exception as exc:
            self._log(f"[SYS] jetson_clocks error: {exc}")

    @staticmethod
    def _file_info(path: Path) -> dict[str, Any]:
        exists = path.exists()
        info: dict[str, Any] = {
            "path": str(path),
            "exists": exists,
        }
        if exists:
            try:
                info["size_bytes"] = int(path.stat().st_size)
            except Exception:
                pass
        return info

    @staticmethod
    def _git_snapshot() -> dict[str, Any]:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(process.WORKSPACE),
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            dirty = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(process.WORKSPACE),
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            return {
                "commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            }
        except Exception:
            return {"commit": None, "dirty": None}

    def _run_output_snapshot(self, run_dir: Path) -> dict[str, Any]:
        artifacts = run_context.run_artifacts(run_dir)
        diagnostics_dir = run_dir / "diagnostics"
        status = run_context.read_json(artifacts.status_json)
        bottle_summary = run_context.read_json(artifacts.bottle_summary_json)
        video_files = sorted(run_dir.glob("raw*.mp4"))
        return {
            "csv_files": {
                "frames": self._file_info(artifacts.frames_csv),
                "errors": self._file_info(artifacts.drop_events_csv),
                "recording": self._file_info(diagnostics_dir / "recording.csv"),
                "camera": self._file_info(diagnostics_dir / "camera.csv"),
                "objects": self._file_info(artifacts.objects_csv),
                "keypoints": self._file_info(artifacts.keypoints_csv),
                "serial": self._file_info(artifacts.serial_csv) if artifacts.serial_csv else None,
                "bottle_measurements": self._file_info(artifacts.bottle_measurements_csv),
            },
            "video_files": [self._file_info(path) for path in video_files],
            "recording_validation": status.get("recording_validation"),
            "capture_reconciliation": status.get("capture_reconciliation"),
            "inference_admission": status.get("inference_admission"),
            "camera_runtime": self._file_info(diagnostics_dir / "camera_runtime.json"),
            "camera_telemetry": self._file_info(diagnostics_dir / "camera.csv"),
            "alignment_summary": self._file_info(run_dir / "alignment_summary.json"),
            "has_analysis": (run_dir / "alignment_summary.json").exists(),
            "bottle_measurements_complete": bool(bottle_summary.get("complete")) if bottle_summary else False,
            "bottle_files": {
                "setup": self._file_info(artifacts.bottle_setup_json),
                "measurements": self._file_info(artifacts.bottle_measurements_csv),
                "summary": self._file_info(artifacts.bottle_summary_json),
            },
        }

    def _bottle_manifest_snapshot(self, run_dir: Path) -> dict[str, Any]:
        artifacts = run_context.run_artifacts(run_dir)
        summary = run_context.read_json(artifacts.bottle_summary_json)
        return {
            "setup": artifacts.bottle_setup_json.name,
            "measurements": artifacts.bottle_measurements_csv.name,
            "summary": artifacts.bottle_summary_json.name,
            "complete": bool(summary.get("complete")) if summary else False,
            "sides": summary.get("sides", {}) if summary else {},
        }

    def _build_run_manifest(self, run_dir: Path) -> dict[str, Any]:
        cfg = self.launch_cfg
        buffer_policy = capture_buffer_policy(cfg.fps)
        artifacts = run_context.run_artifacts(run_dir)
        experiment = (cfg.experiment_name or "").strip() or None
        mouse_id = (cfg.mouse_id or "").strip() or None
        try:
            relative_run_dir = run_dir.relative_to(run_context.RUNS_DIR).as_posix()
        except ValueError:
            relative_run_dir = str(run_dir)
        return {
            "schema_version": "2.0",
            "run_id": run_dir.name,
            "run_directory": str(run_dir),
            "run_directory_relative": relative_run_dir,
            "created_at": self._run_started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "workspace": str(process.WORKSPACE),
            "platform": self._device_context or device_context_snapshot(),
            "native_plugins": {
                "flir_gstreamer_source": file_identity(
                    process.WORKSPACE / "native/flir_gst_source/build/gstflirspinsrc.so"
                ),
                "deepstream_yolo_parser": file_identity(
                    process.WORKSPACE
                    / "native/nvdsinfer_custom_impl_yolo/libnvdsinfer_custom_impl_Yolo.so"
                ),
            },
            "git": self._git_snapshot(),
            "storage": self._run_storage_info,
            "experiment_name": experiment,
            "mouse_id": mouse_id,
            "capture": {
                "backend": str(getattr(cfg, "capture_backend", "flir_direct")),
                "num_cameras": int(getattr(cfg, "num_cameras", 1)),
                "camera_serials": list(getattr(cfg, "camera_serials", ())),
                "width": cfg.width,
                "height": cfg.height,
                "fps": cfg.fps,
                "pixel_format": cfg.pixel_format,
                "trigger_on": cfg.trigger_on,
                "trigger_activation": cfg.trigger_activation,
                "arduino_fps": cfg.arduino_fps,
                "metadata_profile": "scientific",
                "runtime_metadata": "diagnostics/camera_runtime.json",
                "frame_identity": {
                    "camera_frame_id": "FLIR chunk FrameID",
                    "stream_frame_id": "Spinnaker Image.GetFrameID acquisition-local counter",
                    "source_sequence_index": "flirspinsrc emitted-buffer counter",
                    "missing_value_policy": "null; never substitute a sequential counter",
                },
            },
            "inference": {
                "enabled": cfg.inference_enabled,
                "deepstream_config": (str(cfg.ds_cfg) if cfg.ds_cfg else None),
                "model_package": self._model_snapshot,
                "bitrate_kbps": cfg.bitrate,
                "preview_transport": "nvunixfd",
                "preview_sockets": [str(path) for path in cfg.preview_socket_paths],
                "flow_control": "downstream-leaky; latest pending frames retained",
                "admission_field": "frames.csv:inference_admitted",
            },
            "recording": {
                "container": "mp4",
                "file": "raw.mp4",
                "encoder": {
                    "element": "x264enc",
                    "implementation": "software",
                    "input_format": "GRAY8",
                    "speed_preset": "ultrafast",
                    "sliced_threads": False,
                },
                "record_queue_capacity_frames": buffer_policy.record_queue_frames,
                "backpressure_warning_frames": buffer_policy.record_warning_frames,
                "backpressure_failure_frames": buffer_policy.record_failure_frames,
                "source_transport_buffer_count": buffer_policy.source_transport_buffers,
                "backpressure_telemetry": "diagnostics/recording.csv",
                "frame_manifest": artifacts.frames_csv.name,
                "drop_events": "diagnostics/errors.csv",
                "validation": "run_status.json:recording_validation",
                "camera_telemetry": "diagnostics/camera.csv",
                "objects": artifacts.objects_csv.name,
                "keypoints": artifacts.keypoints_csv.name,
                "serial": artifacts.serial_csv.name if cfg.serial_enabled and artifacts.serial_csv else None,
            },
            "task_config": str(cfg.task_cfg) if cfg.task_cfg else None,
            "serial": {
                "enabled": cfg.serial_enabled,
                "port": cfg.serial_port if cfg.serial_enabled else None,
                "baud": cfg.serial_baud if cfg.serial_enabled else None,
            },
            "bottles": self._bottle_manifest_snapshot(run_dir),
            "expected_outputs": {
                "status": run_context.RUN_STATUS_FILENAME,
                "manifest": run_context.RUN_MANIFEST_FILENAME,
                "camera_runtime": "diagnostics/camera_runtime.json",
                "camera_telemetry": "diagnostics/camera.csv",
                "config_dir": "config",
                "frames": artifacts.frames_csv.name,
                "drop_events": "diagnostics/errors.csv",
                "objects": artifacts.objects_csv.name if cfg.inference_enabled else None,
                "keypoints": artifacts.keypoints_csv.name if cfg.inference_enabled else None,
                "serial": artifacts.serial_csv.name if cfg.serial_enabled and artifacts.serial_csv else None,
                "raw_video": artifacts.raw_video.name,
                "recording_path_telemetry": "diagnostics/recording.csv",
                "alignment_summary": "alignment_summary.json" if cfg.serial_enabled else None,
                "bottle_setup": artifacts.bottle_setup_json.name,
                "bottle_measurements": artifacts.bottle_measurements_csv.name,
                "bottle_summary": artifacts.bottle_summary_json.name,
            },
            "actual_outputs": self._run_output_snapshot(run_dir),
        }

    def _write_run_manifest(self, run_dir: Path) -> None:
        try:
            existing = run_context.read_json(
                run_dir / run_context.RUN_MANIFEST_FILENAME
            )
            status = run_context.read_json(
                run_dir / run_context.RUN_STATUS_FILENAME
            )
            terminal_states = {
                "post_run_complete",
                "analysis_complete",
                "finalized",
                "finalization_failed",
                "analysis_failed",
                "failed",
            }
            if existing and (
                self._run_started_at is None
                or status.get("state") in terminal_states
            ):
                # Acquisition configuration and provenance are immutable once a
                # run has completed. A later GUI session may update bottle
                # measurements and the artifact inventory, but it does not have
                # the original in-memory model/configuration snapshot.
                manifest = dict(existing)
                manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                manifest["bottles"] = self._bottle_manifest_snapshot(run_dir)
                manifest["actual_outputs"] = self._run_output_snapshot(run_dir)
            else:
                manifest = self._build_run_manifest(run_dir)
            path = run_context.write_manifest(run_dir, manifest)
            self._log(f"[BACKEND] manifest written → {path}")
        except Exception as exc:
            self._log(f"[BACKEND] manifest write failed: {exc}")

    def _write_bottle_measurements(self, run_dir: Path, bottles: dict[str, Any] | None) -> dict[str, Any]:
        summary = run_context.write_bottle_artifacts(run_dir, bottles)
        state = "complete" if summary.get("complete") else "incomplete"
        self._log(f"[BOTTLES] saved {state} bottle metadata → {run_dir / run_context.BOTTLE_MEASUREMENTS_FILENAME}")
        for warning in summary.get("warnings", []):
            self._log(f"[BOTTLES] warning: {warning}")
        return summary

    def save_bottle_measurements(
        self,
        bottles: dict[str, Any] | None,
        run_dir: Path | None = None,
    ) -> dict[str, Any]:
        target = Path(run_dir) if run_dir is not None else self._maybe_set_run_dir_from_marker()
        if target is None:
            raise RuntimeError("no active run directory for bottle metadata")
        summary = self._write_bottle_measurements(target, bottles)
        self._write_run_manifest(target)
        try:
            run_context.update_status(
                target,
                bottle_measurements_complete=bool(summary.get("complete")),
                outputs=self._run_output_snapshot(target),
            )
        except Exception as exc:
            self._log(f"[BOTTLES] status update failed: {exc}")
        return summary

    def _ensure_metadata(self, run_dir: Path) -> None:
        if self._metadata_written:
            return
        try:
            self._metadata_written = True
            self._write_run_manifest(run_dir)
        except Exception as exc:  # pragma: no cover
            self._log(f"[BACKEND] metadata write failed: {exc}")

    @staticmethod
    def _capture_drain_snapshot(run_dir: Path, camera_count: int) -> tuple[int, ...]:
        paths: list[Path] = []
        for index in range(max(1, int(camera_count))):
            paths.append(run_dir / f"capture_cam{index}.jsonl")
            paths.append(
                run_dir
                / (
                    "record_admission.csv"
                    if index == 0
                    else f"record_admission_cam{index}.csv"
                )
            )
        sizes: list[int] = []
        for path in paths:
            try:
                sizes.append(int(path.stat().st_size))
            except FileNotFoundError:
                sizes.append(-1)
        return tuple(sizes)

    @staticmethod
    def _last_ledger_line(path: Path) -> str | None:
        """Read the last complete nonempty ledger line without scanning the run."""

        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                end = handle.tell()
                if end <= 0:
                    return None
                read_size = min(end, 65_536)
                handle.seek(end - read_size)
                lines = handle.read(read_size).decode(errors="replace").splitlines()
        except OSError:
            return None
        return next((line.strip() for line in reversed(lines) if line.strip()), None)

    @classmethod
    def _capture_drain_counts(
        cls, run_dir: Path, camera_count: int
    ) -> tuple[int | None, ...]:
        """Return source/admission counts inferred from monotonic ledger indices."""

        counts: list[int | None] = []
        for index in range(max(1, int(camera_count))):
            capture_line = cls._last_ledger_line(run_dir / f"capture_cam{index}.jsonl")
            try:
                capture_count = int(json.loads(capture_line or "")["source_sequence_index"]) + 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                capture_count = 0 if capture_line is None else None
            admission_path = run_dir / (
                "record_admission.csv"
                if index == 0
                else f"record_admission_cam{index}.csv"
            )
            admission_line = cls._last_ledger_line(admission_path)
            try:
                first_field = (admission_line or "").split(",", 2)[1]
                admission_count = int(first_field) + 1
            except (IndexError, TypeError, ValueError):
                admission_count = 0 if admission_line in (None, "") else None
            counts.extend((capture_count, admission_count))
        return tuple(counts)

    def _wait_for_capture_drain(
        self, run_dir: Path, *, expected_ttl_count: int | None = None
    ) -> bool:
        """Wait for source and recording ledgers to become quiescent after ACK_STOP."""

        try:
            quiet_s = max(
                0.1, float(os.environ.get("SQUEAKVIEW_CAPTURE_DRAIN_QUIET_S", "0.35"))
            )
            timeout_s = max(
                quiet_s, float(os.environ.get("SQUEAKVIEW_CAPTURE_DRAIN_TIMEOUT_S", "5.0"))
            )
        except ValueError:
            quiet_s, timeout_s = 0.35, 5.0
        run_context.write_status(
            run_dir,
            "capture_draining",
            quiet_period_s=quiet_s,
            timeout_s=timeout_s,
            expected_ttl_count=expected_ttl_count,
        )
        deadline = time.monotonic() + timeout_s
        previous: tuple[int, ...] | None = None
        stable_since: float | None = None
        while time.monotonic() < deadline:
            snapshot = self._capture_drain_snapshot(
                run_dir, int(getattr(self.launch_cfg, "num_cameras", 1))
            )
            counts = self._capture_drain_counts(
                run_dir, int(getattr(self.launch_cfg, "num_cameras", 1))
            )
            now = time.monotonic()
            ledgers_exist = bool(snapshot) and all(size >= 0 for size in snapshot)
            counts_valid = bool(counts) and all(count is not None for count in counts)
            pairs_match = counts_valid and all(
                counts[offset] == counts[offset + 1]
                for offset in range(0, len(counts), 2)
            )
            target_reached = expected_ttl_count is None or (
                counts_valid and all(int(count) >= expected_ttl_count for count in counts)
            )
            if ledgers_exist and pairs_match and target_reached and snapshot == previous:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= quiet_s:
                    run_context.write_status(
                        run_dir,
                        "capture_drained",
                        ledger_sizes=list(snapshot),
                        ledger_frame_counts=list(counts),
                        expected_ttl_count=expected_ttl_count,
                    )
                    return True
            else:
                stable_since = now if ledgers_exist else None
            previous = snapshot
            time.sleep(0.05)
        run_context.write_status(run_dir, "capture_drain_timeout")
        return False

    def _run_capture_finalizer(self, run_dir: Path) -> int:
        self._log("[POST-RUN] starting independent bounded-memory finalizer")
        worker = process.spawn_post_run(
            run_dir,
            camera_count=int(getattr(self.launch_cfg, "num_cameras", 1)),
            enable_infer=bool(getattr(self.launch_cfg, "inference_enabled", True)),
            enable_align=(
                bool(getattr(self.launch_cfg, "serial_enabled", False))
                and os.environ.get("SQUEAKVIEW_AUTO_ALIGN", "1").lower()
                not in {"0", "false", "no", "off"}
            ),
        )
        last_reported = -1
        while worker.poll() is None:
            progress = run_context.read_json(run_dir / "post_run_progress.json")
            processed = int(progress.get("frames_processed") or 0)
            if processed >= last_reported + 100_000:
                self._log(
                    f"[POST-RUN] {progress.get('stage', 'starting')}: "
                    f"{processed} frames"
                )
                last_reported = processed
            time.sleep(0.2)
        return int(worker.returncode or 0)

    def _finalize_run(
        self,
        *,
        final_state: str,
        error: str | None = None,
        terminate_inference: bool = True,
        known_inference_returncode: int | None = None,
    ) -> bool:
        notify_failure = False
        with self._finalize_lock:
            if self._run_finalized:
                return False
            self._stop_requested.set()
            run_dir = self.state.run_dir
            inference = self.state.inference
            inference_running = bool(inference and inference.is_running())
            inference_returncode = known_inference_returncode
            capture_exit_error: str | None = None
            controller_stop_error: str | None = None
            capture_drain_error: str | None = None

            if final_state == "finalized":
                self._log("[BACKEND] stopping run")
                if run_dir:
                    try:
                        run_context.write_status(run_dir, "stopping")
                    except Exception:
                        pass

            serial_handle = self.state.serial
            stopping_capture = bool(terminate_inference and inference_running and inference is not None)
            if (stopping_capture or self._controller_started) and serial_handle:
                serial_handle.log_marker("CAPTURE_STOP_REQUESTED")

            # Stop the controller first so it cannot generate unrecorded trigger
            # pulses while DeepStream drains the recording branch and closes MP4.
            if serial_handle:
                try:
                    serial_handle.log_marker("STOP_SENT")
                    serial_handle.send_line("STOP")
                    if stopping_capture or self._controller_started:
                        stop_acked = serial_handle.wait_for_stop_ack(timeout_s=2.0)
                        serial_handle.log_marker(
                            "CAPTURE_STOP_ACKED" if stop_acked else "CAPTURE_STOP_ACK_TIMEOUT"
                        )
                        if not stop_acked:
                            controller_stop_error = "controller STOP was not acknowledged"
                except Exception as exc:
                    controller_stop_error = f"controller STOP failed: {exc}"
                    self._log(f"[BACKEND] {controller_stop_error}")

            if (
                run_dir is not None
                and serial_handle is not None
                and bool(getattr(self.launch_cfg, "trigger_on", False))
                and self._controller_started
            ):
                drained = self._wait_for_capture_drain(
                    Path(run_dir),
                    expected_ttl_count=getattr(serial_handle, "stop_ack_count", None),
                )
                if serial_handle:
                    serial_handle.log_marker(
                        "CAPTURE_SOURCE_DRAINED"
                        if drained
                        else "CAPTURE_SOURCE_DRAIN_TIMEOUT"
                    )
                if not drained:
                    capture_drain_error = (
                        "capture ledgers did not reach the controller TTL count and become quiet"
                    )

            if stopping_capture and inference is not None:
                # The capture runner now exits after EOS/MP4 closure. Audit and
                # analysis happen in an independent post-run worker.
                inference.terminate_group_graceful(signal.SIGINT, 30.0, True)
                try:
                    inference_returncode = inference.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    capture_exit_error = (
                        "capture process did not exit after shutdown escalation; "
                        "recording files were not validated"
                    )
                    self._log(f"[BACKEND] {capture_exit_error}")
                if serial_handle and inference_returncode is not None:
                    serial_handle.log_marker("CAPTURE_STOP_DONE")
            elif inference is not None and not inference_running:
                # The output-pump thread may already have observed child exit;
                # still collect its concrete code before inspecting artifacts.
                try:
                    inference_returncode = inference.wait(timeout=0)
                except subprocess.TimeoutExpired:
                    capture_exit_error = (
                        "capture process exit could not be confirmed; "
                        "recording files were not validated"
                    )

            if serial_handle:
                try:
                    time.sleep(0.5)
                except Exception:
                    pass
                serial_handle.close()
                self.state.serial = None

            failure_reasons: list[str] = []
            if controller_stop_error:
                failure_reasons.append(controller_stop_error)
            if capture_drain_error:
                failure_reasons.append(capture_drain_error)
            if capture_exit_error:
                failure_reasons.append(capture_exit_error)
            if final_state == "finalized":
                if inference_returncode is None:
                    if inference is not None and not capture_exit_error:
                        failure_reasons.append(
                            "capture process exit was not confirmed; recording files were not validated"
                        )
                elif inference_returncode != 0:
                    failure_reasons.append(f"inference exit code {inference_returncode}")
            if run_dir and inference_returncode is not None:
                finalizer_returncode = self._run_capture_finalizer(Path(run_dir))
                if finalizer_returncode not in (0,):
                    failure_reasons.append(
                        f"post-run finalizer exit code {finalizer_returncode}"
                    )
                if final_state == "finalized":
                    status = run_context.read_json(Path(run_dir) / "run_status.json")
                    validation = status.get("recording_validation")
                    if not isinstance(validation, dict):
                        failure_reasons.append("recording validation was not produced")
                    elif validation.get("passed") is not True:
                        failure_reasons.append("recording frame-count validation failed")
                    integrity = status.get("acquisition_integrity")
                    if not isinstance(integrity, dict):
                        failure_reasons.append("acquisition integrity audit was not produced")
                    elif integrity.get("passed") is not True:
                        failure_reasons.append("camera acquisition integrity validation failed")
            if failure_reasons:
                final_state = "failed"
                error = "; ".join(
                    ([error] if error else []) + failure_reasons
                )
                self._log(f"[BACKEND] run failed validation: {error}")

            self.state.inference = None
            finalizer_status = (
                run_context.read_json(Path(run_dir) / "run_status.json")
                if run_dir
                else {}
            )
            analysis_summary = (
                run_context.read_json(Path(run_dir) / "alignment_summary.json")
                if run_dir
                and (Path(run_dir) / "alignment_summary.json").exists()
                else None
            )
            if run_dir:
                try:
                    self._write_run_manifest(run_dir)
                    updates: dict[str, Any] = {
                        "analysis_complete": bool(
                            analysis_summary is not None
                            and finalizer_status.get("alignment_validated") is True
                        ),
                        "outputs": self._run_output_snapshot(run_dir),
                    }
                    if error:
                        updates["error"] = error
                    run_context.write_status(run_dir, final_state, **updates)
                    self._write_run_manifest(run_dir)
                except Exception as exc:
                    self._log(f"[SAVE] finalize status failed: {exc}")

            self._run_finalized = True
            notify_failure = final_state == "failed"

        if notify_failure and self.on_run_failed is not None:
            try:
                self.on_run_failed(error or "run failed")
            except Exception as exc:
                self._log(f"[BACKEND] run-failure callback failed: {exc}")
        return True


    def start_run(self, cfg: process.LaunchConfig) -> bool:
        if self.state.any_running():
            self._log("[BACKEND] run already active")
            return False
        if not cfg.task_cfg:
            self._log("[BACKEND] task config required; aborting run")
            return False
        cfg.task_cfg = squeakview_config.resolve_workspace_path(cfg.task_cfg)
        if cfg.ds_cfg is not None:
            cfg.ds_cfg = squeakview_config.resolve_workspace_path(cfg.ds_cfg)
        if not Path(cfg.task_cfg).exists():
            self._log(f"[BACKEND] task config missing: {cfg.task_cfg}")
            return False
        self._model_snapshot = None
        if cfg.inference_enabled:
            if cfg.ds_cfg is None:
                self._log("[MODEL] inference is enabled but no model package was selected")
                return False
            try:
                selected_model = model_package.validate_model_package(cfg.ds_cfg)
            except model_package.ModelPackageError as exc:
                self._log(f"[MODEL] selected model package is invalid: {exc}")
                return False
            self._model_snapshot = selected_model.manifest_snapshot()
            self._log(f"[MODEL] selected package: {selected_model.name}")

        try:
            self._run_storage_info = run_context.assert_runs_dir_ready()
        except Exception as exc:
            self._log(f"[SAVE] local run storage unavailable: {exc}")
            return False

        self.launch_cfg = cfg
        self.state.run_dir = None
        self._metadata_written = False
        self._run_started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._inference_ready.clear()
        self._stop_requested.clear()
        self._recording_started = False
        self._controller_started = False
        self._device_context = device_context_snapshot()

        self._set_fan_max()

        mouse_id = (cfg.mouse_id or "").strip()
        experiment_name = (cfg.experiment_name or "").strip()
        run_prefix = mouse_id or "ds"
        try:
            run_dir, _run_id = run_context.create_run_dir(
                experiment_name=experiment_name,
                mouse_id=mouse_id,
                prefix=run_prefix,
            )
        except Exception as exc:
            self._log(f"[SAVE] failed to create run directory: {exc}")
            return False
        cfg.run_dir = run_dir
        cfg.preview_socket_paths = process.preview_socket_paths(run_dir, cfg.num_cameras)
        self.state.run_dir = run_dir
        self._run_finalized = False
        try:
            run_context.write_status(
                run_dir,
                "created",
                run_id=run_dir.name,
                run_directory=str(run_dir),
                experiment_name=experiment_name or None,
                mouse_id=mouse_id or None,
            )
        except Exception as exc:
            self._log(f"[SAVE] status write failed: {exc}")
        try:
            self._write_bottle_measurements(run_dir, getattr(cfg, "bottles", None))
        except Exception as exc:
            self._log(f"[BOTTLES] initial metadata save failed: {exc}")
        self._write_run_manifest(run_dir)

        serial_handle: serial_util.SerialHandle | None = None
        if cfg.serial_enabled:
            if not serial_util.have_pyserial():
                error = (
                    "Serial controller support was requested, but pyserial is not installed. "
                    "Install the project dependencies before starting a scientific run."
                )
                self._log(f"[SER] {error}")
                self._finalize_run(
                    final_state="failed",
                    error=error,
                    terminate_inference=False,
                )
                return False
            else:
                handle = serial_util.SerialHandle(cfg.serial_port, cfg.serial_baud, self._serial_emit)
                if not handle.open(run_dir):
                    error = self._serial_open_failure_message(handle, cfg.serial_port, cfg.serial_baud)
                    self._log(f"[SER] {error.replace(chr(10), ' ')}")
                    self._finalize_run(
                        final_state="failed",
                        error=error,
                        terminate_inference=False,
                    )
                    return False
                serial_handle = handle

        if serial_handle:
            self.state.serial = serial_handle

        self._ensure_metadata(run_dir)

        capture_backend = str(getattr(cfg, "capture_backend", "flir_direct") or "flir_direct").lower().strip()
        if capture_backend != "flir_direct":
            error = f"unsupported capture backend: {capture_backend}"
            self._log(f"[CAP] {error}")
            self._finalize_run(final_state="failed", error=error, terminate_inference=False)
            return False

        self._log("[CAP] FLIR direct capture will be sourced inside DeepStream (flirspinsrc)")
        try:
            run_context.write_status(run_dir, "starting")
            self._write_run_manifest(run_dir)
        except Exception:
            pass
        try:
            handle = process.spawn_inference(cfg, self._inference_emit, on_exit=self._on_inference_exit)
            self.state.inference = handle
        except Exception as exc:
            error = f"failed to launch inference process: {exc}"
            self._log(f"[DS] {error}")
            self._finalize_run(final_state="failed", error=error, terminate_inference=False)
            return False
        if self._run_finalized:
            self.state.inference = None
            return False
        self._log("[DS] inference launched")
        if serial_handle and cfg.trigger_on:
            self._log("[BACKEND] waiting for inference ready before START")
            try:
                ready_timeout = max(1.0, float(os.environ.get("SQUEAKVIEW_INFERENCE_READY_TIMEOUT", "30")))
            except ValueError:
                ready_timeout = 30.0
            ready = self._inference_ready.wait(timeout=ready_timeout)
            if not ready:
                error = f"inference was not ready within {ready_timeout:.1f}s; controller was not started"
                self._log(f"[BACKEND] {error}")
                self._finalize_run(final_state="failed", error=error)
                return False
            if self._run_finalized or not handle.is_running():
                error = "inference exited before the controller could be started"
                self._log(f"[BACKEND] {error}")
                self._finalize_run(final_state="failed", error=error, terminate_inference=False)
                return False
            try:
                self._log("[BACKEND] inference ready; sending START")
                serial_handle.send_line(f"START,{int(cfg.arduino_fps)}")
                serial_handle.log_marker("START_SENT")
                self._controller_started = True
                if not serial_handle.wait_for_ttl(timeout_s=3.0):
                    serial_handle.log_marker("START_TTL_TIMEOUT")
                    error = (
                        "controller START was sent, but no camera TTL was detected within "
                        "3.0s; the run was aborted"
                    )
                    self._log(f"[BACKEND] {error}")
                    self._finalize_run(final_state="failed", error=error)
                    return False
            except Exception as exc:
                error = f"failed to start controller: {exc}"
                self._log(f"[BACKEND] {error}")
                self._finalize_run(final_state="failed", error=error)
                return False
        self._start_run_dir_watch()
        return True

    def stop_run(self) -> None:
        self._maybe_set_run_dir_from_marker()
        self._finalize_run(final_state="finalized")

    def shutdown(self) -> None:
        self.stop_run()
