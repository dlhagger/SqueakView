from __future__ import annotations

"""Backend orchestrator for the operator GUI."""

import os
import importlib.util
import json
import signal
import shutil
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
from squeakview.common import serial as serial_util


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _relocate_summary_outputs(
    summary: dict[str, Any], source_dir: Path, final_dir: Path
) -> dict[str, Any]:
    """Update output paths after the temporary analysis directory is renamed."""
    outputs = summary.get("outputs")
    if not isinstance(outputs, dict):
        return summary
    for key, value in outputs.items():
        try:
            relative = Path(str(value)).relative_to(source_dir)
        except (TypeError, ValueError):
            continue
        outputs[key] = str(final_dir / relative)
    return summary


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
        self._model_snapshot: dict[str, str] | None = None

    def _log(self, message: str) -> None:
        self.emit(f"[{_now()}] {message}")

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
        self._finalize_run(final_state="failed", error=error, terminate_inference=False)

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
    def _csv_data_rows(path: Path) -> int | None:
        if not path.exists():
            return None
        try:
            with path.open(newline="") as f:
                rows = sum(1 for _ in f)
            return max(0, rows - 1)
        except Exception:
            return None

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
        csv_rows = {
            "frames": self._csv_data_rows(artifacts.frames_csv),
            "drop_events": self._csv_data_rows(artifacts.drop_events_csv),
            "detections": self._csv_data_rows(artifacts.detections_csv),
            "serial": self._csv_data_rows(artifacts.serial_csv) if artifacts.serial_csv else None,
            "perf": self._csv_data_rows(run_dir / "perf_stats.csv"),
            "bottle_measurements": self._csv_data_rows(artifacts.bottle_measurements_csv),
        }
        bottle_summary = run_context.read_json(artifacts.bottle_summary_json)
        video_files = sorted(run_dir.glob("raw*.mp4"))
        return {
            "csv_rows": csv_rows,
            "video_files": [self._file_info(path) for path in video_files],
            "analysis_dir": str(run_dir / "analysis"),
            "has_analysis": (run_dir / "analysis").exists(),
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
        artifacts = run_context.run_artifacts(run_dir)
        experiment = (cfg.experiment_name or "").strip() or None
        mouse_id = (cfg.mouse_id or "").strip() or None
        chunked_recording = _env_flag("SQUEAKVIEW_ENABLE_CHUNKED_RECORDING", False)
        try:
            relative_run_dir = run_dir.relative_to(run_context.RUNS_DIR).as_posix()
        except ValueError:
            relative_run_dir = str(run_dir)
        return {
            "schema_version": "1.0",
            "run_id": run_dir.name,
            "run_directory": str(run_dir),
            "run_directory_relative": relative_run_dir,
            "created_at": self._run_started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "workspace": str(process.WORKSPACE),
            "git": self._git_snapshot(),
            "storage": self._run_storage_info,
            "experiment_name": experiment,
            "mouse_id": mouse_id,
            "capture": {
                "backend": str(getattr(cfg, "capture_backend", "flir_direct")),
                "num_cameras": int(getattr(cfg, "num_cameras", 1)),
                "width": cfg.width,
                "height": cfg.height,
                "fps": cfg.fps,
                "pixel_format": cfg.pixel_format,
                "trigger_on": cfg.trigger_on,
                "trigger_activation": cfg.trigger_activation,
                "arduino_fps": cfg.arduino_fps,
            },
            "inference": {
                "enabled": cfg.inference_enabled,
                "deepstream_config": (str(cfg.ds_cfg) if cfg.ds_cfg else None),
                "model_package": self._model_snapshot,
                "bitrate_kbps": cfg.bitrate,
                "preview_window_id": cfg.preview_window_id,
            },
            "recording": {
                "container": "mp4",
                "segmented": chunked_recording,
                "pattern": "raw_%06d.mp4" if chunked_recording else "raw.mp4",
                "segment_seconds": (
                    int(os.environ.get("SQUEAKVIEW_RECORD_SEGMENT_SECONDS", "600")) if chunked_recording else None
                ),
                "video_segments": "video_segments.csv" if chunked_recording else None,
                "frame_manifest": artifacts.frames_csv.name,
                "drop_events": artifacts.drop_events_csv.name,
                "detections": artifacts.detections_csv.name,
                "serial": artifacts.serial_csv.name if artifacts.serial_csv else None,
                "perf": "perf_stats.csv",
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
                "camera_settings": artifacts.metadata_json.name,
                "deepstream_config_dir": "deepstream_config",
                "frames": artifacts.frames_csv.name,
                "drop_events": artifacts.drop_events_csv.name,
                "detections": artifacts.detections_csv.name if cfg.inference_enabled else None,
                "serial": artifacts.serial_csv.name if cfg.serial_enabled and artifacts.serial_csv else None,
                "raw_video": artifacts.raw_video_pattern.name if chunked_recording else artifacts.raw_video.name,
                "video_segments": "video_segments.csv" if chunked_recording else None,
                "bottle_setup": artifacts.bottle_setup_json.name,
                "bottle_measurements": artifacts.bottle_measurements_csv.name,
                "bottle_summary": artifacts.bottle_summary_json.name,
                "analysis": "analysis",
            },
            "actual_outputs": self._run_output_snapshot(run_dir),
        }

    def _write_run_manifest(self, run_dir: Path) -> None:
        try:
            path = run_context.write_manifest(run_dir, self._build_run_manifest(run_dir))
            self._log(f"[BACKEND] manifest written → {path}")
        except Exception as exc:
            self._log(f"[BACKEND] manifest write failed: {exc}")

    def _write_bottle_measurements(self, run_dir: Path, bottles: dict[str, Any] | None) -> dict[str, Any]:
        summary = run_context.write_bottle_artifacts(run_dir, bottles)
        state = "complete" if summary.get("complete") else "incomplete"
        self._log(f"[BOTTLES] saved {state} bottle metadata → {run_dir / run_context.BOTTLE_MEASUREMENTS_FILENAME}")
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
        cfg = self.launch_cfg
        chunked_recording = _env_flag("SQUEAKVIEW_ENABLE_CHUNKED_RECORDING", False)
        payload = {
            "schema_version": "1.0",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_directory": str(run_dir),
            "capture": {
                "backend": str(getattr(cfg, "capture_backend", "flir_direct")),
                "num_cameras": int(getattr(cfg, "num_cameras", 1)),
                "width": cfg.width,
                "height": cfg.height,
                "fps": cfg.fps,
                "pixel_format": cfg.pixel_format,
                "trigger_on": cfg.trigger_on,
                "trigger_activation": cfg.trigger_activation,
                "arduino_fps": cfg.arduino_fps,
            },
            "inference": {
                "enabled": cfg.inference_enabled,
                "deepstream_config": (str(cfg.ds_cfg) if cfg.ds_cfg else None),
                "model_package": self._model_snapshot,
                "bitrate_kbps": cfg.bitrate,
                "preview_window_id": cfg.preview_window_id,
            },
            "recording": {
                "container": "mp4",
                "segmented": chunked_recording,
                "pattern": "raw_%06d.mp4" if chunked_recording else "raw.mp4",
                "segment_seconds": (
                    int(os.environ.get("SQUEAKVIEW_RECORD_SEGMENT_SECONDS", "600")) if chunked_recording else None
                ),
                "video_segments": "video_segments.csv" if chunked_recording else None,
                "frame_manifest": "frames.csv",
                "drop_events": "drop_events.csv",
            },
            "task_config": str(cfg.task_cfg) if cfg.task_cfg else None,
            "serial": {
                "enabled": cfg.serial_enabled,
                "port": cfg.serial_port if cfg.serial_enabled else None,
                "baud": cfg.serial_baud if cfg.serial_enabled else None,
            },
            "mouse_id": cfg.mouse_id,
        }
        try:
            path = run_context.write_metadata(run_dir, payload)
            self._log(f"[BACKEND] metadata written → {path}")
            self._metadata_written = True
            self._write_run_manifest(run_dir)
        except Exception as exc:  # pragma: no cover
            self._log(f"[BACKEND] metadata write failed: {exc}")

    def _run_post_run_alignment(self, run_dir: Path | None) -> dict[str, Any] | None:
        if run_dir is None:
            self._log("[ANALYSIS] skipped; run directory unknown")
            return None

        run_dir = Path(run_dir)
        chunked_video_files = sorted(run_dir.glob("raw_*.mp4"))
        if chunked_video_files:
            ledger_path = run_dir / "video_segments.csv"
            try:
                ledger_rows = self._csv_data_rows(ledger_path)
            except Exception:
                ledger_rows = None
            if not ledger_path.exists() or not ledger_rows:
                error = (
                    "chunked MP4 files exist without writer-owned video_segments.csv; "
                    "segment provenance is not authoritative"
                )
                self._log(f"[ANALYSIS] failed: {error}")
                try:
                    run_context.write_status(run_dir, "analysis_failed", error=error)
                except Exception:
                    pass
                return None

        def skip(reason: str) -> None:
            self._log(f"[ANALYSIS] skipped; {reason}")
            try:
                run_context.write_status(run_dir, "analysis_skipped", reason=reason)
            except Exception:
                pass

        enabled = os.environ.get("SQUEAKVIEW_AUTO_ALIGN", "1").lower()
        if enabled in {"0", "false", "no", "off"}:
            skip("SQUEAKVIEW_AUTO_ALIGN=0")
            return None

        missing = [name for name in ("frames.csv", "serial.csv") if not (run_dir / name).exists()]
        if missing:
            skip(f"missing {', '.join(missing)}")
            return None

        script_path = process.WORKSPACE / "scripts" / "align_run_outputs.py"
        if not script_path.exists():
            skip(f"align script missing: {script_path}")
            return None

        tmp_dir = run_dir / "analysis.tmp"
        final_dir = run_dir / "analysis"
        try:
            self._log("[ANALYSIS] building aligned CSVs")
            run_context.write_status(run_dir, "analyzing")
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            spec = importlib.util.spec_from_file_location("squeakview_align_run_outputs", script_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"could not load {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            summary = module.build_alignment(run_dir, tmp_dir)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            tmp_dir.replace(final_dir)
            summary = _relocate_summary_outputs(summary, tmp_dir, final_dir)
            (final_dir / "alignment_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True)
            )
            counts = summary.get("counts", {})
            validation = summary.get("validation", {})
            self._log(
                "[ANALYSIS] complete → "
                f"{final_dir} "
                f"({counts.get('recorded_frames', 0)} frames, {counts.get('detections', 0)} detections)"
            )

            warning_keys = [
                ("counts", "frame_gaps_detected"),
                ("counts", "frames_missing_ttl"),
                ("counts", "drop_events"),
                ("validation", "detection_mapping_failed_rows"),
                ("validation", "detection_mapping_fallback_rows"),
                ("validation", "detections_missing_frame_count"),
                ("validation", "detection_ts_mismatch_count"),
                ("validation", "detection_pts_mismatch_count"),
            ]
            warnings: list[str] = []
            for section_name, key in warning_keys:
                section = counts if section_name == "counts" else validation
                value = section.get(key)
                if value not in (None, 0):
                    warnings.append(f"{key}={value}")
            if validation.get("video_frame_count_matches_frames_csv") is False:
                warnings.append("video_frame_count_matches_frames_csv=false")
            elif validation.get("video_frame_count_matches_frames_csv") is None:
                warnings.append("video_frame_count_matches_frames_csv=unknown")
            if warnings:
                self._log(f"[ANALYSIS] warnings: {', '.join(warnings)}")
            else:
                self._log("[ANALYSIS] validation passed")
            run_context.write_status(run_dir, "analysis_complete", analysis_summary=summary)
            return summary
        except Exception as exc:
            try:
                failed_dir = run_dir / f"analysis_failed_{int(time.time())}"
                if tmp_dir.exists():
                    tmp_dir.replace(failed_dir)
            except Exception:
                pass
            try:
                run_context.write_status(run_dir, "analysis_failed", error=str(exc))
            except Exception:
                pass
            self._log(f"[ANALYSIS] failed: {exc}")
            return None

    def _finalize_run(
        self,
        *,
        final_state: str,
        error: str | None = None,
        terminate_inference: bool = True,
    ) -> bool:
        notify_failure = False
        with self._finalize_lock:
            if self._run_finalized:
                return False
            self._stop_requested.set()
            run_dir = self.state.run_dir
            inference = self.state.inference
            inference_running = bool(inference and inference.is_running())

            if final_state == "finalized":
                self._log("[BACKEND] stopping run")
                if run_dir:
                    try:
                        run_context.write_status(run_dir, "stopping")
                    except Exception:
                        pass

            if terminate_inference and inference_running and inference is not None:
                if self.state.serial:
                    self.state.serial.log_marker("CAPTURE_STOP_REQUESTED")
                inference.terminate_group_graceful(signal.SIGINT, 10.0, True)
                inference.wait(timeout=2)
                if self.state.serial:
                    self.state.serial.log_marker("CAPTURE_STOP_DONE")

            if self.state.serial:
                try:
                    self.state.serial.log_marker("STOP_SENT")
                    self.state.serial.send_line("STOP")
                except Exception:
                    pass
                try:
                    time.sleep(0.5)
                except Exception:
                    pass
                self.state.serial.close()
                self.state.serial = None

            self.state.inference = None
            analysis_summary = self._run_post_run_alignment(run_dir)
            if run_dir:
                try:
                    self._write_run_manifest(run_dir)
                    updates: dict[str, Any] = {
                        "analysis_complete": analysis_summary is not None,
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
                self._log("[SER] pyserial unavailable; disabling serial")
                cfg.serial_enabled = False
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
                ready_timeout = max(1.0, float(os.environ.get("SQUEAKVIEW_INFERENCE_READY_TIMEOUT", "8")))
            except ValueError:
                ready_timeout = 8.0
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
                serial_handle.log_marker("START_SENT")
                serial_handle.send_line(f"START,{int(cfg.arduino_fps)}")
                serial_handle.wait_for_ttl(timeout_s=3.0)
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

    def set_skeleton_enabled(self, enabled: bool) -> None:
        """Toggle pose skeleton drawing at runtime by touching the control file."""
        run_dir = self._maybe_set_run_dir_from_marker()
        if not run_dir:
            self._log("[BACKEND] skeleton toggle ignored; run dir unknown")
            return
        path = run_dir / "skeleton_toggle.txt"
        try:
            path.write_text("on" if enabled else "off")
            self._log(f"[BACKEND] skeleton {'on' if enabled else 'off'}")
        except Exception as exc:
            self._log(f"[BACKEND] skeleton toggle failed: {exc}")
