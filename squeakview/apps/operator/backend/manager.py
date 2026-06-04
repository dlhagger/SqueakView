from __future__ import annotations

"""Backend orchestrator for the operator GUI."""

import os
import importlib.util
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from squeakview.apps.operator.backend import process
from squeakview import config as squeakview_config
from squeakview.common import run_context
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
    def __init__(self, emit_log: Callable[[str], None], ingest_dashboard: Optional[Callable[[str], None]] = None):
        self.emit = emit_log
        self.ingest = ingest_dashboard
        self.state = RunState()
        self.launch_cfg = process.LaunchConfig()
        self._metadata_written = False
        self._run_dir_watch_thread: threading.Thread | None = None
        self._inference_ready = threading.Event()

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
            self._inference_ready.set()
        if "run dir:" in lower:
            path = lower.split("run dir:", 1)[1].strip()
            if os.path.isdir(path):
                run_path = Path(path)
                self.state.run_dir = run_path
                if self.state.serial:
                    self.state.serial.set_csv_path(run_path)
                self._ensure_metadata(run_path)
        else:
            self._maybe_set_run_dir_from_marker()

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

    def _ensure_metadata(self, run_dir: Path) -> None:
        if self._metadata_written:
            return
        cfg = self.launch_cfg
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
                "bitrate_kbps": cfg.bitrate,
                "preview_window_id": cfg.preview_window_id,
            },
            "recording": {
                "container": "mp4",
                "segmented": True,
                "pattern": "raw_%06d.mp4",
                "segment_seconds": int(os.environ.get("SQUEAKVIEW_RECORD_SEGMENT_SECONDS", "600")),
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
        except Exception as exc:  # pragma: no cover
            self._log(f"[BACKEND] metadata write failed: {exc}")

    def _run_post_run_alignment(self, run_dir: Path | None) -> None:
        enabled = os.environ.get("SQUEAKVIEW_AUTO_ALIGN", "1").lower()
        if enabled in {"0", "false", "no", "off"}:
            self._log("[ANALYSIS] skipped; SQUEAKVIEW_AUTO_ALIGN=0")
            return
        if run_dir is None:
            self._log("[ANALYSIS] skipped; run directory unknown")
            return

        run_dir = Path(run_dir)
        missing = [name for name in ("frames.csv", "serial.csv") if not (run_dir / name).exists()]
        if missing:
            self._log(f"[ANALYSIS] skipped; missing {', '.join(missing)}")
            return

        script_path = process.WORKSPACE / "scripts" / "align_run_outputs.py"
        if not script_path.exists():
            self._log(f"[ANALYSIS] skipped; align script missing: {script_path}")
            return

        try:
            self._log("[ANALYSIS] building aligned CSVs")
            spec = importlib.util.spec_from_file_location("squeakview_align_run_outputs", script_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"could not load {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            summary = module.build_alignment(run_dir, run_dir / "analysis")
            counts = summary.get("counts", {})
            validation = summary.get("validation", {})
            self._log(
                "[ANALYSIS] complete → "
                f"{run_dir / 'analysis'} "
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
            if warnings:
                self._log(f"[ANALYSIS] warnings: {', '.join(warnings)}")
            else:
                self._log("[ANALYSIS] validation passed")
        except Exception as exc:
            self._log(f"[ANALYSIS] failed: {exc}")


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

        self.launch_cfg = cfg
        self.state.run_dir = None
        self._metadata_written = False
        self._inference_ready.clear()

        self._set_fan_max()

        serial_handle: serial_util.SerialHandle | None = None
        if cfg.serial_enabled:
            if not serial_util.have_pyserial():
                self._log("[SER] pyserial unavailable; disabling serial")
                cfg.serial_enabled = False
            else:
                handle = serial_util.SerialHandle(cfg.serial_port, cfg.serial_baud, self._serial_emit)
                if not handle.open(None):
                    self._log("[SER] failed to open port; aborting run")
                    return False
                serial_handle = handle

        mouse_id = (cfg.mouse_id or "").strip()
        experiment_name = (cfg.experiment_name or "").strip()
        run_parent = None
        if experiment_name:
            safe_experiment = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in experiment_name)
            run_parent = run_context.RUNS_DIR / safe_experiment
        if mouse_id:
            safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in mouse_id)
            prefix = safe_id
            run_dir = run_context.timestamped_run_dir(prefix, random_suffix=False, parent=run_parent)
        else:
            run_dir = run_context.timestamped_run_dir("ds", parent=run_parent)
        cfg.run_dir = run_dir
        self.state.run_dir = run_dir

        if serial_handle:
            serial_handle.set_csv_path(run_dir)
            self.state.serial = serial_handle

        self._ensure_metadata(run_dir)

        capture_backend = str(getattr(cfg, "capture_backend", "flir_direct") or "flir_direct").lower().strip()
        if capture_backend != "flir_direct":
            self._log(f"[CAP] unsupported capture backend in SqueakView: {capture_backend}")
            if self.state.serial:
                self.state.serial.close()
                self.state.serial = None
            return False

        self._log("[CAP] FLIR direct capture will be sourced inside DeepStream (flirspinsrc)")
        self.state.inference = process.spawn_inference(cfg, self._inference_emit)
        self._log("[DS] inference launched")
        if serial_handle and cfg.trigger_on:
            try:
                self._log("[BACKEND] waiting for inference ready before START")
                ready = self._inference_ready.wait(timeout=8.0)
                if ready:
                    self._log("[BACKEND] inference ready; sending START")
                else:
                    self._log("[BACKEND] inference ready timeout; sending START anyway")
                serial_handle.log_marker("START_SENT")
                serial_handle.send_line(f"START,{int(cfg.arduino_fps)}")
                serial_handle.wait_for_ttl(timeout_s=3.0)
            except Exception:
                pass
        self._start_run_dir_watch()
        return True

    def stop_run(self) -> None:
        inference_running = bool(self.state.inference and self.state.inference.is_running())
        if not inference_running and not self.state.serial:
            return
        self._log("[BACKEND] stopping run")
        self._maybe_set_run_dir_from_marker()
        if inference_running:
            # In triggered mode, keep TTL pulses alive while DeepStream receives EOS.
            # Stopping the controller first can leave flirspinsrc blocked waiting
            # for a trigger, which turns shutdown into a source ERROR and prevents
            # mp4mux/splitmuxsink from finalizing the raw segment.
            if self.state.serial:
                self.state.serial.log_marker("CAPTURE_STOP_REQUESTED")
            self.state.inference.terminate_group_graceful(signal.SIGINT, 10.0, True)
            self.state.inference.wait(timeout=2)
            if self.state.serial:
                self.state.serial.log_marker("CAPTURE_STOP_DONE")
        if self.state.serial:
            try:
                self.state.serial.log_marker("STOP_SENT")
                self.state.serial.send_line("STOP")
            except Exception:
                pass
            # Drain ACK_STOP and final serial rows before closing the port.
            try:
                time.sleep(0.5)
            except Exception:
                pass
            self.state.serial.close()
            self.state.serial = None
        self.state.inference = None
        self._run_post_run_alignment(self.state.run_dir)

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
