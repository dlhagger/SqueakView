from __future__ import annotations

"""Backend orchestrator for the operator GUI."""

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from squeakview.apps.operator.backend import process
from squeakview.common import run_context
from squeakview.common import serial as serial_util


def _now() -> str:
    return time.strftime("%H:%M:%S")


@dataclass(slots=True)
class RunState:
    capture: Optional[process.ProcessHandle] = None
    inference: Optional[process.ProcessHandle] = None
    serial: Optional[serial_util.SerialHandle] = None
    run_dir: Optional[Path] = None

    def any_running(self) -> bool:
        return bool(
            (self.capture and self.capture.is_running())
            or (self.inference and self.inference.is_running())
        )


class OperatorBackend:
    def __init__(self, emit_log: Callable[[str], None], ingest_dashboard: Optional[Callable[[str], None]] = None):
        self.emit = emit_log
        self.ingest = ingest_dashboard
        self.state = RunState()
        self.launch_cfg = process.LaunchConfig()
        self._metadata_written = False
        self._run_dir_watch_thread: threading.Thread | None = None

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
        """Best-effort attempt to crank the fan; non-fatal on failure."""
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
                "socket_path": cfg.socket_path,
                "deepstream_config": str(cfg.ds_cfg),
                "bitrate_kbps": cfg.bitrate,
                "preview_window_id": cfg.preview_window_id,
            },
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

    def start_run(self, cfg: process.LaunchConfig) -> bool:
        if self.state.any_running():
            self._log("[BACKEND] run already active")
            return False

        self.launch_cfg = cfg
        self.state.run_dir = None
        self._metadata_written = False

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
                if cfg.trigger_on:
                    try:
                        handle.send_line(f"START,{int(cfg.arduino_fps)}")
                        handle.wait_for_ttl(timeout_s=3.0)
                    except Exception:
                        pass

        mouse_id = (cfg.mouse_id or "").strip()
        if mouse_id:
            safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in mouse_id)
            prefix = safe_id
            run_dir = run_context.timestamped_run_dir(prefix, random_suffix=False)
        else:
            run_dir = run_context.timestamped_run_dir("ds")
        cfg.run_dir = run_dir
        self.state.run_dir = run_dir

        if serial_handle:
            serial_handle.set_csv_path(run_dir)
            self.state.serial = serial_handle

        self._ensure_metadata(run_dir)

        self.state.capture = process.spawn_capture(cfg, self._capture_emit)
        self._log("[CAP] capture launched")

        self.state.inference = process.spawn_inference(cfg, self._inference_emit)
        self._log("[DS] inference launched")
        self._start_run_dir_watch()
        return True

    def stop_run(self) -> None:
        self._log("[BACKEND] stopping run")
        self._maybe_set_run_dir_from_marker()
        if self.state.serial:
            try:
                self.state.serial.send_line("STOP")
            except Exception:
                pass
            # Give the Arduino a brief moment to close its loop before we tear everything down.
            try:
                time.sleep(0.1)
            except Exception:
                pass
        if self.state.inference and self.state.inference.is_running():
            self.state.inference.terminate_group_graceful(signal.SIGINT, 10.0, True)
            self.state.inference.wait(timeout=2)
        if self.state.capture and self.state.capture.is_running():
            self.state.capture.terminate_group_graceful(signal.SIGINT, 6.0, True)
            self.state.capture.wait(timeout=2)
        if self.state.serial:
            self.state.serial.close()
            self.state.serial = None
        self.state.capture = None
        self.state.inference = None

    def shutdown(self) -> None:
        self.stop_run()

    def set_preview_enabled(self, enabled: bool) -> None:
        run_dir = self._maybe_set_run_dir_from_marker()
        if not run_dir:
            self._log("[BACKEND] preview toggle ignored; run dir unknown")
            return
        path = run_dir / "preview_toggle.txt"
        try:
            path.write_text("on" if enabled else "off")
            self._log(f"[BACKEND] preview {'on' if enabled else 'off'}")
        except Exception as exc:
            self._log(f"[BACKEND] preview toggle failed: {exc}")

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

    def set_video_enabled(self, enabled: bool) -> None:
        """Toggle whether the live camera feed (vs black background) is shown in the overlay."""
        run_dir = self._maybe_set_run_dir_from_marker()
        if not run_dir:
            self._log("[BACKEND] video toggle ignored; run dir unknown")
            return
        path = run_dir / "video_toggle.txt"
        try:
            path.write_text("on" if enabled else "off")
            self._log(f"[BACKEND] video {'on' if enabled else 'off'}")
        except Exception as exc:
            self._log(f"[BACKEND] video toggle failed: {exc}")
