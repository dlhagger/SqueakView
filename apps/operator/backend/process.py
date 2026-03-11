from __future__ import annotations

"""Subprocess helpers for the operator GUI."""

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from squeakview import config as squeakview_config


def _now() -> str:
    return time.strftime("%H:%M:%S")


WORKSPACE = squeakview_config.WORKSPACE

CAPTURE_ENTRY = "squeakview.apps.capture.main"
INFERENCE_ENTRY = "squeakview.apps.inference.main"


@dataclass(slots=True)
class LaunchConfig:
    capture_backend: str = "flir"
    zed_depth_enabled: bool = False
    zed_depth_mode: str = "NEURAL"
    zed_depth_socket: str = "/tmp/cam_depth.sock"
    zed_depth_record: bool = False
    zed_confidence_threshold: int = 100
    zed_texture_confidence_threshold: int = 100
    zed_depth_minimum_distance_mm: int = 300
    zed_depth_maximum_distance_mm: int = 20000
    zed_fill_mode: bool = False
    zed_depth_stabilization: int = 30
    width: int | None = 1440
    height: int | None = 1080
    fps: int | None = None
    pixel_format: str | None = None
    trigger_on: bool = True
    trigger_activation: str = "rising"
    ds_cfg: Path | None = squeakview_config.DEEPSTREAM_ROOT / "configs" / "yolo11s_distilledFT_fp16.txt"
    inference_enabled: bool = True
    socket_path: str = "/tmp/cam.sock"
    socket_path_2: str = "/tmp/cam1.sock"
    num_cameras: int = 1
    bitrate: int = 4000
    exposure_us: float | None = 10000.0
    serial_enabled: bool = True
    serial_port: str = "/dev/ttyACM0"
    serial_baud: int = 115200
    arduino_fps: int = 30
    preview_window_id: int | None = None
    run_dir: Path | None = None
    capture_ready_path: Path | None = None
    capture_stats_path: Path | None = None
    capture_frame_log_path: Path | None = None
    capture_depth_ready_path: Path | None = None
    capture_depth_stats_path: Path | None = None
    capture_depth_frame_log_path: Path | None = None
    mouse_id: str | None = None
    experiment_name: str | None = None
    draw_skeleton: bool = False
    task_cfg: Path | None = None


class ProcessHandle:
    def __init__(self, name: str, popen: subprocess.Popen[str], emit_fn: Callable[[str], None]):
        self.name = name
        self.p = popen
        self.emit = emit_fn
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        try:
            for line in iter(self.p.stdout.readline, ""):
                if not line:
                    break
                self.emit(f"[{_now()}] {self.name} {line.rstrip()}")
        except Exception as exc:
            self.emit(f"{self.name} output error: {exc}")

    def is_running(self) -> bool:
        return self.p is not None and self.p.poll() is None

    def wait(self, timeout: float | None = None) -> None:
        try:
            self.p.wait(timeout=timeout)
        except Exception:
            pass

    def send_signal_group(self, sig: signal.Signals) -> bool:
        try:
            pgid = os.getpgid(self.p.pid)
            os.killpg(pgid, sig)
            return True
        except Exception as exc:
            self.emit(f"{self.name} signal error: {exc}")
            return False

    def terminate_group_graceful(
        self, first_sig: signal.Signals = signal.SIGINT, wait_s: float = 8.0, escalate: bool = True
    ) -> None:
        if not self.is_running():
            return
        self.emit(f"{self.name} → send {first_sig.name}")
        self.send_signal_group(first_sig)
        t0 = time.time()
        while self.is_running() and (time.time() - t0) < wait_s:
            time.sleep(0.1)
        if not self.is_running() or not escalate:
            return
        if self.is_running():
            self.emit(f"{self.name} still running — SIGTERM")
            self.send_signal_group(signal.SIGTERM)
        t1 = time.time()
        while self.is_running() and (time.time() - t1) < 5.0:
            time.sleep(0.1)
        if self.is_running():
            self.emit(f"{self.name} still running — SIGKILL")
            try:
                os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
            except Exception as exc:
                self.emit(f"{self.name} SIGKILL error: {exc}")


def _spawn(module: str, args: Sequence[str], emit: Callable[[str], None], name: str) -> ProcessHandle:
    cmd = [sys.executable, "-m", module, *args]
    emit(f"{name} CMD: {' '.join(shlex.quote(c) for c in cmd)}")
    env = os.environ.copy()
    pkg_root = str(WORKSPACE.parent)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{pkg_root}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = pkg_root
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(WORKSPACE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
        env=env,
    )
    return ProcessHandle(name, proc, emit)


def spawn_capture(config: LaunchConfig, emit: Callable[[str], None]) -> ProcessHandle:
    return spawn_capture_for_camera(config, emit, camera_index=0)


def spawn_capture_for_camera(
    config: LaunchConfig,
    emit: Callable[[str], None],
    *,
    camera_index: int,
    socket_path: str | None = None,
    socket_path_2: str | None = None,
    ready_path: Path | None = None,
    ready_path_2: Path | None = None,
    stats_path: Path | None = None,
    stats_path_2: Path | None = None,
    frame_log_path: Path | None = None,
    frame_log_path_2: Path | None = None,
    trigger_on: bool | None = None,
) -> ProcessHandle:
    backend = str(getattr(config, "capture_backend", "flir") or "flir").lower().strip()
    if backend == "zed":
        raise RuntimeError("spawn_capture_for_camera() is only valid for FLIR backend")
    args: list[str] = []
    if config.width:
        args += ["--width", str(config.width)]
    if config.height:
        args += ["--height", str(config.height)]
    if config.fps:
        args += ["--fps", str(config.fps)]
    if config.pixel_format and backend != "zed":
        args += ["--pix", config.pixel_format]
    if config.exposure_us is not None:
        args += ["--exposure-us", str(config.exposure_us)]
    args += ["--socket", socket_path or config.socket_path]
    args += ["--camera-index", str(int(camera_index))]
    effective_trigger = config.trigger_on if trigger_on is None else bool(trigger_on)
    args += ["--trigger", "on" if effective_trigger else "off"]
    args += ["--activation", config.trigger_activation]
    ready = ready_path if ready_path is not None else config.capture_ready_path
    stats = stats_path if stats_path is not None else config.capture_stats_path
    frame_log = frame_log_path if frame_log_path is not None else config.capture_frame_log_path
    if ready is not None:
        args += ["--ready-file", str(ready)]
    if stats is not None:
        args += ["--stats-file", str(stats)]
    if frame_log is not None:
        args += ["--frame-log", str(frame_log)]
    return _spawn(CAPTURE_ENTRY, args, emit, f"【CAP{camera_index}】")


def spawn_inference(config: LaunchConfig, emit: Callable[[str], None]) -> ProcessHandle:
    backend = str(getattr(config, "capture_backend", "flir") or "flir").lower().strip()
    args: list[str] = []
    if config.ds_cfg is not None:
        args += ["--cfg", str(config.ds_cfg)]
    args += ["--capture-backend", backend]
    args += ["--num-cameras", str(max(1, int(getattr(config, "num_cameras", 1))))]
    if backend != "zed":
        socks = [config.socket_path]
        if int(getattr(config, "num_cameras", 1)) > 1:
            if config.socket_path_2:
                socks.append(config.socket_path_2)
        for sock in socks:
            args += ["--sock", sock]
    else:
        if bool(getattr(config, "zed_depth_enabled", False)):
            args.append("--zed-depth-enabled")
        args += ["--zed-depth-mode", str(getattr(config, "zed_depth_mode", "NEURAL"))]
        args += ["--zed-depth-socket", str(getattr(config, "zed_depth_socket", "/tmp/cam_depth.sock"))]
        args += ["--zed-confidence-threshold", str(int(getattr(config, "zed_confidence_threshold", 100) or 100))]
        args += [
            "--zed-texture-confidence-threshold",
            str(int(getattr(config, "zed_texture_confidence_threshold", 100) or 100)),
        ]
        args += [
            "--zed-depth-minimum-distance-mm",
            str(int(getattr(config, "zed_depth_minimum_distance_mm", 300) or 300)),
        ]
        args += [
            "--zed-depth-maximum-distance-mm",
            str(int(getattr(config, "zed_depth_maximum_distance_mm", 20000) or 20000)),
        ]
        args += ["--zed-depth-stabilization", str(int(getattr(config, "zed_depth_stabilization", 30) or 30))]
        if bool(getattr(config, "zed_fill_mode", False)):
            args.append("--zed-fill-mode")
        if bool(getattr(config, "zed_depth_record", False)):
            args.append("--zed-depth-record")
    if config.width:
        args += ["--width", str(config.width)]
    if config.height:
        args += ["--height", str(config.height)]
    if config.fps:
        args += ["--fps", str(config.fps)]
    args += ["--bitrate", str(config.bitrate)]
    if config.preview_window_id is not None:
        args += ["--window-xid", str(config.preview_window_id)]
    if config.run_dir is not None:
        args += ["--run-dir", str(config.run_dir)]
    if not config.inference_enabled:
        args.append("--disable-infer")
    if config.draw_skeleton:
        args.append("--draw-skeleton")
    return _spawn(INFERENCE_ENTRY, args, emit, "【DS】")
