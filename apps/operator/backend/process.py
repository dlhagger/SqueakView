from __future__ import annotations

"""Subprocess helpers for the operator GUI."""

import os
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from squeakview import config as squeakview_config


def _now() -> str:
    return time.strftime("%H:%M:%S")


WORKSPACE = squeakview_config.WORKSPACE
CAPTURE_ENV = WORKSPACE / "environments" / "capture"
INFERENCE_ENV = WORKSPACE / "environments" / "inference"

CAPTURE_ENTRY = "squeakview.apps.capture.main"
INFERENCE_ENTRY = "squeakview.apps.inference.main"


@dataclass(slots=True)
class LaunchConfig:
    width: int | None = 1440
    height: int | None = 1080
    fps: int | None = None
    pixel_format: str | None = None
    trigger_on: bool = True
    trigger_activation: str = "rising"
    ds_cfg: Path = squeakview_config.DEEPSTREAM_ROOT / "configs" / "yolo11n_pose_fp32.txt"
    inference_enabled: bool = True
    socket_path: str = "/tmp/cam.sock"
    bitrate: int = 4000
    exposure_us: float | None = 10000.0
    serial_enabled: bool = True
    serial_port: str = "/dev/ttyACM0"
    serial_baud: int = 115200
    arduino_fps: int = 30
    preview_window_id: int | None = None
    run_dir: Path | None = None
    mouse_id: str | None = None
    experiment_name: str | None = None
    draw_skeleton: bool = False


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


def _python_from_env(env_dir: Path) -> Path:
    return env_dir / ".venv" / "bin" / "python"


def _spawn(env_dir: Path, module: str, args: Sequence[str], emit: Callable[[str], None], name: str) -> ProcessHandle:
    python = _python_from_env(env_dir)
    cmd = [str(python), "-m", module, *args]
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
    args: list[str] = []
    if config.width:
        args += ["--width", str(config.width)]
    if config.height:
        args += ["--height", str(config.height)]
    if config.fps:
        args += ["--fps", str(config.fps)]
    if config.pixel_format:
        args += ["--pix", config.pixel_format]
    if config.exposure_us is not None:
        args += ["--exposure-us", str(config.exposure_us)]
    args += ["--trigger", "on" if config.trigger_on else "off"]
    args += ["--activation", config.trigger_activation]
    args += ["--socket", config.socket_path]
    return _spawn(CAPTURE_ENV, CAPTURE_ENTRY, args, emit, "【CAP】")


def spawn_inference(config: LaunchConfig, emit: Callable[[str], None]) -> ProcessHandle:
    args: list[str] = ["--sock", config.socket_path, "--cfg", str(config.ds_cfg)]
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
    return _spawn(INFERENCE_ENV, INFERENCE_ENTRY, args, emit, "【DS】")
