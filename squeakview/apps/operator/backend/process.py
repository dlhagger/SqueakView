from __future__ import annotations

"""Subprocess helpers for the operator GUI."""

import os
import json
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from squeakview import config as squeakview_config


def _now() -> str:
    return time.strftime("%H:%M:%S")


WORKSPACE = squeakview_config.WORKSPACE

INFERENCE_ENTRY = "squeakview.apps.inference.main"


@dataclass(slots=True)
class LaunchConfig:
    capture_backend: str = "flir_direct"
    width: int | None = 1440
    height: int | None = 1080
    fps: int | None = None
    pixel_format: str | None = None
    trigger_on: bool = False
    trigger_activation: str = "rising"
    ds_cfg: Path | None = squeakview_config.DEFAULT_INFER_CONFIG
    inference_enabled: bool = True
    num_cameras: int = 1
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
    task_cfg: Path | None = None
    bottles: dict[str, object] = field(default_factory=dict)


def _prepend_env_path(env: dict[str, str], key: str, paths: list[Path]) -> None:
    values = [str(path) for path in paths if path.exists()]
    current = os.environ.get(key, "")
    if current:
        values.append(current)
    if values:
        env[key] = os.pathsep.join(values)


def _deepstream_runtime_env() -> dict[str, str]:
    env: dict[str, str] = {}
    sdk = squeakview_config.DEEPSTREAM_SDK_ROOT
    _prepend_env_path(env, "LD_LIBRARY_PATH", [sdk / "lib"])
    _prepend_env_path(env, "GST_PLUGIN_PATH", [sdk / "lib" / "gst-plugins"])
    return env


def _resolve_infer_config_path(raw: str, config_dir: Path) -> Path:
    path = Path(raw.strip().strip('"')).expanduser()
    if path.is_absolute():
        resolved = squeakview_config.resolve_workspace_path(path)
        return resolved if resolved is not None else path
    return (config_dir / path).resolve()


def _localize_deepstream_config(config_path: Path, run_dir: Path | None, emit: Callable[[str], None]) -> Path:
    """Write a run-local nvinfer config with paths resolved for this clone."""
    config_path = Path(config_path).expanduser().resolve()
    if run_dir is None:
        return config_path
    try:
        lines = config_path.read_text().splitlines()
    except Exception:
        return config_path

    config_dir = config_path.parent
    path_keys = {"onnx-file", "model-engine-file", "labelfile-path", "custom-lib-path"}
    localized_lines: list[str] = []
    changed = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            localized_lines.append(line)
            continue
        key, raw_value = line.split("=", 1)
        key_name = key.strip()
        if key_name not in path_keys:
            localized_lines.append(line)
            continue
        resolved = _resolve_infer_config_path(raw_value, config_dir)
        new_line = f"{key}={resolved}"
        localized_lines.append(new_line)
        changed = changed or new_line != line

    sidecar_path = config_path.with_name(f"{config_path.stem}.pose.json")
    localized_dir = Path(run_dir) / "deepstream_config"
    localized_dir.mkdir(parents=True, exist_ok=True)
    localized_config = localized_dir / config_path.name
    localized_config.write_text("\n".join(localized_lines) + "\n")

    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text())
            if isinstance(sidecar, dict) and sidecar.get("keypoint_labels_path"):
                kp_path = _resolve_infer_config_path(str(sidecar["keypoint_labels_path"]), sidecar_path.parent)
                sidecar["keypoint_labels_path"] = str(kp_path)
            localized_sidecar = localized_dir / sidecar_path.name
            localized_sidecar.write_text(json.dumps(sidecar, indent=2) + "\n")
        except Exception as exc:
            emit(f"【DS】 WARN: could not localize pose sidecar {sidecar_path}: {exc}")

    if changed:
        emit(f"【DS】 localized DeepStream config → {localized_config}")
    return localized_config


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
                clean = line.rstrip()
                if not clean.strip() or _should_suppress_child_output(clean):
                    continue
                self.emit(f"[{_now()}] {self.name} {clean}")
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


def _should_suppress_child_output(line: str) -> bool:
    if os.environ.get("SQUEAKVIEW_SHOW_PLUGIN_WARNINGS") == "1":
        return False
    return (
        "gst-plugin-scanner" in line
        and (
            "libnvdsgst_inferserver.so" in line
            or "libnvdsgst_udp.so" in line
            or "libtritonserver.so" in line
            or "librivermax.so" in line
        )
    )


def _spawn(
    module: str,
    args: Sequence[str],
    emit: Callable[[str], None],
    name: str,
    extra_env: dict[str, str] | None = None,
) -> ProcessHandle:
    cmd = [sys.executable, "-m", module, *args]
    emit(f"{name} CMD: {' '.join(shlex.quote(c) for c in cmd)}")
    env = os.environ.copy()
    pkg_root = str(WORKSPACE)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{pkg_root}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = pkg_root
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
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


def spawn_inference(config: LaunchConfig, emit: Callable[[str], None]) -> ProcessHandle:
    backend = str(getattr(config, "capture_backend", "flir_direct") or "flir_direct").lower().strip()
    if backend != "flir_direct":
        raise RuntimeError(f"SqueakView only supports capture_backend='flir_direct' (got {backend!r})")

    args: list[str] = []
    if config.ds_cfg is not None:
        ds_cfg = squeakview_config.resolve_workspace_path(config.ds_cfg)
        if ds_cfg is not None:
            ds_cfg = _localize_deepstream_config(ds_cfg, config.run_dir, emit)
            config.ds_cfg = ds_cfg
        args += ["--cfg", str(ds_cfg)]
    args += ["--capture-backend", backend]
    args += ["--num-cameras", str(max(1, int(getattr(config, "num_cameras", 1))))]
    if config.pixel_format:
        args += ["--pixel-format", str(config.pixel_format)]
    args += ["--trigger", "on" if bool(getattr(config, "trigger_on", False)) else "off"]
    args += ["--trigger-activation", str(getattr(config, "trigger_activation", "rising") or "rising")]
    if config.exposure_us is not None:
        args += ["--exposure-us", str(config.exposure_us)]
    args += ["--gain", "-1"]
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
    extra_env = _deepstream_runtime_env()
    if backend == "flir_direct":
        plugin_dir = squeakview_config.FLIR_GST_PLUGIN_DIR
        if plugin_dir.exists():
            existing = extra_env.get("GST_PLUGIN_PATH") or os.environ.get("GST_PLUGIN_PATH", "")
            paths = [str(plugin_dir)]
            if existing:
                paths.append(existing)
            extra_env["GST_PLUGIN_PATH"] = os.pathsep.join(paths)
            emit(f"【DS】 GST_PLUGIN_PATH includes {plugin_dir}")
        else:
            emit(f"【DS】 WARN: FLIR direct plugin build directory not found: {plugin_dir}")
    return _spawn(INFERENCE_ENTRY, args, emit, "【DS】", extra_env=extra_env)
