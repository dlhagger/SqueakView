"""Generic child-process output and process-group supervision."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Sequence


DEFAULT_OUTPUT_LOG_MAX_BYTES = 64 * 1024 * 1024
MAX_CHILD_OUTPUT_CHUNK = 64 * 1024


def _now() -> str:
    return time.strftime("%H:%M:%S")


def should_suppress_child_output(line: str) -> bool:
    if os.environ.get("SQUEAKVIEW_SHOW_PLUGIN_WARNINGS") == "1":
        return False
    return (
        "gst-plugin-scanner" in line
        and any(
            library in line
            for library in (
                "libnvdsgst_inferserver.so",
                "libnvdsgst_udp.so",
                "libtritonserver.so",
                "librivermax.so",
            )
        )
    )


class ProcessHandle:
    """Own one child process, its bounded output pump, and group shutdown."""

    def __init__(
        self,
        name: str,
        popen: subprocess.Popen[str],
        emit_fn: Callable[[str], None],
        on_exit: Callable[[int], None] | None = None,
        output_log_path: Path | None = None,
        output_log_max_bytes: int = DEFAULT_OUTPUT_LOG_MAX_BYTES,
    ) -> None:
        self.name = name
        self.p = popen
        self.emit = emit_fn
        self.on_exit = on_exit
        self.output_log_path = Path(output_log_path) if output_log_path else None
        self.output_log_max_bytes = max(0, int(output_log_max_bytes))
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        log_handle = None
        log_bytes = 0
        log_limit_reported = False
        try:
            if self.output_log_path is not None:
                try:
                    self.output_log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_bytes = (
                        self.output_log_path.stat().st_size
                        if self.output_log_path.exists()
                        else 0
                    )
                    log_handle = self.output_log_path.open(
                        "a", encoding="utf-8", buffering=1
                    )
                except OSError as exc:
                    self.emit(f"{self.name} diagnostic log unavailable: {exc}")
            for line in iter(
                lambda: self.p.stdout.readline(MAX_CHILD_OUTPUT_CHUNK), ""
            ):
                if not line:
                    break
                clean = line.rstrip()
                if not clean.strip() or should_suppress_child_output(clean):
                    continue
                if log_handle is not None and not log_limit_reported:
                    encoded = (clean + "\n").encode("utf-8", errors="replace")
                    remaining = self.output_log_max_bytes - log_bytes
                    if remaining >= len(encoded):
                        log_handle.write(encoded.decode("utf-8"))
                        log_bytes += len(encoded)
                    else:
                        marker = (
                            "[SQUEAKVIEW] diagnostic log size limit reached; "
                            "later child output remains available in the operator log\n"
                        )
                        marker_bytes = marker.encode("utf-8")
                        if remaining >= len(marker_bytes):
                            log_handle.write(marker)
                            log_bytes += len(marker_bytes)
                        log_limit_reported = True
                        self.emit(
                            f"{self.name} diagnostic log size limit reached at "
                            f"{self.output_log_max_bytes} bytes"
                        )
                self.emit(f"[{_now()}] {self.name} {clean}")
        except Exception as exc:
            self.emit(f"{self.name} output error: {exc}")
        finally:
            returncode = self.p.wait()
            if log_handle is not None:
                log_handle.close()
            if self.p.stdout is not None:
                self.p.stdout.close()
            if self.on_exit is not None:
                try:
                    self.on_exit(int(returncode))
                except Exception as exc:
                    self.emit(f"{self.name} exit callback error: {exc}")

    def is_running(self) -> bool:
        return self.p is not None and self.p.poll() is None

    def wait(self, timeout: float | None = None) -> int:
        """Wait for a concrete child exit code; never invent success."""

        return int(self.p.wait(timeout=timeout))

    def send_signal_group(self, sig: signal.Signals) -> bool:
        try:
            os.killpg(os.getpgid(self.p.pid), sig)
            return True
        except Exception as exc:
            self.emit(f"{self.name} signal error: {exc}")
            return False

    def terminate_group_graceful(
        self,
        first_sig: signal.Signals = signal.SIGINT,
        wait_s: float = 8.0,
        escalate: bool = True,
    ) -> None:
        if not self.is_running():
            return
        self.emit(f"{self.name} → send {first_sig.name}")
        self.send_signal_group(first_sig)
        deadline = time.monotonic() + max(0.0, wait_s)
        while self.is_running() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not self.is_running() or not escalate:
            return
        self.emit(f"{self.name} still running — SIGTERM")
        self.send_signal_group(signal.SIGTERM)
        term_deadline = time.monotonic() + 5.0
        while self.is_running() and time.monotonic() < term_deadline:
            time.sleep(0.1)
        if not self.is_running():
            return
        self.emit(f"{self.name} still running — SIGKILL")
        try:
            os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
        except Exception as exc:
            self.emit(f"{self.name} SIGKILL error: {exc}")
        try:
            self.p.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.emit(f"{self.name} did not exit within 5.0s after SIGKILL")


def spawn(
    module: str,
    args: Sequence[str],
    emit: Callable[[str], None],
    name: str,
    *,
    workspace: Path,
    extra_env: dict[str, str] | None = None,
    on_exit: Callable[[int], None] | None = None,
    output_log_path: Path | None = None,
    output_log_max_bytes: int = DEFAULT_OUTPUT_LOG_MAX_BYTES,
    parent_death_signal: signal.Signals | None = None,
) -> ProcessHandle:
    cmd = [sys.executable, "-m", module, *args]
    if parent_death_signal is not None:
        cmd = [
            sys.executable,
            "-m",
            "squeakview.apps.operator.backend.parent_death_exec",
            "--expected-parent-pid",
            str(os.getpid()),
            "--signal",
            str(int(parent_death_signal)),
            "--",
            *cmd,
        ]
    emit(f"{name} CMD: {' '.join(shlex.quote(c) for c in cmd)}")
    env = os.environ.copy()
    package_root = str(workspace)
    env["PYTHONPATH"] = (
        f"{package_root}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else package_root
    )
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    child = subprocess.Popen(
        cmd,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=env,
    )
    return ProcessHandle(
        name,
        child,
        emit,
        on_exit=on_exit,
        output_log_path=output_log_path,
        output_log_max_bytes=output_log_max_bytes,
    )


__all__ = [
    "DEFAULT_OUTPUT_LOG_MAX_BYTES",
    "MAX_CHILD_OUTPUT_CHUNK",
    "ProcessHandle",
    "should_suppress_child_output",
    "spawn",
]
