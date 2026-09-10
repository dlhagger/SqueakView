"""Bounded process recorder for Jetson system telemetry."""

from __future__ import annotations

import atexit
import csv
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, TextIO

from .tegrastats import parse_tegrastats_line, read_platform_metrics


SYSTEM_TELEMETRY_SCHEMA_VERSION = "1.0"
MAX_TELEMETRY_LINE_CHARS = 64 * 1024
MAX_TELEMETRY_SHUTDOWN_TIMEOUT_S = 60.0
SYSTEM_TELEMETRY_HEADERS = [
    "schema_version",
    "host_unix_ns",
    "host_monotonic_ns",
    "sample_index",
    "parse_status",
    "ram_used_mb",
    "ram_total_mb",
    "ram_pct",
    "lfb_blocks",
    "lfb_block_mb",
    "swap_used_mb",
    "swap_total_mb",
    "swap_cached_mb",
    "cpu_online",
    "cpu_total",
    "cpu_util_mean_pct",
    "cpu_util_max_pct",
    "cpu_freq_mean_mhz",
    "cpu_freq_max_mhz",
    "gpu_util_pct",
    "gpu_freq_mhz",
    "emc_util_pct",
    "emc_freq_mhz",
    "emc_freq_hz",
    "emc_max_freq_hz",
    "emc_clock_pct_of_max",
    "temp_cpu_c",
    "temp_gpu_c",
    "temp_tj_c",
    "temp_soc0_c",
    "temp_soc1_c",
    "temp_soc2_c",
    "temp_max_c",
    "thermal_throttled",
    "thermal_throttle_domains",
    "vdd_in_current_mw",
    "vdd_in_avg_mw",
    "vdd_in_peak_mw",
    "vdd_cpu_gpu_cv_current_mw",
    "vdd_cpu_gpu_cv_avg_mw",
    "vdd_cpu_gpu_cv_peak_mw",
    "vdd_soc_current_mw",
    "vdd_soc_avg_mw",
    "vdd_soc_peak_mw",
    "raw_line",
]


def _bounded_shutdown_timeout(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(max(0.0, number), MAX_TELEMETRY_SHUTDOWN_TIMEOUT_S)


class SystemTelemetryRecorder:
    """Stream tegrastats samples directly to CSV with constant memory usage.

    Monitoring is deliberately best-effort: ``start`` returns ``False`` and
    records ``last_error`` when telemetry cannot be opened or launched. It never
    raises a monitoring failure into the scientific capture path.
    """

    def __init__(
        self,
        path: Path,
        *,
        interval_ms: int = 1000,
        command: str = "tegrastats",
        terminate_timeout_s: float = 2.0,
        kill_timeout_s: float = 1.0,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        parse_line: Callable[[str], dict[str, object]] = parse_tegrastats_line,
        platform_reader: Callable[[], dict[str, object]] = read_platform_metrics,
        unix_ns: Callable[[], int] = time.time_ns,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        on_error: Callable[[str], None] | None = None,
        shutdown_requested: Callable[[], bool] = lambda: False,
    ) -> None:
        self.path = Path(path)
        self.interval_ms = max(500, int(interval_ms))
        self.command = command
        self.terminate_timeout_s = _bounded_shutdown_timeout(
            terminate_timeout_s, 2.0
        )
        self.kill_timeout_s = _bounded_shutdown_timeout(kill_timeout_s, 1.0)
        self._popen_factory = popen_factory
        self._parse_line = parse_line
        self._platform_reader = platform_reader
        self._unix_ns = unix_ns
        self._monotonic_ns = monotonic_ns
        self._on_error = on_error
        self._shutdown_requested = shutdown_requested
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self._write_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._sample_index = 0
        self._started = False
        self._closed = False
        self._atexit_registered = False
        self.last_error: str | None = None

    @property
    def sample_count(self) -> int:
        return self._sample_index

    def _report_error(self, message: str) -> None:
        self.last_error = message
        if self._on_error is not None:
            try:
                self._on_error(message)
            except Exception:
                pass

    def start(self) -> bool:
        with self._lifecycle_lock:
            if self._started and not self._closed:
                return True
            if self._closed:
                self._report_error("system telemetry recorder cannot be restarted after close")
                return False
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self.path.open("w", newline="", buffering=1)
                self._writer = csv.DictWriter(
                    self._file,
                    fieldnames=SYSTEM_TELEMETRY_HEADERS,
                    extrasaction="ignore",
                )
                self._writer.writeheader()
                self._process = self._popen_factory(
                    [self.command, "--interval", str(self.interval_ms)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                if self._process.stdout is None:
                    raise RuntimeError("tegrastats stdout pipe is unavailable")
                self._thread = threading.Thread(
                    target=self._pump,
                    name="squeakview-system-telemetry",
                    daemon=True,
                )
                self._started = True
                self._thread.start()
                atexit.register(self.stop)
                self._atexit_registered = True
                return True
            except Exception as exc:
                self._report_error(f"could not start system telemetry: {type(exc).__name__}: {exc}")
                process = self._process
                if process is not None:
                    try:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=self.terminate_timeout_s)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=self.kill_timeout_s)
                    except Exception:
                        pass
                self._close_file()
                return False

    def _pump(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            while True:
                line = process.stdout.readline(MAX_TELEMETRY_LINE_CHARS + 1)
                if not line:
                    break
                if len(line) > MAX_TELEMETRY_LINE_CHARS:
                    raise RuntimeError(
                        "tegrastats output record exceeds "
                        f"{MAX_TELEMETRY_LINE_CHARS} character limit"
                    )
                if not line.strip():
                    continue
                try:
                    row = self._parse_line(line)
                except Exception as exc:
                    row = {
                        "parse_status": "malformed",
                        "raw_line": line.rstrip("\r\n"),
                    }
                    self._report_error(
                        f"system telemetry parse error: {type(exc).__name__}: {exc}"
                    )
                try:
                    row.update(self._platform_reader())
                except Exception as exc:
                    self._report_error(
                        f"system telemetry platform read error: {type(exc).__name__}: {exc}"
                    )
                row.update(
                    schema_version=SYSTEM_TELEMETRY_SCHEMA_VERSION,
                    host_unix_ns=self._unix_ns(),
                    host_monotonic_ns=self._monotonic_ns(),
                    sample_index=self._sample_index,
                )
                with self._write_lock:
                    if self._writer is None:
                        break
                    self._writer.writerow(row)
                    self._sample_index += 1
        except Exception as exc:
            self._report_error(f"system telemetry reader stopped: {type(exc).__name__}: {exc}")
        else:
            returncode = process.poll()
            try:
                expected_shutdown = bool(self._shutdown_requested())
            except Exception:
                expected_shutdown = False
            if not self._closed and not expected_shutdown and returncode is not None:
                self._report_error(
                    "system telemetry process exited unexpectedly "
                    f"with code {returncode}"
                )

    def _close_file(self) -> None:
        with self._write_lock:
            file_handle = self._file
            self._writer = None
            self._file = None
            if file_handle is not None:
                try:
                    file_handle.flush()
                    file_handle.close()
                except OSError as exc:
                    self._report_error(
                        f"could not close system telemetry file: {type(exc).__name__}: {exc}"
                    )

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            thread = self._thread
            if process is not None:
                try:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=self.terminate_timeout_s)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=self.kill_timeout_s)
                except Exception as exc:
                    self._report_error(
                        f"could not stop system telemetry process: {type(exc).__name__}: {exc}"
                    )
                try:
                    if process.stdout is not None:
                        process.stdout.close()
                except OSError:
                    pass
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=self.terminate_timeout_s + self.kill_timeout_s + 0.5)
                if thread.is_alive():
                    self._report_error("system telemetry reader did not stop within its timeout")
            self._close_file()
            if self._atexit_registered:
                try:
                    atexit.unregister(self.stop)
                except Exception:
                    pass
                self._atexit_registered = False

    close = stop

    def __enter__(self) -> "SystemTelemetryRecorder":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop()


__all__ = [
    "MAX_TELEMETRY_LINE_CHARS",
    "SYSTEM_TELEMETRY_HEADERS",
    "SYSTEM_TELEMETRY_SCHEMA_VERSION",
    "SystemTelemetryRecorder",
]
