from __future__ import annotations

"""Jetson system sampling and compact utilization meter widgets."""

import os
import queue
import re
import shutil
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from squeakview import config as squeakview_config
from squeakview.apps.operator.gui.run_health import read_latest_system_telemetry

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


class JetsonMeters(QtCore.QObject):
    """Emit live Jetson and disk utilization without blocking the GUI thread.

    During a capture, the GUI follows the capture-owned system telemetry file.
    Outside a capture it runs its own best-effort ``tegrastats`` process so the
    idle dashboard remains useful.
    """

    updated = QtCore.Signal(float, float, float, float, str)

    def __init__(self, parent=None, interval_ms: int = 500) -> None:
        super().__init__(parent)
        self._have_tegrastats = shutil.which("tegrastats") is not None
        self._proc = None
        self._run_system_csv: Path | None = None
        self._thread = None
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=20)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._drain)
        self._timer.start(interval_ms)
        self._last_io = None  # (timestamp, read_bytes, write_bytes)
        self._io_device = self._resolve_disk_device()

        self._interval_ms = interval_ms
        self._start_tegrastats()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop)

        if psutil:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    def _consume(self) -> None:
        if not self._proc:
            return
        try:
            data = bytes(self._proc.readAll()).decode(errors="replace")
        except Exception:
            return
        for line in data.splitlines():
            try:
                if self._queue.full():
                    self._queue.get_nowait()
                self._queue.put_nowait(line)
            except Exception:
                pass

    def _parse_cpu(self, raw: str) -> float:
        for pat in (
            r"\bCPU@\s*(\d+)%",
            r"\bCPU\s*@\s*(\d+)%",
            r"\bCPU\s+(\d+)%@",
            r"\bCPU\s*\[\s*(\d+)%@",
            r"\bCPU\s*\[\s*(\d+)%\s*\]",
        ):
            match = re.search(pat, raw, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return float("nan")

    def _parse_gpu(self, raw: str) -> float:
        match = re.search(r"\bGR3D[_ ]FREQ\s+(\d+)%", raw, re.IGNORECASE)
        if not match:
            match = re.search(r"\bGR3D\s+(\d+)%", raw, re.IGNORECASE)
        return float(match.group(1)) if match else float("nan")

    def _parse_ram(self, raw: str) -> float:
        match = re.search(r"RAM\s+(\d+)/(\d+)MB", raw)
        if not match:
            return float("nan")
        used, total = float(match.group(1)), float(match.group(2))
        if total <= 0:
            return float("nan")
        return (used / total) * 100.0

    def _drain(self) -> None:
        if self._run_system_csv is not None:
            row = read_latest_system_telemetry(self._run_system_csv)
            disk_pct = self._disk_pct()

            def value(name: str) -> float:
                try:
                    return float((row or {}).get(name, ""))
                except (TypeError, ValueError):
                    return -1.0

            self.updated.emit(
                value("ram_pct"),
                value("gpu_util_pct"),
                value("cpu_util_mean_pct"),
                disk_pct,
                (row or {}).get("raw_line", ""),
            )
            return
        raw = None
        try:
            while not self._queue.empty():
                raw = self._queue.get_nowait()
        except Exception:
            raw = None

        if raw is None:
            cpu_pct = float("nan")
            if psutil:
                try:
                    cpu_pct = psutil.cpu_percent(interval=None)
                except Exception:
                    cpu_pct = float("nan")
            disk_pct = self._disk_pct()
            self.updated.emit(
                float("nan"),
                float("nan"),
                cpu_pct if cpu_pct == cpu_pct else -1.0,
                disk_pct,
                "",
            )
            return

        ram_pct = self._parse_ram(raw)
        gpu_pct = self._parse_gpu(raw)
        cpu_pct = self._parse_cpu(raw)
        disk_pct = self._disk_pct()

        if cpu_pct != cpu_pct and psutil:
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
            except Exception:
                cpu_pct = float("nan")

        def norm(val: float) -> float:
            return val if val == val else -1.0

        self.updated.emit(norm(ram_pct), norm(gpu_pct), norm(cpu_pct), norm(disk_pct), raw)

    @staticmethod
    def _disk_pct() -> float:
        if psutil:
            try:
                usage = psutil.disk_usage("/")
                return (usage.used / usage.total) * 100.0 if usage.total else float("nan")
            except Exception:
                return float("nan")
        try:
            total, used, _ = shutil.disk_usage("/")
            return (used / total) * 100.0 if total else float("nan")
        except Exception:
            return float("nan")

    def _resolve_disk_device(self) -> str | None:
        """Best-effort map the runs directory to a disk device."""
        if not psutil:
            return None
        try:
            target = Path(squeakview_config.RUNS_DIR).resolve()
            best = None
            best_len = -1
            for part in psutil.disk_partitions(all=False):
                try:
                    mountpoint = Path(part.mountpoint).resolve()
                except Exception:
                    continue
                if not str(target).startswith(str(mountpoint)):
                    continue
                if len(str(mountpoint)) > best_len:
                    best = part
                    best_len = len(str(mountpoint))
            if best is None or not best.device:
                return None
            name = os.path.basename(best.device)
            # Preserve the historical best-effort partition normalization.
            return name.rstrip("0123456789")
        except Exception:
            return None

    def _start_tegrastats(self) -> None:
        if not self._have_tegrastats or self._proc is not None:
            return
        process = None
        try:
            process = QtCore.QProcess(self)
            process.setProgram("tegrastats")
            process.setArguments(["--interval", str(self._interval_ms)])
            process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            process.readyRead.connect(self._consume)
            self._proc = process
            process.start()
        except Exception:
            if process is not None:
                try:
                    process.kill()
                    process.waitForFinished(2000)
                    process.close()
                    process.deleteLater()
                except Exception:
                    pass
            self._proc = None

    def _stop_tegrastats(self) -> None:
        process = self._proc
        self._proc = None
        try:
            if process:
                process.terminate()
                if not process.waitForFinished(500):
                    process.kill()
                    process.waitForFinished(2000)
                process.close()
                process.deleteLater()
        except Exception:
            pass

    def follow_run_telemetry(self, path: Path) -> None:
        self._run_system_csv = Path(path)
        self._stop_tegrastats()

    def resume_idle_sampling(self) -> None:
        self._run_system_csv = None
        self._start_tegrastats()

    def stop(self) -> None:
        self._run_system_csv = None
        self._stop_tegrastats()


class MetersBar(QtWidgets.QWidget):
    """Four compact, color-coded utilization meters."""

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        def make_meter(title: str):
            widget = QtWidgets.QWidget()
            meter_layout = QtWidgets.QHBoxLayout(widget)
            meter_layout.setContentsMargins(0, 0, 0, 0)
            meter_layout.setSpacing(8)
            label = QtWidgets.QLabel(title)
            label.setMinimumWidth(44)
            label.setStyleSheet("color: #eef1ff; font-size: 12px; font-weight: 700;")
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%p%")
            bar.setTextVisible(True)
            bar.setFixedHeight(22)
            meter_layout.addWidget(label)
            meter_layout.addWidget(bar, 1)
            return widget, bar

        self.ram_wrap, self.ram_bar = make_meter("RAM")
        self.gpu_wrap, self.gpu_bar = make_meter("GPU")
        self.cpu_wrap, self.cpu_bar = make_meter("CPU")
        self.disk_wrap, self.disk_bar = make_meter("DISK")

        for bar in (self.ram_bar, self.gpu_bar, self.cpu_bar, self.disk_bar):
            bar.setStyleSheet(self._style_for_pct(-1.0))

        layout.addWidget(self.ram_wrap, 0, 0)
        layout.addWidget(self.gpu_wrap, 0, 1)
        layout.addWidget(self.cpu_wrap, 1, 0)
        layout.addWidget(self.disk_wrap, 1, 1)

    @staticmethod
    def _style_for_pct(pct: float) -> str:
        if pct < 0:
            chunk = "#777"
        elif pct < 50.0:
            chunk = "#2ecc71"
        elif pct < 80.0:
            chunk = "#f1c40f"
        else:
            chunk = "#e74c3c"
        return (
            "QProgressBar { background-color: #101526; color: #eef1ff; border: 1px solid #333a55; "
            "border-radius: 5px; text-align: center; font-size: 12px; font-weight: 700; }"
            f" QProgressBar::chunk {{ background-color: {chunk}; }}"
        )

    def _apply(self, bar: QtWidgets.QProgressBar, pct: float | None) -> None:
        if pct is None or pct < 0 or pct != pct:
            bar.setValue(0)
            bar.setFormat("N/A")
            bar.setStyleSheet(self._style_for_pct(-1.0))
        else:
            value = int(round(pct))
            bar.setValue(value)
            bar.setFormat(f"{value}%")
            bar.setStyleSheet(self._style_for_pct(pct))

    def set_ram_pct(self, pct: float | None) -> None:
        self._apply(self.ram_bar, pct)

    def set_gpu_pct(self, pct: float | None) -> None:
        self._apply(self.gpu_bar, pct)

    def set_cpu_pct(self, pct: float | None) -> None:
        self._apply(self.cpu_bar, pct)

    def set_disk_pct(self, pct: float | None) -> None:
        self._apply(self.disk_bar, pct)

__all__ = ["JetsonMeters", "MetersBar"]
