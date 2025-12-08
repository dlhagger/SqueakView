from __future__ import annotations

"""Behavior dashboard widget reused inside the unified operator GUI."""

import os
import queue
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Iterable

from PySide6 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("pyqtgraph is required for the dashboard") from exc

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from squeakview import config as squeakview_config
from squeakview.common import dashboard as dash_util


class JetsonMeters(QtCore.QObject):
    updated = QtCore.Signal(float, float, float, float, str)

    def __init__(self, parent=None, interval_ms: int = 500) -> None:
        super().__init__(parent)
        self._have_tegrastats = shutil.which("tegrastats") is not None
        self._proc = None
        self._thread = None
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=20)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._drain)
        self._timer.start(interval_ms)
        self._last_io = None  # (timestamp, read_bytes, write_bytes)
        self._io_device = self._resolve_disk_device()

        if self._have_tegrastats:
            try:
                self._proc = QtCore.QProcess()
                self._proc.setProgram("tegrastats")
                self._proc.setArguments(["--interval", str(interval_ms)])
                self._proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
                self._proc.readyRead.connect(self._consume)
                self._proc.start()
            except Exception:
                self._proc = None

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
            m = re.search(pat, raw, re.IGNORECASE)
            if m:
                return float(m.group(1))
        return float("nan")

    def _parse_gpu(self, raw: str) -> float:
        m = re.search(r"\bGR3D[_ ]FREQ\s+(\d+)%", raw, re.IGNORECASE)
        if not m:
            m = re.search(r"\bGR3D\s+(\d+)%", raw, re.IGNORECASE)
        return float(m.group(1)) if m else float("nan")

    def _parse_ram(self, raw: str) -> float:
        m = re.search(r"RAM\s+(\d+)/(\d+)MB", raw)
        if not m:
            return float("nan")
        used, total = float(m.group(1)), float(m.group(2))
        if total <= 0:
            return float("nan")
        return (used / total) * 100.0

    def _drain(self) -> None:
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
            self.updated.emit(float("nan"), float("nan"), cpu_pct if cpu_pct == cpu_pct else -1.0, disk_pct, "")
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
        """Best-effort map the runs directory to a disk device for per-disk IO counters."""
        if not psutil:
            return None
        try:
            target = Path(squeakview_config.RUNS_DIR).resolve()
            best = None
            best_len = -1
            for part in psutil.disk_partitions(all=False):
                try:
                    mnt = Path(part.mountpoint).resolve()
                except Exception:
                    continue
                if not str(target).startswith(str(mnt)):
                    continue
                if len(str(mnt)) > best_len:
                    best = part
                    best_len = len(str(mnt))
            if best is None:
                return None
            dev = best.device
            if not dev:
                return None
            name = os.path.basename(dev)
            # drop partition suffix if needed (e.g., nvme0n1p1 -> nvme0n1)
            name = name.rstrip("0123456789")
            return name
        except Exception:
            return None

    def stop(self) -> None:
        try:
            if self._proc:
                self._proc.terminate()
        except Exception:
            pass


class MetersBar(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)

        def mk(title: str):
            widget = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(widget)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(6)
            label = QtWidgets.QLabel(title)
            label.setMinimumWidth(46)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%p%")
            bar.setTextVisible(True)
            bar.setFixedHeight(16)
            hl.addWidget(label)
            hl.addWidget(bar, 1)
            return widget, bar

        self.ram_wrap, self.ram_bar = mk("RAM")
        self.gpu_wrap, self.gpu_bar = mk("GPU")
        self.cpu_wrap, self.cpu_bar = mk("CPU")
        self.disk_wrap, self.disk_bar = mk("DISK")

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
            "QProgressBar { background-color: #2a2a2a; color: #ddd; border: 1px solid #444; border-radius: 4px; }"
            f" QProgressBar::chunk {{ background-color: {chunk}; }}"
        )

    def _apply(self, bar: QtWidgets.QProgressBar, pct: float | None) -> None:
        if pct is None or pct < 0 or pct != pct:
            bar.setValue(0)
            bar.setFormat("N/A")
            bar.setStyleSheet(self._style_for_pct(-1.0))
        else:
            val = int(round(pct))
            bar.setValue(val)
            bar.setFormat(f"{val}%")
            bar.setStyleSheet(self._style_for_pct(pct))

    def set_ram_pct(self, pct: float | None) -> None:
        self._apply(self.ram_bar, pct)

    def set_gpu_pct(self, pct: float | None) -> None:
        self._apply(self.gpu_bar, pct)

    def set_cpu_pct(self, pct: float | None) -> None:
        self._apply(self.cpu_bar, pct)

    def set_disk_pct(self, pct: float | None) -> None:
        self._apply(self.disk_bar, pct)


class BehaviorDashboard(QtWidgets.QWidget):
    def __init__(self, window_sec: float = 300.0, pellet_mode: str = "arrival", parent=None) -> None:
        super().__init__(parent)
        self.window_sec = float(max(30.0, window_sec))
        self.pellet_mode = pellet_mode

        self.counters = {
            "POKE_L": 0,
            "POKE_R": 0,
            "DRINK_L": 0,
            "DRINK_R": 0,
            "PELLET": 0,
            "WELL": 0,
        }
        self.series_x = {k: [] for k in self.counters}
        self.series_y = {k: [] for k in self.counters}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.meters = MetersBar()
        layout.addWidget(self.meters)
        self.counts_label = QtWidgets.QLabel("Counts: --")
        self.counts_label.setStyleSheet("color: #cfd4ea; font-size: 12px; padding: 2px 4px;")
        layout.addWidget(self.counts_label)

        pg.setConfigOptions(antialias=False, useOpenGL=False)
        pg.setConfigOption("background", "k")
        pg.setConfigOption("foreground", "w")

        def mk_plot(title: str):
            axis = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation="bottom", fmt="%H:%M:%S")
            plot = pg.PlotWidget(axisItems={"bottom": axis})
            plot.setTitle(title)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis("left").setTextPen(pg.mkPen("#cfd4ea"))
            plot.getAxis("bottom").setTextPen(pg.mkPen("#cfd4ea"))
            plot.setBackground("#0f1118")
            return plot

        self.plot_poke = mk_plot("POKE")
        layout.addWidget(self.plot_poke, 1)
        self.plot_poke.addLegend(offset=(10, 10), labelTextColor="#cfd4ea", brush=pg.mkBrush(20, 20, 30, 200))
        self.cur_poke_L = self.plot_poke.plot(
            pen=pg.mkPen("#37d67a", width=2.5),
            name="Left poke",
            fillLevel=0,
            brush=pg.mkBrush(55, 214, 122, 60),
        )
        self.cur_poke_R = self.plot_poke.plot(
            pen=pg.mkPen("#6fa8ff", width=2.5),
            name="Right poke",
            fillLevel=0,
            brush=pg.mkBrush(111, 168, 255, 60),
        )

        self.plot_drink = mk_plot("DRINK")
        layout.addWidget(self.plot_drink, 1)
        self.plot_drink.addLegend(offset=(10, 10), labelTextColor="#cfd4ea", brush=pg.mkBrush(20, 20, 30, 200))
        self.cur_drink_L = self.plot_drink.plot(
            pen=pg.mkPen("#a29bfe", width=2.5),
            name="Left drink",
            fillLevel=0,
            brush=pg.mkBrush(162, 155, 254, 60),
        )
        self.cur_drink_R = self.plot_drink.plot(
            pen=pg.mkPen("#ff7eb6", width=2.5),
            name="Right drink",
            fillLevel=0,
            brush=pg.mkBrush(255, 126, 182, 60),
        )

        self.plot_pellet = mk_plot("PELLET & WELL_CHECK")
        layout.addWidget(self.plot_pellet, 1)
        self.plot_pellet.addLegend(offset=(10, 10), labelTextColor="#cfd4ea", brush=pg.mkBrush(20, 20, 30, 200))
        self.cur_pellet = self.plot_pellet.plot(pen=pg.mkPen("#f5a623", width=2.5), name="Pellet", fillLevel=0, brush=pg.mkBrush(245, 166, 35, 60))
        self.cur_well = self.plot_pellet.plot(pen=pg.mkPen("#50e3c2", width=2.5), name="Well check", fillLevel=0, brush=pg.mkBrush(80, 227, 194, 60))

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)

        self._meters = JetsonMeters(self, interval_ms=500)
        self._meters.updated.connect(self._on_meters)

    def detach_meters(self) -> MetersBar:
        """Detach the meters widget so it can be re-parented elsewhere."""
        layout = self.layout()
        if layout is not None:
            for idx in range(layout.count()):
                item = layout.itemAt(idx)
                if item and item.widget() is self.meters:
                    layout.takeAt(idx)
                    break
        self.meters.setParent(None)
        return self.meters

    @QtCore.Slot(float, float, float, float, str)
    def _on_meters(self, ram_pct: float, gpu_pct: float, cpu_pct: float, disk_pct: float, _raw: str) -> None:
        self.meters.set_ram_pct(None if ram_pct < 0 else ram_pct)
        self.meters.set_gpu_pct(None if gpu_pct < 0 else gpu_pct)
        self.meters.set_cpu_pct(None if cpu_pct < 0 else cpu_pct)
        self.meters.set_disk_pct(None if disk_pct < 0 else disk_pct)

    @QtCore.Slot(str)
    def ingest(self, raw: str) -> None:
        data = dash_util.parse_line(raw)
        if not data:
            return
        event = str(data.get("event_uc", ""))
        tsec = dash_util.choose_event_time(data)
        now = time.time()
        if tsec < (now - 2.0 * self.window_sec):
            tsec = now

        if "POKE" in event and dash_util.is_start_event(data):
            key = "POKE_R" if data.get("side_uc") == "R" else "POKE_L"
            self._append_point(key, tsec)
        elif "DRINK" in event and dash_util.is_start_event(data):
            key = "DRINK_R" if data.get("side_uc") == "R" else "DRINK_L"
            self._append_point(key, tsec)
        elif "PELLET" in event:
            ok = (
                (self.pellet_mode == "arrival" and (dash_util.is_start_event(data) or "ARRIVAL" in event))
                or (self.pellet_mode == "retrieval" and (dash_util.is_end_event(data) or "RETRIEVAL" in event))
                or (
                    self.pellet_mode == "both"
                    and (
                        dash_util.is_start_event(data)
                        or dash_util.is_end_event(data)
                        or "ARRIVAL" in event
                        or "RETRIEVAL" in event
                    )
                )
            )
            if ok:
                self._append_point("PELLET", tsec)
        elif "WELL_CHECK" in event and dash_util.is_start_event(data):
            self._append_point("WELL", tsec)

    def _append_point(self, key: str, tsec: float) -> None:
        prev = self.counters[key]
        new_value = prev + 1
        xs, ys = self.series_x[key], self.series_y[key]

        if xs and xs[-1] == tsec:
            ys[-1] = new_value
        else:
            xs.append(tsec)
            ys.append(prev)
            xs.append(tsec)
            ys.append(new_value)

        self.counters[key] = new_value

    def _refresh(self) -> None:
        now = time.time()
        half = self.window_sec / 2.0
        xstart, xend = (now - half), (now + half)

        for key in list(self.series_x.keys()):
            xs, ys = self.series_x[key], self.series_y[key]
            while len(xs) >= 2 and xs[1] < xstart:
                xs.pop(0)
                ys.pop(0)
                xs.pop(0)
                ys.pop(0)
            if xs and xs[0] < xstart:
                xs[0] = xstart

        def set_curve(curve, x, y):
            if x and y:
                if x[-1] < now:
                    curve.setData(x + [now], y + [y[-1]])
                else:
                    curve.setData(x, y)
            else:
                curve.setData([], [])

        set_curve(self.cur_poke_L, self.series_x["POKE_L"], self.series_y["POKE_L"])
        set_curve(self.cur_poke_R, self.series_x["POKE_R"], self.series_y["POKE_R"])
        ymax_poke = max(
            [
                1,
                self.series_y["POKE_L"][-1] if self.series_y["POKE_L"] else 0,
                self.series_y["POKE_R"][-1] if self.series_y["POKE_R"] else 0,
            ]
        )
        self.plot_poke.setXRange(xstart, xend, padding=0.0)
        self.plot_poke.setYRange(0, max(1.0, ymax_poke * 1.2), padding=0.0)

        set_curve(self.cur_drink_L, self.series_x["DRINK_L"], self.series_y["DRINK_L"])
        set_curve(self.cur_drink_R, self.series_x["DRINK_R"], self.series_y["DRINK_R"])
        ymax_drink = max(
            [
                1,
                self.series_y["DRINK_L"][-1] if self.series_y["DRINK_L"] else 0,
                self.series_y["DRINK_R"][-1] if self.series_y["DRINK_R"] else 0,
            ]
        )
        self.plot_drink.setXRange(xstart, xend, padding=0.0)
        self.plot_drink.setYRange(0, max(1.0, ymax_drink * 1.2), padding=0.0)

        set_curve(self.cur_pellet, self.series_x["PELLET"], self.series_y["PELLET"])
        set_curve(self.cur_well, self.series_x["WELL"], self.series_y["WELL"])
        ymax_pw = max(
            [
                1,
                self.series_y["PELLET"][-1] if self.series_y["PELLET"] else 0,
                self.series_y["WELL"][-1] if self.series_y["WELL"] else 0,
            ]
        )
        self.plot_pellet.setXRange(xstart, xend, padding=0.0)
        self.plot_pellet.setYRange(0, max(1.0, ymax_pw * 1.2), padding=0.0)
        # Update counts label
        self.counts_label.setText(
            "Counts: "
            f"POKE L {self.counters['POKE_L']} · POKE R {self.counters['POKE_R']} | "
            f"DRINK L {self.counters['DRINK_L']} · DRINK R {self.counters['DRINK_R']} | "
            f"PELLET {self.counters['PELLET']} · WELL {self.counters['WELL']}"
        )

    def close(self) -> None:
        try:
            self._meters.stop()
        except Exception:
            pass
