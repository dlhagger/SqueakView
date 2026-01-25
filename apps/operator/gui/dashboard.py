from __future__ import annotations

"""Behavior dashboard widget reused inside the unified operator GUI."""

import json
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
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

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
        self.counters: dict[str, int] = {}
        self.series_x: dict[str, list[float]] = {}
        self.series_y: dict[str, list[int]] = {}
        self.series_order: list[str] = []
        self._rules: list[dict] = []
        self._plots: list[dict] = []
        self._task_cfg_path: Path | None = None

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

        self._plot_container = QtWidgets.QWidget(self)
        self._plot_layout = QtWidgets.QHBoxLayout(self._plot_container)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_layout.setSpacing(8)
        self._plot_left = QtWidgets.QWidget(self._plot_container)
        self._plot_left_layout = QtWidgets.QVBoxLayout(self._plot_left)
        self._plot_left_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_left_layout.setSpacing(8)
        self._plot_right = QtWidgets.QWidget(self._plot_container)
        self._plot_right_layout = QtWidgets.QVBoxLayout(self._plot_right)
        self._plot_right_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_right_layout.setSpacing(8)
        self._plot_layout.addWidget(self._plot_left, 3)
        self._plot_layout.addWidget(self._plot_right, 1)
        layout.addWidget(self._plot_container, 1)
        self._build_from_task_config(self._default_task_config())

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)

        self._meters = JetsonMeters(self, interval_ms=500)
        self._meters.updated.connect(self._on_meters)

    def apply_task_config(self, path: Path) -> None:
        cfg = self._load_task_config(path)
        self._task_cfg_path = path
        self._build_from_task_config(cfg)

    def _load_task_config(self, path: Path) -> dict:
        raw = path.read_text()
        suffix = path.suffix.lower()
        if suffix == ".json":
            return json.loads(raw)
        if yaml is None:
            raise RuntimeError("PyYAML is not installed; cannot read task config.")
        return yaml.safe_load(raw)

    @staticmethod
    def _default_task_config() -> dict:
        return {
            "task_name": "SqueakView Default",
            "schema_version": 1,
            "events": [
                {"name": "POKE", "match": {"event_contains": "POKE", "phase": "start"}, "split_by_side": True, "plot": "Poke"},
                {"name": "DRINK", "match": {"event_contains": "DRINK", "phase": "start"}, "split_by_side": True, "plot": "Drink"},
                {"name": "PELLET", "match": {"event_contains": "PELLET", "phase": "retrieval"}, "split_by_side": False, "plot": "Pellet"},
                {"name": "WELL_CHECK", "match": {"event_contains": "WELL_CHECK", "phase": "start"}, "split_by_side": False, "plot": "Pellet"},
            ],
            "dashboard": {
                "plots": [
                    {"id": "Poke", "title": "Pokes", "series": ["POKE_L", "POKE_R"]},
                    {"id": "Drink", "title": "Drinks", "series": ["DRINK_L", "DRINK_R"]},
                    {"id": "Pellet", "title": "Pellet & Well", "series": ["PELLET", "WELL_CHECK"]},
                ]
            },
        }

    def _build_from_task_config(self, cfg: dict) -> None:
        if not isinstance(cfg, dict):
            cfg = {}
        events = cfg.get("events") or []
        dashboard = cfg.get("dashboard") or {}
        plots_cfg = dashboard.get("plots") or []
        if not events or not plots_cfg:
            cfg = self._default_task_config()
            events = cfg["events"]
            plots_cfg = cfg["dashboard"]["plots"]

        self._rules = []
        for rule in events:
            name = str(rule.get("name", "")).strip()
            if not name:
                continue
            match = rule.get("match") or {}
            self._rules.append(
                {
                    "name": name.upper(),
                    "match": {k: str(v).strip() for k, v in match.items() if v is not None},
                    "split_by_side": bool(rule.get("split_by_side", False)),
                }
            )

        series_order: list[str] = []
        for plot in plots_cfg:
            for series in plot.get("series", []) or []:
                key = self._norm_series(series)
                if key and key not in series_order:
                    series_order.append(key)
        for rule in self._rules:
            base = rule["name"]
            if rule["split_by_side"]:
                for side in ("L", "R"):
                    key = f"{base}_{side}"
                    if key not in series_order:
                        series_order.append(key)
            else:
                if base not in series_order:
                    series_order.append(base)

        self.series_order = series_order
        self.counters = {k: 0 for k in series_order}
        self.series_x = {k: [] for k in series_order}
        self.series_y = {k: [] for k in series_order}

        self._clear_plots()
        for plot in plots_cfg:
            title = str(plot.get("title") or plot.get("id") or "Plot")
            plot_type = str(plot.get("type") or "timeseries").lower()
            if plot_type == "matrix":
                matrix = MatrixWidget(title, plot.get("layout") or {})
                self._plot_right_layout.addWidget(matrix, 1)
                series_keys = matrix.series_keys()
                self._plots.append({"kind": "matrix", "plot": matrix, "series": series_keys})
            else:
                widget = self._make_plot(title)
                self._plot_left_layout.addWidget(widget, 1)
                widget.addLegend(offset=(10, 10), labelTextColor="#cfd4ea", brush=pg.mkBrush(20, 20, 30, 200))
                series_keys = []
                curves: dict[str, object] = {}
                for idx, series in enumerate(plot.get("series", []) or []):
                    key = self._norm_series(series)
                    if not key:
                        continue
                    series_keys.append(key)
                    pen, brush = self._series_style(key, idx)
                    curve = widget.plot(pen=pen, name=self._series_label(key), fillLevel=0, brush=brush)
                    curves[key] = curve
                self._plots.append({"kind": "timeseries", "plot": widget, "series": series_keys, "curves": curves})

        self._update_counts_label()

    def _clear_plots(self) -> None:
        for entry in self._plots:
            plot = entry.get("plot")
            if plot is not None:
                try:
                    if entry.get("kind") == "matrix":
                        self._plot_right_layout.removeWidget(plot)
                    else:
                        self._plot_left_layout.removeWidget(plot)
                    plot.setParent(None)
                    plot.deleteLater()
                except Exception:
                    pass
        self._plots = []

    @staticmethod
    def _norm_series(name: str) -> str:
        return str(name).strip().upper()

    @staticmethod
    def _make_plot(title: str) -> pg.PlotWidget:
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

    @staticmethod
    def _series_label(key: str) -> str:
        labels = {
            "POKE_L": "Left poke",
            "POKE_R": "Right poke",
            "DRINK_L": "Left drink",
            "DRINK_R": "Right drink",
            "PELLET": "Pellet",
            "WELL_CHECK": "Well check",
            "WELL": "Well check",
            "GO_CORRECT": "Go correct",
            "GO_INCORRECT": "Go incorrect",
            "NOGO_CORRECT": "NoGo correct",
            "NOGO_INCORRECT": "NoGo incorrect",
        }
        return labels.get(key, key.replace("_", " ").title())

    @staticmethod
    def _series_style(key: str, idx: int):
        color_map = {
            "POKE_L": "#37d67a",
            "POKE_R": "#6fa8ff",
            "DRINK_L": "#a29bfe",
            "DRINK_R": "#ff7eb6",
            "PELLET": "#f5a623",
            "WELL_CHECK": "#50e3c2",
            "WELL": "#50e3c2",
            "GO_CORRECT": "#2ecc71",
            "GO_INCORRECT": "#e74c3c",
            "NOGO_CORRECT": "#4aa3df",
            "NOGO_INCORRECT": "#f39c12",
        }
        palette = ["#ffd166", "#06d6a0", "#118ab2", "#ef476f", "#9b5de5", "#f15bb5"]
        hex_color = color_map.get(key, palette[idx % len(palette)])
        pen = pg.mkPen(hex_color, width=2.5)
        color = QtGui.QColor(hex_color)
        color.setAlpha(60)
        brush = pg.mkBrush(color)
        return pen, brush

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
        if tsec < (now - 2.0 * self.window_sec) or tsec > (now + 2.0 * self.window_sec):
            tsec = now

        if self._rules:
            for rule in self._rules:
                if not self._match_rule(data, event, rule):
                    continue
                if rule["split_by_side"]:
                    side = str(data.get("side_uc", "")).upper()
                    if side not in ("L", "R"):
                        side = "L"
                    key = f"{rule['name']}_{side}"
                else:
                    key = rule["name"]
                if rule.get("use_count_field"):
                    count_val = self._parse_int_field(data.get("count"))
                    self._append_point(key, tsec, new_value=count_val)
                else:
                    self._append_point(key, tsec)
            return

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
            self._append_point("WELL_CHECK", tsec)

    def _append_point(self, key: str, tsec: float, *, new_value: int | None = None) -> None:
        if key not in self.counters:
            return
        prev = self.counters[key]
        if new_value is None:
            new_value = prev + 1
        elif new_value <= prev:
            return
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

        for plot_entry in self._plots:
            if plot_entry.get("kind") == "matrix":
                matrix = plot_entry["plot"]
                series = plot_entry.get("series", [])
                values = {key: self.counters.get(key, 0) for key in series}
                matrix.update_values(values)
                continue
            plot = plot_entry["plot"]
            keys = plot_entry["series"]
            ymax = 1
            for key in keys:
                curve = plot_entry["curves"].get(key)
                if curve is None:
                    continue
                xs = self.series_x.get(key, [])
                ys = self.series_y.get(key, [])
                set_curve(curve, xs, ys)
                if ys:
                    ymax = max(ymax, ys[-1])
            plot.setXRange(xstart, xend, padding=0.0)
            plot.setYRange(0, max(1.0, ymax * 1.2), padding=0.0)

        self._update_counts_label()

    def close(self) -> None:
        try:
            self._meters.stop()
        except Exception:
            pass

    def _match_rule(self, data: dict, event: str, rule: dict) -> bool:
        match = rule.get("match") or {}
        contains = match.get("event_contains")
        if contains and str(contains).upper() not in event:
            return False
        equals = match.get("event_equals")
        if equals and str(equals).upper() != event:
            return False
        reason_equals = match.get("reason_equals")
        if reason_equals and str(reason_equals).upper() != str(data.get("reason", "")).upper():
            return False
        reason_contains = match.get("reason_contains")
        if reason_contains and str(reason_contains).upper() not in str(data.get("reason", "")).upper():
            return False
        value_equals = match.get("value_equals")
        if value_equals is not None and str(value_equals) != str(data.get("value", "")).strip():
            return False
        side = match.get("side")
        if side and str(side).upper() != str(data.get("side_uc", "")).upper():
            return False
        phase = str(match.get("phase", "")).strip().lower()
        if phase in ("start", "arrival"):
            return dash_util.is_start_event(data)
        if phase in ("end", "retrieval"):
            return dash_util.is_end_event(data)
        return True

    def _update_counts_label(self) -> None:
        parts: list[str] = []
        for plot in self._plots:
            series = plot.get("series", [])
            group = []
            for key in series:
                label = key.replace("_", " ")
                group.append(f"{label} {self.counters.get(key, 0)}")
            if group:
                parts.append(" · ".join(group))
        if parts:
            self.counts_label.setText("Counts: " + " | ".join(parts))
        else:
            self.counts_label.setText("Counts: --")

    @staticmethod
    def _parse_int_field(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            try:
                return int(value)
            except Exception:
                return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        try:
            return int(float(text))
        except Exception:
            return None


class MatrixWidget(QtWidgets.QWidget):
    def __init__(self, title: str, layout_cfg: dict, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self._cells: dict[tuple[int, int], tuple[str, QtWidgets.QLabel, QtWidgets.QFrame]] = {}
        self._series_keys: list[str] = []
        self._style_map = self._build_style_map(layout_cfg)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("color: #cfd4ea; font-size: 12px; font-weight: 600;")
        outer.addWidget(title_label)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        outer.addLayout(grid)

        rows = layout_cfg.get("rows") or []
        if not rows:
            rows = [["A", "B"], ["C", "D"]]

        for r_idx, row in enumerate(rows):
            grid.setRowStretch(r_idx, 1)
            for c_idx, series in enumerate(row):
                if r_idx == 0:
                    grid.setColumnStretch(c_idx, 1)
                key = str(series).strip().upper()
                self._series_keys.append(key)
                cell = QtWidgets.QFrame()
                cell.setStyleSheet(self._style_for_key(key))
                cell_layout = QtWidgets.QVBoxLayout(cell)
                cell_layout.setContentsMargins(10, 8, 10, 8)
                cell_layout.setSpacing(4)
                name_label = QtWidgets.QLabel(key.replace("_", " ").title())
                name_label.setStyleSheet("color: #aeb8ff; font-size: 11px;")
                value_label = QtWidgets.QLabel("0")
                value_label.setStyleSheet("color: #e8ebf4; font-size: 18px; font-weight: 700;")
                cell_layout.addWidget(name_label)
                cell_layout.addWidget(value_label)
                grid.addWidget(cell, r_idx, c_idx)
                self._cells[(r_idx, c_idx)] = (key, value_label, cell)

        self.setStyleSheet("QLabel { color: #e8ebf4; }")

    def series_keys(self) -> list[str]:
        return [key for key in self._series_keys if key]

    def update_values(self, values: dict[str, int]) -> None:
        for (r_idx, c_idx), (key, label, _cell) in self._cells.items():
            label.setText(str(values.get(key, 0)))

    def _build_style_map(self, layout_cfg: dict) -> dict[str, str]:
        raw = layout_cfg.get("style_map") or {}
        return {str(k).upper(): str(v) for k, v in raw.items() if k is not None and v is not None}

    def _style_for_key(self, key: str) -> str:
        role = self._style_map.get(key.upper(), "neutral")
        if role == "correct":
            bg = "#1f8f5a"
            border = "#23a166"
        elif role == "incorrect":
            bg = "#8f2f3b"
            border = "#a53644"
        else:
            bg = "#0f1118"
            border = "#2a2d3d"
        return (
            "QFrame { "
            f"background-color: {bg}; border: 1px solid {border}; border-radius: 8px; "
            "}"
        )
