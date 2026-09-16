from __future__ import annotations

"""Behavior dashboard widget reused inside the unified operator GUI."""

import json
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("pyqtgraph is required for the dashboard") from exc

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from squeakview.common import dashboard as dash_util
from squeakview.apps.operator.gui import dashboard_model, dashboard_presentation
from squeakview.apps.operator.gui.dashboard_matrix import MatrixWidget
from squeakview.apps.operator.gui.system_meters import JetsonMeters, MetersBar


class BehaviorDashboard(QtWidgets.QWidget):
    clear_jam_requested = QtCore.Signal()
    jam_state_changed = QtCore.Signal(bool)

    def __init__(
        self,
        window_sec: float = 300.0,
        pellet_mode: str = "auto",
        parent=None,
        *,
        disk_root: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.window_sec = float(max(30.0, window_sec))
        self.pellet_mode = pellet_mode
        self._observed_pellet_mode: str | None = None
        self.counters: dict[str, int] = {}
        self.series_x: dict[str, list[float]] = {}
        self.series_y: dict[str, list[int]] = {}
        self.series_events: dict[str, list[float]] = {}
        self.series_order: list[str] = []
        self._rules: list[dict] = []
        self._plots: list[dict] = []
        self._task_cfg_path: Path | None = None
        self._first_event_at: float | None = None
        self._jam_active = False
        self._jam_reason = ""
        self._jam_detected_at = ""
        self._clear_jam_pending = False
        self.setObjectName("behaviorDashboard")
        self.setAutoFillBackground(True)
        self.setStyleSheet("#behaviorDashboard { background-color: #1a1d2a; }")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        self.meters = MetersBar()
        layout.addWidget(self.meters)
        self.counts_label = QtWidgets.QLabel("Counts: --")
        self.counts_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.counts_label.setWordWrap(True)
        self.counts_label.setMinimumHeight(44)
        self.counts_label.setStyleSheet(
            "background-color: #101526; border: 1px solid #2c3550; border-radius: 8px; "
            "color: #cfd4ea; font-size: 12px; padding: 8px 10px;"
        )
        layout.addWidget(self.counts_label)
        self._jam_banner = self._build_jam_banner()
        layout.addWidget(self._jam_banner)
        self._settings_widget = self._build_settings_panel()

        pg.setConfigOptions(antialias=True, useOpenGL=False)
        pg.setConfigOption("background", "#101526")
        pg.setConfigOption("foreground", "#cfd4ea")

        self._plot_container = QtWidgets.QWidget(self)
        self._plot_layout = QtWidgets.QHBoxLayout(self._plot_container)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_layout.setSpacing(12)
        self._plot_left = QtWidgets.QWidget(self._plot_container)
        self._plot_left_layout = QtWidgets.QVBoxLayout(self._plot_left)
        self._plot_left_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_left_layout.setSpacing(10)
        self._plot_right = QtWidgets.QWidget(self._plot_container)
        self._plot_right_layout = QtWidgets.QVBoxLayout(self._plot_right)
        self._plot_right_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_right_layout.setSpacing(10)
        self._plot_layout.addWidget(self._plot_left, 3)
        self._plot_layout.addWidget(self._plot_right, 1)
        layout.addWidget(self._plot_container, 1)
        self._build_from_task_config(self._default_task_config())

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)

        self._meters = JetsonMeters(
            self,
            interval_ms=500,
            disk_root=disk_root,
        )
        self._meters.updated.connect(self._on_meters)

    def apply_task_config(self, path: Path) -> None:
        cfg = self._load_task_config(path)
        self._task_cfg_path = path
        self._build_from_task_config(cfg)

    def clear_jam_alert(self) -> None:
        """Clear the latch presentation after an authoritative firmware ACK."""

        changed = self._jam_active
        self._jam_active = False
        self._jam_reason = ""
        self._jam_detected_at = ""
        self._clear_jam_pending = False
        self._clear_jam_btn.setEnabled(False)
        self._clear_jam_btn.setText("Clear Jam")
        self._jam_banner.hide()
        if changed:
            self.jam_state_changed.emit(False)

    @property
    def feeder_jammed(self) -> bool:
        return self._jam_active

    @property
    def clear_jam_pending(self) -> bool:
        return self._clear_jam_pending

    def _set_jam_alert(self, reason: str) -> None:
        changed = not self._jam_active
        self._jam_active = True
        self._jam_reason = reason or "The controller reports that its feeder latch is active."
        self._jam_detected_at = time.strftime("%H:%M:%S")
        self._jam_title.setText("FEEDER JAMMED — PHYSICAL INSPECTION REQUIRED")
        self._jam_detail.setText(
            f"{self._jam_detected_at} · Inspect and physically clear the feeder, "
            f"then click Clear Jam. {self._jam_reason}"
        )
        self._clear_jam_btn.setEnabled(not self._clear_jam_pending)
        self._jam_banner.show()
        if changed:
            self.jam_state_changed.emit(True)

    def _request_clear_jam(self) -> None:
        if not self._jam_active or self._clear_jam_pending:
            return
        self._clear_jam_pending = True
        self._clear_jam_btn.setEnabled(False)
        self._clear_jam_btn.setText("Clearing…")
        self._jam_detail.setText(
            "Waiting for the controller. The jam remains latched until "
            "ACK_CLEAR_JAM is received."
        )
        self.clear_jam_requested.emit()

    def clear_jam_failed(self, message: str) -> None:
        """Keep the latch active after a NACK, timeout, or transport failure."""

        self._clear_jam_pending = False
        self._clear_jam_btn.setText("Clear Jam")
        self._clear_jam_btn.setEnabled(self._jam_active)
        self._jam_detail.setText(message)
        if self._jam_active:
            self._jam_banner.show()

    def _load_task_config(self, path: Path) -> dict:
        from squeakview.common.bounded_input import (
            read_json_object,
            read_yaml_mapping,
        )

        suffix = path.suffix.lower()
        if suffix == ".json":
            return read_json_object(
                path, max_bytes=1024 * 1024, label="task config"
            )
        if yaml is None:
            raise RuntimeError("PyYAML is not installed; cannot read task config.")
        return read_yaml_mapping(path, max_bytes=1024 * 1024, label="task config")

    def _infer_pellet_mode(self, data: dict, event: str) -> None:
        self._observed_pellet_mode = dashboard_model.infer_pellet_mode(
            self.pellet_mode, self._observed_pellet_mode, data, event
        )

    def _effective_pellet_mode(self) -> str:
        return dashboard_model.effective_pellet_mode(self.pellet_mode, self._observed_pellet_mode)

    @staticmethod
    def _default_task_config() -> dict:
        return dashboard_model.default_task_config()

    def _build_from_task_config(self, cfg: dict) -> None:
        definition = dashboard_model.compile_task_config(cfg)
        plots_cfg = definition.plots
        settings_panel = definition.settings_panel
        self._settings_widget.setVisible(settings_panel)
        if settings_panel:
            try:
                parent = self._settings_widget.parent()
                if parent is not None:
                    parent.layout().removeWidget(self._settings_widget)
            except Exception:
                pass
            self._plot_right_layout.insertWidget(0, self._settings_widget, 0)

        self._rules = list(definition.rules)
        self.series_order = list(definition.series_order)
        self.counters = {k: 0 for k in self.series_order}
        self.series_x = {k: [] for k in self.series_order}
        self.series_y = {k: [] for k in self.series_order}
        self.series_events = {k: [] for k in self.series_order}
        self._first_event_at = None

        self._clear_plots()
        for plot in plots_cfg:
            title = str(plot.get("title") or plot.get("id") or "Plot")
            plot_type = str(plot.get("type") or "timeseries").lower()
            if plot_type == "matrix":
                matrix = MatrixWidget(title, plot.get("layout") or {})
                self._plot_right_layout.addWidget(matrix, 1)
                series_keys = matrix.series_keys()
                self._plots.append({"kind": "matrix", "plot": matrix, "series": series_keys})
            elif plot_type == "event_raster":
                widget = self._make_plot(title)
                self._plot_left_layout.addWidget(widget, 1)
                series_keys = []
                curves: dict[str, object] = {}
                lanes: dict[str, float] = {}
                configured_series = plot.get("series", []) or []
                lane_count = len(configured_series)
                marker = QtGui.QPainterPath()
                marker.addRect(-0.12, -0.5, 0.24, 1.0)
                for idx, series in enumerate(configured_series):
                    key = self._norm_series(series)
                    if not key:
                        continue
                    lane = float(lane_count - idx - 1)
                    series_keys.append(key)
                    lanes[key] = lane
                    color = self._series_color(key, idx)
                    curves[key] = widget.plot(
                        pen=None,
                        symbol=marker,
                        symbolSize=13,
                        symbolPen=pg.mkPen(color, width=1),
                        symbolBrush=pg.mkBrush(color),
                    )
                ticks = [
                    (lanes[key], self._series_label(key))
                    for key in series_keys
                ]
                widget.getAxis("left").setTicks([ticks])
                widget.getAxis("left").setWidth(86)
                widget.setYRange(-0.6, max(0.6, float(lane_count) - 0.4), padding=0.0)
                self._plots.append(
                    {
                        "kind": "event_raster",
                        "plot": widget,
                        "series": series_keys,
                        "curves": curves,
                        "lanes": lanes,
                    }
                )
            else:
                widget = self._make_plot(title)
                self._plot_left_layout.addWidget(widget, 1)
                legend = widget.addLegend(
                    offset=(12, 8),
                    labelTextColor="#cfd4ea",
                    brush=pg.mkBrush(16, 21, 38, 215),
                    pen=pg.mkPen("#2c3550"),
                )
                try:
                    legend.setLabelTextColor("#cfd4ea")
                except Exception:
                    pass
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

        has_right_panel = settings_panel or any(entry.get("kind") == "matrix" for entry in self._plots)
        self._plot_right.setVisible(has_right_panel)
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
        return dashboard_model.normalize_series(name)

    @staticmethod
    def _make_plot(title: str) -> pg.PlotWidget:
        axis = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation="bottom", fmt="%H:%M:%S")
        plot = pg.PlotWidget(axisItems={"bottom": axis})
        plot.setMinimumHeight(118)
        plot.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        plot.setTitle(f"<span style='color:#eef1ff; font-size:12pt; font-weight:700'>{title}</span>")
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.showGrid(x=True, y=True, alpha=0.08)
        plot.setBackground("#101526")
        plot.getViewBox().setBackgroundColor("#101526")
        for axis_name in ("left", "bottom"):
            item = plot.getAxis(axis_name)
            item.setPen(pg.mkPen("#2c3550"))
            item.setTextPen(pg.mkPen("#9aa7cc"))
            item.setStyle(tickFont=QtGui.QFont("Sans Serif", 9), tickTextOffset=8)
        plot.getPlotItem().setContentsMargins(8, 8, 12, 8)
        return plot

    @staticmethod
    def _series_label(key: str) -> str:
        return dashboard_presentation.series_label(key)

    @staticmethod
    def _series_color(key: str, idx: int) -> str:
        return dashboard_presentation.series_color(key, idx)

    @classmethod
    def _series_style(cls, key: str, idx: int):
        hex_color = cls._series_color(key, idx)
        pen = pg.mkPen(hex_color, width=2.5)
        color = QtGui.QColor(hex_color)
        color.setAlpha(42)
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

    def detach_task_panel(self) -> QtWidgets.QWidget:
        """Detach the live task-state panel so it can be mounted outside the plot area."""
        self._plot_layout.removeWidget(self._plot_right)
        self._plot_right.setParent(None)
        self._plot_right.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        return self._plot_right

    @QtCore.Slot(float, float, float, float, str)
    def _on_meters(self, ram_pct: float, gpu_pct: float, cpu_pct: float, disk_pct: float, _raw: str) -> None:
        self.meters.set_ram_pct(None if ram_pct < 0 else ram_pct)
        self.meters.set_gpu_pct(None if gpu_pct < 0 else gpu_pct)
        self.meters.set_cpu_pct(None if cpu_pct < 0 else cpu_pct)
        self.meters.set_disk_pct(None if disk_pct < 0 else disk_pct)

    @QtCore.Slot(str)
    def ingest(self, raw: str) -> None:
        """Compatibility edge for local callers that still supply raw lines."""

        event = dash_util.DashboardEvent.parse(raw)
        if event is not None:
            self.ingest_event(event)

    @QtCore.Slot(object)
    def ingest_event(self, event: dash_util.DashboardEvent) -> None:
        data = dashboard_model.event_data(event)
        if not data:
            return
        jam_event = dashboard_model.feeder_jam_event(event)
        if jam_event == dashboard_model.FeederJamEvent.JAMMED:
            reason = str(data.get("reason", "")).strip()
            self._set_jam_alert(reason)
        elif jam_event == dashboard_model.FeederJamEvent.CLEAR_ACK:
            self.clear_jam_alert()
        elif jam_event == dashboard_model.FeederJamEvent.CLEAR_FEED_ACTIVE:
            self.clear_jam_failed(
                "Clear Jam was rejected: wait until the current feed stops, "
                "then inspect the mechanism and try again."
            )
        elif jam_event == dashboard_model.FeederJamEvent.CLEAR_NOT_JAMMED:
            self.clear_jam_failed(
                "The firmware reports that no jam is currently latched. "
                "The warning remains active because no ACK_CLEAR_JAM was received."
            )
        event = str(data.get("event_uc", ""))
        if event == "TASK_INFO":
            self._update_settings_from_task_info(data)
        elif event == "NOGO_STAGE_INFO":
            self._update_settings_from_nogo_stage_info(data)
        elif event in ("SIDE_SET", "TRIAL_START"):
            self._update_settings_from_side_set(data)

        self._infer_pellet_mode(data, event)
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
            mode = self._effective_pellet_mode()
            ok = (
                (mode == "arrival" and (dash_util.is_start_event(data) or "ARRIVAL" in event))
                or (mode == "retrieval" and (dash_util.is_end_event(data) or "RETRIEVAL" in event))
                or (
                    mode == "both"
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
        self.series_events[key].append(tsec)
        if self._first_event_at is None:
            self._first_event_at = tsec
        dashboard_presentation.cap_event_times(self.series_events[key])
        xs, ys = self.series_x[key], self.series_y[key]

        if xs and xs[-1] == tsec:
            ys[-1] = new_value
        else:
            xs.append(tsec)
            ys.append(prev)
            xs.append(tsec)
            ys.append(new_value)
            dashboard_presentation.cap_step_series(xs, ys)

        self.counters[key] = new_value

    def _refresh(self) -> None:
        now = time.time()
        xstart, xend = dashboard_presentation.window_bounds(now, self.window_sec)

        for key in list(self.series_x.keys()):
            xs, ys = self.series_x[key], self.series_y[key]
            dashboard_presentation.trim_step_series(xs, ys, xstart=xstart)
            dashboard_presentation.trim_event_times(
                self.series_events[key], xstart=xstart
            )

        def set_curve(curve, x, y):
            plot_x, plot_y = dashboard_presentation.curve_points(x, y, now=now)
            curve.setData(plot_x, plot_y)

        for plot_entry in self._plots:
            if plot_entry.get("kind") == "matrix":
                matrix = plot_entry["plot"]
                series = plot_entry.get("series", [])
                values = {key: self.counters.get(key, 0) for key in series}
                matrix.update_values(values)
                continue
            plot = plot_entry["plot"]
            keys = plot_entry["series"]
            if plot_entry.get("kind") == "event_raster":
                lanes = plot_entry["lanes"]
                raster_start, raster_end = dashboard_presentation.raster_window_bounds(
                    now, self.window_sec, self._first_event_at
                )
                for key in keys:
                    timestamps = self.series_events.get(key, [])
                    plot_entry["curves"][key].setData(
                        timestamps, [lanes[key]] * len(timestamps)
                    )
                plot.setXRange(raster_start, raster_end, padding=0.0)
                continue
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

    def follow_run_telemetry(self, path: Path) -> None:
        self._meters.follow_run_telemetry(path)

    def resume_idle_system_sampling(self) -> None:
        self._meters.resume_idle_sampling()

    def _match_rule(self, data: dict, event: str, rule: dict) -> bool:
        return dashboard_model.rule_matches(
            data, event, rule, pellet_mode=self._effective_pellet_mode()
        )

    def _update_counts_label(self) -> None:
        self.counts_label.setText(
            dashboard_presentation.counts_html(
                [plot.get("series", []) for plot in self._plots], self.counters
            )
        )

    def _build_settings_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setObjectName("settingsPanel")
        panel.setStyleSheet(
            "#settingsPanel { background-color: #141622; border: 1px solid #2b2f3b; border-radius: 6px; }"
        )
        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)
        title = QtWidgets.QLabel("Current Task Settings")
        title.setStyleSheet("color: #cfd4ea; font-size: 12px; font-weight: 600;")
        outer.addWidget(title)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)
        outer.addLayout(grid)

        self._settings_labels: dict[str, QtWidgets.QLabel] = {}
        rows = [
            ("Stage", "stage"),
            ("Hold (ms)", "hold_ms"),
            ("Go (ms)", "go_ms"),
            ("NoGo (ms)", "nogo_ms"),
            ("Go %", "go_pct"),
            ("Side", "side"),
            ("Reason", "reason"),
        ]
        for r, (label_text, key) in enumerate(rows):
            label = QtWidgets.QLabel(label_text)
            label.setStyleSheet("color: #b9c0d6; font-size: 12px;")
            value = QtWidgets.QLabel("--")
            value.setStyleSheet("color: #e4e7f2; font-size: 12px; font-weight: 600;")
            grid.addWidget(label, r, 0)
            grid.addWidget(value, r, 1)
            self._settings_labels[key] = value
        return panel

    def _build_jam_banner(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame(self)
        panel.setObjectName("jamBanner")
        panel.setStyleSheet(
            "#jamBanner { background-color: #3a1014; border: 1px solid #b33a44; border-radius: 6px; }"
            "#jamBanner QLabel { background: transparent; }"
            "#jamBanner QPushButton {"
            " background-color: #c64b55; color: #ffffff; border: none; border-radius: 4px; padding: 5px 10px;"
            " font-weight: 600; }"
            "#jamBanner QPushButton:hover { background-color: #d85d67; }"
        )
        layout = QtWidgets.QHBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        text_wrap = QtWidgets.QVBoxLayout()
        text_wrap.setContentsMargins(0, 0, 0, 0)
        text_wrap.setSpacing(2)
        self._jam_title = QtWidgets.QLabel("FEEDER JAMMED — PHYSICAL INSPECTION REQUIRED")
        self._jam_title.setStyleSheet("color: #ffd8dc; font-size: 12px; font-weight: 700;")
        self._jam_detail = QtWidgets.QLabel("--")
        self._jam_detail.setStyleSheet("color: #ffb6be; font-size: 12px;")
        self._jam_detail.setWordWrap(True)
        text_wrap.addWidget(self._jam_title)
        text_wrap.addWidget(self._jam_detail)

        self._clear_jam_btn = QtWidgets.QPushButton("Clear Jam")
        self._clear_jam_btn.setEnabled(False)
        self._clear_jam_btn.clicked.connect(self._request_clear_jam)

        layout.addLayout(text_wrap, 1)
        layout.addWidget(self._clear_jam_btn, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        panel.hide()
        return panel

    def _update_settings_from_task_info(self, data: dict) -> None:
        self._apply_settings_update(dashboard_model.task_settings_update(data, "TASK_INFO"))

    def _update_settings_from_nogo_stage_info(self, data: dict) -> None:
        self._apply_settings_update(dashboard_model.task_settings_update(data, "NOGO_STAGE_INFO"))

    def _update_settings_from_side_set(self, data: dict) -> None:
        self._apply_settings_update(dashboard_model.task_settings_update(data, "SIDE_SET"))

    def _apply_settings_update(self, updates: dict[str, str]) -> None:
        if not hasattr(self, "_settings_labels"):
            return
        for key, value in updates.items():
            label = self._settings_labels.get(key)
            if label is not None:
                label.setText(value)

    @staticmethod
    def _parse_int_field(value: object) -> int | None:
        return dashboard_model.parse_int_field(value)
