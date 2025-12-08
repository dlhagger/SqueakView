from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview.apps.operator.backend import process
from squeakview.apps.operator.backend.manager import OperatorBackend
from squeakview.apps.operator.gui.config_dialog import ConfigDialog
from squeakview.apps.operator.gui.dashboard import BehaviorDashboard


class PreviewWidget(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NativeWindow)
        self.setMinimumHeight(320)
        self.setStyleSheet("background-color: #0f1118; border: 1px solid #24283b; border-radius: 10px;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel("Live preview will appear here once DeepStream starts…")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #8088a6; letter-spacing: 0.2px;")
        layout.addWidget(self.label, 1)

        self.logo_label = QtWidgets.QLabel(self)
        self.logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setStyleSheet("background: rgba(15,17,24,0.85);")
        self.logo_label.hide()
        self._logo_pixmap = self._load_logo()

        self.status_badge = QtWidgets.QLabel("Idle", self)
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setStyleSheet("""
            QLabel#statusBadge {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3a3f5c, stop:1 #2d3046);
                color: #e8ebf4;
                padding: 4px 10px;
                border-radius: 8px;
                font-weight: 700;
                font-size: 11px;
            }
        """)
        self.info_label = QtWidgets.QLabel("", self)
        self.info_label.setStyleSheet(
            "color: #a5adc8; background: rgba(15,17,24,0.6); padding: 4px 8px; border-radius: 8px; font-size: 11px;"
        )
        self.info_label.hide()
        self._preview_enabled = True

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reposition_overlays()
        self._update_logo_scale()

    def _reposition_overlays(self) -> None:
        margin = 12
        self.status_badge.adjustSize()
        self.status_badge.move(margin, margin)
        if self.info_label.isVisible():
            self.info_label.adjustSize()
            info_y = margin + self.status_badge.height() + 6
            self.info_label.move(margin, info_y)
        else:
            info_y = margin
        self.logo_label.setGeometry(0, 0, self.width(), self.height())

    def set_status(self, text: str, *, color: str | None = None) -> None:
        self.status_badge.setText(text)
        if color:
            self.status_badge.setStyleSheet(
                self.status_badge.styleSheet() + f"\nQLabel#statusBadge {{ background-color: {color}; }}"
            )
        self._reposition_overlays()

    def set_info(self, text: str | None) -> None:
        if text:
            self.info_label.setText(text)
            self.info_label.show()
        else:
            self.info_label.hide()
        self._reposition_overlays()

    def window_id(self) -> int:
        return int(self.winId())

    def show_hint(self, visible: bool) -> None:
        self.label.setVisible(visible)

    def set_preview_enabled(self, enabled: bool) -> None:
        self._preview_enabled = enabled
        if enabled:
            self.logo_label.hide()
            self.label.hide()
        else:
            self.label.setText("Preview disabled")
            self.label.show()
            self.logo_label.show()
            self._update_logo_scale()
        self._reposition_overlays()

    def _load_logo(self) -> QtGui.QPixmap | None:
        try:
            logo_path = Path(__file__).resolve().parents[2] / "SqueakView_logo.png"
            if logo_path.exists():
                pix = QtGui.QPixmap(str(logo_path))
                return pix if not pix.isNull() else None
        except Exception:
            return None
        return None

    def _update_logo_scale(self) -> None:
        if not self.logo_label.isVisible():
            return
        if self._logo_pixmap is None:
            self.logo_label.setText("Preview disabled")
            return
        pix = self._logo_pixmap.scaled(
            self.logo_label.size() * 0.6,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.logo_label.setPixmap(pix)


class MainWindow(QtWidgets.QMainWindow):
    log_msg = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.log_msg.connect(self._append_log)
        self.setWindowTitle("SqueakView")
        self.resize(1280, 820)
        self.setMinimumSize(900, 600)

        self._config_data: dict | None = None
        self._preview_window_id: int | None = None
        self._preview_enabled: bool = True

        self.backend = OperatorBackend(self._emit_log, self._forward_dashboard)

        self._build_ui()
        self._apply_brand_theme()
        QtCore.QTimer.singleShot(0, self._capture_preview_window_id)

        if not self._show_config_dialog(initial=True):
            QtCore.QTimer.singleShot(0, self.close)
        else:
            self.statusBar().showMessage("Ready to record.")
        self.preview.set_status("Idle")

    # ---- UI -------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Main content grid: top row (preview + meters/summary), bottom row (graphs)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        preview_group = QtWidgets.QGroupBox("Live Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        self.preview = PreviewWidget(self)
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        preview_layout.addWidget(self.preview)
        grid.addWidget(preview_group, 0, 0, 1, 2)

        # Dashboard (plots) and meters (top-right)
        self.dashboard = BehaviorDashboard(window_sec=300.0, pellet_mode="arrival")
        meters_only = self.dashboard.detach_meters()
        meters_group = QtWidgets.QGroupBox("System Load")
        meters_layout = QtWidgets.QVBoxLayout(meters_group)
        meters_layout.setContentsMargins(12, 12, 12, 12)
        meters_layout.addWidget(meters_only)

        self.summary_label = QtWidgets.QLabel("No configuration loaded.")
        self.summary_label.setObjectName("summaryBanner")
        self.summary_label.setWordWrap(True)
        meters_layout.addSpacing(6)
        meters_layout.addWidget(self.summary_label)

        # Move run controls into the system load panel
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch(1)
        self.configure_btn = QtWidgets.QPushButton("Configure…")
        self.configure_btn.setObjectName("secondaryButton")
        self.configure_btn.clicked.connect(self._on_configure)
        btn_row.addWidget(self.configure_btn)
        self.run_btn = QtWidgets.QPushButton("Start Recording")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)
        self.video_btn = QtWidgets.QPushButton("Video: On")
        self.video_btn.setCheckable(True)
        self.video_btn.setChecked(True)
        self.video_btn.setEnabled(False)
        self.video_btn.clicked.connect(self._on_video_toggle)
        btn_row.addWidget(self.video_btn)
        self.skeleton_btn = QtWidgets.QPushButton("Skeleton: Off")
        self.skeleton_btn.setCheckable(True)
        self.skeleton_btn.setEnabled(False)
        self.skeleton_btn.clicked.connect(self._on_skeleton_toggle)
        btn_row.addWidget(self.skeleton_btn)
        self.preview_btn = QtWidgets.QPushButton("Preview: On")
        self.preview_btn.setCheckable(True)
        self.preview_btn.setChecked(True)
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self._on_preview_toggle)
        btn_row.addWidget(self.preview_btn)
        self.stop_btn = QtWidgets.QPushButton("Stop Recording")
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.stop_btn)
        meters_layout.addSpacing(6)
        meters_layout.addLayout(btn_row)

        grid.addWidget(meters_group, 0, 2, 1, 1)

        dashboard_group = QtWidgets.QGroupBox("Behavior Dashboard")
        dash_layout = QtWidgets.QVBoxLayout(dashboard_group)
        dash_layout.setContentsMargins(14, 14, 14, 14)
        self.dashboard.setMinimumHeight(260)
        dash_layout.addWidget(self.dashboard)
        dashboard_group.setMinimumHeight(300)
        grid.addWidget(dashboard_group, 1, 0, 1, 3)

        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 0)
        grid.setColumnStretch(2, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        layout.addLayout(grid, 1)

        layout.setStretch(0, 0)
        layout.setStretch(1, 5)

    def _apply_brand_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow {
                background-color: #171821;
                color: #e8ebf4;
            }
            QStatusBar {
                background-color: #171821;
                color: #e8ebf4;
            }
            QStatusBar QLabel {
                color: #e8ebf4;
            }
            QLabel {
                color: #e8ebf4;
            }
            QGroupBox {
                border: 1px solid #2a2d3d;
                border-radius: 10px;
                margin-top: 16px;
                padding-top: 16px;
            }
            QGroupBox::title {
                color: #aeb8ff;
                subcontrol-origin: margin;
                left: 14px;
                top: 10px;
                padding: 0 6px;
                background-color: transparent;
            }
            QFrame#brandHeader {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2b2f46, stop:1 #202336);
                border: 1px solid #333650;
                border-radius: 10px;
            }
            QLabel#brandTitle {
                font-size: 24px;
                font-weight: 700;
                color: #ffffff;
            }
            QLabel#brandSubtitle {
                color: #aeb8ff;
                font-size: 13px;
            }
            QLabel#summaryBanner {
                background-color: rgba(46, 52, 80, 0.6);
                border: 1px solid #38405d;
                border-radius: 8px;
                padding: 10px;
                color: #e0e5ff;
            }
            QPushButton {
                padding: 8px 18px;
                border-radius: 6px;
                font-weight: 600;
                color: #e8ebf4;
                background-color: #2c3146;
                border: 1px solid #404663;
            }
            QPushButton:hover {
                background-color: #353b55;
            }
            QPushButton:disabled {
                background-color: #3b3f4f;
                color: #7d8299;
                border-color: #3b3f4f;
            }
            QPushButton#primaryButton {
                background-color: #5c6df5;
                border: 1px solid #5c6df5;
                color: white;
            }
            QPushButton#primaryButton:hover {
                background-color: #4959e6;
            }
            QPushButton#dangerButton {
                background-color: #d9536f;
                border: 1px solid #d9536f;
            }
            QPushButton#dangerButton:hover {
                background-color: #c13d59;
            }
            QPushButton#secondaryButton {
                background-color: #353a4d;
            }
        """)

    # ---- Configuration --------------------------------------------------
    def _show_config_dialog(self, *, initial: bool = False) -> bool:
        dialog = ConfigDialog(self, config=self._config_data)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        result = dialog.result_config
        if not result:
            return False
        self._config_data = result
        self._apply_config(result)
        return True

    def _apply_config(self, data: dict) -> None:
        summary_parts = [
            f"Resolution: {data['width']}×{data['height']} @ {data['fps']} FPS",
            f"Pixel: {data['pixel_format']}",
            f"Trigger: {'On' if data['trigger_on'] else 'Off'}",
            f"Config: {Path(data['ds_cfg']).name}",
        ]
        if data.get("mouse_id"):
            summary_parts.append(f"Mouse: {data['mouse_id']}")
        inference_on = data.get("inference_enabled", True)
        summary_parts.append(f"Inference: {'On' if inference_on else 'Off'}")
        if data.get("serial_enabled", True):
            summary_parts.append(f"Serial: {data['serial_port']} @ {data['serial_baud']}")
        else:
            summary_parts.append("Serial: disabled")
        self.summary_label.setText(" · ".join(summary_parts))
        info = f"{data['width']}×{data['height']} · {data['fps']} FPS · {'Trig' if data['trigger_on'] else 'Free'} · {data['bitrate']} kbps"
        self.preview.set_info(info)
        self.statusBar().showMessage("Configuration committed.", 4000)

        cfg = self.backend.launch_cfg
        cfg.width = data["width"]
        cfg.height = data["height"]
        cfg.fps = data["fps"]
        cfg.pixel_format = data["pixel_format"]
        cfg.trigger_on = data["trigger_on"]
        cfg.exposure_us = data.get("exposure_us", 10000)
        cfg.ds_cfg = Path(data["ds_cfg"])
        cfg.inference_enabled = inference_on
        cfg.draw_skeleton = data.get("draw_skeleton", False)
        cfg.socket_path = data["socket_path"]
        cfg.bitrate = data["bitrate"]
        cfg.serial_enabled = data["serial_enabled"]
        cfg.serial_port = data["serial_port"]
        cfg.serial_baud = data["serial_baud"]
        cfg.arduino_fps = data["arduino_fps"]
        cfg.mouse_id = data.get("mouse_id", "")
        self.run_btn.setEnabled(True)
        self._set_skeleton_button_state(enabled=False, checked=cfg.draw_skeleton)
        self._set_video_button_state(enabled=False, checked=True)

    def _build_launch_config(self) -> process.LaunchConfig:
        if not self._config_data:
            raise RuntimeError("Configuration not set")
        data = self._config_data
        cfg = process.LaunchConfig(
            width=data["width"],
            height=data["height"],
            fps=data["fps"],
            pixel_format=data["pixel_format"],
            trigger_on=data["trigger_on"],
            exposure_us=data.get("exposure_us", 10000),
            ds_cfg=Path(data["ds_cfg"]),
            inference_enabled=data.get("inference_enabled", True),
            socket_path=data["socket_path"],
            bitrate=data["bitrate"],
            serial_enabled=data["serial_enabled"],
            serial_port=data["serial_port"],
            serial_baud=data["serial_baud"],
            arduino_fps=data["arduino_fps"],
            mouse_id=data.get("mouse_id", ""),
            draw_skeleton=data.get("draw_skeleton", False),
        )
        cfg.preview_window_id = self._preview_window_id
        return cfg

    # ---- Helpers --------------------------------------------------------
    def _capture_preview_window_id(self) -> None:
        try:
            wid = int(self.preview.window_id())
        except Exception:
            wid = 0
        if wid and wid != self._preview_window_id:
            self._preview_window_id = wid
            self.preview.show_hint(False)
            self.preview.set_status("Ready")
        if not wid:
            QtCore.QTimer.singleShot(200, self._capture_preview_window_id)

    def _forward_dashboard(self, raw: str) -> None:
        try:
            QtCore.QMetaObject.invokeMethod(
                self.dashboard,
                "ingest",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(str, raw),
            )
        except Exception:
            pass

    # ---- Actions --------------------------------------------------------
    def _on_configure(self) -> None:
        if self._show_config_dialog(initial=False):
            self.statusBar().showMessage("Configuration updated.", 5000)

    def _on_run(self) -> None:
        if self._preview_window_id is None:
            self._emit_log("[GUI] Preview still initializing; please wait a moment and try again.")
            QtCore.QTimer.singleShot(200, self._capture_preview_window_id)
            return
        try:
            config = self._build_launch_config()
        except RuntimeError:
            self._emit_log("[GUI] Please configure SqueakView first.")
            return
        config.preview_window_id = self._preview_window_id
        if not self.backend.start_run(config):
            self._emit_log("[GUI] Failed to start run")
            return
        self.preview.show_hint(False)
        self.preview.set_status("Live", color="#5c6df5")
        self.preview.set_preview_enabled(True)
        self._preview_enabled = True
        self._emit_log("[GUI] Run started")
        self.run_btn.setEnabled(False)
        self.preview_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.configure_btn.setEnabled(False)
        # Enable skeleton toggle once a run is active (only meaningful when inference is on)
        self._set_skeleton_button_state(enabled=config.inference_enabled, checked=config.draw_skeleton)
        self._set_video_button_state(enabled=config.inference_enabled, checked=True)

    def _on_stop(self) -> None:
        self.backend.stop_run()
        self.preview.show_hint(True)
        self.preview.set_status("Idle")
        self.preview.set_preview_enabled(True)
        self._emit_log("[GUI] Run stopped")
        self.run_btn.setEnabled(True)
        self.preview_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.configure_btn.setEnabled(True)
        self._set_skeleton_button_state(enabled=False, checked=self.skeleton_btn.isChecked())
        self._set_video_button_state(enabled=False, checked=self.video_btn.isChecked())

    # ---- Logging --------------------------------------------------------
    def _emit_log(self, msg: str) -> None:
        self.log_msg.emit(msg)

    @QtCore.Slot(str)
    def _append_log(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 5000)
        print(msg, flush=True)

    def _on_preview_toggle(self, enabled: bool | None = None) -> None:
        state = enabled if isinstance(enabled, bool) else self.preview_btn.isChecked()
        self._preview_enabled = state
        self.preview_btn.setText(f"Preview: {'On' if state else 'Off'}")
        self.preview_btn.setChecked(state)
        self.preview.set_preview_enabled(state)
        try:
            self.backend.set_preview_enabled(state)
        except Exception:
            pass

    def _set_skeleton_button_state(self, *, enabled: bool, checked: bool) -> None:
        self.skeleton_btn.setEnabled(enabled)
        self.skeleton_btn.setChecked(checked)
        self.skeleton_btn.setText(f"Skeleton: {'On' if checked else 'Off'}")

    def _on_skeleton_toggle(self) -> None:
        state = self.skeleton_btn.isChecked()
        self._set_skeleton_button_state(enabled=True, checked=state)
        try:
            self.backend.set_skeleton_enabled(state)
        except Exception:
            pass

    def _set_video_button_state(self, *, enabled: bool, checked: bool) -> None:
        self.video_btn.setEnabled(enabled)
        self.video_btn.setChecked(checked)
        self.video_btn.setText(f"Video: {'On' if checked else 'Off'}")

    def _on_video_toggle(self) -> None:
        state = self.video_btn.isChecked()
        self._set_video_button_state(enabled=True, checked=state)
        try:
            self.backend.set_video_enabled(state)
        except Exception:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        try:
            self.backend.shutdown()
        except Exception:
            pass
        try:
            self.dashboard.close()
        except Exception:
            pass
        super().closeEvent(event)
