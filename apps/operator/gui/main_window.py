from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
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
        self._logo_scale = 1.0
        self._logo_anim: QtCore.QVariantAnimation | None = None
        self._logo_opacity = QtWidgets.QGraphicsOpacityEffect(self.logo_label)
        self.logo_label.setGraphicsEffect(self._logo_opacity)
        self._logo_opacity.setOpacity(1.0)

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
            self._stop_logo_anim()
            self.logo_label.hide()
            self.label.hide()
        else:
            self.label.setText("Preview disabled")
            self.label.show()
            self.logo_label.show()
            self._start_logo_anim()
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
            self.logo_label.size() * (0.6 * self._logo_scale),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.logo_label.setPixmap(pix)

    def _start_logo_anim(self) -> None:
        self._stop_logo_anim()
        self._logo_scale = 0.75
        self._logo_opacity.setOpacity(0.0)
        self._update_logo_scale()
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(180)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        def _on_value(value: object) -> None:
            t = float(value)
            self._logo_scale = 0.75 + (0.25 * t)
            self._logo_opacity.setOpacity(t)
            self._update_logo_scale()

        anim.valueChanged.connect(_on_value)
        self._logo_anim = anim
        anim.start()

    def _stop_logo_anim(self) -> None:
        if self._logo_anim is not None:
            self._logo_anim.stop()
            self._logo_anim.deleteLater()
            self._logo_anim = None
        self._logo_scale = 1.0
        self._logo_opacity.setOpacity(1.0)


class DepthPreviewWidget(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setStyleSheet("background-color: #0f1118; border: 1px solid #24283b; border-radius: 10px;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.label = QtWidgets.QLabel("Depth preview idle")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #9097b6;")
        self.label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        layout.addWidget(self.label, 1)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._cap = None
        self._socket_path = "/tmp/cam_depth.sock"
        self._width = 1920
        self._height = 1200
        self._fps = 30
        self._preview_lo_pct = 2.0
        self._preview_hi_pct = 98.0
        self._preview_gamma = 0.8
        self._start_token = 0
        self._start_attempts = 0

    def start_stream(self, *, socket_path: str, width: int, height: int, fps: int) -> None:
        self.stop_stream()
        self._socket_path = socket_path
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._start_attempts = 0
        self._start_token += 1
        token = self._start_token
        self._try_open_stream(token)

    def _try_open_stream(self, token: int) -> None:
        if token != self._start_token:
            return
        if not os.path.exists(self._socket_path):
            self._start_attempts += 1
            if self._start_attempts < 12:
                self.label.setText("Depth preview connecting…")
                QtCore.QTimer.singleShot(400, lambda t=token: self._try_open_stream(t))
            else:
                self.label.setText("Depth preview unavailable")
                self._cap = None
            return
        preview_w = 640
        preview_h = 400
        preview_fps = min(15, max(1, self._fps))
        pipelines = [
            (
                f"shmsrc socket-path={self._socket_path} is-live=true do-timestamp=true ! "
                f"video/x-raw,format=BGRA,width={preview_w},height={preview_h},framerate={preview_fps}/1 ! "
                "queue max-size-buffers=1 leaky=downstream ! "
                "autovideoconvert ! appsink sync=false drop=true max-buffers=1"
            ),
            (
                f"shmsrc socket-path={self._socket_path} is-live=true do-timestamp=true ! "
                "video/x-raw,format=BGRA ! "
                "queue max-size-buffers=1 leaky=downstream ! "
                "autovideoconvert ! appsink sync=false drop=true max-buffers=1"
            ),
            (
                f"shmsrc socket-path={self._socket_path} is-live=true do-timestamp=true ! "
                "video/x-raw,format=BGRx ! "
                "queue max-size-buffers=1 leaky=downstream ! "
                "videoconvert ! appsink sync=false drop=true max-buffers=1"
            ),
            (
                f"shmsrc socket-path={self._socket_path} is-live=true do-timestamp=true ! "
                "queue max-size-buffers=1 leaky=downstream ! "
                "videoconvert ! appsink sync=false drop=true max-buffers=1"
            ),
        ]
        cap = None
        for pipeline in pipelines:
            probe = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if probe.isOpened():
                cap = probe
                break
            probe.release()
        if cap is None:
            self._start_attempts += 1
            if self._start_attempts < 12:
                self.label.setText("Depth preview connecting…")
                QtCore.QTimer.singleShot(400, lambda t=token: self._try_open_stream(t))
            else:
                self.label.setText("Depth preview unavailable")
                self._cap = None
            return
        self._cap = cap
        self.label.setText("Depth preview live")
        self.label.show()
        self._timer.start()

    def stop_stream(self) -> None:
        self._start_token += 1
        if self._timer.isActive():
            self._timer.stop()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.label.setText("Depth preview idle")
        self.label.show()

    def _tick(self) -> None:
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return

        if frame.ndim == 2:
            depth_like = frame.astype("float32")
        elif frame.ndim == 3 and frame.dtype == "uint8" and frame.shape[2] == 2:
            h0, w0 = frame.shape[:2]
            depth_like = frame.view("<u2").reshape(h0, w0).astype("float32")
        elif frame.ndim == 3 and frame.shape[2] == 4:
            depth_like = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY).astype("float32")
        elif frame.ndim == 3:
            depth_like = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32")
        else:
            depth_like = frame.astype("float32")

        h, w = depth_like.shape[:2]
        if h <= 0 or w <= 0:
            return

        valid = depth_like[depth_like > 0]
        if valid.size == 0:
            rgb = cv2.cvtColor(np.zeros((h, w), dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            lo = np.percentile(valid, self._preview_lo_pct)
            hi = np.percentile(valid, self._preview_hi_pct)
            if hi <= lo:
                hi = lo + 1.0
            clipped = np.clip(depth_like, lo, hi)
            norm = ((clipped - lo) * (255.0 / (hi - lo))).astype("uint8")
            if self._preview_gamma != 1.0:
                norm_f = np.power(norm.astype("float32") / 255.0, self._preview_gamma)
                norm = np.clip(norm_f * 255.0, 0, 255).astype("uint8")
            color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(pix)


class MainWindow(QtWidgets.QMainWindow):
    log_msg = QtCore.Signal(str)
    stop_done = QtCore.Signal()
    stop_failed = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.log_msg.connect(self._append_log)
        self.setWindowTitle("SqueakView")
        self.resize(1280, 820)
        self.setMinimumSize(900, 600)

        self._config_data: dict | None = None
        self._preview_window_id: int | None = None
        self._stop_in_progress = False
        self._stop_thread: threading.Thread | None = None
        self._upload_thread: threading.Thread | None = None
        self._upload_in_progress = False
        self._depth_preview_enabled = False
        self.stop_done.connect(self._on_stop_complete)
        self.stop_failed.connect(self._on_stop_failed)

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
        # Count pellets on retrieval events by default (arrival-only counting was confusing)
        self.dashboard = BehaviorDashboard(window_sec=300.0, pellet_mode="retrieval")
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
        self.depth_preview = DepthPreviewWidget(self)
        self.depth_preview.hide()
        meters_layout.addSpacing(6)
        meters_layout.addWidget(self.depth_preview)

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
        self.skeleton_btn = QtWidgets.QPushButton("Skeleton: Off")
        self.skeleton_btn.setCheckable(True)
        self.skeleton_btn.setEnabled(False)
        self.skeleton_btn.clicked.connect(self._on_skeleton_toggle)
        btn_row.addWidget(self.skeleton_btn)
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

        self.stop_overlay = QtWidgets.QFrame(central)
        self.stop_overlay.setObjectName("stopOverlay")
        self.stop_overlay.hide()
        self.stop_overlay.raise_()
        overlay_layout = QtWidgets.QVBoxLayout(self.stop_overlay)
        overlay_layout.setContentsMargins(24, 24, 24, 24)
        overlay_layout.setSpacing(10)
        overlay_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.stop_overlay_title = QtWidgets.QLabel("Finalizing Run…", self.stop_overlay)
        self.stop_overlay_title.setObjectName("stopOverlayTitle")
        self.stop_overlay_msg = QtWidgets.QLabel(
            "Sending STOP, shutting down capture/inference, and closing files.",
            self.stop_overlay,
        )
        self.stop_overlay_msg.setObjectName("stopOverlayMsg")
        self.stop_overlay_msg.setWordWrap(True)
        self.stop_overlay_msg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.stop_overlay_bar = QtWidgets.QProgressBar(self.stop_overlay)
        self.stop_overlay_bar.setRange(0, 0)
        self.stop_overlay_bar.setTextVisible(False)
        self.stop_overlay_bar.setFixedWidth(360)
        overlay_layout.addWidget(self.stop_overlay_title, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addWidget(self.stop_overlay_msg, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addWidget(self.stop_overlay_bar, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        self._resize_stop_overlay()

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
            QFrame#stopOverlay {
                background-color: rgba(10, 12, 20, 230);
                border: 1px solid #2a2d3d;
                border-radius: 12px;
            }
            QLabel#stopOverlayTitle {
                color: #eef1ff;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#stopOverlayMsg {
                color: #b8c0de;
                font-size: 13px;
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
            f"Backend: {str(data.get('capture_backend', 'flir')).upper()}",
            f"Cameras: {data.get('num_cameras', 1)}",
            f"Trigger: {'On' if data['trigger_on'] else 'Off'}",
            f"Config: {Path(data['ds_cfg']).name if data.get('ds_cfg') else 'N/A (inference off)'}",
            f"Task: {Path(data['task_cfg']).name}",
        ]
        if data.get("mouse_id"):
            summary_parts.append(f"Mouse: {data['mouse_id']}")
        inference_on = data.get("inference_enabled", True)
        summary_parts.append(f"Inference: {'On' if inference_on else 'Off'}")
        if str(data.get("capture_backend", "flir")).lower() == "zed":
            summary_parts.append(
                f"Depth: {'On' if data.get('zed_depth_enabled', False) else 'Off'} ({str(data.get('zed_depth_mode', 'NEURAL')).upper()})"
            )
            zed_conf = int(data.get("zed_confidence_threshold", 100) or 100)
            summary_parts.append(f"Depth confidence: {zed_conf if zed_conf > 0 else 'Off'}")
        if data.get("serial_enabled", True):
            summary_parts.append(f"Serial: {data['serial_port']} @ {data['serial_baud']}")
        else:
            summary_parts.append("Serial: disabled")
        self.summary_label.setText(" · ".join(summary_parts))
        info = f"{data['width']}×{data['height']} · {data['fps']} FPS · {'Trig' if data['trigger_on'] else 'Free'} · {data['bitrate']} kbps"
        self.preview.set_info(info)
        self.statusBar().showMessage("Configuration committed.", 4000)
        try:
            self.dashboard.apply_task_config(Path(data["task_cfg"]))
        except Exception as exc:
            self._emit_log(f"[GUI] Task config load failed: {exc}")

        cfg = self.backend.launch_cfg
        cfg.width = data["width"]
        cfg.height = data["height"]
        cfg.fps = data["fps"]
        cfg.pixel_format = data["pixel_format"]
        cfg.capture_backend = str(data.get("capture_backend", "flir"))
        cfg.zed_depth_enabled = bool(data.get("zed_depth_enabled", False))
        cfg.zed_depth_mode = str(data.get("zed_depth_mode", "NEURAL"))
        cfg.zed_depth_socket = str(data.get("zed_depth_socket", "/tmp/cam_depth.sock"))
        cfg.zed_depth_record = bool(data.get("zed_depth_record", False))
        cfg.zed_confidence_threshold = int(data.get("zed_confidence_threshold", 100) or 100)
        cfg.zed_texture_confidence_threshold = int(data.get("zed_texture_confidence_threshold", 100) or 100)
        cfg.zed_depth_minimum_distance_mm = int(data.get("zed_depth_minimum_distance_mm", 300) or 300)
        cfg.zed_depth_maximum_distance_mm = int(data.get("zed_depth_maximum_distance_mm", 20000) or 20000)
        cfg.zed_fill_mode = bool(data.get("zed_fill_mode", False))
        cfg.zed_depth_stabilization = int(data.get("zed_depth_stabilization", 30) or 30)
        cfg.trigger_on = data["trigger_on"]
        cfg.exposure_us = data.get("exposure_us", 10000)
        ds_cfg_raw = data.get("ds_cfg")
        cfg.ds_cfg = Path(ds_cfg_raw) if ds_cfg_raw else None
        cfg.inference_enabled = inference_on
        cfg.draw_skeleton = data.get("draw_skeleton", False)
        cfg.socket_path = data["socket_path"]
        cfg.socket_path_2 = data.get("socket_path_2", "/tmp/cam1.sock")
        cfg.num_cameras = int(data.get("num_cameras", 1))
        cfg.bitrate = data["bitrate"]
        cfg.serial_enabled = data["serial_enabled"]
        cfg.serial_port = data["serial_port"]
        cfg.serial_baud = data["serial_baud"]
        cfg.arduino_fps = data["arduino_fps"]
        cfg.mouse_id = data.get("mouse_id", "")
        cfg.task_cfg = Path(data["task_cfg"])
        self._depth_preview_enabled = bool(
            str(data.get("capture_backend", "flir")).lower() == "zed" and data.get("zed_depth_enabled", False)
        )
        self.depth_preview.setVisible(self._depth_preview_enabled)
        if not self._depth_preview_enabled:
            self.depth_preview.stop_stream()
        self.run_btn.setEnabled(True)
        self._set_skeleton_button_state(enabled=False, checked=cfg.draw_skeleton)

    def _build_launch_config(self) -> process.LaunchConfig:
        if not self._config_data:
            raise RuntimeError("Configuration not set")
        data = self._config_data
        cfg = process.LaunchConfig(
            capture_backend=str(data.get("capture_backend", "flir")),
            zed_depth_enabled=bool(data.get("zed_depth_enabled", False)),
            zed_depth_mode=str(data.get("zed_depth_mode", "NEURAL")),
            zed_depth_socket=str(data.get("zed_depth_socket", "/tmp/cam_depth.sock")),
            zed_depth_record=bool(data.get("zed_depth_record", False)),
            zed_confidence_threshold=int(data.get("zed_confidence_threshold", 100) or 100),
            zed_texture_confidence_threshold=int(data.get("zed_texture_confidence_threshold", 100) or 100),
            zed_depth_minimum_distance_mm=int(data.get("zed_depth_minimum_distance_mm", 300) or 300),
            zed_depth_maximum_distance_mm=int(data.get("zed_depth_maximum_distance_mm", 20000) or 20000),
            zed_fill_mode=bool(data.get("zed_fill_mode", False)),
            zed_depth_stabilization=int(data.get("zed_depth_stabilization", 30) or 30),
            width=data["width"],
            height=data["height"],
            fps=data["fps"],
            pixel_format=data["pixel_format"],
            trigger_on=data["trigger_on"],
            exposure_us=data.get("exposure_us", 10000),
            ds_cfg=(Path(data["ds_cfg"]) if data.get("ds_cfg") else None),
            inference_enabled=data.get("inference_enabled", True),
            socket_path=data["socket_path"],
            socket_path_2=data.get("socket_path_2", "/tmp/cam1.sock"),
            num_cameras=int(data.get("num_cameras", 1)),
            bitrate=data["bitrate"],
            serial_enabled=data["serial_enabled"],
            serial_port=data["serial_port"],
            serial_baud=data["serial_baud"],
            arduino_fps=data["arduino_fps"],
            mouse_id=data.get("mouse_id", ""),
            draw_skeleton=data.get("draw_skeleton", False),
            task_cfg=Path(data["task_cfg"]),
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
    def _run_preflight(self) -> bool:
        if os.environ.get("SQUEAKVIEW_SKIP_PREFLIGHT") == "1":
            self._emit_log("[GUI] Preflight skipped via SQUEAKVIEW_SKIP_PREFLIGHT=1")
            return True

        script = Path(__file__).resolve().parents[3] / "scripts" / "preflight.sh"
        if not script.exists():
            self._emit_log(f"[GUI] Preflight script not found: {script}")
            return True

        env = os.environ.copy()
        env["PYTHON_BIN"] = sys.executable
        backend = str(self._config_data.get("capture_backend", "flir")).lower() if self._config_data else "flir"
        env["CAPTURE_BACKEND"] = backend
        try:
            result = subprocess.run(
                ["bash", str(script)],
                cwd=str(Path(__file__).resolve().parents[3]),
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception as exc:
            self._emit_log(f"[GUI] Preflight execution failed: {exc}")
            return False

        output = (result.stdout or "") + (result.stderr or "")
        output = output.strip()
        if result.returncode == 0:
            self._emit_log("[GUI] Preflight passed")
            if output:
                for line in output.splitlines():
                    self._emit_log(f"[PREFLIGHT] {line}")
            return True

        self._emit_log("[GUI] Preflight failed")
        if output:
            for line in output.splitlines():
                self._emit_log(f"[PREFLIGHT] {line}")
        QtWidgets.QMessageBox.critical(
            self,
            "Preflight Failed",
            "System prerequisites are missing.\n\n"
            "See the log output for details and suggested fixes.\n\n"
            "Set SQUEAKVIEW_SKIP_PREFLIGHT=1 to bypass this check temporarily.",
        )
        return False

    def _on_configure(self) -> None:
        if self._show_config_dialog(initial=False):
            self.statusBar().showMessage("Configuration updated.", 5000)

    @staticmethod
    def _read_ds_batch_size(cfg_path: Path) -> int | None:
        try:
            for raw in Path(cfg_path).read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("batch-size"):
                    return int(line.split("=", 1)[1].strip())
        except Exception:
            return None
        return None

    @staticmethod
    def _effective_batch_camera_count(config: process.LaunchConfig) -> int:
        backend = str(getattr(config, "capture_backend", "flir") or "flir").lower().strip()
        cam_count = max(1, int(getattr(config, "num_cameras", 1)))
        if backend == "zed":
            if bool(getattr(config, "zed_depth_enabled", False)):
                return 1
            return 2 if cam_count > 1 else 1
        return cam_count

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
        if config.inference_enabled:
            cam_count = self._effective_batch_camera_count(config)
            cfg_batch = self._read_ds_batch_size(config.ds_cfg) if config.ds_cfg else None
            if cfg_batch is not None and cfg_batch != cam_count:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Batch Size Mismatch",
                    (
                        f"You selected {cam_count} camera(s), but this DeepStream config uses "
                        f"batch-size={cfg_batch}.\n\n"
                        f"Please choose a config with batch-size={cam_count} and ensure the TensorRT "
                        "engine was built for that same batch size."
                    ),
                )
                self._emit_log(
                    f"[GUI] blocked: camera count ({cam_count}) != config batch-size ({cfg_batch})"
                )
                return
        if not self._run_preflight():
            self._emit_log("[GUI] Run blocked: preflight checks failed.")
            return
        config.preview_window_id = self._preview_window_id
        if not self.backend.start_run(config):
            self._emit_log("[GUI] Failed to start run")
            return
        self.preview.show_hint(False)
        self.preview.set_status("Live", color="#5c6df5")
        self.preview.set_preview_enabled(True)
        self._emit_log("[GUI] Run started")
        if self._depth_preview_enabled:
            socket_path = str(getattr(config, "zed_depth_socket", "/tmp/cam_depth.sock"))
            QtCore.QTimer.singleShot(
                1500,
                lambda: self.depth_preview.start_stream(
                    socket_path=socket_path,
                    width=int(config.width or 1920),
                    height=int(config.height or 1200),
                    fps=int(config.fps or 30),
                ),
            )
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.configure_btn.setEnabled(False)
        # Enable skeleton toggle once a run is active (only meaningful when inference is on)
        self._set_skeleton_button_state(enabled=config.inference_enabled, checked=config.draw_skeleton)

    def _on_stop(self) -> None:
        if self._stop_in_progress:
            return
        self._stop_in_progress = True
        self._show_stop_overlay()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.configure_btn.setEnabled(False)
        self._set_skeleton_button_state(enabled=False, checked=self.skeleton_btn.isChecked())
        self.preview.set_status("Finalizing")
        self.depth_preview.stop_stream()
        self._emit_log("[GUI] Stopping run…")

        def _worker() -> None:
            try:
                self.backend.stop_run()
            except Exception as exc:
                self.stop_failed.emit(str(exc))
                return
            self.stop_done.emit()

        self._stop_thread = threading.Thread(target=_worker, daemon=True)
        self._stop_thread.start()

    @QtCore.Slot()
    def _on_stop_complete(self) -> None:
        self._stop_in_progress = False
        self._hide_stop_overlay()
        self.preview.show_hint(True)
        self.preview.set_status("Idle")
        self.preview.set_preview_enabled(True)
        if self._depth_preview_enabled:
            self.depth_preview.stop_stream()
        self._emit_log("[GUI] Run stopped")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.configure_btn.setEnabled(True)
        self._set_skeleton_button_state(enabled=False, checked=self.skeleton_btn.isChecked())
        self._prompt_upload_latest_run()

    def _prompt_upload_latest_run(self) -> None:
        run_dir = self.backend.state.run_dir
        if run_dir is None or not Path(run_dir).exists():
            self._emit_log("[UPLOAD] no run directory found; skipping upload prompt")
            return
        if self._upload_in_progress:
            self._emit_log("[UPLOAD] upload already in progress; skipping prompt")
            return

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Upload to Google Drive?")
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg_box.setText(
            "Run completed.\n\n"
            "Upload a copy of this run to Google Drive now?\n"
            "(Video files .mp4/.svo/.svo2 will be excluded.)"
        )
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
        msg_box.setStyleSheet(
            """
            QMessageBox {
                background-color: #171821;
                color: #e8ebf4;
            }
            QMessageBox QLabel {
                color: #e8ebf4;
            }
            QMessageBox QPushButton {
                background-color: #4a70d6;
                color: #ffffff;
                padding: 6px 14px;
                border-radius: 4px;
                font-weight: 600;
            }
            QMessageBox QPushButton:hover {
                background-color: #3e64c4;
            }
            """
        )
        answer = msg_box.exec()
        if answer != int(QtWidgets.QMessageBox.StandardButton.Yes):
            self._emit_log("[UPLOAD] skipped by user")
            return
        self._start_drive_upload(Path(run_dir))

    def _start_drive_upload(self, run_dir: Path) -> None:
        if self._upload_in_progress:
            return
        self._upload_in_progress = True

        dest = f"gdrive:SqueakViewUploads/{run_dir.name}"
        cmd = [
            "rclone",
            "copy",
            str(run_dir),
            dest,
            "--exclude",
            "*.mp4",
            "--exclude",
            "*.svo",
            "--exclude",
            "*.svo2",
            "--progress",
        ]
        self._emit_log(f"[UPLOAD] starting: {run_dir} -> {dest}")

        def _worker() -> None:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    self._emit_log(f"[UPLOAD] complete: {dest}")
                else:
                    err = (result.stderr or result.stdout or "").strip()
                    if len(err) > 500:
                        err = err[-500:]
                    self._emit_log(f"[UPLOAD] failed (rc={result.returncode}): {err}")
            except Exception as exc:
                self._emit_log(f"[UPLOAD] error: {exc}")
            finally:
                self._upload_in_progress = False

        self._upload_thread = threading.Thread(target=_worker, daemon=True)
        self._upload_thread.start()

    @QtCore.Slot(str)
    def _on_stop_failed(self, err: str) -> None:
        self._stop_in_progress = False
        self._hide_stop_overlay()
        self._emit_log(f"[GUI] Stop failed: {err}")
        self.depth_preview.stop_stream()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.configure_btn.setEnabled(True)
        self._set_skeleton_button_state(enabled=False, checked=self.skeleton_btn.isChecked())

    def _resize_stop_overlay(self) -> None:
        central = self.centralWidget()
        if central is None:
            return
        self.stop_overlay.setGeometry(0, 0, central.width(), central.height())

    def _show_stop_overlay(self) -> None:
        self._resize_stop_overlay()
        self.stop_overlay.raise_()
        self.stop_overlay.show()

    def _hide_stop_overlay(self) -> None:
        self.stop_overlay.hide()

    # ---- Logging --------------------------------------------------------
    def _emit_log(self, msg: str) -> None:
        self.log_msg.emit(msg)

    @QtCore.Slot(str)
    def _append_log(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 5000)
        print(msg, flush=True)

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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        try:
            self.backend.shutdown()
        except Exception:
            pass
        try:
            self.dashboard.close()
        except Exception:
            pass
        try:
            self.depth_preview.stop_stream()
        except Exception:
            pass
        super().closeEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._resize_stop_overlay()
