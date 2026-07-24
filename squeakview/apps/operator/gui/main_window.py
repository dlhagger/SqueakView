from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview.apps.operator.backend import process
from squeakview.apps.operator.backend.manager import OperatorBackend
from squeakview.apps.operator.gui.config_dialog import (
    ConfigDialog,
    CreateExperimentDialog,
    CreateSubjectDialog,
    SessionLauncherDialog,
    apply_dark_combo_popups,
    center_window,
)
from squeakview.apps.operator.gui.dashboard import BehaviorDashboard
from squeakview.apps.operator.gui.ipc_preview import IpcPreviewController
from squeakview.common.profiles import ExperimentProfile, ProfileStore, SubjectProfile
from squeakview import config as squeakview_config
from squeakview import model_package

BOTTLE_FLUID_PRESETS = ["", "water", "sucrose", "quinine", "ethanol", "saline", "custom"]
RUN_FAILURE_DIALOG_STYLESHEET = """
    QMessageBox {
        background-color: #171821;
    }
    QMessageBox QLabel {
        color: #eef1ff;
        background-color: transparent;
        font-size: 13px;
    }
    QMessageBox QLabel#qt_msgbox_label,
    QMessageBox QLabel#qt_msgbox_informativelabel {
        min-width: 520px;
        max-width: 560px;
    }
    QMessageBox QPushButton {
        min-width: 84px;
        min-height: 30px;
        padding: 4px 14px;
        color: #ffffff;
        background-color: #4f5ed7;
        border: 1px solid #7180ff;
        border-radius: 5px;
        font-weight: 700;
    }
    QMessageBox QPushButton:hover {
        background-color: #5c6df5;
    }
"""


class PreviewWidget(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NativeWindow)
        self.setMinimumHeight(260)
        self.setMinimumWidth(320)
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
        self._target_aspect = 4.0 / 3.0

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
        self._update_aspect_constraint()
        self._reposition_overlays()
        self._update_logo_scale()

    def _update_aspect_constraint(self) -> None:
        height = max(1, self.height())
        target_width = max(360, int(round(height * self._target_aspect)))
        if self.maximumWidth() != target_width:
            self.setMaximumWidth(target_width)

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
            logo_path = squeakview_config.WORKSPACE / "SqueakView_logo.png"
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


class MainWindow(QtWidgets.QMainWindow):
    log_msg = QtCore.Signal(str)
    stop_done = QtCore.Signal()
    stop_failed = QtCore.Signal(str)
    run_started = QtCore.Signal()
    run_failed = QtCore.Signal(str)

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
        self._profile_store = ProfileStore()
        self._experiments: list[ExperimentProfile] = []
        self._subjects: list[SubjectProfile] = []
        self._profile_selection_updating = False
        self._centered_once = False
        self.stop_done.connect(self._on_stop_complete)
        self.stop_failed.connect(self._on_stop_failed)
        self.run_started.connect(self._on_backend_run_started)
        self.run_failed.connect(self._on_backend_run_failed)

        self.backend = OperatorBackend(
            self._emit_log,
            self._forward_dashboard,
            on_run_started=self.run_started.emit,
            on_run_failed=self.run_failed.emit,
        )

        self._build_ui()
        self._preview_controller = IpcPreviewController(self._emit_log, self)
        self._preview_controller.ready.connect(self._on_preview_ready)
        self._preview_controller.failed.connect(self._on_preview_failed)
        self._preview_controller.ended.connect(self._on_preview_ended)
        self._apply_brand_theme()
        apply_dark_combo_popups(self)
        QtCore.QTimer.singleShot(0, self._capture_preview_window_id)
        self._config_data = self._default_config_data()
        if not self._show_launch_dialog():
            QtCore.QTimer.singleShot(0, self.close)
        else:
            self.statusBar().showMessage("Ready to record.")
        self.preview.set_status("Idle")

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._centered_once and not (self.isMaximized() or self.isFullScreen()):
            self._centered_once = True
            center_window(self)

    # ---- UI -------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        # Main content grid: top row (preview + meters/summary/task), bottom row (graphs)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)

        preview_group = QtWidgets.QGroupBox("Live Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        self.preview = PreviewWidget(self)
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        preview_layout.addWidget(self.preview, 1, QtCore.Qt.AlignmentFlag.AlignHCenter)
        grid.addWidget(preview_group, 0, 0, 1, 1)

        # Dashboard (plots), meters, and task viewer
        # Count pellets on retrieval events by default (arrival-only counting was confusing)
        self.dashboard = BehaviorDashboard(window_sec=300.0, pellet_mode="retrieval")
        meters_only = self.dashboard.detach_meters()
        task_state_panel = self.dashboard.detach_task_panel()
        meters_group = QtWidgets.QGroupBox("System Load")
        meters_layout = QtWidgets.QVBoxLayout(meters_group)
        meters_layout.setContentsMargins(12, 12, 12, 12)
        meters_layout.addWidget(meters_only)

        profile_group = QtWidgets.QGroupBox("Profiles", self)
        self._profile_group = profile_group
        profile_form = QtWidgets.QFormLayout(profile_group)
        profile_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        profile_form.setHorizontalSpacing(10)
        profile_form.setVerticalSpacing(8)

        exp_row = QtWidgets.QHBoxLayout()
        self.experiment_combo = QtWidgets.QComboBox(self)
        self.experiment_combo.currentIndexChanged.connect(self._on_experiment_selected)
        exp_row.addWidget(self.experiment_combo, 1)
        self.new_experiment_btn = QtWidgets.QPushButton("New…", self)
        self.new_experiment_btn.clicked.connect(self._on_new_experiment)
        exp_row.addWidget(self.new_experiment_btn, 0)
        profile_form.addRow("Experiment:", exp_row)

        subj_row = QtWidgets.QHBoxLayout()
        self.subject_combo = QtWidgets.QComboBox(self)
        self.subject_combo.currentIndexChanged.connect(self._on_subject_selected)
        subj_row.addWidget(self.subject_combo, 1)
        self.new_subject_btn = QtWidgets.QPushButton("New…", self)
        self.new_subject_btn.clicked.connect(self._on_new_subject)
        subj_row.addWidget(self.new_subject_btn, 0)
        profile_form.addRow("Subject:", subj_row)
        meters_layout.addSpacing(6)
        meters_layout.addWidget(profile_group)
        profile_group.hide()

        self.summary_label = QtWidgets.QLabel("No configuration loaded.")
        self.summary_label.setObjectName("summaryBanner")
        self.summary_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.summary_label.setWordWrap(True)
        meters_layout.addSpacing(6)
        meters_layout.addWidget(self.summary_label)

        bottle_group = QtWidgets.QGroupBox("Bottles", self)
        bottle_layout = QtWidgets.QGridLayout(bottle_group)
        bottle_layout.setHorizontalSpacing(8)
        bottle_layout.setVerticalSpacing(8)
        bottle_layout.addWidget(QtWidgets.QLabel("Side", self), 0, 0)
        bottle_layout.addWidget(QtWidgets.QLabel("Fluid", self), 0, 1)
        bottle_layout.addWidget(QtWidgets.QLabel("Initial g", self), 0, 2)
        bottle_layout.addWidget(QtWidgets.QLabel("Final g", self), 0, 3)

        self.left_fluid_combo = self._make_fluid_combo()
        self.left_initial_weight_edit = self._make_weight_edit("initial")
        self.left_final_weight_edit = self._make_weight_edit("final")
        self.right_fluid_combo = self._make_fluid_combo()
        self.right_initial_weight_edit = self._make_weight_edit("initial")
        self.right_final_weight_edit = self._make_weight_edit("final")

        bottle_layout.addWidget(QtWidgets.QLabel("Left", self), 1, 0)
        bottle_layout.addWidget(self.left_fluid_combo, 1, 1)
        bottle_layout.addWidget(self.left_initial_weight_edit, 1, 2)
        bottle_layout.addWidget(self.left_final_weight_edit, 1, 3)
        bottle_layout.addWidget(QtWidgets.QLabel("Right", self), 2, 0)
        bottle_layout.addWidget(self.right_fluid_combo, 2, 1)
        bottle_layout.addWidget(self.right_initial_weight_edit, 2, 2)
        bottle_layout.addWidget(self.right_final_weight_edit, 2, 3)

        self.bottle_status_label = QtWidgets.QLabel("Bottle info pending for next run.", self)
        self.bottle_status_label.setObjectName("bottleStatus")
        self.bottle_status_label.setWordWrap(True)
        self.save_bottles_btn = QtWidgets.QPushButton("Save Bottle Info", self)
        self.save_bottles_btn.setObjectName("secondaryButton")
        self.save_bottles_btn.clicked.connect(self._on_save_bottles)
        bottle_action_row = QtWidgets.QHBoxLayout()
        bottle_action_row.addWidget(self.bottle_status_label, 1)
        bottle_action_row.addWidget(self.save_bottles_btn, 0)
        bottle_layout.addLayout(bottle_action_row, 3, 0, 1, 4)
        bottle_layout.setColumnStretch(1, 1)
        bottle_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )

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
        self.stop_btn = QtWidgets.QPushButton("Stop Recording")
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.stop_btn)
        meters_layout.addSpacing(6)
        meters_layout.addLayout(btn_row)

        grid.addWidget(meters_group, 0, 1, 1, 1)

        right_column = QtWidgets.QWidget(self)
        right_column_layout = QtWidgets.QVBoxLayout(right_column)
        right_column_layout.setContentsMargins(0, 0, 0, 0)
        right_column_layout.setSpacing(14)

        self.task_state_group = QtWidgets.QGroupBox("Live Task State")
        task_state_layout = QtWidgets.QVBoxLayout(self.task_state_group)
        task_state_layout.setContentsMargins(12, 12, 12, 12)
        task_state_layout.addWidget(task_state_panel, 1)
        self.task_state_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        right_column_layout.addWidget(self.task_state_group, 1)
        right_column_layout.addWidget(bottle_group, 0)
        grid.addWidget(right_column, 0, 2, 1, 1)

        dashboard_group = QtWidgets.QGroupBox("Behavior Dashboard")
        dash_layout = QtWidgets.QVBoxLayout(dashboard_group)
        dash_layout.setContentsMargins(16, 16, 16, 16)
        self.dashboard.setMinimumHeight(360)
        dash_layout.addWidget(self.dashboard)
        dashboard_group.setMinimumHeight(390)
        grid.addWidget(dashboard_group, 1, 0, 1, 3)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 2)

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
            "Finalizing capture/inference, closing files, and stopping serial control.",
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
            QWidget {
                selection-background-color: #5967d8;
                selection-color: #ffffff;
            }
            QLabel {
                color: #e8ebf4;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
                background-color: #12172a;
                color: #e8ecff;
                border: 1px solid #333a55;
                border-radius: 6px;
                padding: 6px 8px;
                min-height: 24px;
                selection-background-color: #5967d8;
                selection-color: #ffffff;
            }
            QLineEdit {
                placeholder-text-color: #7f8aac;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
            QTextEdit:focus, QPlainTextEdit:focus {
                border-color: #6f7dff;
                background-color: #151b31;
            }
            QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
            QTextEdit:disabled, QPlainTextEdit:disabled {
                background-color: #171b29;
                color: #7f8aac;
                border-color: #2a3046;
            }
            QComboBox {
                padding-right: 30px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left: 1px solid #333a55;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
                background-color: #192039;
            }
            QComboBox::down-arrow {
                width: 10px;
                height: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #11162a;
                color: #e8ecff;
                border: 1px solid #333a55;
                selection-background-color: #5967d8;
                selection-color: #ffffff;
                outline: 0;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item {
                min-height: 26px;
                padding: 4px 8px;
            }
            QCheckBox {
                color: #d7ddf5;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #46506d;
                background-color: #12172a;
            }
            QCheckBox::indicator:checked {
                background-color: #5c6df5;
                border-color: #5c6df5;
            }
            QGroupBox {
                border: 1px solid #2a2d3d;
                border-radius: 10px;
                background-color: #1a1d2a;
                margin-top: 16px;
                padding: 16px 12px 12px 12px;
            }
            QGroupBox::title {
                color: #aeb8ff;
                subcontrol-origin: margin;
                left: 14px;
                top: 10px;
                padding: 0 6px;
                background-color: #171821;
            }
            QMenu {
                background-color: #11162a;
                color: #e8ecff;
                border: 1px solid #333a55;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 18px;
            }
            QMenu::item:selected {
                background-color: #5967d8;
                color: #ffffff;
            }
            QToolTip {
                background-color: #11162a;
                color: #eef1ff;
                border: 1px solid #333a55;
                padding: 4px 6px;
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
            QLabel#bottleStatus {
                color: #9aa7cc;
                font-size: 11px;
            }
            QLabel#taskTitle {
                color: #eef1ff;
                font-size: 14px;
                font-weight: 800;
            }
            QLabel#taskPath {
                color: #9aa7cc;
                font-size: 11px;
            }
            QTextBrowser#taskSummary {
                background-color: #101526;
                color: #d7ddf5;
                border: 1px solid #2c3550;
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
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
            QProgressBar {
                background-color: #11162a;
                border: 1px solid #333a55;
                border-radius: 6px;
                min-height: 10px;
            }
            QProgressBar::chunk {
                background-color: #5c6df5;
                border-radius: 5px;
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

    def _default_config_data(self) -> dict:
        defaults = process.LaunchConfig()
        default_task = squeakview_config.TASKS_DIR / "default.yaml"
        task_cfg = default_task if default_task.exists() else squeakview_config.TASKS_DIR / "gonogo_auto.yaml"
        ds_cfg = defaults.ds_cfg if defaults.ds_cfg else None
        return {
            "width": int(defaults.width or 1440),
            "height": int(defaults.height or 1080),
            "fps": int(defaults.fps or 30),
            "pixel_format": defaults.pixel_format or "Mono8",
            "capture_backend": defaults.capture_backend,
            "trigger_on": defaults.trigger_on,
            "exposure_us": int(defaults.exposure_us or 10000),
            "arduino_fps": defaults.arduino_fps,
            "serial_enabled": defaults.serial_enabled,
            "serial_port": defaults.serial_port,
            "serial_baud": defaults.serial_baud,
            "ds_cfg": str(ds_cfg) if ds_cfg else "",
            "inference_enabled": defaults.inference_enabled,
            "task_cfg": str(task_cfg),
            "num_cameras": max(1, defaults.num_cameras),
            "bitrate": defaults.bitrate,
            "mouse_id": "",
            "experiment_name": "",
            "experiment_mode": "sandbox",
        }

    def _make_fluid_combo(self) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(self)
        combo.setEditable(True)
        combo.addItems(BOTTLE_FLUID_PRESETS)
        combo.setMinimumWidth(96)
        return combo

    def _make_weight_edit(self, phase: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(self)
        edit.setPlaceholderText(phase)
        edit.setMaximumWidth(82)
        validator = QtGui.QDoubleValidator(0.0, 100000.0, 4, edit)
        validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        edit.setValidator(validator)
        return edit

    @staticmethod
    def _parse_weight(text: str, label: str, *, strict: bool) -> float | None:
        cleaned = text.strip()
        if not cleaned:
            return None
        try:
            value = float(cleaned)
        except ValueError:
            if strict:
                raise ValueError(f"{label} must be a number.")
            return None
        if value < 0:
            if strict:
                raise ValueError(f"{label} cannot be negative.")
            return None
        return round(value, 6)

    def _collect_bottle_payload(self, *, include_final: bool, strict: bool) -> dict[str, object]:
        sides = {
            "left": {
                "fluid": self.left_fluid_combo.currentText().strip(),
                "initial_edit": self.left_initial_weight_edit,
                "final_edit": self.left_final_weight_edit,
            },
            "right": {
                "fluid": self.right_fluid_combo.currentText().strip(),
                "initial_edit": self.right_initial_weight_edit,
                "final_edit": self.right_final_weight_edit,
            },
        }
        payload: dict[str, object] = {}
        for side, widgets in sides.items():
            label = side.title()
            fluid = widgets["fluid"]
            initial = self._parse_weight(widgets["initial_edit"].text(), f"{label} initial weight", strict=strict)
            final = (
                self._parse_weight(widgets["final_edit"].text(), f"{label} final weight", strict=strict)
                if include_final
                else None
            )
            if strict and (initial is not None or final is not None) and not fluid:
                raise ValueError(f"{label} fluid is required when saving bottle weights.")
            payload[side] = {
                "fluid": fluid,
                "initial_weight_g": initial,
                "final_weight_g": final,
            }
        return payload

    def _clear_bottle_final_fields(self) -> None:
        self.left_final_weight_edit.clear()
        self.right_final_weight_edit.clear()

    def _set_bottle_status(self, text: str) -> None:
        self.bottle_status_label.setText(text)

    @QtCore.Slot()
    def _on_save_bottles(self) -> None:
        try:
            payload = self._collect_bottle_payload(include_final=True, strict=True)
        except ValueError as exc:
            message = str(exc)
            self._set_bottle_status(message)
            QtWidgets.QMessageBox.warning(self, "Bottle Weight", message)
            return

        run_dir = self.backend.state.run_dir
        if run_dir is None:
            self._set_bottle_status("Bottle info pending for next run.")
            self._emit_log("[BOTTLES] no active run; values will be saved when the next run starts")
            return

        try:
            summary = self.backend.save_bottle_measurements(payload, run_dir=run_dir)
        except Exception as exc:
            self._set_bottle_status("Bottle save failed.")
            self._emit_log(f"[BOTTLES] save failed: {exc}")
            return

        state = "complete" if summary.get("complete") else "saved; missing one or more weights"
        self._set_bottle_status(f"Bottle info {state}.")

    def _reload_profiles(self) -> None:
        self._experiments = self._profile_store.list_experiments()
        self._subjects = self._profile_store.list_subjects()
        self._refresh_profile_selectors()

    def _refresh_profile_selectors(self) -> None:
        selected_exp = self.current_experiment_slug()
        selected_subject = self.current_subject_id()
        self._profile_selection_updating = True
        try:
            self.experiment_combo.clear()
            self.experiment_combo.addItem("No experiment", "")
            for profile in self._experiments:
                self.experiment_combo.addItem(profile.name, profile.slug)
            exp_index = max(0, self.experiment_combo.findData(selected_exp))
            self.experiment_combo.setCurrentIndex(exp_index)

            self.subject_combo.clear()
            self.subject_combo.addItem("No subject", "")
            for profile in self._subjects:
                self.subject_combo.addItem(profile.name, profile.subject_id)
            subj_index = max(0, self.subject_combo.findData(selected_subject))
            self.subject_combo.setCurrentIndex(subj_index)
        finally:
            self._profile_selection_updating = False

    def _apply_profile_defaults(self) -> None:
        if self._subjects and not self.current_subject_id():
            first_subject = self._subjects[0]
            idx = self.subject_combo.findData(first_subject.subject_id)
            if idx >= 0:
                self.subject_combo.setCurrentIndex(idx)
        if self._experiments and not self.current_experiment_slug():
            first_experiment = self._experiments[0]
            idx = self.experiment_combo.findData(first_experiment.slug)
            if idx >= 0:
                self.experiment_combo.setCurrentIndex(idx)
        self._apply_profile_selection()

    def current_experiment_slug(self) -> str:
        return str(self.experiment_combo.currentData() or "")

    def current_subject_id(self) -> str:
        return str(self.subject_combo.currentData() or "")

    def _find_experiment(self, slug: str) -> ExperimentProfile | None:
        for profile in self._experiments:
            if profile.slug == slug:
                return profile
        return None

    def _find_subject(self, subject_id: str) -> SubjectProfile | None:
        for profile in self._subjects:
            if profile.subject_id == subject_id:
                return profile
        return None

    def _apply_profile_selection(self) -> None:
        data = dict(self._config_data or self._default_config_data())
        experiment = self._find_experiment(self.current_experiment_slug())
        subject = self._find_subject(self.current_subject_id())
        if experiment:
            data["experiment_name"] = experiment.slug
            data.update(dict(experiment.config or {}))
        else:
            data["experiment_name"] = ""
        if subject:
            data["mouse_id"] = subject.subject_id
        else:
            data["mouse_id"] = ""
        self._config_data = data
        self._apply_config(data)

    @QtCore.Slot()
    def _on_experiment_selected(self) -> None:
        if self._profile_selection_updating:
            return
        self._apply_profile_selection()

    @QtCore.Slot()
    def _on_subject_selected(self) -> None:
        if self._profile_selection_updating:
            return
        subject = self._find_subject(self.current_subject_id())
        if subject and subject.default_experiment:
            idx = self.experiment_combo.findData(subject.default_experiment)
            if idx >= 0 and idx != self.experiment_combo.currentIndex():
                self._profile_selection_updating = True
                try:
                    self.experiment_combo.setCurrentIndex(idx)
                finally:
                    self._profile_selection_updating = False
        self._apply_profile_selection()

    @QtCore.Slot()
    def _on_new_experiment(self) -> None:
        dialog = CreateExperimentDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        name = dialog.experiment_name
        data = self._config_data or self._default_config_data()
        profile = ExperimentProfile(
            name=name,
            slug=name,
            config={key: (str(value) if isinstance(value, Path) else value) for key, value in data.items()},
        )
        path = self._profile_store.save_experiment(profile)
        self._emit_log(f"[GUI] experiment profile saved → {path}")
        self._reload_profiles()
        idx = self.experiment_combo.findData(path.stem)
        if idx >= 0:
            self.experiment_combo.setCurrentIndex(idx)

    @QtCore.Slot()
    def _on_new_subject(self) -> None:
        dialog = CreateSubjectDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        profile = SubjectProfile(
            name=dialog.subject_name,
            subject_id=dialog.subject_id,
            default_experiment=(self.current_experiment_slug() or None),
        )
        path = self._profile_store.save_subject(profile)
        self._emit_log(f"[GUI] subject profile saved → {path}")
        self._reload_profiles()
        idx = self.subject_combo.findData(profile.subject_id)
        if idx >= 0:
            self.subject_combo.setCurrentIndex(idx)

    # ---- Configuration --------------------------------------------------
    def _show_launch_dialog(self) -> bool:
        dialog = SessionLauncherDialog(self, base_config=self._config_data)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        result = dialog.result_config
        if not result:
            return False
        self._config_data = result
        self._apply_config(result)
        return True

    def _show_config_dialog(self, *, initial: bool = False) -> bool:
        dialog = ConfigDialog(self, config=self._config_data, show_session_setup=False)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        result = dialog.result_config
        if not result:
            return False
        if self._config_data and "experiment_name" not in result:
            result["experiment_name"] = self._config_data.get("experiment_name", "")
        self._config_data = result
        self._apply_config(result)
        return True

    def _apply_config(self, data: dict) -> None:
        ds_cfg = squeakview_config.resolve_workspace_path(data.get("ds_cfg")) if data.get("ds_cfg") else None
        task_cfg = squeakview_config.resolve_workspace_path(data.get("task_cfg"))
        if ds_cfg is not None:
            data["ds_cfg"] = str(ds_cfg)
        if task_cfg is not None:
            data["task_cfg"] = str(task_cfg)
        inference_on = data.get("inference_enabled", True)
        serial_text = (
            f"{data['serial_port']} @ {data['serial_baud']}" if data.get("serial_enabled", True) else "disabled"
        )
        session_bits = []
        if data.get("experiment_name"):
            session_bits.append(str(data["experiment_name"]))
        if data.get("mouse_id"):
            session_bits.append(f"Subject {data['mouse_id']}")
        session_text = " / ".join(session_bits) if session_bits else "No session profile"
        model_name = "Inference off"
        if ds_cfg is not None:
            try:
                model_name = model_package.validate_model_package(ds_cfg).name
            except model_package.ModelPackageError:
                model_name = f"Invalid: {ds_cfg.name}"
        rows = [
            ("Camera", f"{data['width']}×{data['height']} @ {data['fps']} FPS · {data['pixel_format']} · {data.get('num_cameras', 1)} cam"),
            ("Run", f"Trigger {'On' if data['trigger_on'] else 'Off'} · Inference {'On' if inference_on else 'Off'}"),
            ("Model", model_name),
            ("Task", task_cfg.name if task_cfg else "N/A"),
            ("Serial", serial_text),
            ("Session", session_text),
        ]
        self.summary_label.setText(
            "<table cellspacing='0' cellpadding='2'>"
            + "".join(
                "<tr>"
                f"<td style='color:#9aa7cc; font-weight:700; padding-right:10px;'>{label}</td>"
                f"<td style='color:#e8ecff;'>{value}</td>"
                "</tr>"
                for label, value in rows
            )
            + "</table>"
        )
        info = f"{data['width']}×{data['height']} · {data['fps']} FPS · {'Trig' if data['trigger_on'] else 'Free'} · {data['bitrate']} kbps"
        self.preview.set_info(info)
        self.statusBar().showMessage("Configuration committed.", 4000)
        try:
            if task_cfg is not None:
                self.dashboard.apply_task_config(task_cfg)
        except Exception as exc:
            self._emit_log(f"[GUI] Task config load failed: {exc}")

        cfg = self.backend.launch_cfg
        cfg.width = data["width"]
        cfg.height = data["height"]
        cfg.fps = data["fps"]
        cfg.pixel_format = data["pixel_format"]
        cfg.capture_backend = str(data.get("capture_backend", "flir_direct"))
        cfg.trigger_on = data["trigger_on"]
        cfg.exposure_us = data.get("exposure_us", 10000)
        cfg.ds_cfg = ds_cfg
        cfg.inference_enabled = inference_on
        cfg.num_cameras = int(data.get("num_cameras", 1))
        cfg.bitrate = data["bitrate"]
        cfg.serial_enabled = data["serial_enabled"]
        cfg.serial_port = data["serial_port"]
        cfg.serial_baud = data["serial_baud"]
        cfg.arduino_fps = data["arduino_fps"]
        cfg.mouse_id = data.get("mouse_id", "")
        cfg.experiment_name = data.get("experiment_name", "")
        cfg.task_cfg = task_cfg
        self.run_btn.setEnabled(True)

    def _build_launch_config(self) -> process.LaunchConfig:
        if not self._config_data:
            raise RuntimeError("Configuration not set")
        data = self._config_data
        ds_cfg = squeakview_config.resolve_workspace_path(data.get("ds_cfg")) if data.get("ds_cfg") else None
        task_cfg = squeakview_config.resolve_workspace_path(data.get("task_cfg"))
        cfg = process.LaunchConfig(
            capture_backend=str(data.get("capture_backend", "flir_direct")),
            width=data["width"],
            height=data["height"],
            fps=data["fps"],
            pixel_format=data["pixel_format"],
            trigger_on=data["trigger_on"],
            exposure_us=data.get("exposure_us", 10000),
            ds_cfg=ds_cfg,
            inference_enabled=data.get("inference_enabled", True),
            num_cameras=int(data.get("num_cameras", 1)),
            bitrate=data["bitrate"],
            serial_enabled=data["serial_enabled"],
            serial_port=data["serial_port"],
            serial_baud=data["serial_baud"],
            arduino_fps=data["arduino_fps"],
            mouse_id=data.get("mouse_id", ""),
            experiment_name=data.get("experiment_name", ""),
            task_cfg=task_cfg,
        )
        cfg.bottles = self._collect_bottle_payload(include_final=False, strict=False)
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

        script = squeakview_config.WORKSPACE / "scripts" / "preflight.sh"
        if not script.exists():
            self._emit_log(f"[GUI] Preflight script not found: {script}")
            return True

        env = os.environ.copy()
        env["PYTHON_BIN"] = sys.executable
        backend = (
            str(self._config_data.get("capture_backend", "flir_direct")).lower()
            if self._config_data
            else "flir_direct"
        )
        env["CAPTURE_BACKEND"] = backend
        inference_enabled = bool(self._config_data.get("inference_enabled", True)) if self._config_data else True
        env["INFERENCE_ENABLED"] = "1" if inference_enabled else "0"
        if self._config_data and self._config_data.get("ds_cfg"):
            ds_cfg = squeakview_config.resolve_workspace_path(self._config_data["ds_cfg"])
            if ds_cfg is not None:
                env["DS_CFG"] = str(ds_cfg)
        try:
            result = subprocess.run(
                ["bash", str(script)],
                cwd=str(squeakview_config.WORKSPACE),
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
        return max(1, int(getattr(config, "num_cameras", 1)))

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
        self._start_ipc_preview(config)
        self._clear_bottle_final_fields()
        if self.backend.state.run_dir is not None:
            self._set_bottle_status("Initial bottle info saved with current run.")
        self.dashboard.clear_jam_alert()
        self.preview.show_hint(False)
        self.preview.set_status("Starting")
        self.preview.set_preview_enabled(True)
        self._emit_log("[GUI] Run is waiting for inference readiness")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.configure_btn.setEnabled(False)

    @QtCore.Slot()
    def _on_backend_run_started(self) -> None:
        self.preview.show_hint(False)
        self.preview.set_status("Live", color="#5c6df5")
        self._emit_log("[GUI] Run started")

    def _start_ipc_preview(self, config: process.LaunchConfig) -> None:
        sockets = tuple(config.preview_socket_paths)
        if not sockets or self._preview_window_id is None:
            self._on_preview_failed("Preview unavailable: no IPC socket was configured")
            return
        if len(sockets) > 1:
            self._emit_log(
                f"[PREVIEW] {len(sockets)} camera streams are available; displaying camera 0"
            )
        self._preview_controller.start(sockets[0], self._preview_window_id)

    @QtCore.Slot()
    def _on_preview_ready(self) -> None:
        self.preview.show_hint(False)
        self.preview.set_status("Live", color="#5c6df5")

    @QtCore.Slot(str)
    def _on_preview_failed(self, error: str) -> None:
        self.preview.label.setText(error)
        self.preview.show_hint(True)
        self.preview.set_status("Recording (no preview)")

    @QtCore.Slot()
    def _on_preview_ended(self) -> None:
        if not self._stop_in_progress:
            self.preview.label.setText("Preview stream ended")
            self.preview.show_hint(True)
            self.preview.set_status("Preview ended")

    @QtCore.Slot(str)
    def _on_backend_run_failed(self, error: str) -> None:
        self._preview_controller.stop()
        self._stop_in_progress = False
        self._hide_stop_overlay()
        self.preview.show_hint(True)
        self.preview.set_status("Failed")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.configure_btn.setEnabled(True)
        self._emit_log(f"[GUI] Run failed: {error}")
        self._set_bottle_status("Run failed; inspect the log and run status before retrying.")
        dialog = QtWidgets.QMessageBox(self)
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        dialog.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        dialog.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        dialog.setStyleSheet(RUN_FAILURE_DIALOG_STYLESHEET)
        if error.startswith("Could not open serial port"):
            dialog.setWindowTitle("Serial Port Unavailable")
            dialog.setText("SqueakView could not connect to the experiment controller.")
        else:
            dialog.setWindowTitle("Run Failed")
            dialog.setText("SqueakView could not start or continue the run.")
        dialog.setInformativeText(error)
        dialog.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        dialog.exec()

    def _on_stop(self) -> None:
        if self._stop_in_progress:
            return
        self._stop_in_progress = True
        self._show_stop_overlay()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.configure_btn.setEnabled(False)
        self.preview.set_status("Finalizing")
        self._emit_log("[GUI] Stopping run…")
        self._preview_controller.stop()

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
        self._emit_log("[GUI] Run stopped")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.configure_btn.setEnabled(True)
        run_dir = self.backend.state.run_dir
        if run_dir is not None:
            self._emit_log(f"[SAVE] local run finalized: {run_dir}")
            self._set_bottle_status("Run finalized; final weights can be saved.")

    @QtCore.Slot(str)
    def _on_stop_failed(self, err: str) -> None:
        self._stop_in_progress = False
        self._hide_stop_overlay()
        self._emit_log(f"[GUI] Stop failed: {err}")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.configure_btn.setEnabled(True)

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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._preview_controller.stop()
        try:
            self.backend.shutdown()
        except Exception:
            pass
        try:
            self.dashboard.close()
        except Exception:
            pass
        super().closeEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._resize_stop_overlay()
