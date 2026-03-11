from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview import config as squeakview_config


class SensorDetectWorker(QtCore.QObject):
    finished = QtCore.Signal(int, int, int, str)

    @QtCore.Slot()
    def run(self) -> None:
        flir_count = 0
        zed_count = 0
        warnings: list[str] = []

        # FLIR detection via PySpin.
        system = None
        cam_list = None
        try:
            import PySpin  # type: ignore
            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()
            flir_count = int(cam_list.GetSize())
        except Exception as exc:
            warnings.append(f"FLIR detection unavailable (PySpin): {exc}")
        finally:
            try:
                if cam_list is not None:
                    cam_list.Clear()
            except Exception:
                pass
            try:
                if system is not None:
                    system.ReleaseInstance()
            except Exception:
                pass

        # ZED detection via pyzed.
        try:
            import pyzed.sl as sl  # type: ignore

            try:
                devices = sl.Camera.get_device_list()
                zed_count = int(len(devices))
            except Exception:
                zed_count = 0
        except Exception as exc:
            warnings.append(f"ZED detection unavailable (pyzed): {exc}")

        # ZED stereo camera counts as two capture sources for current dual-camera UI.
        zed_sources = 2 if zed_count > 0 else 0
        total_sources = flir_count + zed_sources
        warning_text = "\n".join(warnings)
        self.finished.emit(total_sources, flir_count, zed_count, warning_text)


class ConfigDialog(QtWidgets.QDialog):
    """Modal dialog to configure SqueakView capture + inference parameters."""

    def __init__(self, parent=None, *, title: str = "Configure SqueakView", config: Optional[dict] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(440)

        self.setStyleSheet("""
            QDialog {
                background-color: #0f1118;
            }
            QLabel {
                color: #d7ddf5;
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 1px solid #24283b;
                border-radius: 8px;
                background-color: #14192a;
                top: -1px;
            }
            QTabBar::tab {
                background: #181f31;
                color: #aeb8da;
                border: 1px solid #24283b;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 7px 12px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #1f2840;
                color: #e7ebff;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
            QLineEdit, QComboBox {
                background-color: #11162a;
                color: #e8ecff;
                border: 1px solid #2c3550;
                border-radius: 4px;
                padding: 4px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #7480ff;
            }
            QComboBox QAbstractItemView {
                background-color: #11162a;
                color: #e8ecff;
                selection-background-color: #283a7a;
                selection-color: #ffffff;
                border: 1px solid #2c3550;
            }
            QCheckBox {
                color: #d7ddf5;
            }
            QPushButton {
                background-color: #2f4daa;
                color: #ffffff;
                padding: 6px 14px;
                border-radius: 4px;
                border: 1px solid #3557bf;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3a5fc9;
            }
            QPushButton:disabled {
                background-color: #2a3248;
                color: #7f8aac;
                border-color: #313c5d;
            }
            QProgressDialog {
                background-color: #14192a;
                color: #d7ddf5;
            }
            QMessageBox {
                background-color: #14192a;
                color: #d7ddf5;
            }
        """)
        cfg = config or {}
        self._detected_cameras = max(0, min(2, int(cfg.get("num_cameras", 0))))
        self._detect_thread: QtCore.QThread | None = None
        self._detect_worker: SensorDetectWorker | None = None
        self._detect_progress: QtWidgets.QProgressDialog | None = None

        form = QtWidgets.QFormLayout()
        self._run_form = form
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)

        int_validator = QtGui.QIntValidator(1, 4096, self)

        self.width_edit = QtWidgets.QLineEdit(str(cfg.get("width", 1440)))
        self.width_edit.setValidator(int_validator)
        form.addRow("Width:", self.width_edit)

        self.height_edit = QtWidgets.QLineEdit(str(cfg.get("height", 1080)))
        self.height_edit.setValidator(int_validator)
        form.addRow("Height:", self.height_edit)

        self.fps_edit = QtWidgets.QLineEdit(str(cfg.get("fps", 30)))
        self.fps_edit.setValidator(QtGui.QIntValidator(1, 240, self))
        form.addRow("FPS:", self.fps_edit)

        self.capture_backend_combo = QtWidgets.QComboBox()
        self.capture_backend_combo.addItem("FLIR (PySpin)", "flir")
        self.capture_backend_combo.addItem("ZED (DeepStream zedsrc)", "zed")
        current_backend = str(cfg.get("capture_backend", "flir")).lower().strip()
        idx = self.capture_backend_combo.findData(current_backend)
        if idx >= 0:
            self.capture_backend_combo.setCurrentIndex(idx)
        form.addRow("Capture Backend:", self.capture_backend_combo)

        self.trigger_chk = QtWidgets.QCheckBox("Enable camera trigger")
        self.trigger_chk.setChecked(cfg.get("trigger_on", True))
        form.addRow("", self.trigger_chk)

        self.arduino_fps_edit = QtWidgets.QLineEdit(str(cfg.get("arduino_fps", 30)))
        self.arduino_fps_edit.setValidator(QtGui.QIntValidator(1, 240, self))
        form.addRow("Arduino FPS:", self.arduino_fps_edit)

        self.serial_enable = QtWidgets.QCheckBox("Enable Arduino serial logging")
        self.serial_enable.setChecked(cfg.get("serial_enabled", True))
        form.addRow("", self.serial_enable)

        self.inference_enable = QtWidgets.QCheckBox("Enable YOLO inference (DeepStream)")
        self.inference_enable.setChecked(cfg.get("inference_enabled", True))
        form.addRow("", self.inference_enable)

        self.skeleton_chk = QtWidgets.QCheckBox("Draw skeleton overlay (pose only)")
        self.skeleton_chk.setChecked(cfg.get("draw_skeleton", False))
        form.addRow("", self.skeleton_chk)

        self.mouse_id_edit = QtWidgets.QLineEdit(str(cfg.get("mouse_id", "")))
        form.addRow("Mouse ID:", self.mouse_id_edit)

        task_default = cfg.get("task_cfg", "")
        if not task_default:
            candidate = squeakview_config.TASKS_DIR / "default.yaml"
            task_default = str(candidate) if candidate.exists() else ""
        self.task_cfg_edit = QtWidgets.QLineEdit(str(task_default))
        task_browse_btn = QtWidgets.QPushButton("Browse…")
        task_browse_btn.clicked.connect(self._on_browse_task_cfg)
        task_layout = QtWidgets.QHBoxLayout()
        task_layout.addWidget(self.task_cfg_edit, 1)
        task_layout.addWidget(task_browse_btn, 0)
        form.addRow("Task config:", task_layout)

        serial_row = QtWidgets.QHBoxLayout()
        self.serial_port_edit = QtWidgets.QLineEdit(cfg.get("serial_port", "/dev/ttyACM0"))
        serial_row.addWidget(QtWidgets.QLabel("Port:"))
        serial_row.addWidget(self.serial_port_edit)
        self.serial_baud_edit = QtWidgets.QLineEdit(str(cfg.get("serial_baud", 115200)))
        serial_row.addWidget(QtWidgets.QLabel("Baud:"))
        serial_row.addWidget(self.serial_baud_edit)
        self.serial_row_widget = QtWidgets.QWidget(self)
        self.serial_row_widget.setLayout(serial_row)
        form.addRow("", self.serial_row_widget)

        default_cfg = cfg.get("ds_cfg", "")
        self.cfg_edit = QtWidgets.QLineEdit(str(default_cfg))
        self.cfg_browse_btn = QtWidgets.QPushButton("Browse…")
        self.cfg_browse_btn.clicked.connect(self._on_browse_cfg)
        cfg_layout = QtWidgets.QHBoxLayout()
        cfg_layout.addWidget(self.cfg_edit, 1)
        cfg_layout.addWidget(self.cfg_browse_btn, 0)
        self.cfg_label = QtWidgets.QLabel("DeepStream config:")
        self.cfg_row_widget = QtWidgets.QWidget(self)
        self.cfg_row_widget.setLayout(cfg_layout)
        form.addRow(self.cfg_label, self.cfg_row_widget)

        self.flir_panel = QtWidgets.QGroupBox("FLIR Capture Panel", self)
        flir_form = QtWidgets.QFormLayout(self.flir_panel)
        flir_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        flir_form.setHorizontalSpacing(12)
        flir_form.setVerticalSpacing(8)

        self.pix_combo = QtWidgets.QComboBox()
        self.pix_combo.addItems(["Mono8", "BGR8", "GRAY8"])
        current_pix = cfg.get("pixel_format", "Mono8")
        if current_pix in [self.pix_combo.itemText(i) for i in range(self.pix_combo.count())]:
            self.pix_combo.setCurrentText(current_pix)
        flir_form.addRow("Pixel Format:", self.pix_combo)

        self.exposure_edit = QtWidgets.QLineEdit(str(cfg.get("exposure_us", 10000)))
        self.exposure_edit.setValidator(QtGui.QIntValidator(10, 10_000_000, self))
        flir_form.addRow("Exposure (us):", self.exposure_edit)

        self.socket_edit = QtWidgets.QLineEdit(str(cfg.get("socket_path", "/tmp/cam.sock")))
        flir_form.addRow("Shared socket:", self.socket_edit)

        self.socket2_edit = QtWidgets.QLineEdit(str(cfg.get("socket_path_2", "/tmp/cam1.sock")))
        flir_form.addRow("Second socket:", self.socket2_edit)
        form.addRow("", self.flir_panel)

        self.zed_panel = QtWidgets.QGroupBox("ZED Capture Panel", self)
        zed_form = QtWidgets.QFormLayout(self.zed_panel)
        zed_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        zed_form.setHorizontalSpacing(12)
        zed_form.setVerticalSpacing(8)

        self.zed_depth_chk = QtWidgets.QCheckBox("Enable ZED depth stream")
        self.zed_depth_chk.setChecked(bool(cfg.get("zed_depth_enabled", False)))
        zed_form.addRow("", self.zed_depth_chk)

        self.zed_depth_mode_combo = QtWidgets.QComboBox()
        self.zed_depth_mode_combo.addItems(["NEURAL", "NEURAL_LIGHT", "NEURAL_PLUS", "PERFORMANCE", "QUALITY"])
        current_depth_mode = str(cfg.get("zed_depth_mode", "NEURAL")).upper().strip()
        if current_depth_mode in [self.zed_depth_mode_combo.itemText(i) for i in range(self.zed_depth_mode_combo.count())]:
            self.zed_depth_mode_combo.setCurrentText(current_depth_mode)
        zed_form.addRow("ZED Depth Mode:", self.zed_depth_mode_combo)

        self.zed_depth_record_chk = QtWidgets.QCheckBox("Record depth video (higher load)")
        self.zed_depth_record_chk.setChecked(bool(cfg.get("zed_depth_record", False)))
        zed_form.addRow("", self.zed_depth_record_chk)

        self.zed_confidence_95_chk = QtWidgets.QCheckBox("Apply ZED confidence threshold 100")
        self.zed_confidence_95_chk.setChecked(int(cfg.get("zed_confidence_threshold", 100) or 100) >= 100)
        zed_form.addRow("", self.zed_confidence_95_chk)

        self.zed_adv_group = QtWidgets.QGroupBox("ZED Advanced Depth", self.zed_panel)
        zed_adv_form = QtWidgets.QFormLayout(self.zed_adv_group)
        zed_adv_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        zed_adv_form.setHorizontalSpacing(10)
        zed_adv_form.setVerticalSpacing(6)

        self.zed_texture_conf_edit = QtWidgets.QLineEdit(str(cfg.get("zed_texture_confidence_threshold", 100)))
        self.zed_texture_conf_edit.setValidator(QtGui.QIntValidator(0, 100, self))
        zed_adv_form.addRow("Texture confidence:", self.zed_texture_conf_edit)

        self.zed_depth_min_mm_edit = QtWidgets.QLineEdit(str(cfg.get("zed_depth_minimum_distance_mm", 300)))
        self.zed_depth_min_mm_edit.setValidator(QtGui.QIntValidator(100, 3000, self))
        zed_adv_form.addRow("Depth min (mm):", self.zed_depth_min_mm_edit)

        self.zed_depth_max_mm_edit = QtWidgets.QLineEdit(str(cfg.get("zed_depth_maximum_distance_mm", 20000)))
        self.zed_depth_max_mm_edit.setValidator(QtGui.QIntValidator(500, 40000, self))
        zed_adv_form.addRow("Depth max (mm):", self.zed_depth_max_mm_edit)

        self.zed_fill_mode_chk = QtWidgets.QCheckBox("Enable depth fill mode")
        self.zed_fill_mode_chk.setChecked(bool(cfg.get("zed_fill_mode", False)))
        zed_adv_form.addRow("", self.zed_fill_mode_chk)

        self.zed_depth_stab_edit = QtWidgets.QLineEdit(str(cfg.get("zed_depth_stabilization", 30)))
        self.zed_depth_stab_edit.setValidator(QtGui.QIntValidator(0, 100, self))
        zed_adv_form.addRow("Depth stabilization:", self.zed_depth_stab_edit)
        zed_form.addRow("", self.zed_adv_group)
        form.addRow("", self.zed_panel)

        self.bitrate_edit = QtWidgets.QLineEdit(str(cfg.get("bitrate", 4000)))
        self.bitrate_edit.setValidator(QtGui.QIntValidator(100, 50000, self))
        form.addRow("Bitrate (kbps):", self.bitrate_edit)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel("Configure capture and inference parameters before starting SqueakView.")
        header.setWordWrap(True)
        layout.addWidget(header)
        layout.addSpacing(6)
        tabs = QtWidgets.QTabWidget(self)
        sensor_tab = QtWidgets.QWidget(self)
        sensor_layout = QtWidgets.QVBoxLayout(sensor_tab)
        sensor_layout.setContentsMargins(10, 10, 10, 10)
        sensor_form = QtWidgets.QFormLayout()
        sensor_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        sensor_form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        sensor_form.setHorizontalSpacing(14)
        sensor_form.setVerticalSpacing(10)
        self.detected_sensors_label = QtWidgets.QLabel("")
        sensor_form.addRow("Detected sensors:", self.detected_sensors_label)
        self.detected_sources_label = QtWidgets.QLabel("")
        sensor_form.addRow("Sensor details:", self.detected_sources_label)
        self.detect_btn = QtWidgets.QPushButton("Detect Sensors")
        self.detect_btn.clicked.connect(self._on_detect_sensors)
        sensor_form.addRow("", self.detect_btn)
        self.continue_btn = QtWidgets.QPushButton("Continue to Run Settings")
        self.continue_btn.clicked.connect(self._on_continue_to_run_tab)
        sensor_form.addRow("", self.continue_btn)
        sensor_help = QtWidgets.QLabel(
            "Detects connected FLIR cameras (PySpin) and ZED devices (pyzed), then configures camera count automatically (0-2). "
            "Run-time ZED capture uses DeepStream zedsrc."
        )
        sensor_help.setWordWrap(True)
        sensor_help.setStyleSheet("color: #9aa7cc;")
        sensor_layout.addLayout(sensor_form)
        sensor_layout.addWidget(sensor_help)
        sensor_layout.addStretch(1)

        run_tab = QtWidgets.QWidget(self)
        run_layout = QtWidgets.QVBoxLayout(run_tab)
        run_layout.setContentsMargins(10, 10, 10, 10)
        run_layout.addLayout(form)
        run_layout.addStretch(1)

        tabs.addTab(sensor_tab, "Configure Sensor Source")
        tabs.addTab(run_tab, "Configure Run")
        tabs.setCurrentIndex(0)
        tabs.setTabEnabled(1, False)
        self.tabs = tabs
        layout.addWidget(tabs)
        layout.addSpacing(12)
        layout.addWidget(button_box)
        self.ok_button = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if self.ok_button is not None:
            self.ok_button.setEnabled(False)

        self._result: dict | None = None
        self.inference_enable.toggled.connect(self._on_inference_toggled)
        self.capture_backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self._on_inference_toggled(self.inference_enable.isChecked())
        self._set_detected_cameras(self._detected_cameras, from_detection=True)
        self._on_backend_changed()

    def _apply_zed_autoconfig_defaults(self) -> None:
        """Prefill sane defaults for ZED runs; user may edit afterward."""
        self.width_edit.setText("1920")
        self.height_edit.setText("1200")
        self.fps_edit.setText("30")
        self.arduino_fps_edit.setText("30")
        self.pix_combo.setCurrentText("GRAY8")
        self.trigger_chk.setChecked(False)
        self.socket_edit.setText("/tmp/cam.sock")
        self.socket2_edit.setText("/tmp/cam1.sock")
        self.bitrate_edit.setText("4000")
        self.zed_depth_chk.setChecked(True)
        self.zed_depth_mode_combo.setCurrentText("NEURAL")
        self.zed_texture_conf_edit.setText("100")
        self.zed_depth_min_mm_edit.setText("300")
        self.zed_depth_max_mm_edit.setText("20000")
        self.zed_fill_mode_chk.setChecked(False)
        self.zed_depth_stab_edit.setText("30")
        self.zed_confidence_95_chk.setChecked(True)
        # Prefer an existing batch=2 config for dual-source ZED left/right ingest.
        preferred_cfg = squeakview_config.DEEPSTREAM_ROOT / "configs" / "26n_pose_DSFT_fp16.txt"
        if preferred_cfg.exists():
            self.cfg_edit.setText(str(preferred_cfg))

    def _set_detected_cameras(self, count: int, *, from_detection: bool) -> None:
        count = max(0, min(2, int(count)))
        self._detected_cameras = count
        prefix = "Detected" if from_detection else "Configured"
        suffix = "camera" if count == 1 else "cameras"
        self.detected_sensors_label.setText(f"{prefix}: {count} {suffix}")
        if not from_detection:
            self.detected_sources_label.setText("FLIR: unknown | ZED devices: unknown")
        self.socket2_edit.setEnabled(count > 1)

    @QtCore.Slot(bool)
    def _on_inference_toggled(self, enabled: bool) -> None:
        self.cfg_label.setVisible(enabled)
        self.cfg_row_widget.setVisible(enabled)
        if not enabled:
            self.skeleton_chk.setChecked(False)
        self.skeleton_chk.setEnabled(enabled)

    def _set_form_row_visible(self, field: QtWidgets.QWidget, visible: bool) -> None:
        field.setVisible(visible)
        label = self._run_form.labelForField(field)
        if label is not None:
            label.setVisible(visible)

    @QtCore.Slot()
    def _on_backend_changed(self) -> None:
        backend = str(self.capture_backend_combo.currentData() or "flir").lower().strip()
        flir_visible = backend == "flir"
        zed_visible = backend == "zed"

        self.flir_panel.setVisible(flir_visible)
        self.zed_panel.setVisible(zed_visible)

    @QtCore.Slot()
    def _on_continue_to_run_tab(self) -> None:
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(1)
        if self.ok_button is not None:
            self.ok_button.setEnabled(True)

    def _on_detect_sensors(self) -> None:
        if self._detect_thread is not None:
            return

        self.detect_btn.setEnabled(False)
        progress = QtWidgets.QProgressDialog("Detecting sensors...", "", 0, 0, self)
        progress.setWindowTitle("Detecting Sensors")
        progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        self._detect_progress = progress

        thread = QtCore.QThread(self)
        worker = SensorDetectWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_detect_sensors_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_detect_thread_finished)
        self._detect_worker = worker
        self._detect_thread = thread
        thread.start()

    @QtCore.Slot(int, int, int, str)
    def _on_detect_sensors_finished(self, count: int, flir_count: int, zed_count: int, warning: str) -> None:
        capped = max(0, min(2, count))
        self._set_detected_cameras(capped, from_detection=True)
        zed_sources = 2 if zed_count > 0 else 0
        self.detected_sources_label.setText(
            f"FLIR: {flir_count} | ZED devices: {zed_count} | ZED usable sources: {zed_sources}"
        )
        if zed_count > 0 and flir_count == 0:
            idx = self.capture_backend_combo.findData("zed")
            if idx >= 0:
                self.capture_backend_combo.setCurrentIndex(idx)
        elif flir_count > 0 and zed_count == 0:
            idx = self.capture_backend_combo.findData("flir")
            if idx >= 0:
                self.capture_backend_combo.setCurrentIndex(idx)
        if zed_count > 0 and str(self.capture_backend_combo.currentData() or "").lower() == "zed":
            self._apply_zed_autoconfig_defaults()
            QtWidgets.QMessageBox.information(
                self,
                "ZED Defaults Applied",
                "ZED was detected, so capture defaults were prefilled:\n"
                "1920x1200 @ 30 FPS, trigger off, GRAY8, /tmp/cam.sock + /tmp/cam1.sock,\n"
                "and batch-size=2 config when available.\n\n"
                "You can change any value before starting.",
            )
        if warning:
            QtWidgets.QMessageBox.information(self, "Sensor Detection Notes", warning)
        if count > 2:
            QtWidgets.QMessageBox.information(
                self,
                "Sensor Detection",
                f"Detected {count} cameras. SqueakView currently uses up to 2 cameras.",
            )
        if count == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Sensors Detected",
                "No sensors detected, please ensure sensors are cprrectly plugged in.\n"
                "If you are attempting to use ZED sensors, please close SqueakView, run:\n\n"
                "sudo service zed_x_daemon restart\n\n"
                "Then reopen SqueakView.",
            )
        self._on_backend_changed()

    @QtCore.Slot()
    def _on_detect_thread_finished(self) -> None:
        if self._detect_progress is not None:
            self._detect_progress.close()
            self._detect_progress.deleteLater()
            self._detect_progress = None
        self.detect_btn.setEnabled(True)
        self._detect_worker = None
        self._detect_thread = None

    def _on_browse_cfg(self) -> None:
        start_dir = Path(self.cfg_edit.text()).parent if self.cfg_edit.text() else squeakview_config.DEEPSTREAM_ROOT
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select DeepStream nvinfer config",
            str(start_dir),
            "DeepStream config (*.txt *.cfg);;All files (*)",
        )
        if path:
            self.cfg_edit.setText(path)

    def _on_browse_task_cfg(self) -> None:
        start_dir = (
            Path(self.task_cfg_edit.text()).parent
            if self.task_cfg_edit.text()
            else squeakview_config.TASKS_DIR
        )
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select task config",
            str(start_dir),
            "Task config (*.yaml *.yml *.json);;All files (*)",
        )
        if path:
            self.task_cfg_edit.setText(path)

    def accept(self) -> None:
        try:
            width = int(self.width_edit.text()) or 1280
            height = int(self.height_edit.text()) or 720
            fps = int(self.fps_edit.text()) or 30
            bitrate = int(self.bitrate_edit.text()) or 4000
            arduino_fps = int(self.arduino_fps_edit.text()) or 30
            serial_baud = int(self.serial_baud_edit.text()) or 115200
            exposure_us = int(self.exposure_edit.text()) if self.exposure_edit.text() else 10000
            zed_texture_conf = int(self.zed_texture_conf_edit.text()) if self.zed_texture_conf_edit.text() else 100
            zed_depth_min_mm = int(self.zed_depth_min_mm_edit.text()) if self.zed_depth_min_mm_edit.text() else 300
            zed_depth_max_mm = int(self.zed_depth_max_mm_edit.text()) if self.zed_depth_max_mm_edit.text() else 20000
            zed_depth_stab = int(self.zed_depth_stab_edit.text()) if self.zed_depth_stab_edit.text() else 30
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid input", "Please enter valid numeric values for size, FPS, bitrate, and baud.")
            return

        self._result = {
            "width": width,
            "height": height,
            "fps": fps,
            "pixel_format": self.pix_combo.currentText() or "Mono8",
            "capture_backend": str(self.capture_backend_combo.currentData() or "flir"),
            "zed_depth_enabled": self.zed_depth_chk.isChecked(),
            "zed_depth_mode": self.zed_depth_mode_combo.currentText() or "NEURAL",
            "zed_depth_socket": "/tmp/cam_depth.sock",
            "zed_depth_record": self.zed_depth_record_chk.isChecked(),
            "zed_confidence_threshold": 100 if self.zed_confidence_95_chk.isChecked() else 0,
            "zed_texture_confidence_threshold": max(0, min(100, zed_texture_conf)),
            "zed_depth_minimum_distance_mm": max(100, min(3000, zed_depth_min_mm)),
            "zed_depth_maximum_distance_mm": max(500, min(40000, zed_depth_max_mm)),
            "zed_fill_mode": self.zed_fill_mode_chk.isChecked(),
            "zed_depth_stabilization": max(0, min(100, zed_depth_stab)),
            "trigger_on": self.trigger_chk.isChecked(),
            "exposure_us": exposure_us,
            "arduino_fps": arduino_fps,
            "serial_enabled": self.serial_enable.isChecked(),
            "serial_port": self.serial_port_edit.text().strip() or "/dev/ttyACM0",
            "serial_baud": serial_baud,
            "ds_cfg": Path(self.cfg_edit.text().strip()) if self.inference_enable.isChecked() else None,
            "inference_enabled": self.inference_enable.isChecked(),
            "draw_skeleton": self.skeleton_chk.isChecked(),
            "task_cfg": Path(self.task_cfg_edit.text().strip()),
            "socket_path": self.socket_edit.text().strip() or "/tmp/cam.sock",
            "socket_path_2": self.socket2_edit.text().strip() or "/tmp/cam1.sock",
            "num_cameras": int(self._detected_cameras),
            "bitrate": bitrate,
            "mouse_id": self.mouse_id_edit.text().strip(),
        }
        if self._result["num_cameras"] < 1:
            QtWidgets.QMessageBox.warning(
                self,
                "No Sensors Detected",
                "No cameras are currently detected.\n\nOpen the 'Configure Sensor Source' tab and click 'Detect Sensors'.",
            )
            return
        if self._result["inference_enabled"]:
            if not self._result["ds_cfg"]:
                QtWidgets.QMessageBox.warning(self, "Config missing", "DeepStream config is required when inference is enabled.")
                return
            if not self._result["ds_cfg"].exists():
                QtWidgets.QMessageBox.warning(self, "Config missing", f"DeepStream config not found:\n{self._result['ds_cfg']}")
                return
        if not str(self._result["task_cfg"]):
            QtWidgets.QMessageBox.warning(self, "Task config required", "Please select a task config before starting.")
            return
        if not self._result["task_cfg"].exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Task config missing",
                f"Task config not found:\n{self._result['task_cfg']}",
            )
            return
        super().accept()

    @property
    def result_config(self) -> dict | None:
        return self._result
