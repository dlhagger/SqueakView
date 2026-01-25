from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview import config as squeakview_config


class ConfigDialog(QtWidgets.QDialog):
    """Modal dialog to configure SqueakView capture + inference parameters."""

    def __init__(self, parent=None, *, title: str = "Configure SqueakView", config: Optional[dict] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(440)

        self.setStyleSheet("""
            QDialog { background-color: #f1f2f6; }
            QLabel { color: #2c2f33; font-size: 13px; }
            QLineEdit, QComboBox { background-color: #ffffff; border: 1px solid #c5c9d6; border-radius: 4px; padding: 4px; }
            QLineEdit:focus, QComboBox:focus { border-color: #7480ff; }
            QPushButton { background-color: #4a70d6; color: #ffffff; padding: 6px 14px; border-radius: 4px; font-weight: 600; }
            QPushButton:hover { background-color: #3e64c4; }
            QPushButton:disabled { background-color: #9aa3bd; color: #eceff5; }
        """)
        cfg = config or {}

        form = QtWidgets.QFormLayout()
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

        self.pix_combo = QtWidgets.QComboBox()
        self.pix_combo.addItems(["Mono8", "BGR8", "GRAY8"])
        current_pix = cfg.get("pixel_format", "Mono8")
        if current_pix in [self.pix_combo.itemText(i) for i in range(self.pix_combo.count())]:
            self.pix_combo.setCurrentText(current_pix)
        form.addRow("Pixel Format:", self.pix_combo)

        self.trigger_chk = QtWidgets.QCheckBox("Enable camera trigger")
        self.trigger_chk.setChecked(cfg.get("trigger_on", True))
        form.addRow("", self.trigger_chk)

        self.exposure_edit = QtWidgets.QLineEdit(str(cfg.get("exposure_us", 10000)))
        self.exposure_edit.setValidator(QtGui.QIntValidator(10, 10_000_000, self))
        form.addRow("Exposure (us):", self.exposure_edit)

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
        form.addRow("", serial_row)

        default_cfg = cfg.get("ds_cfg", "")
        self.cfg_edit = QtWidgets.QLineEdit(str(default_cfg))
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_cfg)
        cfg_layout = QtWidgets.QHBoxLayout()
        cfg_layout.addWidget(self.cfg_edit, 1)
        cfg_layout.addWidget(browse_btn, 0)
        form.addRow("DeepStream config:", cfg_layout)

        self.socket_edit = QtWidgets.QLineEdit(str(cfg.get("socket_path", "/tmp/cam.sock")))
        form.addRow("Shared socket:", self.socket_edit)

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
        layout.addLayout(form)
        layout.addSpacing(12)
        layout.addWidget(button_box)

        self._result: dict | None = None

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
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid input", "Please enter valid numeric values for size, FPS, bitrate, and baud.")
            return

        self._result = {
            "width": width,
            "height": height,
            "fps": fps,
            "pixel_format": self.pix_combo.currentText() or "Mono8",
            "trigger_on": self.trigger_chk.isChecked(),
            "exposure_us": exposure_us,
            "arduino_fps": arduino_fps,
            "serial_enabled": self.serial_enable.isChecked(),
            "serial_port": self.serial_port_edit.text().strip() or "/dev/ttyACM0",
            "serial_baud": serial_baud,
            "ds_cfg": Path(self.cfg_edit.text().strip()),
            "inference_enabled": self.inference_enable.isChecked(),
            "draw_skeleton": self.skeleton_chk.isChecked(),
            "task_cfg": Path(self.task_cfg_edit.text().strip()),
            "socket_path": self.socket_edit.text().strip() or "/tmp/cam.sock",
            "bitrate": bitrate,
            "mouse_id": self.mouse_id_edit.text().strip(),
        }
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
