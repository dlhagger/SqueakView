"""Widget construction for the operator configuration dialog."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview.common.profiles import ExperimentProfile, SubjectProfile


@dataclass(frozen=True, slots=True)
class ConfigViewCallbacks:
    create_experiment: Callable[[], None]
    create_subject: Callable[[], None]
    add_subject: Callable[[], None]
    browse_task_config: Callable[[], None]
    browse_model_config: Callable[[], None]
    accept: Callable[[], None]
    reject: Callable[[], None]


@dataclass(frozen=True, slots=True)
class ConfigView:
    run_form: QtWidgets.QFormLayout
    mode_combo: QtWidgets.QComboBox
    existing_experiment_combo: QtWidgets.QComboBox
    new_experiment_btn: QtWidgets.QPushButton
    existing_subject_combo: QtWidgets.QComboBox
    new_subject_btn: QtWidgets.QPushButton
    add_subject_btn: QtWidgets.QPushButton
    width_edit: QtWidgets.QLineEdit
    height_edit: QtWidgets.QLineEdit
    fps_edit: QtWidgets.QLineEdit
    trigger_chk: QtWidgets.QCheckBox
    arduino_fps_edit: QtWidgets.QLineEdit
    serial_enable: QtWidgets.QCheckBox
    inference_enable: QtWidgets.QCheckBox
    task_cfg_edit: QtWidgets.QLineEdit
    task_browse_btn: QtWidgets.QPushButton
    serial_port_edit: QtWidgets.QLineEdit
    serial_baud_edit: QtWidgets.QLineEdit
    serial_row_widget: QtWidgets.QWidget
    cfg_edit: QtWidgets.QLineEdit
    cfg_browse_btn: QtWidgets.QPushButton
    cfg_label: QtWidgets.QLabel
    cfg_row_widget: QtWidgets.QWidget
    flir_panel: QtWidgets.QGroupBox
    pix_combo: QtWidgets.QComboBox
    exposure_edit: QtWidgets.QLineEdit
    bitrate_edit: QtWidgets.QLineEdit
    experiment_profile_name_edit: QtWidgets.QLineEdit | None
    session_hint_label: QtWidgets.QLabel
    session_summary_label: QtWidgets.QLabel
    session_group: QtWidgets.QGroupBox
    button_box: QtWidgets.QDialogButtonBox


def _header_text(
    *,
    show_session_setup: bool,
    show_experiment_profile_editor: bool,
) -> str:
    if show_experiment_profile_editor:
        return (
            "Edit the experiment profile and the saved run defaults used when "
            "that experiment starts."
        )
    if show_session_setup:
        return (
            "Choose the session mode and set model, camera, trigger, serial, "
            "and task settings."
        )
    return (
        "Set model, camera, trigger, serial, and task settings for this run or "
        "experiment profile."
    )


def build_config_view(
    dialog: QtWidgets.QDialog,
    *,
    config: Mapping[str, Any],
    experiments: Sequence[ExperimentProfile],
    subjects: Sequence[SubjectProfile],
    callbacks: ConfigViewCallbacks,
    tasks_dir: Path,
    show_session_setup: bool,
    experiment_profile_name: str,
    show_experiment_profile_editor: bool,
) -> ConfigView:
    """Build the complete form without loading profiles or applying policy."""

    form = QtWidgets.QFormLayout()
    form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
    form.setFormAlignment(
        QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
    )
    form.setHorizontalSpacing(14)
    form.setVerticalSpacing(10)

    mode_combo = QtWidgets.QComboBox(dialog)
    mode_combo.addItem("Experiment Session", "existing")
    mode_combo.addItem("Sandbox / Manual Mode", "sandbox")

    existing_experiment_combo = QtWidgets.QComboBox(dialog)
    existing_experiment_combo.addItem("Select experiment…", "")
    for profile in experiments:
        existing_experiment_combo.addItem(profile.name, profile.slug)
    new_experiment_btn = QtWidgets.QPushButton("Create…", dialog)
    new_experiment_btn.clicked.connect(callbacks.create_experiment)
    existing_experiment_row = QtWidgets.QWidget(dialog)
    existing_experiment_layout = QtWidgets.QHBoxLayout(existing_experiment_row)
    existing_experiment_layout.setContentsMargins(0, 0, 0, 0)
    existing_experiment_layout.setSpacing(8)
    existing_experiment_layout.addWidget(existing_experiment_combo, 1)
    existing_experiment_layout.addWidget(new_experiment_btn, 0)

    existing_subject_combo = QtWidgets.QComboBox(dialog)
    existing_subject_combo.addItem("No saved subject", "")
    for profile in subjects:
        existing_subject_combo.addItem(profile.name, profile.subject_id)
    new_subject_btn = QtWidgets.QPushButton("Create…", dialog)
    new_subject_btn.clicked.connect(callbacks.create_subject)
    add_subject_btn = QtWidgets.QPushButton("Add Existing…", dialog)
    add_subject_btn.clicked.connect(callbacks.add_subject)
    existing_subject_row = QtWidgets.QWidget(dialog)
    existing_subject_layout = QtWidgets.QHBoxLayout(existing_subject_row)
    existing_subject_layout.setContentsMargins(0, 0, 0, 0)
    existing_subject_layout.setSpacing(8)
    existing_subject_layout.addWidget(existing_subject_combo, 1)
    existing_subject_layout.addWidget(new_subject_btn, 0)
    existing_subject_layout.addWidget(add_subject_btn, 0)

    initial_mode = str(config.get("experiment_mode", "sandbox"))
    existing_slug = str(config.get("experiment_name", ""))
    if existing_slug and existing_experiment_combo.findData(existing_slug) >= 0:
        initial_mode = "existing"
    mode_combo.setCurrentIndex(max(0, mode_combo.findData(initial_mode)))

    size_validator = QtGui.QIntValidator(1, 4096, dialog)
    width_edit = QtWidgets.QLineEdit(str(config.get("width", 1440)))
    width_edit.setValidator(size_validator)
    form.addRow("Width:", width_edit)
    height_edit = QtWidgets.QLineEdit(str(config.get("height", 1080)))
    height_edit.setValidator(size_validator)
    form.addRow("Height:", height_edit)
    fps_edit = QtWidgets.QLineEdit(str(config.get("fps", 30)))
    fps_edit.setValidator(QtGui.QIntValidator(1, 240, dialog))
    form.addRow("FPS:", fps_edit)

    trigger_chk = QtWidgets.QCheckBox("Enable camera trigger")
    trigger_chk.setChecked(config.get("trigger_on", False))
    form.addRow("", trigger_chk)
    arduino_fps_edit = QtWidgets.QLineEdit(str(config.get("arduino_fps", 30)))
    arduino_fps_edit.setValidator(QtGui.QIntValidator(1, 240, dialog))
    form.addRow("Arduino FPS:", arduino_fps_edit)
    serial_enable = QtWidgets.QCheckBox("Enable Arduino serial logging")
    serial_enable.setChecked(config.get("serial_enabled", True))
    form.addRow("", serial_enable)
    inference_enable = QtWidgets.QCheckBox("Enable YOLO inference (DeepStream)")
    inference_enable.setChecked(config.get("inference_enabled", True))
    form.addRow("", inference_enable)

    task_default = config.get("task_cfg", "")
    if not task_default:
        candidate = tasks_dir / "default.yaml"
        task_default = str(candidate) if candidate.exists() else ""
    task_cfg_edit = QtWidgets.QLineEdit(str(task_default))
    task_browse_btn = QtWidgets.QPushButton("Browse…")
    task_browse_btn.clicked.connect(callbacks.browse_task_config)
    task_layout = QtWidgets.QHBoxLayout()
    task_layout.addWidget(task_cfg_edit, 1)
    task_layout.addWidget(task_browse_btn, 0)
    form.addRow("Task config:", task_layout)

    serial_row = QtWidgets.QHBoxLayout()
    serial_port_edit = QtWidgets.QLineEdit(
        str(config.get("serial_port", "/dev/ttyACM0"))
    )
    serial_row.addWidget(QtWidgets.QLabel("Port:"))
    serial_row.addWidget(serial_port_edit)
    serial_baud_edit = QtWidgets.QLineEdit(str(config.get("serial_baud", 115200)))
    serial_row.addWidget(QtWidgets.QLabel("Baud:"))
    serial_row.addWidget(serial_baud_edit)
    serial_row_widget = QtWidgets.QWidget(dialog)
    serial_row_widget.setLayout(serial_row)
    form.addRow("", serial_row_widget)

    cfg_edit = QtWidgets.QLineEdit(str(config.get("ds_cfg", "")))
    cfg_browse_btn = QtWidgets.QPushButton("Browse…")
    cfg_browse_btn.clicked.connect(callbacks.browse_model_config)
    cfg_layout = QtWidgets.QHBoxLayout()
    cfg_layout.addWidget(cfg_edit, 1)
    cfg_layout.addWidget(cfg_browse_btn, 0)
    cfg_label = QtWidgets.QLabel("DeepStream config:")
    cfg_row_widget = QtWidgets.QWidget(dialog)
    cfg_row_widget.setLayout(cfg_layout)
    form.addRow(cfg_label, cfg_row_widget)

    flir_panel = QtWidgets.QGroupBox("FLIR Capture Panel", dialog)
    flir_form = QtWidgets.QFormLayout(flir_panel)
    flir_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
    flir_form.setHorizontalSpacing(12)
    flir_form.setVerticalSpacing(8)
    pix_combo = QtWidgets.QComboBox()
    pix_combo.addItems(["Mono8", "BGR8", "GRAY8"])
    current_pix = config.get("pixel_format", "Mono8")
    if current_pix in [pix_combo.itemText(i) for i in range(pix_combo.count())]:
        pix_combo.setCurrentText(str(current_pix))
    flir_form.addRow("Pixel Format:", pix_combo)
    exposure_edit = QtWidgets.QLineEdit(str(config.get("exposure_us", 10000)))
    exposure_edit.setValidator(QtGui.QIntValidator(10, 10_000_000, dialog))
    flir_form.addRow("Exposure (us):", exposure_edit)
    form.addRow("", flir_panel)
    bitrate_edit = QtWidgets.QLineEdit(str(config.get("bitrate", 4000)))
    bitrate_edit.setValidator(QtGui.QIntValidator(100, 50000, dialog))
    form.addRow("Bitrate (kbps):", bitrate_edit)

    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        QtCore.Qt.Orientation.Horizontal,
        dialog,
    )
    button_box.accepted.connect(callbacks.accept)
    button_box.rejected.connect(callbacks.reject)

    layout = QtWidgets.QVBoxLayout(dialog)
    header = QtWidgets.QLabel(
        _header_text(
            show_session_setup=show_session_setup,
            show_experiment_profile_editor=show_experiment_profile_editor,
        )
    )
    header.setWordWrap(True)
    layout.addWidget(header)
    layout.addSpacing(6)

    experiment_profile_name_edit: QtWidgets.QLineEdit | None
    if show_experiment_profile_editor:
        experiment_group = QtWidgets.QGroupBox("Experiment Profile", dialog)
        experiment_form = QtWidgets.QFormLayout(experiment_group)
        experiment_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        experiment_form.setHorizontalSpacing(14)
        experiment_form.setVerticalSpacing(10)
        experiment_help = QtWidgets.QLabel(
            "Edit both the experiment name and the saved Configure Run defaults "
            "in this window."
        )
        experiment_help.setWordWrap(True)
        experiment_help.setStyleSheet("color: #9aa7cc;")
        experiment_profile_name_edit = QtWidgets.QLineEdit(
            experiment_profile_name, dialog
        )
        experiment_form.addRow("", experiment_help)
        experiment_form.addRow("Experiment name:", experiment_profile_name_edit)
        layout.addWidget(experiment_group)
    else:
        experiment_profile_name_edit = None

    session_group = QtWidgets.QGroupBox("Session Setup", dialog)
    session_form = QtWidgets.QFormLayout(session_group)
    session_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
    session_form.setHorizontalSpacing(14)
    session_form.setVerticalSpacing(10)
    session_hint_label = QtWidgets.QLabel(
        "Experiment sessions load saved Configure Run defaults. Subjects are "
        "managed separately and attached to experiments."
    )
    session_hint_label.setWordWrap(True)
    session_hint_label.setStyleSheet("color: #9aa7cc;")
    session_form.addRow("", session_hint_label)
    session_summary_label = QtWidgets.QLabel("")
    session_summary_label.setWordWrap(True)
    session_summary_label.setStyleSheet(
        "color: #e7ebff; background-color: #11162a; border: 1px solid "
        "#2c3550; border-radius: 6px; padding: 8px;"
    )
    session_form.addRow("Summary:", session_summary_label)
    session_form.addRow("Mode:", mode_combo)
    session_form.addRow("Experiment:", existing_experiment_row)
    session_form.addRow("Subject:", existing_subject_row)
    layout.addWidget(session_group)
    if not show_session_setup:
        session_group.hide()

    run_group = QtWidgets.QGroupBox("Run Configuration", dialog)
    run_layout = QtWidgets.QVBoxLayout(run_group)
    run_layout.setContentsMargins(12, 12, 12, 12)
    run_layout.addLayout(form)
    layout.addWidget(run_group)
    layout.addSpacing(12)
    layout.addWidget(button_box)

    return ConfigView(
        run_form=form,
        mode_combo=mode_combo,
        existing_experiment_combo=existing_experiment_combo,
        new_experiment_btn=new_experiment_btn,
        existing_subject_combo=existing_subject_combo,
        new_subject_btn=new_subject_btn,
        add_subject_btn=add_subject_btn,
        width_edit=width_edit,
        height_edit=height_edit,
        fps_edit=fps_edit,
        trigger_chk=trigger_chk,
        arduino_fps_edit=arduino_fps_edit,
        serial_enable=serial_enable,
        inference_enable=inference_enable,
        task_cfg_edit=task_cfg_edit,
        task_browse_btn=task_browse_btn,
        serial_port_edit=serial_port_edit,
        serial_baud_edit=serial_baud_edit,
        serial_row_widget=serial_row_widget,
        cfg_edit=cfg_edit,
        cfg_browse_btn=cfg_browse_btn,
        cfg_label=cfg_label,
        cfg_row_widget=cfg_row_widget,
        flir_panel=flir_panel,
        pix_combo=pix_combo,
        exposure_edit=exposure_edit,
        bitrate_edit=bitrate_edit,
        experiment_profile_name_edit=experiment_profile_name_edit,
        session_hint_label=session_hint_label,
        session_summary_label=session_summary_label,
        session_group=session_group,
        button_box=button_box,
    )
