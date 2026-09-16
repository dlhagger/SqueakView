"""Widget assembly for the SqueakView operator main window.

This module deliberately contains presentation construction only.  Runtime,
session, and recording behavior remain owned by :class:`MainWindow`; callers
provide the callbacks that connect the view to those behaviors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets

from squeakview.apps.operator.gui.bottle_measurements import BottleMeasurementPanel
from squeakview.apps.operator.gui.dashboard import BehaviorDashboard
from squeakview.apps.operator.gui.dock_workspace import DockWorkspace, build_layout_menu
from squeakview.apps.operator.gui.finalization_overlay import FinalizationOverlay
from squeakview.apps.operator.gui.preview import AspectRatioPreviewHost, PreviewWidget


@dataclass(frozen=True, slots=True)
class MainViewCallbacks:
    """Behavior entry points needed by the widget tree."""

    configure: Callable[[], None]
    start_run: Callable[[], None]
    stop_run: Callable[[], None]
    select_experiment: Callable[[int], None]
    create_experiment: Callable[[], None]
    select_subject: Callable[[int], None]
    create_subject: Callable[[], None]
    save_bottles: Callable[[], None]
    copy_events: Callable[[], None]
    open_run_folder: Callable[[], None]


@dataclass(frozen=True, slots=True)
class MainView:
    """Stable references consumed by the MainWindow presenter/controller code."""

    central: QtWidgets.QWidget
    run_state_label: QtWidgets.QLabel
    run_identity_label: QtWidgets.QLabel
    run_elapsed_label: QtWidgets.QLabel
    capture_health_label: QtWidgets.QLabel
    layout_btn: QtWidgets.QPushButton
    configure_btn: QtWidgets.QPushButton
    run_btn: QtWidgets.QPushButton
    stop_btn: QtWidgets.QPushButton
    preview: PreviewWidget
    dashboard: BehaviorDashboard
    profile_group: QtWidgets.QGroupBox
    experiment_combo: QtWidgets.QComboBox
    new_experiment_btn: QtWidgets.QPushButton
    subject_combo: QtWidgets.QComboBox
    new_subject_btn: QtWidgets.QPushButton
    summary_label: QtWidgets.QLabel
    bottle_panel: BottleMeasurementPanel
    task_state_group: QtWidgets.QWidget
    stop_overlay: FinalizationOverlay
    workspace: DockWorkspace
    event_dock: QtWidgets.QDockWidget
    event_log: QtWidgets.QPlainTextEdit


def build_main_view(
    window: QtWidgets.QMainWindow,
    callbacks: MainViewCallbacks,
) -> MainView:
    """Build and connect the operator view without starting application logic."""

    central = QtWidgets.QWidget(window)
    central.setMaximumSize(0, 0)
    window.setCentralWidget(central)

    header = QtWidgets.QWidget(window)
    header.setObjectName("workspaceHeader")
    header_layout = QtWidgets.QVBoxLayout(header)
    header_layout.setContentsMargins(14, 14, 14, 8)
    header_layout.setSpacing(0)
    window.setMenuWidget(header)

    run_control = QtWidgets.QFrame(window)
    run_control.setObjectName("runControlBar")
    run_control_layout = QtWidgets.QHBoxLayout(run_control)
    run_control_layout.setContentsMargins(12, 9, 12, 9)
    run_control_layout.setSpacing(10)
    run_state_label = QtWidgets.QLabel("READY", window)
    run_state_label.setObjectName("runStateBadge")
    run_identity_label = QtWidgets.QLabel("No session selected", window)
    run_identity_label.setObjectName("runIdentity")
    run_elapsed_label = QtWidgets.QLabel("00:00:00", window)
    run_elapsed_label.setObjectName("runElapsed")
    capture_health_label = QtWidgets.QLabel("Camera --  ·  Queue --  ·  Disk --", window)
    capture_health_label.setObjectName("captureHealth")
    capture_health_label.setTextInteractionFlags(
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )
    run_control_layout.addWidget(run_state_label, 0)
    run_control_layout.addWidget(run_identity_label, 0)
    run_control_layout.addWidget(run_elapsed_label, 0)
    run_control_layout.addWidget(capture_health_label, 1)

    layout_btn = QtWidgets.QPushButton("Layout", window)
    layout_btn.setObjectName("secondaryButton")
    layout_btn.setToolTip("Unlock, rearrange, or reset workspace cards")
    configure_btn = QtWidgets.QPushButton("Configure…", window)
    configure_btn.setObjectName("secondaryButton")
    configure_btn.clicked.connect(callbacks.configure)
    run_btn = QtWidgets.QPushButton("Start Recording", window)
    run_btn.setObjectName("primaryButton")
    run_btn.setEnabled(False)
    run_btn.clicked.connect(callbacks.start_run)
    stop_btn = QtWidgets.QPushButton("Stop Recording", window)
    stop_btn.setObjectName("dangerButton")
    stop_btn.setEnabled(False)
    stop_btn.clicked.connect(callbacks.stop_run)
    run_control_layout.addWidget(layout_btn, 0)
    run_control_layout.addWidget(configure_btn, 0)
    run_control_layout.addWidget(run_btn, 0)
    run_control_layout.addWidget(stop_btn, 0)
    header_layout.addWidget(run_control, 0)

    workspace = DockWorkspace(window)

    preview_group = QtWidgets.QWidget()
    preview_group.setObjectName("workspaceCardContent")
    preview_layout = QtWidgets.QVBoxLayout(preview_group)
    preview_layout.setContentsMargins(10, 10, 10, 10)
    preview = PreviewWidget()
    preview_host = AspectRatioPreviewHost(preview, parent=preview_group)
    preview_layout.addWidget(preview_host, 1)

    dashboard = BehaviorDashboard(
        window_sec=300.0,
        pellet_mode="auto",
        disk_root=(
            window.project.paths.runs
            if getattr(window, "project", None) is not None
            else None
        ),
    )
    meters_only = dashboard.detach_meters()
    task_state_panel = dashboard.detach_task_panel()
    meters_group = QtWidgets.QWidget()
    meters_group.setObjectName("workspaceCardContent")
    meters_layout = QtWidgets.QVBoxLayout(meters_group)
    meters_layout.setContentsMargins(12, 12, 12, 12)
    meters_layout.addWidget(meters_only)

    profile_group = QtWidgets.QGroupBox("Profiles", window)
    profile_form = QtWidgets.QFormLayout(profile_group)
    profile_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
    profile_form.setHorizontalSpacing(10)
    profile_form.setVerticalSpacing(8)

    exp_row = QtWidgets.QHBoxLayout()
    experiment_combo = QtWidgets.QComboBox(window)
    experiment_combo.currentIndexChanged.connect(callbacks.select_experiment)
    exp_row.addWidget(experiment_combo, 1)
    new_experiment_btn = QtWidgets.QPushButton("New…", window)
    new_experiment_btn.clicked.connect(callbacks.create_experiment)
    exp_row.addWidget(new_experiment_btn, 0)
    profile_form.addRow("Experiment:", exp_row)

    subj_row = QtWidgets.QHBoxLayout()
    subject_combo = QtWidgets.QComboBox(window)
    subject_combo.currentIndexChanged.connect(callbacks.select_subject)
    subj_row.addWidget(subject_combo, 1)
    new_subject_btn = QtWidgets.QPushButton("New…", window)
    new_subject_btn.clicked.connect(callbacks.create_subject)
    subj_row.addWidget(new_subject_btn, 0)
    profile_form.addRow("Subject:", subj_row)
    meters_layout.addSpacing(6)
    meters_layout.addWidget(profile_group)
    profile_group.hide()

    summary_label = QtWidgets.QLabel("No configuration loaded.")
    summary_label.setObjectName("summaryBanner")
    summary_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
    summary_label.setWordWrap(True)
    meters_layout.addSpacing(6)
    meters_layout.addWidget(summary_label)

    bottle_panel = BottleMeasurementPanel(window)
    bottle_panel.setTitle("")
    bottle_panel.setProperty("embeddedCard", True)
    bottle_panel.save_requested.connect(callbacks.save_bottles)

    task_state_group = QtWidgets.QWidget()
    task_state_group.setObjectName("workspaceCardContent")
    task_state_layout = QtWidgets.QVBoxLayout(task_state_group)
    task_state_layout.setContentsMargins(12, 12, 12, 12)
    task_state_layout.addWidget(task_state_panel, 1)
    task_state_group.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding,
        QtWidgets.QSizePolicy.Policy.Expanding,
    )

    dashboard_group = QtWidgets.QWidget()
    dashboard_group.setObjectName("workspaceCardContent")
    dash_layout = QtWidgets.QVBoxLayout(dashboard_group)
    dash_layout.setContentsMargins(16, 16, 16, 16)
    dashboard.setMinimumHeight(210)
    dash_layout.addWidget(dashboard)

    event_panel = QtWidgets.QWidget()
    event_panel.setObjectName("workspaceCardContent")
    event_layout = QtWidgets.QHBoxLayout(event_panel)
    event_layout.setContentsMargins(12, 12, 12, 12)
    event_layout.setSpacing(8)
    event_actions = QtWidgets.QVBoxLayout()
    copy_events_btn = QtWidgets.QPushButton("Copy", event_panel)
    copy_events_btn.setFixedSize(160, 36)
    copy_events_btn.clicked.connect(callbacks.copy_events)
    open_run_btn = QtWidgets.QPushButton("Open Run Folder", event_panel)
    open_run_btn.setFixedSize(160, 36)
    open_run_btn.clicked.connect(callbacks.open_run_folder)
    event_actions.addWidget(
        copy_events_btn,
        0,
        QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight,
    )
    event_actions.addWidget(
        open_run_btn,
        0,
        QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight,
    )
    event_actions.addStretch(1)
    event_log = QtWidgets.QPlainTextEdit(event_panel)
    event_log.setObjectName("eventLog")
    event_log.setReadOnly(True)
    event_log.setMaximumBlockCount(500)
    event_log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
    event_layout.addWidget(event_log, 5)
    event_layout.addLayout(event_actions, 1)
    workspace.add_card("preview", "Live Preview", preview_group)
    workspace.add_card("system", "System Load", meters_group)
    workspace.add_card("task", "Live Task State", task_state_group)
    workspace.add_card("bottles", "Bottles", bottle_panel)
    workspace.add_card("behavior", "Behavior Dashboard", dashboard_group)
    event_dock = workspace.add_card("events", "Operator Events", event_panel)
    workspace.establish_default_layout()
    build_layout_menu(layout_btn, workspace)

    stop_overlay = FinalizationOverlay(window)

    return MainView(
        central=central,
        run_state_label=run_state_label,
        run_identity_label=run_identity_label,
        run_elapsed_label=run_elapsed_label,
        capture_health_label=capture_health_label,
        layout_btn=layout_btn,
        configure_btn=configure_btn,
        run_btn=run_btn,
        stop_btn=stop_btn,
        preview=preview,
        dashboard=dashboard,
        profile_group=profile_group,
        experiment_combo=experiment_combo,
        new_experiment_btn=new_experiment_btn,
        subject_combo=subject_combo,
        new_subject_btn=new_subject_btn,
        summary_label=summary_label,
        bottle_panel=bottle_panel,
        task_state_group=task_state_group,
        stop_overlay=stop_overlay,
        workspace=workspace,
        event_dock=event_dock,
        event_log=event_log,
    )
