from __future__ import annotations

import os
import threading
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview.apps.operator.backend import manifest, process
from squeakview.apps.operator.backend.contracts import BackendProtocol
from squeakview.apps.operator.backend.events import BackendEvent
from squeakview.apps.operator.backend.manager import OperatorBackend
from squeakview.apps.operator.gui.backend_proxy import (
    ALLOW_INPROCESS_BACKEND_ENV,
    SupervisorBackendProxy,
    supervisor_socket_from_environment,
)
from squeakview.apps.operator.gui.config_dialog import (
    ConfigDialog,
    CreateExperimentDialog,
    CreateSubjectDialog,
    SessionLauncherDialog,
    apply_dark_combo_popups,
    center_window,
)
from squeakview.apps.operator.gui.bottle_measurements import (
    BOTTLE_FLUID_PRESETS,
    BottleMeasurementPanel,
    parse_weight,
    present_saved_bottles,
)
from squeakview.apps.operator.gui.config_presentation import present_config
from squeakview.apps.operator.gui.dashboard import BehaviorDashboard
from squeakview.apps.operator.gui.finalization_overlay import FinalizationOverlay
from squeakview.apps.operator.gui.ipc_preview import IpcPreviewController
from squeakview.apps.operator.gui.main_theme import apply_main_window_theme
from squeakview.apps.operator.gui.main_view import MainViewCallbacks, build_main_view
from squeakview.apps.operator.gui.preview import PreviewWidget
from squeakview.apps.operator.gui.run_presenter import (
    RUN_FAILURE_DIALOG_STYLESHEET,
    RunLifecycleController,
    _effective_batch_camera_count,
    _elapsed_text,
    _last_csv_row,
    _read_ds_batch_size,
    _tail_text_line,
)
from squeakview.common.dashboard import DashboardEvent
from squeakview.apps.operator.gui.session_controller import (
    SessionConfigController,
    build_launch_config,
    default_config_data,
    resolve_config_paths,
)
from squeakview.common.profiles import ExperimentProfile, ProfileStore, SubjectProfile
from squeakview import config as squeakview_config


GUI_HEARTBEAT_INTERVAL_MS = 1_000


def _create_supervisor_heartbeat_timer(
    parent: QtCore.QObject,
    callback,
) -> QtCore.QTimer:
    """Create the lease source on the owning Qt event-loop thread."""

    timer = QtCore.QTimer(parent)
    timer.setInterval(GUI_HEARTBEAT_INTERVAL_MS)
    timer.timeout.connect(callback)
    timer.start()
    return timer


def _production_backend_factory(
    emit_log,
    ingest_dashboard=None,
    on_run_started=None,
    on_run_failed=None,
):
    """Build the supervised GUI facade, with only an explicit dev escape hatch."""

    socket_value = os.environ.get("SQUEAKVIEW_SUPERVISOR_SOCKET", "").strip()
    if socket_value:
        return SupervisorBackendProxy(
            supervisor_socket_from_environment(),
            emit_log,
            ingest_dashboard,
            on_run_started,
            on_run_failed,
        )
    allow_inprocess = os.environ.get(ALLOW_INPROCESS_BACKEND_ENV, "").lower()
    if allow_inprocess in {"1", "true", "yes", "on"}:
        emit_log(
            "[GUI] WARNING: explicit development mode uses an in-process backend"
        )
        return OperatorBackend(
            emit_log,
            ingest_dashboard,
            on_run_started=on_run_started,
            on_run_failed=on_run_failed,
            acquisition_owner=manifest.IN_PROCESS_DEV_OWNER,
        )
    # Raise the actionable error from the shared environment parser.  A GUI
    # started outside the durable launcher must never acquire cameras itself.
    supervisor_socket_from_environment()
    raise AssertionError("unreachable")


class MainWindow(QtWidgets.QMainWindow):
    log_msg = QtCore.Signal(str)
    start_finished = QtCore.Signal(bool)
    start_error = QtCore.Signal(str)
    stop_done = QtCore.Signal()
    stop_failed = QtCore.Signal(str)
    run_started = QtCore.Signal()
    run_failed = QtCore.Signal(str)
    backend_event = QtCore.Signal(object)
    dashboard_event = QtCore.Signal(object)
    clear_jam_finished = QtCore.Signal(str)
    clear_jam_failed = QtCore.Signal(str)

    def __init__(self, *, backend_factory=None) -> None:
        super().__init__()
        self.log_msg.connect(self._append_log)
        self.setWindowTitle("SqueakView")
        self.resize(1280, 820)
        self.setMinimumSize(1024, 700)

        self._config_data: dict | None = None
        self._preview_window_id: int | None = None
        self._profile_store = ProfileStore()
        self._experiments: list[ExperimentProfile] = []
        self._subjects: list[SubjectProfile] = []
        self._profile_selection_updating = False
        self._centered_once = False
        self.start_finished.connect(self._on_start_finished)
        self.start_error.connect(self._on_start_error)
        self.stop_done.connect(self._on_stop_complete)
        self.stop_failed.connect(self._on_stop_failed)
        self.run_started.connect(self._on_backend_run_started)
        self.run_failed.connect(self._on_backend_run_failed)
        self.backend_event.connect(self._on_backend_event)
        self.dashboard_event.connect(self._on_dashboard_event)
        self.clear_jam_finished.connect(self._on_clear_jam_finished)
        self.clear_jam_failed.connect(self._on_clear_jam_failed)
        self._clear_jam_thread = None

        self.backend: BackendProtocol = (backend_factory or _production_backend_factory)(
            self._emit_log,
            self._forward_dashboard,
            on_run_started=self.run_started.emit,
            on_run_failed=self.run_failed.emit,
        )
        self.backend.subscribe(self.backend_event.emit)
        # Start the only GUI-lease source as soon as the proxy is connected.
        # QDialog.exec() below runs a nested Qt event loop, so the launch dialog
        # continues to renew the lease while the operator is configuring a run.
        self._heartbeat_timer = _create_supervisor_heartbeat_timer(
            self, self._send_supervisor_heartbeat
        )

        self._build_ui()
        self._session_controller = SessionConfigController(
            self,
            self.experiment_combo,
            self.subject_combo,
            store=self._profile_store,
            commit=self._apply_config,
            emit=self._emit_log,
            launcher_dialog=SessionLauncherDialog,
            config_dialog=ConfigDialog,
            experiment_dialog=CreateExperimentDialog,
            subject_dialog=CreateSubjectDialog,
        )
        self._preview_controller = IpcPreviewController(self._emit_log, self)
        self._preview_controller.ready.connect(self._on_preview_ready)
        self._preview_controller.failed.connect(self._on_preview_failed)
        self._preview_controller.ended.connect(self._on_preview_ended)
        self._run_controller = RunLifecycleController(self)
        self._runtime_timer = QtCore.QTimer(self)
        self._runtime_timer.setInterval(500)
        self._runtime_timer.timeout.connect(self._refresh_runtime_status)
        self._runtime_timer.start()
        self._apply_brand_theme()
        apply_dark_combo_popups(self)
        QtCore.QTimer.singleShot(0, self._capture_preview_window_id)
        self._config_data = self._default_config_data()
        if not self._show_launch_dialog():
            QtCore.QTimer.singleShot(0, self.close)
        else:
            self._emit_log("[GUI] Ready to record.")
        self.preview.set_status("Idle")

    # Compatibility aliases for callers/tests that inspected lifecycle state on
    # MainWindow before it was moved into RunLifecycleController.
    @property
    def _start_in_progress(self) -> bool:
        return self._run_controller.start_in_progress

    @_start_in_progress.setter
    def _start_in_progress(self, value: bool) -> None:
        self._run_controller.start_in_progress = value

    @property
    def _start_thread(self):
        return self._run_controller.start_thread

    @_start_thread.setter
    def _start_thread(self, value) -> None:
        self._run_controller.start_thread = value

    @property
    def _pending_start_config(self):
        return self._run_controller.pending_start_config

    @_pending_start_config.setter
    def _pending_start_config(self, value) -> None:
        self._run_controller.pending_start_config = value

    @property
    def _close_after_start(self) -> bool:
        return self._run_controller.close_after_start

    @_close_after_start.setter
    def _close_after_start(self, value: bool) -> None:
        self._run_controller.close_after_start = value

    @property
    def _start_failure_reported(self) -> bool:
        return self._run_controller.start_failure_reported

    @_start_failure_reported.setter
    def _start_failure_reported(self, value: bool) -> None:
        self._run_controller.start_failure_reported = value

    @property
    def _stop_in_progress(self) -> bool:
        return self._run_controller.stop_in_progress

    @_stop_in_progress.setter
    def _stop_in_progress(self, value: bool) -> None:
        self._run_controller.stop_in_progress = value

    @property
    def _stop_thread(self):
        return self._run_controller.stop_thread

    @_stop_thread.setter
    def _stop_thread(self, value) -> None:
        self._run_controller.stop_thread = value

    @property
    def _close_after_stop(self) -> bool:
        return self._run_controller.close_after_stop

    @_close_after_stop.setter
    def _close_after_stop(self, value: bool) -> None:
        self._run_controller.close_after_stop = value

    @property
    def _recording_active(self) -> bool:
        return self._run_controller.recording_active

    @_recording_active.setter
    def _recording_active(self, value: bool) -> None:
        self._run_controller.recording_active = value

    @property
    def _recording_started_monotonic(self) -> float | None:
        return self._run_controller.recording_started_monotonic

    @_recording_started_monotonic.setter
    def _recording_started_monotonic(self, value: float | None) -> None:
        self._run_controller.recording_started_monotonic = value

    @property
    def _stop_started_monotonic(self) -> float | None:
        return self._run_controller.stop_started_monotonic

    @_stop_started_monotonic.setter
    def _stop_started_monotonic(self, value: float | None) -> None:
        self._run_controller.stop_started_monotonic = value

    @property
    def _last_health_poll_monotonic(self) -> float:
        return self._run_controller.last_health_poll_monotonic

    @_last_health_poll_monotonic.setter
    def _last_health_poll_monotonic(self, value: float) -> None:
        self._run_controller.last_health_poll_monotonic = value

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._centered_once and not (self.isMaximized() or self.isFullScreen()):
            self._centered_once = True
            center_window(self)

    # ---- UI -------------------------------------------------------------
    def _build_ui(self) -> None:
        view = build_main_view(
            self,
            MainViewCallbacks(
                configure=self._on_configure,
                start_run=self._on_run,
                stop_run=self._on_stop,
                select_experiment=self._on_experiment_selected,
                create_experiment=self._on_new_experiment,
                select_subject=self._on_subject_selected,
                create_subject=self._on_new_subject,
                save_bottles=self._on_save_bottles,
                copy_events=self._copy_event_log,
                open_run_folder=self._open_run_folder,
            ),
        )

        self.run_state_label = view.run_state_label
        self.run_identity_label = view.run_identity_label
        self.run_elapsed_label = view.run_elapsed_label
        self.capture_health_label = view.capture_health_label
        self.layout_btn = view.layout_btn
        self.configure_btn = view.configure_btn
        self.run_btn = view.run_btn
        self.stop_btn = view.stop_btn
        self.preview = view.preview
        self.dashboard = view.dashboard
        clear_jam_handler = getattr(self, "_on_clear_jam_requested", None)
        if callable(clear_jam_handler):
            self.dashboard.clear_jam_requested.connect(clear_jam_handler)
        self._profile_group = view.profile_group
        self.experiment_combo = view.experiment_combo
        self.new_experiment_btn = view.new_experiment_btn
        self.subject_combo = view.subject_combo
        self.new_subject_btn = view.new_subject_btn
        self.summary_label = view.summary_label
        self.bottle_panel = view.bottle_panel
        self.task_state_group = view.task_state_group
        self.stop_overlay = view.stop_overlay
        self.workspace = view.workspace
        self.event_dock = view.event_dock
        self.event_log = view.event_log
        self.workspace.restore_layout()

        # Compatibility aliases retained for existing MainWindow integrations.
        self.bottle_group = self.bottle_panel
        self.left_fluid_combo = self.bottle_panel.left_fluid_combo
        self.left_initial_weight_edit = self.bottle_panel.left_initial_weight_edit
        self.left_final_weight_edit = self.bottle_panel.left_final_weight_edit
        self.right_fluid_combo = self.bottle_panel.right_fluid_combo
        self.right_initial_weight_edit = self.bottle_panel.right_initial_weight_edit
        self.right_final_weight_edit = self.bottle_panel.right_final_weight_edit
        self.bottle_status_label = self.bottle_panel.status_label
        self.save_bottles_btn = self.bottle_panel.save_button
        self.stop_overlay_title = self.stop_overlay.title_label
        self.stop_overlay_msg = self.stop_overlay.message_label
        self.stop_overlay_bar = self.stop_overlay.progress_bar

    def _apply_brand_theme(self) -> None:
        apply_main_window_theme(self)

    def _default_config_data(self) -> dict:
        return default_config_data()

    def _make_fluid_combo(self) -> QtWidgets.QComboBox:
        return self.bottle_panel._make_fluid_combo()

    def _make_weight_edit(self, phase: str) -> QtWidgets.QLineEdit:
        return self.bottle_panel._make_weight_edit(phase)

    @staticmethod
    def _parse_weight(text: str, label: str, *, strict: bool) -> float | None:
        return parse_weight(text, label, strict=strict)

    def _collect_bottle_payload(self, *, include_final: bool, strict: bool) -> dict[str, object]:
        return self.bottle_panel.collect_payload(
            include_final=include_final,
            strict=strict,
        )

    def _clear_bottle_final_fields(self) -> None:
        self.bottle_panel.clear_final_fields()

    def _set_bottle_status(self, text: str) -> None:
        self.bottle_panel.set_status(text)

    def _set_bottle_completion_pending(self, pending: bool) -> None:
        self.bottle_panel.set_completion_pending(pending)

    @QtCore.Slot()
    def _on_save_bottles(self) -> None:
        try:
            payload = self._collect_bottle_payload(include_final=True, strict=True)
        except ValueError as exc:
            message = str(exc)
            self._set_bottle_status(message)
            QtWidgets.QMessageBox.warning(self, "Bottle Weight", message)
            return

        run_dir = self.backend.current_snapshot.run_dir
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

        presentation = present_saved_bottles(summary)
        self._set_bottle_completion_pending(presentation.completion_pending)
        self._set_bottle_status(presentation.status_text)
        if presentation.warning_message is not None:
            QtWidgets.QMessageBox.warning(
                self,
                "Bottle Weight",
                presentation.warning_message,
            )

    def _reload_profiles(self) -> None:
        self._session_controller.reload()
        self._sync_profile_compat_state()

    def _sync_profile_compat_state(self) -> None:
        """Keep legacy inspection attributes available during decomposition."""

        self._experiments = self._session_controller.experiments
        self._subjects = self._session_controller.subjects
        self._profile_selection_updating = self._session_controller.selection_updating

    def _refresh_profile_selectors(self) -> None:
        self._session_controller.refresh_selectors()
        self._sync_profile_compat_state()

    def _apply_profile_defaults(self) -> None:
        self._session_controller.apply_defaults(self._config_data)
        self._sync_profile_compat_state()

    def current_experiment_slug(self) -> str:
        return self._session_controller.current_experiment_slug()

    def current_subject_id(self) -> str:
        return self._session_controller.current_subject_id()

    def _find_experiment(self, slug: str) -> ExperimentProfile | None:
        return self._session_controller.find_experiment(slug)

    def _find_subject(self, subject_id: str) -> SubjectProfile | None:
        return self._session_controller.find_subject(subject_id)

    def _apply_profile_selection(self) -> None:
        self._session_controller.apply_selection(self._config_data)

    @QtCore.Slot()
    def _on_experiment_selected(self) -> None:
        self._session_controller.experiment_selected(self._config_data)
        self._sync_profile_compat_state()

    @QtCore.Slot()
    def _on_subject_selected(self) -> None:
        self._session_controller.subject_selected(self._config_data)
        self._sync_profile_compat_state()

    @QtCore.Slot()
    def _on_new_experiment(self) -> None:
        self._session_controller.create_experiment(self._config_data)
        self._sync_profile_compat_state()

    @QtCore.Slot()
    def _on_new_subject(self) -> None:
        self._session_controller.create_subject()
        self._sync_profile_compat_state()

    # ---- Configuration --------------------------------------------------
    def _show_launch_dialog(self) -> bool:
        result = self._session_controller.show_launcher(self._config_data)
        if result is None:
            return False
        self._apply_config(result)
        return True

    def _show_config_dialog(self, *, initial: bool = False) -> bool:
        del initial  # retained in the compatibility signature
        result = self._session_controller.show_config(self._config_data)
        if result is None:
            return False
        self._apply_config(result)
        return True

    def _apply_config(self, data: dict) -> None:
        resolved = resolve_config_paths(data)
        data = resolved.data
        ds_cfg = resolved.ds_cfg
        task_cfg = resolved.task_cfg
        self._config_data = data
        presentation = present_config(data, ds_cfg=ds_cfg, task_cfg=task_cfg)
        inference_on = presentation.inference_enabled
        self.run_identity_label.setText(presentation.session_text)
        if not self._recording_active and not self._start_in_progress and not self._stop_in_progress:
            self._set_run_state("ready")
        self.summary_label.setText(presentation.summary_html)
        self.preview.set_info(presentation.preview_info)
        self._emit_log("[GUI] Configuration committed.")
        try:
            if task_cfg is not None:
                self.dashboard.apply_task_config(task_cfg)
        except Exception as exc:
            self._emit_log(f"[GUI] Task config load failed: {exc}")

        self.run_btn.setEnabled(True)

    def _build_launch_config(self) -> process.LaunchConfig:
        return build_launch_config(
            self._config_data,
            bottles=self._collect_bottle_payload(include_final=False, strict=False),
            preview_window_id=self._preview_window_id,
        )

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

    def _forward_dashboard(self, event: DashboardEvent | str) -> None:
        """Cross the backend-thread boundary with one immutable typed event."""

        if isinstance(event, str):
            event = DashboardEvent.parse(event)
        if event is not None:
            self.dashboard_event.emit(event)

    @QtCore.Slot(object)
    def _on_dashboard_event(self, event: DashboardEvent) -> None:
        was_jammed = self.dashboard.feeder_jammed
        self.dashboard.ingest_event(event)
        if self.dashboard.feeder_jammed and not was_jammed:
            behavior_dock = self.workspace.cards.get("behavior")
            if behavior_dock is not None:
                behavior_dock.show()
                behavior_dock.raise_()

    @QtCore.Slot()
    def _on_clear_jam_requested(self) -> None:
        if self._clear_jam_thread is not None and self._clear_jam_thread.is_alive():
            return
        self._emit_log("[GUI] Sending CLEAR_JAM after operator confirmation")

        def worker() -> None:
            try:
                response = self.backend.clear_feeder_jam()
            except Exception as exc:
                self.clear_jam_failed.emit(str(exc))
                return
            self.clear_jam_finished.emit(response)

        self._clear_jam_thread = threading.Thread(
            target=worker,
            daemon=True,
            name="squeakview-clear-feeder-jam",
        )
        self._clear_jam_thread.start()

    @QtCore.Slot(str)
    def _on_clear_jam_finished(self, response: str) -> None:
        if response == "ACK_CLEAR_JAM":
            self.dashboard.clear_jam_alert()
            self._emit_log("[GUI] Feeder jam latch cleared by ACK_CLEAR_JAM")
        elif response == "NACK,CLEAR_JAM,FEED_ACTIVE":
            message = (
                "Clear Jam was rejected: wait until the current feed stops, "
                "then inspect the mechanism and try again."
            )
            self.dashboard.clear_jam_failed(message)
            self._emit_log(f"[GUI] {message}")
        elif response == "NACK,CLEAR_JAM,NOT_JAMMED":
            message = (
                "The firmware reports that no jam is currently latched. "
                "The warning remains active because no ACK_CLEAR_JAM was received."
            )
            self.dashboard.clear_jam_failed(message)
            self._emit_log(f"[GUI] {message}")
        else:
            self._on_clear_jam_failed(f"unrecognized controller response: {response}")

    @QtCore.Slot(str)
    def _on_clear_jam_failed(self, error: str) -> None:
        message = (
            f"Clear Jam failed: {error}. Inspect and physically clear the feeder; "
            "the jam warning remains active."
        )
        self.dashboard.clear_jam_failed(message)
        self._emit_log(f"[GUI] {message}")

    @QtCore.Slot()
    def _send_supervisor_heartbeat(self) -> None:
        """Renew the GUI lease only when the Qt main event loop is responsive."""

        heartbeat = getattr(self.backend, "heartbeat", None)
        if callable(heartbeat):
            heartbeat()

    @QtCore.Slot()
    def _copy_event_log(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.event_log.toPlainText())
        self._emit_log("[GUI] Operator events copied to clipboard.")

    @QtCore.Slot()
    def _open_run_folder(self) -> None:
        target = self.backend.current_snapshot.run_dir or squeakview_config.RUNS_DIR
        path = Path(target)
        if not path.exists():
            self._emit_log(f"[GUI] Run folder does not exist: {path}")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _set_run_state(self, state: str) -> None:
        states = {
            "ready": ("READY", "#4357bd"),
            "starting": ("STARTING", "#a66a20"),
            "recording": ("RECORDING", "#c4425f"),
            "finalizing": ("FINALIZING", "#6855c7"),
            "complete": ("COMPLETE", "#267a58"),
            "failed": ("FAILED", "#a93750"),
        }
        label, color = states.get(state, (state.upper(), "#34394f"))
        self.run_state_label.setText(label)
        self.run_state_label.setStyleSheet(
            "QLabel#runStateBadge {"
            f"background-color: {color}; color: #ffffff; border-radius: 8px; "
            "padding: 5px 10px; font-size: 11px; font-weight: 800; }"
        )

    @QtCore.Slot(object)
    def _on_backend_event(self, event: BackendEvent) -> None:
        self._run_controller.handle_backend_event(event)

    def _set_capture_health(self, text: str, level: str = "ok") -> None:
        color = {"ok": "#a7d9c2", "warning": "#ffd28b", "error": "#ff8ca4"}.get(
            level, "#a7d9c2"
        )
        self.capture_health_label.setText(text)
        self.capture_health_label.setStyleSheet(f"color: {color}; font-size: 11px;")

    def _refresh_runtime_status(self) -> None:
        self._run_controller.refresh_runtime_status()

    def _refresh_capture_health(self) -> None:
        self._run_controller.refresh_capture_health()

    def _refresh_stop_progress(self) -> None:
        self._run_controller.refresh_stop_progress()

    # ---- Actions --------------------------------------------------------
    def _on_configure(self) -> None:
        self._show_config_dialog(initial=False)

    @staticmethod
    def _read_ds_batch_size(cfg_path: Path) -> int | None:
        return _read_ds_batch_size(cfg_path)

    @staticmethod
    def _effective_batch_camera_count(config: process.LaunchConfig) -> int:
        return _effective_batch_camera_count(config)

    def _on_run(self) -> None:
        self._run_controller.start()

    @QtCore.Slot(bool)
    def _on_start_finished(self, started: bool) -> None:
        self._run_controller.on_start_finished(started)

    @QtCore.Slot(str)
    def _on_start_error(self, error: str) -> None:
        self._run_controller.on_start_error(error)

    @QtCore.Slot()
    def _on_backend_run_started(self) -> None:
        self._run_controller.on_backend_run_started()

    def _start_ipc_preview(self, config: process.LaunchConfig) -> None:
        self._run_controller.start_ipc_preview(config)

    @QtCore.Slot()
    def _on_preview_ready(self) -> None:
        self._run_controller.on_preview_ready()

    @QtCore.Slot(str)
    def _on_preview_failed(self, error: str) -> None:
        self._run_controller.on_preview_failed(error)

    @QtCore.Slot()
    def _on_preview_ended(self) -> None:
        self._run_controller.on_preview_ended()

    @QtCore.Slot(str)
    def _on_backend_run_failed(self, error: str) -> None:
        self._run_controller.on_backend_run_failed(error)

    def _on_stop(self) -> None:
        self._run_controller.stop()

    @QtCore.Slot()
    def _on_stop_complete(self) -> None:
        self._run_controller.on_stop_complete()

    @QtCore.Slot(str)
    def _on_stop_failed(self, err: str) -> None:
        self._run_controller.on_stop_failed(err)

    def _resize_stop_overlay(self) -> None:
        self.stop_overlay.resize_to_parent()

    def _show_stop_overlay(self) -> None:
        self.stop_overlay.show_overlay()

    def _hide_stop_overlay(self) -> None:
        self.stop_overlay.hide_overlay()

    # ---- Logging --------------------------------------------------------
    def _emit_log(self, msg: str) -> None:
        self.log_msg.emit(msg)

    @QtCore.Slot(str)
    def _append_log(self, msg: str) -> None:
        if hasattr(self, "event_log"):
            self.event_log.appendPlainText(msg)
            self.event_log.moveCursor(QtGui.QTextCursor.MoveOperation.End)
            self.event_log.ensureCursorVisible()
        try:
            print(msg, flush=True)
        except (BrokenPipeError, OSError, ValueError):
            # Losing the launching terminal must not interrupt scientific
            # shutdown or prevent the GUI from closing a supervised child.
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if not self._run_controller.request_window_close(event):
            return
        try:
            self.workspace.save_layout()
        except Exception as exc:
            self._emit_log(f"[GUI] Could not save workspace layout: {exc}")
        try:
            self.dashboard.close()
        except Exception:
            pass
        self._runtime_timer.stop()
        self._heartbeat_timer.stop()
        super().closeEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._resize_stop_overlay()
