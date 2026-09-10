from __future__ import annotations

"""Qt controller orchestration for an operator run lifecycle."""

import threading
import time
from pathlib import Path
from typing import Any, Callable

from PySide6 import QtCore, QtWidgets

from squeakview import config as squeakview_config
from squeakview.apps.operator.backend import process
from squeakview.apps.operator.backend.events import BackendEvent, RunPhase
from squeakview.apps.operator.gui.run_presentation import (
    CaptureHealthPresentation,
    CaptureHealthSnapshot,
    FinalizationPresentation,
    RunPresentation,
    _effective_batch_camera_count,
    _elapsed_text,
    _last_csv_row,
    _read_ds_batch_size,
    _tail_text_line,
    disk_free_text,
    present_capture_health,
    present_finalization,
    present_run_phase,
    scan_capture_health,
)
from squeakview.common import run_context


RUN_FAILURE_DIALOG_STYLESHEET = """
    QMessageBox { background-color: #171821; }
    QMessageBox QLabel { color: #eef1ff; background-color: transparent; font-size: 13px; }
    QMessageBox QLabel#qt_msgbox_label,
    QMessageBox QLabel#qt_msgbox_informativelabel { min-width: 520px; max-width: 560px; }
    QMessageBox QPushButton {
        min-width: 84px; min-height: 30px; padding: 4px 14px; color: #ffffff;
        background-color: #4f5ed7; border: 1px solid #7180ff;
        border-radius: 5px; font-weight: 700;
    }
    QMessageBox QPushButton:hover { background-color: #5c6df5; }
"""


class RunLifecycleController(QtCore.QObject):
    """Coordinate one window's run lifecycle without owning its widgets."""

    def __init__(
        self,
        view: Any,
        *,
        thread_factory: Callable[..., threading.Thread] = threading.Thread,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(view if isinstance(view, QtCore.QObject) else None)
        self.view = view
        self.thread_factory = thread_factory
        self.monotonic = monotonic
        self.start_in_progress = False
        self.start_thread: threading.Thread | None = None
        self.pending_start_config: process.LaunchConfig | None = None
        self.close_after_start = False
        self.start_failure_reported = False
        self.stop_in_progress = False
        self.stop_thread: threading.Thread | None = None
        self.close_after_stop = False
        self.recording_active = False
        self.recording_started_monotonic: float | None = None
        self.stop_started_monotonic: float | None = None
        self.last_health_poll_monotonic = 0.0
        self.active = True

    def handle_backend_event(self, event: BackendEvent) -> None:
        if not self.active:
            return
        presentation = present_run_phase(event.phase)
        if self.stop_in_progress and event.phase == RunPhase.FINALIZED:
            # The backend has persisted its terminal phase, but the stop worker
            # has not yet handed control back to the GUI for fail-closed status
            # reconciliation.  Keep every action disabled until stop_done.
            presentation = RunPresentation(
                "finalizing", "Finalizing", False, False, False
            )
        elif self.stop_in_progress and event.phase == RunPhase.FAILED:
            presentation = RunPresentation("failed", "Failed", False, False, False)
        self.view._set_run_state(presentation.badge_state)
        self.view.preview.set_status(presentation.preview_status)
        self._set_buttons(presentation)

    def _set_buttons(self, presentation: RunPresentation) -> None:
        self.view.run_btn.setEnabled(presentation.run_enabled)
        self.view.stop_btn.setEnabled(presentation.stop_enabled)
        self.view.configure_btn.setEnabled(presentation.configure_enabled)

    def refresh_runtime_status(self) -> None:
        if not self.active:
            return
        now = self.monotonic()
        if self.recording_active and self.recording_started_monotonic is not None:
            self.view.run_elapsed_label.setText(
                _elapsed_text(now - self.recording_started_monotonic)
            )
        elif self.start_in_progress:
            self.view.run_elapsed_label.setText("STARTING")
        elif self.stop_in_progress and self.stop_started_monotonic is not None:
            self.view.run_elapsed_label.setText(
                f"STOP +{_elapsed_text(now - self.stop_started_monotonic)}"
            )
        if self.stop_in_progress:
            self.refresh_stop_progress()
        if now - self.last_health_poll_monotonic >= 1.0:
            self.last_health_poll_monotonic = now
            self.refresh_capture_health()

    def refresh_capture_health(self) -> None:
        if not self.active:
            return
        run_dir = self.view.backend.current_snapshot.run_dir
        disk_target = Path(run_dir) if run_dir is not None else squeakview_config.RUNS_DIR
        current_disk_text = disk_free_text(disk_target)
        if not self.recording_active:
            if self.start_in_progress or self.stop_in_progress:
                return
            if self.view.run_state_label.text() in {"COMPLETE", "FAILED"}:
                return
            self.view._set_capture_health(
                f"Camera --  ·  Queue --  ·  Disk {current_disk_text}"
            )
            return
        if run_dir is None:
            self.view._set_capture_health("Waiting for run diagnostics…", "warning")
            return
        presentation = present_capture_health(
            scan_capture_health(Path(run_dir)), disk_text=current_disk_text
        )
        self.view._set_capture_health(presentation.text, presentation.level)

    def refresh_stop_progress(self) -> None:
        if not self.active:
            return
        run_dir = self.view.backend.current_snapshot.run_dir
        if run_dir is None:
            return
        run_path = Path(run_dir)
        self.view.stop_overlay.update_progress(
            run_context.read_json(run_path / "run_status.json"),
            run_context.read_json(run_path / "post_run_progress.json"),
        )

    def start(self) -> None:
        if not self.active:
            return
        view = self.view
        if self.start_in_progress or self.stop_in_progress:
            return
        if view._preview_window_id is None:
            view._emit_log("[GUI] Preview still initializing; please wait a moment and try again.")
            QtCore.QTimer.singleShot(200, view._capture_preview_window_id)
            return
        try:
            config = view._build_launch_config()
        except RuntimeError:
            view._emit_log("[GUI] Please configure SqueakView first.")
            return
        if config.inference_enabled:
            cam_count = _effective_batch_camera_count(config)
            cfg_batch = _read_ds_batch_size(config.ds_cfg) if config.ds_cfg else None
            if cfg_batch is not None and cfg_batch != cam_count:
                QtWidgets.QMessageBox.critical(
                    view, "Batch Size Mismatch",
                    f"You selected {cam_count} camera(s), but this DeepStream config uses "
                    f"batch-size={cfg_batch}.\n\nPlease choose a config with "
                    f"batch-size={cam_count} "
                    "and ensure the TensorRT engine was built for that same batch size."
                )
                view._emit_log(
                    f"[GUI] blocked: camera count ({cam_count}) != config batch-size ({cfg_batch})"
                )
                return
        self.start_in_progress = True
        self.start_failure_reported = False
        self.pending_start_config = config
        self.recording_active = False
        self.recording_started_monotonic = None
        view.run_elapsed_label.setText("STARTING")
        view._set_run_state("starting")
        view.preview.show_hint(False)
        view.preview.set_status("Starting")
        view.preview.set_preview_enabled(True)
        view._set_capture_health("Running preflight…", "warning")
        view._emit_log("[GUI] Running preflight and starting capture…")
        self._set_buttons(RunPresentation("starting", "Starting", False, False, False))

        def worker() -> None:
            try:
                if not view.backend.start_run(config):
                    backend_error = getattr(
                        view.backend.current_snapshot, "error", None
                    )
                    view.start_error.emit(
                        backend_error
                        or "The backend preflight or acquisition startup did not pass. "
                        "Check Operator Events for details."
                    )
                    return
            except Exception as exc:
                view.start_error.emit(str(exc))
                return
            view.start_finished.emit(True)

        self.start_thread = self.thread_factory(target=worker, daemon=True)
        self.start_thread.start()

    def on_start_finished(self, started: bool) -> None:
        if not self.active:
            return
        view = self.view
        config = self.pending_start_config
        self.pending_start_config = None
        self.start_in_progress = False
        if not started or config is None:
            self._set_buttons(RunPresentation("failed", "Failed", True, False, True))
            if not self.start_failure_reported:
                view._set_run_state("failed")
            if self.close_after_start:
                self.close_after_start = False
                QtCore.QTimer.singleShot(0, view.close)
            return
        if self.close_after_start:
            # The window is waiting only so startup cannot orphan a process.
            # Do not create a preview pipeline or announce a transient run;
            # immediately transfer ownership to the ordered stop path.
            self.close_after_start = False
            self.close_after_stop = True
            self.stop()
            return
        # Startup assigns the run directory and per-run preview socket paths
        # inside the backend.  The immutable request created by the GUI still
        # contains the pre-start values, so always prefer the backend's
        # prepared runtime config after start_run() succeeds.
        runtime_config = view.backend.runtime_config
        self.start_ipc_preview(runtime_config)
        view._clear_bottle_final_fields()
        view._set_bottle_completion_pending(False)
        if view.backend.current_snapshot.run_dir is not None:
            view._set_bottle_status("Initial bottle info saved with current run.")
        view.dashboard.clear_jam_alert()
        view.preview.show_hint(False)
        view.preview.set_preview_enabled(True)
        # start_run confirms that startup orchestration handed off a live
        # capture process.  Only on_backend_run_started confirms that required
        # metadata is durable and the scientific run is actually recording.
        # That ready callback may be queued before this worker-complete callback.
        if self.recording_active:
            view._set_run_state("recording")
            view.preview.set_status("Recording")
            view._emit_log("[GUI] Capture startup completed; recording was confirmed")
            self._set_buttons(
                RunPresentation("recording", "Recording", False, True, False)
            )
        else:
            view._set_run_state("starting")
            view.preview.set_status("Starting")
            view._emit_log(
                "[GUI] Capture process started; waiting for recording confirmation"
            )
            self._set_buttons(
                RunPresentation("starting", "Starting", False, True, False)
            )

    def on_start_error(self, error: str) -> None:
        if not self.active:
            return
        view = self.view
        close_requested = self.close_after_start
        if self.start_failure_reported:
            self.on_start_finished(False)
            return
        self.start_failure_reported = True
        self.on_start_finished(False)
        self.recording_active = False
        view.run_elapsed_label.setText("00:00:00")
        view._set_run_state("failed")
        view._set_capture_health("Run could not start — open Events", "error")
        view.preview.show_hint(True)
        view.preview.set_status("Failed")
        view.event_dock.show()
        view._emit_log(f"[GUI] Start failed: {error}")
        if not close_requested:
            dialog = QtWidgets.QMessageBox(view)
            dialog.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            dialog.setWindowTitle("Run Could Not Start")
            dialog.setText("SqueakView could not start the run.")
            dialog.setInformativeText(error)
            dialog.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            dialog.setStyleSheet(RUN_FAILURE_DIALOG_STYLESHEET)
            dialog.exec()

    def on_backend_run_started(self) -> None:
        if not self.active:
            return
        view = self.view
        self.recording_active = True
        self.recording_started_monotonic = self.monotonic()
        view._set_run_state("recording")
        view.preview.show_hint(False)
        view.preview.set_status("Recording")
        run_dir = view.backend.current_snapshot.run_dir
        if run_dir is not None:
            view.dashboard.follow_run_telemetry(Path(run_dir) / "diagnostics" / "system.csv")
        view._emit_log("[GUI] Run started")

    def start_ipc_preview(self, config: process.LaunchConfig) -> None:
        view = self.view
        if not config.preview_enabled:
            view.preview.set_preview_enabled(False)
            view.preview.set_status("Live preview disabled for this qualification run")
            view._emit_log("[PREVIEW] disabled by SQUEAKVIEW_DISABLE_PREVIEW")
            return
        sockets = tuple(config.preview_socket_paths)
        if not sockets or view._preview_window_id is None:
            self.on_preview_failed("Preview unavailable: no IPC socket was configured")
            return
        if len(sockets) > 1:
            view._emit_log(
                f"[PREVIEW] {len(sockets)} camera streams are available; displaying camera 0"
            )
        view._preview_controller.start(sockets[0], view._preview_window_id)

    def on_preview_ready(self) -> None:
        if not self.active:
            return
        self.view.preview.show_hint(False)
        self.view.preview.set_status(
            "Recording" if self.recording_active else "Starting (preview ready)"
        )

    def on_preview_failed(self, error: str) -> None:
        if not self.active:
            return
        self.view.preview.label.setText(error)
        self.view.preview.show_hint(True)
        self.view.preview.set_status(
            "Recording (no preview)"
            if self.recording_active
            else "Starting (no preview)"
        )

    def on_preview_ended(self) -> None:
        if not self.active:
            return
        if not self.stop_in_progress:
            self.view.preview.label.setText("Preview stream ended")
            self.view.preview.show_hint(True)
            self.view.preview.set_status("Preview ended")

    def on_backend_run_failed(self, error: str) -> None:
        if not self.active:
            return
        view = self.view
        self.start_failure_reported = True
        # Backend failure notification is emitted only after finalization has
        # released capture/serial ownership.  Honor a close request even when
        # the stop worker's completion signal is still queued behind this one.
        close_after_failure = self.close_after_stop or self.close_after_start
        self.close_after_start = False
        view._preview_controller.stop()
        self.start_in_progress = False
        self.pending_start_config = None
        self.stop_in_progress = False
        self.recording_active = False
        self._freeze_elapsed()
        view._hide_stop_overlay()
        view._set_run_state("failed")
        view._set_capture_health("Run failed — open Events", "error")
        view.preview.show_hint(True)
        view.preview.set_status("Failed")
        view.event_dock.show()
        self._set_buttons(RunPresentation("failed", "Failed", True, False, True))
        view._emit_log(f"[GUI] Run failed: {error}")
        view._set_bottle_status("Run failed; inspect the log and run status before retrying.")
        if not close_after_failure:
            dialog = QtWidgets.QMessageBox(view)
            dialog.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            dialog.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            dialog.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            dialog.setStyleSheet(RUN_FAILURE_DIALOG_STYLESHEET)
            if error.startswith("Could not open serial port"):
                dialog.setWindowTitle("Serial Port Unavailable")
                dialog.setText(
                    "SqueakView could not connect to the experiment controller."
                )
            else:
                dialog.setWindowTitle("Run Failed")
                dialog.setText("SqueakView could not start or continue the run.")
            dialog.setInformativeText(error)
            dialog.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            dialog.exec()
        if close_after_failure:
            self.close_after_stop = False
            QtCore.QTimer.singleShot(0, view.close)

    def stop(self) -> None:
        if not self.active:
            return
        view = self.view
        if self.stop_in_progress or self.start_in_progress:
            return
        self.stop_in_progress = True
        self.recording_active = False
        self.stop_started_monotonic = self.monotonic()
        view._set_run_state("finalizing")
        view._show_stop_overlay()
        self._set_buttons(RunPresentation("finalizing", "Finalizing", False, False, False))
        view.preview.set_status("Finalizing")
        view._emit_log("[GUI] Stopping run…")
        view._preview_controller.stop()

        def worker() -> None:
            try:
                view.backend.stop_run()
            except Exception as exc:
                view.stop_failed.emit(str(exc))
                return
            view.stop_done.emit()

        self.stop_thread = self.thread_factory(target=worker, daemon=True)
        self.stop_thread.start()

    def on_stop_complete(self) -> None:
        if not self.active:
            return
        view = self.view
        close_after_stop = self.close_after_stop
        self.close_after_stop = False
        self.stop_in_progress = False
        self.recording_active = False
        if not close_after_stop:
            view.dashboard.resume_idle_system_sampling()
        self._freeze_elapsed()
        self.stop_started_monotonic = None
        view._hide_stop_overlay()
        view.preview.show_hint(True)
        view.preview.set_preview_enabled(True)
        run_dir = view.backend.current_snapshot.run_dir
        status = run_context.read_json(Path(run_dir) / "run_status.json") if run_dir else {}
        presentation = present_finalization(status)
        view._set_run_state(presentation.badge_state)
        view.preview.set_status(presentation.preview_status)
        view._set_capture_health(presentation.health_text, presentation.health_level)
        self._set_buttons(
            RunPresentation(
                presentation.badge_state,
                presentation.preview_status,
                True,
                False,
                True,
            )
        )
        if presentation.passed:
            view._emit_log("[GUI] Run stopped and capture validation passed")
        else:
            view.event_dock.show()
            view._emit_log("[GUI] Run stopped, but validation did not pass")
        if run_dir is not None:
            view._emit_log(f"[SAVE] local run finalized: {run_dir}")
            if not presentation.passed:
                view._set_bottle_status(
                    "Run validation failed; inspect Events before completing bottle data."
                )
                view._set_bottle_completion_pending(False)
            elif bool(status.get("outputs", {}).get("bottle_measurements_complete")):
                view._set_bottle_status("Run finalized; bottle measurements complete.")
                view._set_bottle_completion_pending(False)
            else:
                view._set_bottle_status("Capture validated. Enter final bottle weights, then save.")
                view._set_bottle_completion_pending(True)
        if close_after_stop:
            QtCore.QTimer.singleShot(0, view.close)

    def on_stop_failed(self, error: str) -> None:
        if not self.active:
            return
        view = self.view
        self.close_after_stop = False
        self.stop_in_progress = False
        self.recording_active = False
        self._freeze_elapsed()
        self.stop_started_monotonic = None
        view._hide_stop_overlay()
        view._set_run_state("failed")
        view.preview.set_status("Failed")
        view._set_capture_health("Shutdown failed — open Events", "error")
        view.event_dock.show()
        view._emit_log(f"[GUI] Stop failed: {error}")
        # A stop exception does not prove that the capture child exited.  Keep
        # new runs/configuration disabled and expose only an explicit retry.
        self._set_buttons(RunPresentation("failed", "Failed", False, True, False))

    def request_window_close(self, event: Any) -> bool:
        """Return true only after the backend is synchronously safe to release."""

        view = self.view
        view._preview_controller.stop()
        if not self.active:
            return True
        if self.start_in_progress:
            self.close_after_start = True
            event.ignore()
            return False
        try:
            snapshot = view.backend.current_snapshot
            capture_running = snapshot.capture_running
            finalizing = snapshot.finalization_in_progress
        except Exception as exc:
            self._present_close_failure(
                f"could not confirm backend activity: {type(exc).__name__}: {exc}"
            )
            event.ignore()
            return False
        if self.stop_in_progress or capture_running or finalizing:
            self.close_after_stop = True
            if capture_running and not self.stop_in_progress and not finalizing:
                self.stop()
            event.ignore()
            return False

        try:
            view.backend.shutdown()
            snapshot = view.backend.current_snapshot
            capture_running = snapshot.capture_running
            finalizing = snapshot.finalization_in_progress
        except Exception as exc:
            self._present_close_failure(
                f"backend shutdown raised {type(exc).__name__}: {exc}"
            )
            event.ignore()
            return False
        if capture_running or finalizing:
            self._present_close_failure(
                "backend shutdown returned before capture/finalization became inactive"
            )
            event.ignore()
            return False
        self.deactivate()
        return True

    def _present_close_failure(self, error: str) -> None:
        view = self.view
        self.close_after_start = False
        self.close_after_stop = False
        view._set_run_state("failed")
        view._set_capture_health(
            "Safe shutdown was not confirmed — SqueakView remains open", "error"
        )
        view.preview.set_status("Shutdown unconfirmed")
        view.event_dock.show()
        self._set_buttons(RunPresentation("failed", "Failed", False, True, False))
        view._emit_log(f"[GUI] Refusing to close: {error}")

    def deactivate(self) -> None:
        """Ignore queued GUI callbacks after a terminal close is committed."""

        self.active = False

    def _freeze_elapsed(self) -> None:
        if self.recording_started_monotonic is not None:
            stop_mark = self.stop_started_monotonic or self.monotonic()
            self.view.run_elapsed_label.setText(
                _elapsed_text(stop_mark - self.recording_started_monotonic)
            )


__all__ = [
    "CaptureHealthPresentation", "CaptureHealthSnapshot",
    "FinalizationPresentation", "RUN_FAILURE_DIALOG_STYLESHEET",
    "RunLifecycleController", "RunPresentation", "_effective_batch_camera_count",
    "_elapsed_text", "_last_csv_row", "_read_ds_batch_size", "_tail_text_line",
    "disk_free_text", "present_capture_health", "present_finalization",
    "present_run_phase", "scan_capture_health",
]
