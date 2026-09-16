from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from squeakview.apps.operator.backend.events import BackendEvent, RunPhase, RunSnapshot
from squeakview.apps.operator.gui.run_presenter import (
    RunLifecycleController,
    present_finalization,
    present_run_phase,
)


def _lifecycle_view() -> SimpleNamespace:
    return SimpleNamespace(
        backend=SimpleNamespace(
            current_snapshot=RunSnapshot(RunPhase.IDLE, None, None),
            runtime_config=SimpleNamespace(
                preview_enabled=False, preview_socket_paths=()
            ),
            snapshot=mock.Mock(return_value=RunSnapshot(RunPhase.IDLE, None, None)),
            shutdown=mock.Mock(),
            stop_run=mock.Mock(),
        ),
        run_btn=SimpleNamespace(setEnabled=mock.Mock()),
        stop_btn=SimpleNamespace(setEnabled=mock.Mock()),
        configure_btn=SimpleNamespace(setEnabled=mock.Mock()),
        run_elapsed_label=SimpleNamespace(setText=mock.Mock()),
        run_state_label=SimpleNamespace(text=mock.Mock(return_value="READY")),
        preview=SimpleNamespace(
            label=SimpleNamespace(setText=mock.Mock()),
            show_hint=mock.Mock(),
            set_status=mock.Mock(),
            set_preview_enabled=mock.Mock(),
        ),
        dashboard=SimpleNamespace(
            clear_jam_alert=mock.Mock(),
            reset_run_data=mock.Mock(),
            follow_run_telemetry=mock.Mock(),
            resume_idle_system_sampling=mock.Mock(),
        ),
        event_dock=SimpleNamespace(show=mock.Mock()),
        stop_overlay=SimpleNamespace(update_progress=mock.Mock()),
        _preview_controller=SimpleNamespace(stop=mock.Mock(), start=mock.Mock()),
        _set_run_state=mock.Mock(),
        _set_capture_health=mock.Mock(),
        _show_stop_overlay=mock.Mock(),
        _hide_stop_overlay=mock.Mock(),
        _emit_log=mock.Mock(),
        _clear_bottle_final_fields=mock.Mock(),
        _set_bottle_completion_pending=mock.Mock(),
        _set_bottle_status=mock.Mock(),
        close=mock.Mock(),
        stop_done=SimpleNamespace(emit=mock.Mock()),
        stop_failed=SimpleNamespace(emit=mock.Mock()),
    )


class RunPresenterTests(unittest.TestCase):
    def test_clock_preflight_event_populates_operator_evidence(self) -> None:
        view = _lifecycle_view()
        view.clock_labels = {
            name: SimpleNamespace(setText=mock.Mock())
            for name in (
                "validation_state",
                "ntp_synchronized",
                "rtc_valid",
                "median_offset_seconds",
                "median_round_trip_ms",
                "correction_state",
                "validation_timestamp",
                "evidence_path",
            )
        }
        controller = RunLifecycleController(view)
        controller.handle_backend_event(
            BackendEvent(
                type="clock_preflight",
                phase=RunPhase.STARTING,
                run_dir=None,
                payload={
                    "validation_state": "CORRECTED_AND_VERIFIED",
                    "ntp_synchronized": True,
                    "rtc_valid": True,
                    "median_offset_seconds": 1.117,
                    "median_round_trip_ms": 2.5,
                    "correction_requested": True,
                    "correction_applied": True,
                    "validation_timestamp": "2026-09-16T12:00:00+00:00",
                    "evidence_path": "/run/diagnostics/clock_validation.json",
                },
            )
        )
        view.clock_labels["validation_state"].setText.assert_called_with(
            "CORRECTED_AND_VERIFIED"
        )
        view.clock_labels["median_offset_seconds"].setText.assert_called_with(
            "+1.117000 s"
        )
        view.clock_labels["correction_state"].setText.assert_called_with(
            "Applied and verified"
        )

    def test_button_matrix_for_every_phase(self) -> None:
        expected = {
            RunPhase.IDLE: (True, False, True),
            RunPhase.CREATED: (False, True, False),
            RunPhase.STARTING: (False, True, False),
            RunPhase.RECORDING: (False, True, False),
            RunPhase.STOPPING: (False, False, False),
            RunPhase.DRAINING: (False, False, False),
            RunPhase.CAPTURE_CLOSED: (False, False, False),
            RunPhase.VALIDATING: (False, False, False),
            RunPhase.FINALIZED: (True, False, True),
            RunPhase.FAILED: (True, False, True),
        }
        for phase, buttons in expected.items():
            with self.subTest(phase=phase):
                presentation = present_run_phase(phase)
                self.assertEqual(
                    (
                        presentation.run_enabled,
                        presentation.stop_enabled,
                        presentation.configure_enabled,
                    ),
                    buttons,
                )

    def test_layout_editing_is_available_only_in_idle_presentations(self) -> None:
        view = _lifecycle_view()
        view.workspace = SimpleNamespace(set_runtime_locked=mock.Mock())
        view.layout_btn = SimpleNamespace(setEnabled=mock.Mock())
        controller = RunLifecycleController(view)

        controller._set_buttons(
            present_run_phase(RunPhase.RECORDING)
        )
        view.workspace.set_runtime_locked.assert_called_with(True)
        view.layout_btn.setEnabled.assert_called_with(True)

        controller._set_buttons(present_run_phase(RunPhase.IDLE))
        view.workspace.set_runtime_locked.assert_called_with(False)
        view.layout_btn.setEnabled.assert_called_with(True)

    def test_terminal_presentation_fails_closed_without_explicit_validation(self) -> None:
        for status in (
            {},
            {"state": "finalized"},
            {"state": "finalized", "recording_validation": {"passed": False}},
            {"state": "analysis_failed", "recording_validation": {"passed": True}},
        ):
            with self.subTest(status=status):
                result = present_finalization(status)
                self.assertFalse(result.passed)
                self.assertEqual(result.badge_state, "failed")

    def test_terminal_presentation_accepts_only_validated_nonfailure_state(self) -> None:
        result = present_finalization(
            {"state": "finalized", "recording_validation": {"passed": True}}
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.badge_state, "complete")
        self.assertIn("validation PASS", result.health_text)

    def test_controller_applies_backend_phase_to_view_atomically(self) -> None:
        view = SimpleNamespace(
            run_btn=mock.Mock(),
            stop_btn=mock.Mock(),
            configure_btn=mock.Mock(),
            preview=mock.Mock(),
            _set_run_state=mock.Mock(),
        )
        controller = RunLifecycleController(view)
        event = SimpleNamespace(phase=RunPhase.DRAINING)

        controller.handle_backend_event(event)

        view._set_run_state.assert_called_once_with("finalizing")
        view.preview.set_status.assert_called_once_with("Finalizing")
        view.run_btn.setEnabled.assert_called_once_with(False)
        view.stop_btn.setEnabled.assert_called_once_with(False)
        view.configure_btn.setEnabled.assert_called_once_with(False)

    def test_runtime_clock_freezes_elapsed_at_stop_boundary(self) -> None:
        view = SimpleNamespace(run_elapsed_label=mock.Mock())
        controller = RunLifecycleController(view, monotonic=lambda: 500.0)
        controller.recording_started_monotonic = 100.0
        controller.stop_started_monotonic = 372.9

        controller._freeze_elapsed()

        view.run_elapsed_label.setText.assert_called_once_with("00:04:32")

    def test_stop_enters_finalization_ui_before_backend_shutdown(self) -> None:
        calls: list[str] = []

        class ImmediateThread:
            def __init__(self, *, target, daemon):
                self.target = target
                self.daemon = daemon

            def start(self) -> None:
                calls.append("thread:start")
                self.target()

        def mark(name: str):
            return mock.Mock(side_effect=lambda *args: calls.append(name))

        view = SimpleNamespace(
            backend=SimpleNamespace(stop_run=mark("backend:stop")),
            run_btn=SimpleNamespace(setEnabled=mark("button:run")),
            stop_btn=SimpleNamespace(setEnabled=mark("button:stop")),
            configure_btn=SimpleNamespace(setEnabled=mark("button:configure")),
            preview=SimpleNamespace(set_status=mark("preview:status")),
            _preview_controller=SimpleNamespace(stop=mark("preview:stop")),
            _set_run_state=mark("state:finalizing"),
            _show_stop_overlay=mark("overlay:show"),
            _emit_log=mark("log:stop"),
            stop_done=SimpleNamespace(emit=mark("signal:done")),
            stop_failed=SimpleNamespace(emit=mark("signal:failed")),
        )
        controller = RunLifecycleController(
            view, thread_factory=ImmediateThread, monotonic=lambda: 25.0
        )

        controller.stop()

        self.assertTrue(controller.stop_in_progress)
        self.assertLess(calls.index("overlay:show"), calls.index("backend:stop"))
        self.assertLess(calls.index("preview:stop"), calls.index("backend:stop"))
        self.assertEqual(calls[-1], "signal:done")

    def test_start_worker_completion_waits_for_recording_ready_event(self) -> None:
        view = _lifecycle_view()
        controller = RunLifecycleController(view)
        config = SimpleNamespace(preview_enabled=False, preview_socket_paths=())
        runtime_config = SimpleNamespace(
            preview_enabled=True,
            preview_socket_paths=("/run/squeakview-preview.sock",),
        )
        view.backend.runtime_config = runtime_config
        controller.start_in_progress = True
        controller.pending_start_config = config
        controller.start_ipc_preview = mock.Mock()

        controller.on_start_finished(True)

        self.assertFalse(controller.recording_active)
        view._set_run_state.assert_called_once_with("starting")
        view.preview.set_status.assert_called_with("Starting")
        view.run_btn.setEnabled.assert_called_with(False)
        view.stop_btn.setEnabled.assert_called_with(True)
        view.configure_btn.setEnabled.assert_called_with(False)
        self.assertNotIn(
            mock.call("recording"),
            view._set_run_state.call_args_list,
        )
        self.assertIn(
            "waiting for recording confirmation",
            view._emit_log.call_args.args[0],
        )
        controller.start_ipc_preview.assert_called_once_with(runtime_config)

    def test_preview_callbacks_do_not_claim_recording_before_confirmation(self) -> None:
        view = _lifecycle_view()
        controller = RunLifecycleController(view)

        controller.on_preview_ready()
        view.preview.set_status.assert_called_with("Starting (preview ready)")
        controller.on_preview_failed("socket unavailable")
        view.preview.set_status.assert_called_with("Starting (no preview)")

        controller.recording_active = True
        controller.on_preview_ready()
        view.preview.set_status.assert_called_with("Recording")

    def test_start_completion_preserves_an_earlier_recording_confirmation(self) -> None:
        view = _lifecycle_view()
        controller = RunLifecycleController(view)
        controller.start_in_progress = True
        controller.recording_active = True
        controller.pending_start_config = SimpleNamespace(
            preview_enabled=False,
            preview_socket_paths=(),
        )
        controller.start_ipc_preview = mock.Mock()

        controller.on_start_finished(True)

        view._set_run_state.assert_called_once_with("recording")
        view.preview.set_status.assert_called_with("Recording")
        self.assertTrue(controller.recording_active)

    def test_deferred_close_skips_preview_and_transfers_start_to_stop(self) -> None:
        view = _lifecycle_view()
        controller = RunLifecycleController(view)
        controller.start_in_progress = True
        controller.close_after_start = True
        controller.pending_start_config = mock.sentinel.config
        controller.start_ipc_preview = mock.Mock()
        controller.stop = mock.Mock()

        controller.on_start_finished(True)

        controller.start_ipc_preview.assert_not_called()
        controller.stop.assert_called_once_with()
        self.assertFalse(controller.close_after_start)
        self.assertTrue(controller.close_after_stop)

    def test_stop_completion_during_close_does_not_restart_idle_sampler(self) -> None:
        view = _lifecycle_view()
        controller = RunLifecycleController(view)
        controller.stop_in_progress = True
        controller.close_after_stop = True
        with mock.patch(
            "squeakview.apps.operator.gui.run_presenter.QtCore.QTimer.singleShot"
        ) as single_shot:
            controller.on_stop_complete()

        view.dashboard.resume_idle_system_sampling.assert_not_called()
        single_shot.assert_called_once_with(0, view.close)

    def test_terminal_backend_event_cannot_claim_success_before_stop_reconciliation(self) -> None:
        view = _lifecycle_view()
        controller = RunLifecycleController(view)
        controller.stop_in_progress = True

        controller.handle_backend_event(SimpleNamespace(phase=RunPhase.FINALIZED))

        view._set_run_state.assert_called_once_with("finalizing")
        view.preview.set_status.assert_called_once_with("Finalizing")
        view.run_btn.setEnabled.assert_called_once_with(False)
        view.stop_btn.setEnabled.assert_called_once_with(False)
        view.configure_btn.setEnabled.assert_called_once_with(False)

    def test_close_during_start_is_deferred_without_touching_backend(self) -> None:
        view = _lifecycle_view()
        controller = RunLifecycleController(view)
        controller.start_in_progress = True
        event = mock.Mock()

        self.assertFalse(controller.request_window_close(event))

        self.assertTrue(controller.close_after_start)
        event.ignore.assert_called_once_with()
        view.backend.shutdown.assert_not_called()
        view._preview_controller.stop.assert_called_once_with()

    def test_close_during_recording_requests_stop_and_waits(self) -> None:
        view = _lifecycle_view()
        view.backend.current_snapshot = RunSnapshot(
            RunPhase.RECORDING, None, None, capture_running=True
        )
        controller = RunLifecycleController(view)
        controller.stop = mock.Mock()
        event = mock.Mock()

        self.assertFalse(controller.request_window_close(event))

        self.assertTrue(controller.close_after_stop)
        controller.stop.assert_called_once_with()
        event.ignore.assert_called_once_with()
        view.backend.shutdown.assert_not_called()

    def test_idle_close_requires_confirmed_shutdown_then_deactivates_callbacks(self) -> None:
        view = _lifecycle_view()
        controller = RunLifecycleController(view)
        event = mock.Mock()

        self.assertTrue(controller.request_window_close(event))

        view.backend.shutdown.assert_called_once_with()
        event.ignore.assert_not_called()
        self.assertFalse(controller.active)
        view._set_run_state.reset_mock()
        view.preview.set_status.reset_mock()

        controller.handle_backend_event(SimpleNamespace(phase=RunPhase.RECORDING))
        controller.on_backend_run_started()
        controller.on_preview_ready()
        controller.on_stop_failed("late callback")

        view._set_run_state.assert_not_called()
        view.preview.set_status.assert_not_called()

    def test_shutdown_exception_refuses_close_and_exposes_only_stop_retry(self) -> None:
        view = _lifecycle_view()
        view.backend.shutdown.side_effect = RuntimeError("shutdown fault")
        controller = RunLifecycleController(view)
        event = mock.Mock()

        self.assertFalse(controller.request_window_close(event))

        self.assertTrue(controller.active)
        event.ignore.assert_called_once_with()
        view._set_run_state.assert_called_with("failed")
        view._set_capture_health.assert_called_with(
            "Safe shutdown was not confirmed — SqueakView remains open", "error"
        )
        view.run_btn.setEnabled.assert_called_with(False)
        view.stop_btn.setEnabled.assert_called_with(True)
        view.configure_btn.setEnabled.assert_called_with(False)
        self.assertIn("Refusing to close", view._emit_log.call_args.args[0])

    def test_shutdown_return_with_live_child_refuses_close(self) -> None:
        view = _lifecycle_view()
        view.backend.shutdown.side_effect = lambda: setattr(
            view.backend,
            "current_snapshot",
            RunSnapshot(RunPhase.RECORDING, None, None, capture_running=True),
        )
        controller = RunLifecycleController(view)
        event = mock.Mock()

        self.assertFalse(controller.request_window_close(event))

        view.backend.shutdown.assert_called_once_with()
        event.ignore.assert_called_once_with()
        self.assertTrue(controller.active)

    def test_stop_exception_never_reenables_new_run(self) -> None:
        view = _lifecycle_view()
        controller = RunLifecycleController(view)
        controller.stop_in_progress = True

        controller.on_stop_failed("cannot terminate capture")

        view.run_btn.setEnabled.assert_called_with(False)
        view.stop_btn.setEnabled.assert_called_with(True)
        view.configure_btn.setEnabled.assert_called_with(False)

    def test_failure_callback_during_requested_close_never_opens_modal_dialog(self) -> None:
        view = _lifecycle_view()
        controller = RunLifecycleController(view)
        controller.close_after_stop = True
        controller.stop_in_progress = True
        with (
            mock.patch(
                "squeakview.apps.operator.gui.run_presenter.QtWidgets.QMessageBox"
            ) as message_box,
            mock.patch(
                "squeakview.apps.operator.gui.run_presenter.QtCore.QTimer.singleShot"
            ) as single_shot,
        ):
            controller.on_backend_run_failed("capture failed during close")

        message_box.assert_not_called()
        single_shot.assert_called_once_with(0, view.close)

    def test_start_error_during_requested_close_never_opens_modal_dialog(self) -> None:
        view = _lifecycle_view()
        controller = RunLifecycleController(view)
        controller.start_in_progress = True
        controller.pending_start_config = mock.sentinel.config
        controller.close_after_start = True
        with (
            mock.patch(
                "squeakview.apps.operator.gui.run_presenter.QtWidgets.QMessageBox"
            ) as message_box,
            mock.patch(
                "squeakview.apps.operator.gui.run_presenter.QtCore.QTimer.singleShot"
            ) as single_shot,
        ):
            controller.on_start_error("preflight failed during close")

        message_box.assert_not_called()
        single_shot.assert_called_once_with(0, view.close)


if __name__ == "__main__":
    unittest.main()
