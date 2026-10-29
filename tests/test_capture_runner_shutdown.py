from __future__ import annotations

import csv
import json
import tempfile
import threading
import tracemalloc
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from squeakview.apps.inference import service_maker_runner as runner
from squeakview.apps.inference import (
    contracts,
    debug_instrumentation,
    frame_audit,
    main as inference_main,
    pipeline_builder,
    post_run,
    video_probe,
)
from squeakview.apps.inference.pose_pipeline import PoseClass, PoseSchema, decode_yolo26_rows, load_pose_schema
from squeakview.common import run_context
from squeakview.common.capture_policy import capture_buffer_policy
from squeakview.common.child_events import decode_child_event
from squeakview.common.failure_injection import FailurePlan


from tests.support.service_maker_runner import ServiceMakerCase, FakePipeline

class CaptureShutdownTests(ServiceMakerCase):
    def test_safe_print_cannot_break_shutdown_when_parent_pipe_is_closed(self) -> None:
        with mock.patch("builtins.print", side_effect=BrokenPipeError):
            runner._safe_print("shutdown")


    def test_ready_message_emits_versioned_event(self) -> None:
        app = runner.ServiceMakerApp(self.config(enable_infer=False))
        app._ready_origin = "sink"

        class ReadyMessage:
            new_state = runner.PipelineState.PLAYING
            origin = "sink"

        with (
            mock.patch.object(runner, "StateTransitionMessage", ReadyMessage),
            mock.patch.object(runner, "_safe_print") as safe_print,
        ):
            app._on_message(ReadyMessage())

        events = [
            decode_child_event(call.args[0])
            for call in safe_print.call_args_list
        ]
        ready = [event for event in events if event is not None]
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].type, "pipeline_ready")
        self.assertEqual(ready[0].payload["run_dir"], str(app.run_dir))


    def test_unsolicited_live_eos_is_fatal_but_requested_eos_is_clean(self) -> None:
        class EndMessage:
            pass

        with mock.patch.object(runner, "EOSMessage", EndMessage):
            unexpected = runner.ServiceMakerApp(self.config(enable_infer=False))
            unexpected._on_message(EndMessage())
            requested = runner.ServiceMakerApp(self.config(enable_infer=False))
            requested.request_stop()
            requested._on_message(EndMessage())

        self.assertEqual(unexpected.exit_code, 1)
        self.assertEqual(requested.exit_code, 0)


    def test_stop_attempts_every_closer_and_persists_all_errors(self) -> None:
        app = runner.ServiceMakerApp(self.config(enable_infer=False))
        pipeline = mock.Mock()
        pipeline.stop.side_effect = OSError("pipeline stop failed")
        observations = mock.Mock()
        observations.close.side_effect = OSError("observation flush failed")
        admission_failed = mock.Mock()
        admission_failed.close.side_effect = OSError("admission flush failed")
        admission_ok = mock.Mock()
        telemetry = mock.Mock()
        frames = mock.Mock()
        app.pipeline = pipeline
        app.observations = observations
        app.record_admissions = [admission_failed, admission_ok]
        app.record_telemetry = [telemetry]
        app.frames = frames

        app.stop()

        pipeline.stop.assert_called_once_with()
        pipeline.wait.assert_called_once_with()
        observations.close.assert_called_once_with()
        admission_failed.close.assert_called_once_with()
        admission_ok.close.assert_called_once_with()
        telemetry.close.assert_called_once_with()
        frames.close.assert_called_once_with()
        self.assertEqual(app.exit_code, 1)
        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertEqual(status["state"], "capture_closed")
        self.assertEqual(len(status["capture_close_errors"]), 3)
        self.assertIn("pipeline stop failed", status["capture_close_error"])


    def test_pipeline_shutdown_timeout_returns_and_marks_capture_invalid(self) -> None:
        release = threading.Event()
        app = runner.ServiceMakerApp(self.config(enable_infer=False))
        pipeline = mock.Mock()
        pipeline.stop.side_effect = release.wait
        app.pipeline = pipeline

        with mock.patch.dict(
            "os.environ", {"SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S": "0.1"}
        ):
            app.stop()
        release.set()

        self.assertEqual(app.exit_code, 1)
        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertIn("timed out", status["capture_close_error"])


    def test_recording_integrity_fault_bypasses_native_pipeline_shutdown(self) -> None:
        app = runner.ServiceMakerApp(self.config(enable_infer=False))
        pipeline = mock.Mock()
        app.pipeline = pipeline

        app._recording_fault("recording queue backlog reached fatal threshold")
        app.stop()

        pipeline.stop.assert_not_called()
        pipeline.wait.assert_not_called()
        self.assertEqual(app.exit_code, 4)
        self.assertTrue(app._force_process_exit)
        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertEqual(status["capture_exit_code"], 4)
        self.assertIn("native shutdown bypassed", status["capture_close_error"])


    def test_storage_reserve_fault_uses_normal_pipeline_drain(self) -> None:
        app = runner.ServiceMakerApp(self.config(enable_infer=False))
        pipeline = mock.Mock()
        app.pipeline = pipeline

        with mock.patch.object(runner, "_safe_print") as safe_print:
            app._storage_reserve_fault("reserve exhausted")
            app.stop()

        pipeline.stop.assert_called_once_with()
        pipeline.wait.assert_called_once_with()
        self.assertEqual(app.exit_code, 5)
        self.assertFalse(app._force_process_exit)
        events = [
            decode_child_event(call.args[0]) for call in safe_print.call_args_list
        ]
        fatal = [event for event in events if event is not None and event.type == "fatal"]
        self.assertEqual(len(fatal), 1)
        self.assertIn("reserve exhausted", fatal[0].payload["error"])


    def test_capture_process_owns_system_telemetry_lifecycle(self) -> None:
        order: list[str] = []
        telemetry_kwargs: dict[str, object] = {}

        class OrderedPipeline(FakePipeline):
            def start(self, callback):
                del callback
                order.append("pipeline_start")
                return super().start(None)

            def stop(self):
                order.append("pipeline_stop")
                return super().stop()

            def wait(self):
                order.append("pipeline_wait")
                return super().wait()

        class FakeSystemTelemetry:
            def __init__(self, path, **kwargs):
                self.path = path
                self.sample_count = 2
                self.last_error = None
                telemetry_kwargs.update(kwargs)

            def start(self):
                order.append("telemetry_start")
                return True

            def stop(self):
                order.append("telemetry_stop")

        app = runner.ServiceMakerApp(
            self.config(enable_infer=False),
            pipeline_factory=OrderedPipeline,
            probe_factory=lambda name, operator: (name, operator),
            system_telemetry_factory=FakeSystemTelemetry,
        )
        app.build()
        app.request_stop()

        self.assertEqual(app.run(), 0)
        self.assertLess(order.index("telemetry_start"), order.index("pipeline_start"))
        self.assertLess(order.index("pipeline_wait"), order.index("telemetry_stop"))
        shutdown_requested = telemetry_kwargs["shutdown_requested"]
        self.assertTrue(callable(shutdown_requested))
        self.assertTrue(shutdown_requested())
        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertEqual(status["system_telemetry"]["sample_count"], 2)


    def test_system_telemetry_loss_is_persisted_without_aborting_recording(self) -> None:
        app = runner.ServiceMakerApp(self.config(enable_infer=False))

        app._system_telemetry_fault("tegrastats reader stopped")

        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertTrue(status["system_telemetry_degraded"])
        self.assertEqual(
            status["system_telemetry_error"], "tegrastats reader stopped"
        )
        self.assertFalse(app._stop_event.is_set())


    def test_stop_wakes_extracted_recording_liveness_monitor_before_join(self) -> None:
        app = runner.ServiceMakerApp(self.config(enable_infer=False))
        app._recording_liveness_monitor = runner.RecordingLivenessMonitor(
            [],
            app._stop_event,
            app._recording_fault,
            runner.resolve_recording_liveness_policy(
                app.config.fps,
                failure_injection=False,
            ),
        )
        app._recording_liveness_thread = threading.Thread(
            target=app._recording_liveness_monitor.run,
            daemon=True,
        )
        app._recording_liveness_thread.start()

        app.stop()

        self.assertTrue(app._stop_event.is_set())
        self.assertFalse(app._recording_liveness_thread.is_alive())


    def test_unexpected_pipeline_end_wakes_runner_and_fails_capture(self) -> None:
        app = runner.ServiceMakerApp(
            self.config(enable_infer=False),
            pipeline_factory=FakePipeline,
            probe_factory=lambda name, operator: (name, operator),
        )
        app.build()

        result = app.run()

        self.assertEqual(result, 1)
        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertEqual(status["capture_exit_code"], 1)


