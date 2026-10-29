from __future__ import annotations

import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from squeakview.apps.operator.backend import manager, process
from squeakview.apps.operator.backend.events import RunPhase
from squeakview.common.child_events import EVENT_PREFIX, encode_child_event
from squeakview.common import run_context
from squeakview.common.dashboard import DashboardEvent
from squeakview.project import (
    AppPaths,
    Project,
    ProjectMetadata,
    ProjectPaths,
    RuntimeContext,
    UserPaths,
)


from tests.support.lifecycle import BackendLifecycleCase, _runtime_context, FakeProcessHandle, FakeHungFinalizer, FakeSerialHandle

class LifecycleStartupTests(BackendLifecycleCase):
    def test_ready_marker_transitions_starting_to_recording(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))
        self.assertEqual(self.status()["state"], "starting")
        self.assertTrue(self.status()["preflight"]["passed"])
        self.assertEqual(
            (self.run_dir / "config/task.yaml").read_text(),
            self.task_cfg.read_text(),
        )
        self.assertEqual(
            self.backend._task_config_snapshot["snapshot_path"],
            "config/task.yaml",
        )
        self.backend._inference_emit(
            encode_child_event("pipeline_ready", run_dir=str(self.run_dir))
        )

        status = self.status()
        self.assertEqual(status["state"], "recording")
        self.assertEqual(self.started, [True])
        self.assertIn("starting_at", status)


    def test_inference_start_uses_prepared_localized_config_and_parser_identity(self) -> None:
        _package, config, parser, selected = self._inference_model_fixture()

        with mock.patch.object(
            manager.model_package,
            "validate_model_package",
            return_value=selected,
        ):
            started = self.backend.start_run(
                self.config(inference_enabled=True, ds_cfg=config)
            )

        self.assertTrue(started)
        effective = self.backend._effective_deepstream_config
        self.assertIsNotNone(effective)
        assert effective is not None
        self.assertEqual(self.backend.launch_cfg.ds_cfg, effective.path)
        self.assertEqual(effective.parser_identity["path"], str(parser.resolve()))
        self.assertEqual(self.spawn_kwargs["effective_config"], effective)


    def test_bound_qualification_case_is_persisted_in_initial_status(self) -> None:
        matrix_path = self.root / "matrix.yaml"
        matrix_path.write_text(
            """schema_version: '1.0'
matrix_id: lifecycle-binding-test
capture_profiles:
  - {id: capture, width: 10, height: 20, fps: 30, camera_count: 1, pixel_format: Mono8, trigger_on: false, trigger_activation: rising, arduino_fps: 30, exposure_us: 10000.0, bitrate_kbps: 4000, serial_enabled: false, serial_port: /dev/ttyACM0, serial_baud: 115200}
durations:
  - {id: short, minimum_seconds: 5}
inference_enabled: [false]
preview_enabled: [false]
power_modes: [25W]
"""
        )
        case_id = "capture--short--infer-off--preview-off--25w"
        with (
            mock.patch.dict(
                manager.os.environ,
                {
                    "SQUEAKVIEW_QUALIFICATION_CASE_ID": case_id,
                    "SQUEAKVIEW_QUALIFICATION_MATRIX": str(matrix_path),
                },
            ),
            mock.patch.object(
                manager,
                "device_context_snapshot",
                return_value={"nvpmodel": "NV Power Mode: 25W\n0"},
            ),
        ):
            started = self.backend.start_run(
                self.config(
                    width=10,
                    height=20,
                    fps=30,
                    pixel_format="Mono8",
                    preview_enabled=False,
                )
            )

        self.assertTrue(started)
        binding = self.status()["qualification"]
        self.assertEqual(binding["matrix_id"], "lifecycle-binding-test")
        self.assertEqual(binding["case_id"], case_id)
        self.assertEqual(binding["expected_factors"]["preview_enabled"], False)
        self.assertEqual(len(binding["matrix_sha256"]), 64)


    def test_backend_owned_preflight_blocks_run_creation(self) -> None:
        with mock.patch.object(
            manager.preflight,
            "run_preflight",
            return_value=manager.preflight.PreflightResult(
                False, "Automatic suspend policy is unsafe", "[FAIL] suspend"
            ),
        ):
            started = self.backend.start_run(self.config())

        self.assertFalse(started)
        self.assertIsNone(self.exit_callback)
        self.assertEqual(self.backend.snapshot().phase, RunPhase.IDLE)
        self.assertFalse(self.backend._preflight_evidence["passed"])


    def test_task_snapshot_failure_aborts_before_capture_spawn(self) -> None:
        (self.run_dir / run_context.RUN_STATUS_FILENAME).unlink()
        self.task_cfg.write_bytes(
            b"x" * (manager.manifest.MAX_TASK_CONFIG_BYTES + 1)
        )

        result = self.backend.start_run(self.config())

        self.assertFalse(result)
        self.assertIsNone(self.backend.state.inference)
        self.assertFalse((self.run_dir / "config/task.yaml").exists())
        self.assertFalse((self.run_dir / run_context.RUN_STATUS_FILENAME).exists())


    def test_typed_events_cover_complete_backend_lifecycle(self) -> None:
        phases: list[RunPhase] = []
        self.backend.subscribe(lambda event: phases.append(event.phase))
        self.assertTrue(self.backend.start_run(self.config()))
        self.backend._inference_emit(
            encode_child_event("pipeline_ready", run_dir=str(self.run_dir))
        )

        self.backend.stop_run()

        self.assertEqual(
            phases,
            [
                RunPhase.CREATED,
                RunPhase.STARTING,
                RunPhase.RECORDING,
                RunPhase.STOPPING,
                RunPhase.CAPTURE_CLOSED,
                RunPhase.VALIDATING,
                RunPhase.FINALIZED,
            ],
        )
        self.assertEqual(self.backend.snapshot().phase, RunPhase.FINALIZED)


    def test_only_structured_ready_event_marks_recording(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))

        self.backend._inference_emit("[12:00:00] [READY] inference playing")
        self.assertEqual(self.status()["state"], "starting")
        self.assertTrue(
            any("ignored legacy text readiness marker" in message for message in self.logs)
        )

        self.backend._inference_emit(
            encode_child_event("pipeline_ready", run_dir=str(self.run_dir))
        )

        self.assertEqual(self.status()["state"], "recording")


    def test_malformed_structured_event_fails_run_without_marking_ready(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))

        self.backend._inference_emit(EVENT_PREFIX + "not-json")
        self.assertTrue(self.failure_event.wait(timeout=2.0))

        self.assertEqual(self.status()["state"], "failed")
        self.assertEqual(self.started, [])
        self.assertIn("invalid or unsupported structured event", self.failures[-1])


    def test_structured_fatal_event_fails_run_with_child_detail(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))

        self.backend._inference_emit(
            encode_child_event(
                "fatal", run_dir=str(self.run_dir), error="recording sink stalled"
            )
        )
        self.assertTrue(self.failure_event.wait(timeout=2.0))

        self.assertEqual(self.status()["state"], "failed")
        self.assertEqual(self.started, [])
        self.assertIn("recording sink stalled", self.failures[-1])


    def test_unexpected_exit_marks_run_failed_and_notifies_gui(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))
        self.handle.running = False

        assert self.exit_callback is not None
        self.exit_callback(7)

        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("exit code 7", status["error"])
        self.assertIn("failed_at", status)
        self.assertEqual(len(self.failures), 1)
        self.assertIsNone(self.backend.state.inference)
        self.backend._run_capture_finalizer.assert_called_once_with(self.run_dir)


    def test_spawn_failure_marks_created_run_failed(self) -> None:
        with mock.patch.object(manager.process, "spawn_inference", side_effect=OSError("spawn denied")):
            result = self.backend.start_run(self.config())

        self.assertFalse(result)
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("spawn denied", status["error"])
        self.assertEqual(len(self.failures), 1)


    def test_starting_status_enospc_fails_before_capture_launch(self) -> None:
        original_write_status = run_context.write_status

        def fail_starting(run_dir: Path, state: str, **updates):
            if state == "starting":
                raise OSError("starting status ENOSPC")
            return original_write_status(run_dir, state, **updates)

        with mock.patch.object(
            manager.run_context, "write_status", side_effect=fail_starting
        ):
            result = self.backend.start_run(self.config())

        self.assertFalse(result)
        manager.process.spawn_inference.assert_not_called()
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("starting status ENOSPC", status["error"])


    def test_invalid_model_is_rejected_before_run_creation(self) -> None:
        missing_config = (
            self.root / "project" / "models" / "missing" / "configs" / "missing.txt"
        )
        (self.run_dir / run_context.RUN_STATUS_FILENAME).unlink()

        result = self.backend.start_run(self.config(inference_enabled=True, ds_cfg=missing_config))

        self.assertFalse(result)
        self.assertFalse((self.run_dir / run_context.RUN_STATUS_FILENAME).exists())
        self.assertTrue(any("model package is invalid" in line for line in self.logs))


    def test_unsafe_trigger_policy_is_actionable_before_preflight_or_run_creation(self) -> None:
        (self.run_dir / run_context.RUN_STATUS_FILENAME).unlink()
        with mock.patch.object(manager.preflight, "run_preflight") as preflight_run:
            result = self.backend.start_run(
                self.config(
                    trigger_on=True,
                    serial_enabled=False,
                    fps=30,
                    arduino_fps=30,
                )
            )

        self.assertFalse(result)
        preflight_run.assert_not_called()
        self.assertIn("requires serial controller", self.backend.snapshot().error or "")
        self.assertFalse((self.run_dir / run_context.RUN_STATUS_FILENAME).exists())
        manager.process.spawn_inference.assert_not_called()


