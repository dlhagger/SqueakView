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

class LifecycleSerialTests(BackendLifecycleCase):
    def test_serial_dashboard_line_is_parsed_once_into_typed_event(self) -> None:
        received: list[DashboardEvent] = []
        backend = manager.OperatorBackend(
            self.logs.append,
            received.append,
            runtime_context=_runtime_context(self.root),
        )

        with mock.patch.object(
            manager.dashboard_util.DashboardEvent,
            "parse",
            wraps=DashboardEvent.parse,
        ) as parse:
            backend._serial_emit(
                "[12:00:00] 【SER】 POKE_START,1000000,2000000,L,1,2,3,4,ON,ok"
            )

        parse.assert_called_once()
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], DashboardEvent)
        self.assertEqual(received[0].event_uc, "POKE_START")


    def test_clock_preflight_evidence_is_persisted_and_associated_with_run(self) -> None:
        FakeSerialHandle.instances.clear()
        with (
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
        ):
            self.assertTrue(
                self.backend.start_run(self.config(serial_enabled=True))
            )
        evidence_path = self.run_dir / "diagnostics" / "clock_validation.json"
        evidence = run_context.read_json(evidence_path)
        self.assertEqual(evidence["result"], "PASS")
        self.assertEqual(evidence["reason"], "CLOCK_WITHIN_TOLERANCE")
        self.assertEqual(evidence["evidence_path"], str(evidence_path))
        self.assertEqual(self.status()["clock_validation"]["result"], "PASS")


    def test_model_loaded_recheck_blocks_controller_start_after_artifact_tamper(self) -> None:
        package, config, _parser, selected = self._inference_model_fixture()
        FakeSerialHandle.instances.clear()

        def pipeline_ready(*, timeout: float) -> bool:
            self.assertGreater(timeout, 0)
            (package / "engines/model.engine").write_bytes(b"tampered after spawn")
            self.backend._inference_emit(
                encode_child_event("pipeline_ready", run_dir=str(self.run_dir))
            )
            return True

        with (
            mock.patch.object(
                manager.model_package,
                "validate_model_package",
                return_value=selected,
            ),
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
            mock.patch.object(
                self.backend._inference_ready,
                "wait",
                side_effect=pipeline_ready,
            ),
        ):
            started = self.backend.start_run(
                self.config(
                    inference_enabled=True,
                    ds_cfg=config,
                    serial_enabled=True,
                    trigger_on=True,
                )
            )

        self.assertFalse(started)
        serial_handle = FakeSerialHandle.instances[-1]
        self.assertFalse(any(line.startswith("START,") for line in serial_handle.sent))
        self.assertNotIn("START_SENT", serial_handle.markers)
        self.assertEqual(self.started, [])
        self.assertTrue(self.failure_event.wait(2.0))
        self.assertEqual(self.status()["state"], "failed")
        self.assertIn("engine", self.status()["error"])


    def test_unexpected_exit_still_requires_controller_stop_ack(self) -> None:
        FakeSerialHandle.instances.clear()
        with (
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
            mock.patch.object(self.backend._inference_ready, "wait", return_value=True),
        ):
            self.assertTrue(
                self.backend.start_run(
                    self.config(serial_enabled=True, trigger_on=True)
                )
            )
        serial_handle = FakeSerialHandle.instances[-1]
        serial_handle.wait_for_stop_ack = mock.Mock(return_value=True)
        self.handle.running = False

        assert self.exit_callback is not None
        with mock.patch.object(self.backend, "_wait_for_capture_drain", return_value=True):
            self.exit_callback(7)

        serial_handle.wait_for_stop_ack.assert_called_once_with(timeout_s=2.0)
        self.assertIn("CAPTURE_STOP_ACKED", serial_handle.markers)
        self.assertTrue(serial_handle.closed)


    def test_created_status_enospc_fails_before_serial_or_capture_start(self) -> None:
        FakeSerialHandle.instances.clear()
        original_write_status = run_context.write_status

        def fail_created(run_dir: Path, state: str, **updates):
            if state == "created":
                raise OSError("No space left on device")
            return original_write_status(run_dir, state, **updates)

        with (
            mock.patch.object(
                manager.run_context, "write_status", side_effect=fail_created
            ),
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
        ):
            result = self.backend.start_run(self.config(serial_enabled=True))

        self.assertFalse(result)
        manager.process.spawn_inference.assert_not_called()
        self.assertEqual(FakeSerialHandle.instances, [])
        self.assertIn("pre-start run metadata", self.failures[0])
        self.assertIn("No space left on device", self.failures[0])


    def test_manifest_enospc_fails_before_serial_or_capture_start(self) -> None:
        FakeSerialHandle.instances.clear()

        def fail_required_manifest(_run_dir: Path, *, required: bool = False):
            if required:
                raise OSError("manifest ENOSPC")
            return False

        self.backend._write_run_manifest.side_effect = fail_required_manifest
        with (
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
        ):
            result = self.backend.start_run(self.config(serial_enabled=True))

        self.assertFalse(result)
        manager.process.spawn_inference.assert_not_called()
        self.assertEqual(FakeSerialHandle.instances, [])
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("manifest ENOSPC", status["error"])


    def test_bottle_metadata_enospc_fails_before_capture_launch(self) -> None:
        self.backend._write_bottle_measurements.side_effect = OSError(
            "bottle metadata ENOSPC"
        )

        result = self.backend.start_run(self.config())

        self.assertFalse(result)
        manager.process.spawn_inference.assert_not_called()
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("bottle metadata ENOSPC", status["error"])


    def test_recording_manifest_failure_prevents_controller_start(self) -> None:
        FakeSerialHandle.instances.clear()
        pipeline_ready = False

        def fail_recording_manifest(run_dir: Path, *, required: bool = False):
            del run_dir
            if required and pipeline_ready:
                raise OSError("recording manifest ENOSPC")
            return True

        self.backend._write_run_manifest.side_effect = fail_recording_manifest

        def become_ready(*_args, **_kwargs) -> bool:
            nonlocal pipeline_ready
            pipeline_ready = True
            self.backend._inference_emit(
                encode_child_event("pipeline_ready", run_dir=str(self.run_dir))
            )
            return True

        with (
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
            mock.patch.object(
                self.backend._inference_ready, "wait", side_effect=become_ready
            ),
        ):
            result = self.backend.start_run(
                self.config(serial_enabled=True, trigger_on=True)
            )
            self.assertTrue(self.failure_event.wait(timeout=2.0))

        serial_handle = FakeSerialHandle.instances[-1]
        self.assertFalse(result)
        self.assertFalse(any(line.startswith("START,") for line in serial_handle.sent))
        self.assertEqual(self.status()["state"], "failed")
        self.assertIn("before controller START", self.status()["error"])


    def test_stop_halts_controller_before_draining_capture(self) -> None:
        FakeSerialHandle.instances.clear()
        with (
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
        ):
            self.assertTrue(self.backend.start_run(self.config(serial_enabled=True)))

        serial_handle = FakeSerialHandle.instances[-1]
        events: list[str] = []

        def log_marker(marker: str) -> None:
            events.append(f"marker:{marker}")

        def send_line(line: str) -> None:
            events.append(f"send:{line}")

        def terminate(*_args, **_kwargs) -> None:
            events.append("inference:terminate")
            self.handle.running = False

        serial_handle.log_marker = log_marker
        serial_handle.send_line = send_line
        serial_handle.wait_for_stop_ack = lambda **_kwargs: events.append("wait:ACK_STOP") or True
        serial_handle.close = lambda: events.append("serial:close")
        self.handle.terminate_group_graceful = terminate

        self.backend.stop_run()

        self.assertEqual(
            events,
            [
                "marker:CAPTURE_STOP_REQUESTED",
                "marker:STOP_SENT",
                "send:STOP",
                "wait:ACK_STOP",
                "marker:CAPTURE_STOP_ACKED",
                "inference:terminate",
                "marker:CAPTURE_STOP_DONE",
                "serial:close",
            ],
        )
        self.assertEqual(self.status()["state"], "finalized")


    def test_serial_flush_failure_during_close_fails_finalization(self) -> None:
        FakeSerialHandle.instances.clear()
        with (
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
        ):
            self.assertTrue(self.backend.start_run(self.config(serial_enabled=True)))

        serial_handle = FakeSerialHandle.instances[-1]

        def fail_close() -> None:
            serial_handle.closed = True
            serial_handle.fatal_error = "serial CSV close/flush failed: disk full"

        serial_handle.close = fail_close
        self.backend.stop_run()

        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("integrity failed during shutdown", status["error"])
        self.assertIn("disk full", status["error"])


    def test_capture_drain_does_not_accept_quiet_ledgers_below_ttl_count(self) -> None:
        (self.run_dir / "capture_cam0.jsonl").write_text(
            '{"source_sequence_index": 0}\n'
        )
        (self.run_dir / "record_admission.csv").write_text(
            "stream_id,record_frame_index,pts_ns\n0,0,100\n"
        )
        self.backend.launch_cfg = self.config(num_cameras=1)

        with mock.patch.dict(
            manager.os.environ,
            {
                "SQUEAKVIEW_CAPTURE_DRAIN_QUIET_S": "0.1",
                "SQUEAKVIEW_CAPTURE_DRAIN_TIMEOUT_S": "0.2",
            },
        ):
            drained = self.backend._wait_for_capture_drain(
                self.run_dir, expected_ttl_count=2
            )

        self.assertFalse(drained)
        self.assertEqual(self.status()["state"], "capture_drain_timeout")


    def test_free_running_serial_logging_does_not_request_trigger_alignment(self) -> None:
        self.backend.launch_cfg = self.config(serial_enabled=True, trigger_on=False)

        with mock.patch.object(
            manager.finalizer, "run_capture_finalizer", return_value=0
        ) as run_finalizer:
            manager.OperatorBackend._run_capture_finalizer(
                self.backend, self.run_dir
            )

        self.assertFalse(run_finalizer.call_args.kwargs["enable_align"])


    def test_trigger_timeout_never_sends_start(self) -> None:
        FakeSerialHandle.instances.clear()
        self.patches.extend(
            [
                mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
                mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
                mock.patch.object(self.backend._inference_ready, "wait", return_value=False),
            ]
        )
        for patcher in self.patches[-3:]:
            patcher.start()

        result = self.backend.start_run(self.config(serial_enabled=True, trigger_on=True))

        self.assertFalse(result)
        serial_handle = FakeSerialHandle.instances[-1]
        self.assertNotIn("START_SENT", serial_handle.markers)
        self.assertFalse(any(line.startswith("START,") for line in serial_handle.sent))
        self.assertIn("STOP", serial_handle.sent)
        self.assertTrue(serial_handle.closed)
        self.assertEqual(self.status()["state"], "failed")
        self.assertIn("controller was not started", self.status()["error"])


    def test_requested_serial_fails_closed_when_pyserial_is_unavailable(self) -> None:
        with mock.patch.object(manager.serial_util, "have_pyserial", return_value=False):
            result = self.backend.start_run(self.config(serial_enabled=True))

        self.assertFalse(result)
        self.assertTrue(self.backend.launch_cfg.serial_enabled)
        self.assertIsNone(self.backend.state.inference)
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("pyserial is not installed", status["error"])
        self.assertEqual(self.failures, [status["error"]])


    def test_runtime_serial_integrity_failure_stops_and_fails_run(self) -> None:
        FakeSerialHandle.instances.clear()
        with (
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
        ):
            self.assertTrue(self.backend.start_run(self.config(serial_enabled=True)))
            serial_handle = FakeSerialHandle.instances[-1]

            serial_handle.fail("serial CSV write/flush failed: disk full")

            self.assertTrue(self.failure_event.wait(timeout=2.0))

        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("serial acquisition integrity failed", status["error"])
        self.assertIn("disk full", status["error"])
        self.assertIn("STOP", serial_handle.sent)
        self.assertTrue(serial_handle.closed)
        self.assertEqual(self.handle.terminate_calls, 1)


    def test_serial_failure_plan_is_gated_and_passed_to_controller(self) -> None:
        plan_path = self.root / "project/qualification/serial-failure.json"
        plan_path.write_text(
            '{"schema_version":"1.0","target":"serial_controller",'
            '"kind":"read_error","after_frames":3}'
        )
        FakeSerialHandle.instances.clear()
        with (
            mock.patch.dict(
                manager.os.environ,
                {"SQUEAKVIEW_ENABLE_FAILURE_INJECTION": "1"},
            ),
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
        ):
            self.assertTrue(
                self.backend.start_run(
                    self.config(serial_enabled=True, failure_plan=plan_path)
                )
            )

        injected = FakeSerialHandle.instances[-1].failure_plan
        self.assertIsNotNone(injected)
        self.assertEqual(injected.target, "serial_controller")
        self.assertFalse(self.status()["production_eligible"])


    def test_controller_start_write_failure_aborts_with_failed_epoch_marker(self) -> None:
        FakeSerialHandle.instances.clear()
        serial_handle = FakeSerialHandle("/dev/test", 115200, self.logs.append)

        def send_line(line: str) -> None:
            if line.startswith("START,"):
                raise RuntimeError("USB write failed")
            serial_handle.sent.append(line)

        serial_handle.send_line = send_line
        with (
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", return_value=serial_handle),
            mock.patch.object(self.backend._inference_ready, "wait", return_value=True),
        ):
            result = self.backend.start_run(
                self.config(serial_enabled=True, trigger_on=True)
            )

        self.assertFalse(result)
        self.assertIn("START_SENT", serial_handle.markers)
        self.assertIn("STOP", serial_handle.sent)
        self.assertTrue(serial_handle.closed)
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("USB write failed", status["error"])


    def test_missing_first_ttl_aborts_triggered_run(self) -> None:
        FakeSerialHandle.instances.clear()

        def pipeline_ready(*, timeout: float) -> bool:
            self.assertGreater(timeout, 0)
            self.backend._inference_emit(
                encode_child_event("pipeline_ready", run_dir=str(self.run_dir))
            )
            return True

        def no_first_ttl(_handle, *, timeout_s: float) -> bool:
            self.assertEqual(timeout_s, 3.0)
            self.assertEqual(self.backend.snapshot().phase, RunPhase.STARTING)
            self.assertEqual(self.status()["state"], "starting")
            self.assertEqual(self.started, [])
            return False

        with (
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
            mock.patch.object(self.backend._inference_ready, "wait", side_effect=pipeline_ready),
            mock.patch.object(FakeSerialHandle, "wait_for_ttl", no_first_ttl),
        ):
            result = self.backend.start_run(
                self.config(serial_enabled=True, trigger_on=True)
            )

        self.assertFalse(result)
        serial_handle = FakeSerialHandle.instances[-1]
        self.assertTrue(any(line.startswith("START,") for line in serial_handle.sent))
        self.assertIn("START_SENT", serial_handle.markers)
        self.assertIn("START_TTL_TIMEOUT", serial_handle.markers)
        self.assertIn("STOP", serial_handle.sent)
        self.assertTrue(serial_handle.closed)
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("no camera TTL was detected", status["error"])
        self.assertEqual(self.started, [])


    def test_successful_controller_handshake_starts_triggered_run(self) -> None:
        FakeSerialHandle.instances.clear()

        def pipeline_ready(*, timeout: float) -> bool:
            self.assertGreater(timeout, 0)
            self.backend._inference_emit(
                encode_child_event("pipeline_ready", run_dir=str(self.run_dir))
            )
            return True

        def first_ttl(_handle, *, timeout_s: float) -> bool:
            self.assertEqual(timeout_s, 3.0)
            self.assertEqual(self.backend.snapshot().phase, RunPhase.STARTING)
            self.assertEqual(self.status()["state"], "starting")
            self.assertEqual(self.started, [])
            return True

        with (
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
            mock.patch.object(self.backend._inference_ready, "wait", side_effect=pipeline_ready),
            mock.patch.object(FakeSerialHandle, "wait_for_ttl", first_ttl),
        ):
            result = self.backend.start_run(
                self.config(serial_enabled=True, trigger_on=True)
            )

        self.assertTrue(result)
        serial_handle = FakeSerialHandle.instances[-1]
        self.assertIn("START,30", serial_handle.sent)
        self.assertIn("START_SENT", serial_handle.markers)
        self.assertNotIn("START_TTL_TIMEOUT", serial_handle.markers)
        self.assertFalse(serial_handle.closed)
        self.assertIs(self.backend.state.inference, self.handle)
        self.assertEqual(self.backend.snapshot().phase, RunPhase.RECORDING)
        self.assertEqual(self.status()["state"], "recording")
        self.assertEqual(self.started, [True])


    def test_experimental_watchdog_uses_negotiate_arm_and_disarm_path(self) -> None:
        FakeSerialHandle.instances.clear()

        def pipeline_ready(*, timeout: float) -> bool:
            self.backend._inference_emit(
                encode_child_event("pipeline_ready", run_dir=str(self.run_dir))
            )
            return True

        with (
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
            mock.patch.object(
                self.backend._inference_ready, "wait", side_effect=pipeline_ready
            ),
        ):
            result = self.backend.start_run(
                self.config(
                    serial_enabled=True,
                    trigger_on=True,
                    controller_protocol="watchdog_v1_experimental",
                    controller_watchdog_lease_ms=900,
                )
            )

        self.assertTrue(result)
        serial_handle = FakeSerialHandle.instances[-1]
        self.assertIn("HELLO_TEST,900,2.0", serial_handle.sent)
        self.assertIn("ARM_TEST,30,2.0", serial_handle.sent)
        self.assertNotIn("START,30", serial_handle.sent)
        self.assertIn("WATCHDOG_V1_NEGOTIATED", serial_handle.markers)
        self.assertIn("WATCHDOG_V1_ARM_ACKED", serial_handle.markers)

        self.backend.stop_run()
        self.assertIn("DISARM_TEST,2.0", serial_handle.sent)
        self.assertNotIn("STOP", serial_handle.sent)
        self.assertIn("WATCHDOG_V1_DISARM_ACKED", serial_handle.markers)


    def test_serial_permission_error_is_preserved_for_gui(self) -> None:
        denied_handle = mock.Mock()
        denied_handle.open.return_value = False
        denied_handle.last_error = "[Errno 13] Permission denied: '/dev/ttyACM0'"
        serial_patches = [
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", return_value=denied_handle),
        ]
        self.patches.extend(serial_patches)
        for patcher in serial_patches:
            patcher.start()

        result = self.backend.start_run(self.config(serial_enabled=True, serial_port="/dev/ttyACM0"))

        self.assertFalse(result)
        error = self.status()["error"]
        self.assertIn("/dev/ttyACM0", error)
        self.assertIn("Permission denied", error)
        self.assertIn("bash scripts/setup_jetson.sh", error)
        self.assertIn("reboot", error.lower())
        self.assertEqual(self.failures, [error])


