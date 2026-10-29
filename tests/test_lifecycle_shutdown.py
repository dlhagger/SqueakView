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

class LifecycleShutdownTests(BackendLifecycleCase):
    def test_shutdown_coordinator_exception_fails_and_releases_lock(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))
        self.assertTrue(self.backend._acquisition_lock.held)

        with mock.patch.object(
            manager.lifecycle,
            "finalize_run",
            side_effect=RuntimeError("unexpected boundary failure"),
        ):
            self.backend.stop_run()

        self.assertFalse(self.backend._acquisition_lock.held)
        self.assertEqual(self.backend.snapshot().phase, RunPhase.FAILED)
        self.assertEqual(self.status()["state"], "failed")
        self.assertIn("shutdown coordinator failed", self.failures[-1])


    def test_stop_finalizes_even_when_child_already_exited(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))
        self.handle.running = False

        self.backend.stop_run()
        self.backend.stop_run()

        self.assertEqual(self.status()["state"], "finalized")
        self.assertEqual(self.handle.terminate_calls, 0)


    def test_terminal_status_persistence_failure_is_reported_fail_closed(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))
        original_write_status = run_context.write_status

        def fail_finalized(run_dir: Path, state: str, **updates):
            if state == "finalized":
                raise OSError("terminal status disk full")
            return original_write_status(run_dir, state, **updates)

        with mock.patch.object(
            manager.run_context,
            "write_status",
            side_effect=fail_finalized,
        ):
            self.backend.stop_run()

        self.assertEqual(self.backend.snapshot().phase, RunPhase.FAILED)
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("final run metadata could not be persisted", status["error"])
        self.assertIn("terminal status disk full", status["error"])


    def test_stop_marks_failed_when_recording_validation_fails(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))
        run_context.update_status(
            self.run_dir, recording_validation={"passed": False}
        )

        self.backend.stop_run()

        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("recording frame-count validation failed", status["error"])
        self.assertEqual(self.failures, [status["error"]])


    def test_stop_marks_failed_when_inference_process_exits_nonzero(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))
        self.handle.returncode = 9

        self.backend.stop_run()

        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("inference exit code 9", status["error"])


    def test_stop_skips_validation_when_capture_exit_is_not_confirmed(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))
        self.handle.wait = mock.Mock(
            side_effect=subprocess.TimeoutExpired("capture", 2.0)
        )

        with mock.patch.object(self.backend, "_run_capture_finalizer") as finalizer:
            self.backend.stop_run()

        finalizer.assert_not_called()
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("did not exit", status["error"])


    def test_capture_drain_waits_for_quiet_source_and_admission_ledgers(self) -> None:
        (self.run_dir / "capture_cam0.jsonl").write_text(
            '{"source_sequence_index": 122}\n'
        )
        (self.run_dir / "record_admission.csv").write_text(
            "stream_id,record_frame_index,pts_ns\n0,122,100\n"
        )
        self.backend.launch_cfg = self.config(num_cameras=1)

        with mock.patch.dict(
            manager.os.environ,
            {
                "SQUEAKVIEW_CAPTURE_DRAIN_QUIET_S": "0.1",
                "SQUEAKVIEW_CAPTURE_DRAIN_TIMEOUT_S": "0.5",
            },
        ):
            drained = self.backend._wait_for_capture_drain(
                self.run_dir, expected_ttl_count=123
            )

        self.assertTrue(drained)
        status = self.status()
        self.assertEqual(status["state"], "capture_drained")
        self.assertEqual(status["expected_ttl_count"], 123)
        self.assertEqual(status["ledger_frame_counts"], [123, 123])


    def test_stop_marks_failed_when_independent_finalizer_fails(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))
        with mock.patch.object(self.backend, "_run_capture_finalizer", return_value=1):
            self.backend.stop_run()

        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("post-run finalizer exit code 1", status["error"])


    def test_hung_finalizer_is_killed_and_stop_cannot_claim_success(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))
        worker = FakeHungFinalizer()
        signals: list[signal.Signals] = []

        def kill_group(_pid: int, sig: signal.Signals) -> None:
            signals.append(sig)
            if sig == signal.SIGKILL:
                worker.returncode = -int(signal.SIGKILL)

        self.backend._run_capture_finalizer.side_effect = (
            lambda run_dir: manager.OperatorBackend._run_capture_finalizer(
                self.backend, run_dir
            )
        )
        with (
            mock.patch.object(manager.process, "spawn_post_run", return_value=worker),
            mock.patch.object(manager.os, "killpg", side_effect=kill_group),
            mock.patch.dict(
                manager.os.environ,
                {
                    "SQUEAKVIEW_POST_RUN_TIMEOUT_S": "0.1",
                    "SQUEAKVIEW_POST_RUN_TERMINATE_GRACE_S": "0.1",
                    "SQUEAKVIEW_POST_RUN_KILL_GRACE_S": "0.1",
                },
            ),
        ):
            self.backend.stop_run()

        status = self.status()
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(worker.wait_timeouts, [0.1, 0.1])
        self.assertEqual(status["state"], "failed")
        self.assertIn("post-run finalizer timed out", status["error"])
        self.assertIn(
            "finalization_failed",
            [entry["state"] for entry in status["history"]],
        )
        self.assertNotIn("finalized", [entry["state"] for entry in status["history"]])


    def test_timed_out_finalizer_returning_zero_after_sigterm_still_fails(self) -> None:
        worker = FakeHungFinalizer(pid=4343)

        def terminate_group(_pid: int, sig: signal.Signals) -> None:
            self.assertEqual(sig, signal.SIGTERM)
            worker.returncode = 0

        with (
            mock.patch.object(manager.process, "spawn_post_run", return_value=worker),
            mock.patch.object(manager.os, "killpg", side_effect=terminate_group),
            mock.patch.object(manager.time, "monotonic", side_effect=[10.0, 11.0]),
            mock.patch.dict(
                manager.os.environ,
                {"SQUEAKVIEW_POST_RUN_TIMEOUT_S": "0.1"},
            ),
        ):
            returncode = manager.OperatorBackend._run_capture_finalizer(
                self.backend, self.run_dir
            )

        self.assertEqual(returncode, 124)
        self.assertEqual(worker.wait_timeouts, [30.0])
        self.assertEqual(self.status()["state"], "finalization_failed")


    def test_triggered_legacy_run_defers_alignment_to_explicit_analysis(self) -> None:
        self.backend.launch_cfg = self.config(
            serial_enabled=True,
            trigger_on=True,
            controller_protocol="legacy_v1",
        )
        with mock.patch.object(
            manager.finalizer, "run_capture_finalizer", return_value=0
        ) as run_finalizer:
            manager.OperatorBackend._run_capture_finalizer(
                self.backend, self.run_dir
            )

        self.assertFalse(run_finalizer.call_args.kwargs["enable_align"])


    def test_triggered_v2_run_skips_legacy_alignment_during_finalization(self) -> None:
        self.backend.launch_cfg = self.config(
            serial_enabled=True,
            trigger_on=True,
            controller_protocol="v2",
        )
        with mock.patch.object(
            manager.finalizer, "run_capture_finalizer", return_value=0
        ) as run_finalizer:
            manager.OperatorBackend._run_capture_finalizer(
                self.backend, self.run_dir
            )

        self.assertFalse(run_finalizer.call_args.kwargs["enable_align"])


    def test_shutdown_stop_ack_timeout_is_deterministic_and_fails_run(self) -> None:
        plan_path = self.root / "project/qualification/shutdown-failure.json"
        plan_path.write_text(
            '{"schema_version":"1.0","target":"shutdown",'
            '"kind":"stop_ack_timeout","after_frames":1}'
        )
        FakeSerialHandle.instances.clear()
        with (
            mock.patch.dict(
                manager.os.environ,
                {"SQUEAKVIEW_ENABLE_FAILURE_INJECTION": "1"},
            ),
            mock.patch.object(manager.serial_util, "have_pyserial", return_value=True),
            mock.patch.object(manager.serial_util, "SerialHandle", FakeSerialHandle),
            mock.patch.object(FakeSerialHandle, "wait_for_stop_ack") as wait_for_ack,
        ):
            self.assertTrue(
                self.backend.start_run(
                    self.config(serial_enabled=True, failure_plan=plan_path)
                )
            )
            self.backend.stop_run()

        wait_for_ack.assert_not_called()
        self.assertEqual(self.status()["state"], "failed")
        self.assertIn("STOP was not acknowledged", self.status()["error"])


    def test_shutdown_capture_exit_unconfirmed_skips_artifact_validation(self) -> None:
        plan_path = self.root / "project/qualification/capture-exit-failure.json"
        plan_path.write_text(
            '{"schema_version":"1.0","target":"shutdown",'
            '"kind":"capture_exit_unconfirmed","after_frames":1}'
        )
        self.handle.wait = mock.Mock(return_value=0)
        with mock.patch.dict(
            manager.os.environ,
            {"SQUEAKVIEW_ENABLE_FAILURE_INJECTION": "1"},
        ):
            self.assertTrue(
                self.backend.start_run(self.config(failure_plan=plan_path))
            )
            self.backend.stop_run()

        self.handle.wait.assert_not_called()
        self.backend._run_capture_finalizer.assert_not_called()
        self.assertEqual(self.status()["state"], "failed")
        self.assertIn("did not exit", self.status()["error"])


