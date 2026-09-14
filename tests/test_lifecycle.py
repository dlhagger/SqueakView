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


class FakeProcessHandle:
    def __init__(self) -> None:
        self.running = True
        self.terminate_calls = 0
        self.returncode = 0

    def is_running(self) -> bool:
        return self.running

    def terminate_group_graceful(self, *_args, **_kwargs) -> None:
        self.terminate_calls += 1
        self.running = False

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode


class FakeHungFinalizer:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("post-run", timeout)
        return self.returncode


class FakeSerialHandle:
    instances: list["FakeSerialHandle"] = []

    def __init__(
        self, port: str, baud: int, emit_fn, on_fatal=None, failure_plan=None
    ) -> None:
        self.port = port
        self.baud = baud
        self.emit_fn = emit_fn
        self.on_fatal = on_fatal
        self.failure_plan = failure_plan
        self.fatal_error: str | None = None
        self.sent: list[str] = []
        self.markers: list[str] = []
        self.closed = False
        self.__class__.instances.append(self)

    def open(self, _run_dir: Path | None = None) -> bool:
        return True

    def set_csv_path(self, _run_dir: Path) -> bool:
        return True

    def log_marker(self, marker: str) -> None:
        self.markers.append(marker)

    def send_line(self, text: str) -> None:
        self.sent.append(text)

    def send_start(self, fps: int) -> None:
        self.log_marker("START_SENT")
        self.send_line(f"START,{fps}")

    def wait_for_ttl(self, timeout_s: float = 3.0) -> bool:
        del timeout_s
        return True

    def wait_for_stop_ack(self, timeout_s: float = 2.0) -> bool:
        del timeout_s
        return True

    def negotiate_watchdog_v1(
        self, *, requested_lease_ms: int, timeout_s: float = 2.0
    ) -> object:
        self.sent.append(f"HELLO_TEST,{requested_lease_ms},{timeout_s}")
        return object()

    def arm_watchdog_v1(self, fps: int, *, timeout_s: float = 2.0) -> None:
        self.sent.append(f"ARM_TEST,{fps},{timeout_s}")

    def disarm_watchdog_v1(self, *, timeout_s: float = 2.0) -> None:
        self.sent.append(f"DISARM_TEST,{timeout_s}")

    @property
    def watchdog_snapshot(self) -> dict[str, object] | None:
        return None

    def close(self) -> None:
        self.closed = True

    def fail(self, message: str) -> None:
        self.fatal_error = message
        if self.on_fatal is not None:
            self.on_fatal(message)


class BackendTimeoutPolicyTests(unittest.TestCase):
    def test_inference_readiness_timeout_is_finite_and_bounded(self) -> None:
        for raw in ("nan", "inf", "-inf", "0", "601", "invalid"):
            with self.subTest(raw=raw), mock.patch.dict(
                manager.os.environ,
                {"SQUEAKVIEW_INFERENCE_READY_TIMEOUT": raw},
            ):
                self.assertEqual(manager._bounded_ready_timeout(), 30.0)


class ManifestPersistenceTests(unittest.TestCase):
    def test_post_run_bottle_save_preserves_immutable_manifest_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            original = {
                "schema_version": "2.0",
                "run_id": "run",
                "created_at": "2026-07-28T17:41:50",
                "updated_at": "2026-07-29T09:37:57",
                "git": {"commit": "original", "dirty": False},
                "storage": {"free_bytes": 123},
                "capture": {"fps": 30, "width": 1440, "height": 1080},
                "inference": {
                    "enabled": True,
                    "model_package": {"name": "humans", "config": "original-config"},
                },
                "bottles": {"complete": False},
                "actual_outputs": {"has_analysis": True},
            }
            run_context.write_manifest(run_dir, original)
            run_context.atomic_write_json(
                run_dir / run_context.RUN_STATUS_FILENAME,
                {"state": "finalized"},
            )
            backend = manager.OperatorBackend(lambda _message: None)
            backend.state.run_dir = run_dir

            summary = backend.save_bottle_measurements(
                {
                    "left": {
                        "fluid": "water",
                        "initial_weight_g": 10,
                        "final_weight_g": 9,
                    },
                    "right": {
                        "fluid": "water",
                        "initial_weight_g": 12,
                        "final_weight_g": 11,
                    },
                },
                run_dir=run_dir,
            )

            manifest = run_context.read_json(
                run_dir / run_context.RUN_MANIFEST_FILENAME
            )
            self.assertTrue(summary["complete"])
            self.assertEqual(manifest["created_at"], original["created_at"])
            self.assertEqual(manifest["git"], original["git"])
            self.assertEqual(manifest["storage"], original["storage"])
            self.assertEqual(manifest["capture"], original["capture"])
            self.assertEqual(manifest["inference"], original["inference"])
            self.assertTrue(manifest["bottles"]["complete"])
            self.assertTrue(manifest["actual_outputs"]["bottle_measurements_complete"])


class BackendLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.task_cfg = self.root / "task.yaml"
        self.task_cfg.write_text("task_name: test\n")
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        run_context.atomic_write_json(
            self.run_dir / "run_status.json",
            {
                "recording_validation": {"passed": True},
                "acquisition_integrity": {"passed": True},
            },
        )
        self.logs: list[str] = []
        self.started: list[bool] = []
        self.failures: list[str] = []
        self.failure_event = threading.Event()

        def on_run_failed(message: str) -> None:
            self.failures.append(message)
            self.failure_event.set()

        self.backend = manager.OperatorBackend(
            self.logs.append,
            on_run_started=lambda: self.started.append(True),
            on_run_failed=on_run_failed,
        )
        self.backend._acquisition_lock = manager.AcquisitionLock(
            self.root / ".acquisition.lock"
        )
        self.handle = FakeProcessHandle()
        self.exit_callback = None
        self.spawn_kwargs = None

        def spawn(_cfg, emit, on_exit=None, **kwargs):
            del emit
            self.exit_callback = on_exit
            self.spawn_kwargs = kwargs
            return self.handle

        def finalize(run_dir: Path) -> int:
            if self.backend.launch_cfg.serial_enabled:
                run_context.update_status(run_dir, alignment_validated=True)
            return 0

        self.patches = [
            mock.patch.object(
                manager.preflight,
                "run_preflight",
                return_value=manager.preflight.PreflightResult(
                    True,
                    "Preflight passed",
                    "[PASS] ffprobe is installed (/usr/bin/ffprobe)\n"
                    "[PASS] DeepStream new nvstreammux VIC/NVMM path works\n"
                    "[PASS] Jetson H.264 full-decode validation path works "
                    "(decoded 1 frame(s) with h264_nvv4l2dec)\n"
                    "[PASS] Automatic desktop suspend on AC power is disabled",
                ),
            ),
            mock.patch.object(manager.run_context, "assert_runs_dir_ready", return_value={"free_bytes": 10_000}),
            mock.patch.object(manager.run_context, "create_run_dir", return_value=(self.run_dir, self.run_dir.name)),
            mock.patch.object(manager.process, "spawn_inference", side_effect=spawn),
            mock.patch.object(self.backend, "_write_run_manifest"),
            mock.patch.object(self.backend, "_write_bottle_measurements", return_value={}),
            mock.patch.object(self.backend, "_ensure_metadata"),
            mock.patch.object(self.backend, "_run_capture_finalizer", side_effect=finalize),
            mock.patch.object(self.backend, "_run_output_snapshot", return_value={}),
            mock.patch.object(manager.time, "sleep"),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self) -> None:
        self.backend._acquisition_lock.release()
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temp_dir.cleanup()

    def config(self, **updates) -> process.LaunchConfig:
        values = {
            "task_cfg": self.task_cfg,
            "serial_enabled": False,
            "inference_enabled": False,
        }
        values.update(updates)
        return process.LaunchConfig(**values)

    def status(self) -> dict:
        return run_context.read_json(self.run_dir / run_context.RUN_STATUS_FILENAME)

    def test_serial_dashboard_line_is_parsed_once_into_typed_event(self) -> None:
        received: list[DashboardEvent] = []
        backend = manager.OperatorBackend(self.logs.append, received.append)

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

    def _inference_model_fixture(self):
        package = self.root / "models/selected"
        config = package / "configs/selected.txt"
        parser = package / "lib/parser.so"
        config.parent.mkdir(parents=True)
        parser.parent.mkdir(parents=True)
        parser.write_bytes(b"package parser")
        for path in (
            package / "labels/classes.txt",
            package / "labels/keypoints.txt",
            package / "onnx/model.onnx",
            package / "engines/model.engine",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode())
        config.write_text(
            "[property]\n"
            "onnx-file=../onnx/model.onnx\n"
            "model-engine-file=../engines/model.engine\n"
            "labelfile-path=../labels/classes.txt\n"
            "custom-lib-path=../lib/parser.so\n"
            "batch-size=1\n"
        )
        config.with_name("selected.pose.json").write_text(
            '{"keypoint_labels_path":"../labels/keypoints.txt"}\n'
        )
        selected = SimpleNamespace(
            name="selected",
            config=config,
            engine_build_identity={"validated": True},
            manifest_snapshot=lambda: {
                "name": "selected",
                "config": str(config),
                "pose_sidecar": str(config.with_name("selected.pose.json")),
                "classes": str(package / "labels/classes.txt"),
                "keypoint_labels": str(package / "labels/keypoints.txt"),
                "onnx": str(package / "onnx/model.onnx"),
                "engine": str(package / "engines/model.engine"),
                "parser_library": str(parser),
            },
        )
        return package, config, parser, selected

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

    def test_spawn_failure_marks_created_run_failed(self) -> None:
        with mock.patch.object(manager.process, "spawn_inference", side_effect=OSError("spawn denied")):
            result = self.backend.start_run(self.config())

        self.assertFalse(result)
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("spawn denied", status["error"])
        self.assertEqual(len(self.failures), 1)

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

    def test_invalid_model_is_rejected_before_run_creation(self) -> None:
        missing_config = self.root / "models" / "missing" / "configs" / "missing.txt"
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

    def test_stop_finalizes_even_when_child_already_exited(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))
        self.handle.running = False

        self.backend.stop_run()
        self.backend.stop_run()

        self.assertEqual(self.status()["state"], "finalized")
        self.assertEqual(self.handle.terminate_calls, 0)

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

    def test_triggered_run_generates_alignment_during_finalization(self) -> None:
        self.backend.launch_cfg = self.config(serial_enabled=True, trigger_on=True)
        with mock.patch.object(
            manager.finalizer, "run_capture_finalizer", return_value=0
        ) as run_finalizer:
            manager.OperatorBackend._run_capture_finalizer(
                self.backend, self.run_dir
            )

        self.assertTrue(run_finalizer.call_args.kwargs["enable_align"])

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
        plan_path = self.root / "serial-failure.json"
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

    def test_shutdown_stop_ack_timeout_is_deterministic_and_fails_run(self) -> None:
        plan_path = self.root / "shutdown-failure.json"
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
        plan_path = self.root / "capture-exit-failure.json"
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


class ProcessHandleTests(unittest.TestCase):
    def test_wait_does_not_convert_timeout_into_unknown_success(self) -> None:
        handle = process.ProcessHandle.__new__(process.ProcessHandle)
        handle.p = mock.Mock()
        handle.p.wait.side_effect = subprocess.TimeoutExpired("capture", 1.0)

        with self.assertRaises(subprocess.TimeoutExpired):
            handle.wait(timeout=1.0)

    def test_preview_socket_paths_are_short_unique_and_per_camera(self) -> None:
        paths = process.preview_socket_paths(
            Path("/a/very/long/run/directory/that/cannot/be/a/unix/socket/path"),
            2,
        )

        self.assertEqual(len(paths), 2)
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(path.parent == Path("/tmp") for path in paths))
        self.assertTrue(all(len(str(path).encode()) < 108 for path in paths))

    def test_spawn_inference_passes_ipc_sockets_not_window_id(self) -> None:
        root = Path("/tmp/squeakview-process-test")
        cfg = process.LaunchConfig(
            ds_cfg=None,
            inference_enabled=False,
            num_cameras=2,
            preview_window_id=12345,
            run_dir=root,
        )
        sentinel = object()
        with mock.patch.object(process, "_spawn", return_value=sentinel) as spawn:
            result = process.spawn_inference(cfg, lambda _line: None)

        self.assertIs(result, sentinel)
        args = spawn.call_args.args[1]
        self.assertNotIn("--window-xid", args)
        self.assertEqual(args.count("--preview-socket"), 2)
        for socket_path in process.preview_socket_paths(root, 2):
            self.assertIn(str(socket_path), args)

    def test_spawn_inference_can_omit_preview_branch_for_matrix_run(self) -> None:
        cfg = process.LaunchConfig(
            ds_cfg=None,
            inference_enabled=False,
            preview_enabled=False,
            run_dir=Path("/tmp/squeakview-no-preview"),
        )
        with mock.patch.object(process, "_spawn", return_value=object()) as spawn:
            process.spawn_inference(cfg, lambda _line: None)

        self.assertNotIn("--preview-socket", spawn.call_args.args[1])

    def test_spawn_inference_localizes_config_without_mutating_frozen_request(self) -> None:
        original = Path("configs/source.txt")
        localized = Path("/tmp/squeakview-process-test/config/source.txt")
        cfg = process.LaunchConfig(ds_cfg=original, run_dir=localized.parents[1])

        with (
            mock.patch.object(
                process.squeakview_config,
                "resolve_workspace_path",
                return_value=original,
            ),
            mock.patch.object(
                process,
                "_localize_deepstream_config",
                return_value=localized,
            ),
            mock.patch.object(process, "_spawn", return_value=object()) as spawn,
        ):
            process.spawn_inference(cfg, lambda _line: None)

        self.assertEqual(cfg.ds_cfg, original)
        args = spawn.call_args.args[1]
        self.assertEqual(args[args.index("--cfg") + 1], str(localized))

    def test_spawn_inference_forwards_failure_plan(self) -> None:
        plan = Path("/tmp/squeakview-failure-plan.json")
        cfg = process.LaunchConfig(
            ds_cfg=None,
            inference_enabled=False,
            failure_plan=plan,
        )
        with mock.patch.object(process, "_spawn", return_value=object()) as spawn:
            process.spawn_inference(cfg, lambda _line: None)

        args = spawn.call_args.args[1]
        self.assertEqual(args[args.index("--failure-plan") + 1], str(plan))

    def test_debug_profile_enables_nvidia_latency_environment(self) -> None:
        cfg = process.LaunchConfig(
            ds_cfg=None,
            inference_enabled=False,
            run_dir=Path("/tmp/squeakview-profile-test"),
        )
        logs: list[str] = []
        with (
            mock.patch.dict(
                process.os.environ,
                {"SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE": "1"},
            ),
            mock.patch.object(process, "_spawn", return_value=object()) as spawn,
        ):
            process.spawn_inference(cfg, logs.append)

        extra_env = spawn.call_args.kwargs["extra_env"]
        self.assertEqual(extra_env["NVDS_ENABLE_LATENCY_MEASUREMENT"], "1")
        self.assertEqual(extra_env["USE_NEW_NVSTREAMMUX"], "yes")
        self.assertEqual(
            extra_env["NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT"], "1"
        )
        self.assertTrue(any("debug latency profiling enabled" in line for line in logs))
        self.assertEqual(
            spawn.call_args.kwargs["output_log_path"],
            cfg.run_dir / "diagnostics" / "deepstream.log",
        )

    def test_process_handle_bounds_run_local_child_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "diagnostics" / "deepstream.log"
            exited = threading.Event()
            child = subprocess.Popen(
                [sys.executable, "-c", "print('first'); print('x' * 100)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            emitted: list[str] = []
            process.ProcessHandle(
                "test",
                child,
                emitted.append,
                on_exit=lambda _code: exited.set(),
                output_log_path=log_path,
                output_log_max_bytes=70,
            )

            self.assertTrue(exited.wait(timeout=5))
            self.assertLessEqual(log_path.stat().st_size, 70)
            self.assertIn("first", log_path.read_text())
            self.assertTrue(any("size limit" in line for line in emitted))

    def test_exit_callback_receives_child_return_code(self) -> None:
        exited = threading.Event()
        returncodes: list[int] = []
        child = subprocess.Popen(
            [sys.executable, "-c", "print('child output'); raise SystemExit(3)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def on_exit(returncode: int) -> None:
            returncodes.append(returncode)
            exited.set()

        process.ProcessHandle("test", child, lambda _line: None, on_exit=on_exit)

        self.assertTrue(exited.wait(timeout=5))
        self.assertEqual(returncodes, [3])


if __name__ == "__main__":
    unittest.main()
