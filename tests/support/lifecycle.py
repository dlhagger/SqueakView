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


def _runtime_context(root: Path) -> RuntimeContext:
    app_root = root / "app"
    project_root = root / "project"
    app_root.mkdir(exist_ok=True)
    project_root.mkdir(exist_ok=True)
    (project_root / "runs").mkdir(exist_ok=True)
    (project_root / "models").mkdir(exist_ok=True)
    (project_root / "tasks").mkdir(exist_ok=True)
    (project_root / "qualification").mkdir(exist_ok=True)
    return RuntimeContext(
        app=AppPaths.from_root(app_root),
        project=Project(
            paths=ProjectPaths.from_existing_root(project_root),
            metadata=ProjectMetadata.create("Test"),
        ),
        user=UserPaths(
            config=root / "user/config",
            state=root / "user/state",
            runtime=root / "user/runtime",
            projects_parent=root / "projects",
        ),
    )



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

    def send_behavior_start(self, *, timeout_s: float = 3.0) -> None:
        del timeout_s
        self.log_marker("START_BEHAVIOR_SENT")
        self.send_line("START_BEHAVIOR")

    def wait_for_ttl(self, timeout_s: float = 3.0) -> bool:
        del timeout_s
        return True

    def wait_for_stop_ack(self, timeout_s: float = 2.0) -> bool:
        del timeout_s
        return True

    def wait_for_camera_stop(self, timeout_s: float = 3.0) -> bool:
        del timeout_s
        return True

    def negotiate_protocol_v2(self, *, timeout_s: float = 3.0) -> None:
        self.sent.append(f"PROTO_V2_TEST,{timeout_s}")

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



class BackendLifecycleCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.task_cfg = self.root / "project" / "tasks" / "task.yaml"
        self.task_cfg.parent.mkdir(parents=True)
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
            runtime_context=_runtime_context(self.root),
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
            if _cfg.serial_enabled and not _cfg.trigger_on:
                self.backend._inference_ready.set()
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
            mock.patch.object(
                manager.clock_validation,
                "host_time_status",
                return_value={
                    "ntp_synchronized": True,
                    "timezone": "UTC",
                    "checked_utc": "2026-09-16T12:00:00+00:00",
                },
            ),
            mock.patch.object(
                manager.clock_validation,
                "validate_clock",
                return_value={
                    "schema_version": "1.0",
                    "result": "PASS",
                    "reason": "CLOCK_WITHIN_TOLERANCE",
                    "validation_state": "WITHIN_TOLERANCE",
                    "host": {"ntp_synchronized": True, "timezone": "UTC"},
                    "before": {
                        "all_rtc_valid": True,
                        "median_offset_seconds": 0.1,
                        "median_round_trip_ms": 1.0,
                    },
                    "correction_requested": False,
                    "correction_applied": False,
                    "completed_utc": "2026-09-16T12:00:01+00:00",
                },
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


    def _inference_model_fixture(self):
        package = self.root / "project/models/selected"
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


