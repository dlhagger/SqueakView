from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

from squeakview.apps.operator.backend import startup
from squeakview.apps.operator.backend.contracts import RunRequest


class _Capture:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def is_running(self) -> bool:
        self.events.append("capture-running")
        return True


class _Serial:
    last_error = None
    fatal_error = None

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def open(self, _run_dir: Path) -> bool:
        self.events.append("serial-open")
        return True

    def send_line(self, line: str) -> None:
        self.events.append(f"serial-send:{line}")

    def log_marker(self, marker: str) -> None:
        self.events.append(f"serial-marker:{marker}")

    def wait_for_ttl(self, *, timeout_s: float) -> bool:
        self.events.append(f"serial-ttl:{timeout_s}")
        return True

    def negotiate_watchdog_v1(
        self, *, requested_lease_ms: int, timeout_s: float
    ) -> object:
        self.events.append(f"serial-negotiate:{requested_lease_ms}:{timeout_s}")
        return object()


class StartupServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.task = self.root / "task.yaml"
        self.task.write_text("task_name: startup-test\n")
        self.run_dir = self.root / "run"
        self.events: list[str] = []
        self.capture = _Capture(self.events)
        self.serial = _Serial(self.events)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def hooks(self, *, persist_error: Exception | None = None, create_error: Exception | None = None) -> startup.StartupHooks:
        def create_run_dir(**_kwargs: object) -> tuple[Path, str]:
            self.events.append("create-run-dir")
            if create_error is not None:
                raise create_error
            self.run_dir.mkdir(exist_ok=True)
            return self.run_dir, self.run_dir.name

        def persist(_prepared: startup.PreparedRun) -> None:
            self.events.append("persist-prestart")
            if persist_error is not None:
                raise persist_error

        def start_controller(handle, fps: int) -> bool:
            handle.send_line(f"START,{fps}")
            handle.log_marker("START_SENT")
            self.events.append("controller-started")
            return True

        return startup.StartupHooks(
            log=lambda message: self.events.append(f"log:{message}"),
            resolve_task_path=lambda path: Path(path).resolve(),
            resolve_model_path=lambda path: Path(path).resolve(),
            resolve_failure_plan_path=lambda path: Path(path).resolve(),
            load_failure_plan=lambda _path: (_ for _ in ()).throw(AssertionError("unexpected failure plan")),
            validate_model=lambda _path: (_ for _ in ()).throw(AssertionError("unexpected model")),
            assert_storage_ready=lambda: self.events.append("storage") or {"free_bytes": 1_000_000},
            acquire_lock=lambda: self.events.append("lock-acquire") or True,
            release_lock=lambda: self.events.append("lock-release"),
            now_iso=lambda: "2026-09-08T12:00:00",
            device_context=lambda: self.events.append("device") or {"model": "Orin Nano"},
            set_fan_max=lambda: self.events.append("fan"),
            create_run_dir=create_run_dir,
            preview_socket_paths=lambda _run, _count: (Path("/tmp/preview.sock"),),
            initialize_runtime=lambda *_args: self.events.append("initialize"),
            establish_run=lambda _prepared: self.events.append("establish"),
            persist_prestart=persist,
            serial_available=lambda: self.events.append("serial-available") or True,
            create_serial=lambda _cfg, _plan: self.events.append("serial-create") or self.serial,
            set_serial=lambda handle: self.events.append(f"serial-set:{handle is not None}"),
            arm_serial_runtime=lambda: self.events.append("serial-arm"),
            spawn_capture=lambda _cfg: self.events.append("capture-spawn") or self.capture,
            set_capture=lambda handle: self.events.append(f"capture-set:{handle is not None}"),
            after_capture_spawn=lambda _run: self.events.append("capture-barrier"),
            run_finalized=lambda: False,
            finalize_failure=lambda error, terminate: self.events.append(f"finalize:{terminate}:{error}"),
            wait_ready=lambda timeout: self.events.append(f"ready-wait:{timeout}") or True,
            stop_requested=lambda: False,
            start_controller=start_controller,
            start_run_dir_watch=lambda: self.events.append("watch"),
            ready_timeout=lambda: 30.0,
            serial_open_failure_message=lambda *_args: "serial open failure",
        )

    def test_success_preserves_metadata_serial_capture_and_controller_order(self) -> None:
        result = startup.start_run(
            startup.StartupRequest(
                config=RunRequest(
                    task_cfg=self.task,
                    inference_enabled=False,
                    serial_enabled=True,
                    trigger_on=True,
                    fps=50,
                    arduino_fps=50,
                ),
                already_running=False,
            ),
            self.hooks(),
        )

        self.assertTrue(result.started)
        ordered = [
            "storage",
            "lock-acquire",
            "device",
            "initialize",
            "fan",
            "create-run-dir",
            "establish",
            "persist-prestart",
            "serial-available",
            "serial-create",
            "serial-set:True",
            "serial-open",
            "serial-arm",
            "capture-spawn",
            "capture-set:True",
            "ready-wait:30.0",
            "capture-running",
            "serial-send:START,50",
            "serial-marker:START_SENT",
            "controller-started",
            "serial-ttl:3.0",
            "watch",
        ]
        positions = [self.events.index(event) for event in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(self.events.count("storage"), 1)
        self.assertEqual(result.prepared.config.run_dir, self.run_dir)
        self.assertEqual(result.prepared.config.preview_socket_paths, (Path("/tmp/preview.sock"),))

    def test_experimental_watchdog_negotiates_before_capture_spawn(self) -> None:
        result = startup.start_run(
            startup.StartupRequest(
                RunRequest(
                    task_cfg=self.task,
                    inference_enabled=False,
                    serial_enabled=True,
                    trigger_on=True,
                    controller_protocol="watchdog_v1_experimental",
                    controller_watchdog_lease_ms=900,
                ),
                False,
            ),
            self.hooks(),
        )

        self.assertTrue(result.started)
        self.assertLess(
            self.events.index("serial-negotiate:900:2.0"),
            self.events.index("capture-spawn"),
        )
        self.assertIn("serial-marker:WATCHDOG_V1_NEGOTIATED", self.events)

    def test_experimental_watchdog_negotiation_failure_prevents_capture(self) -> None:
        self.serial.negotiate_watchdog_v1 = lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("no nonce-bound CAPS")
        )
        result = startup.start_run(
            startup.StartupRequest(
                RunRequest(
                    task_cfg=self.task,
                    inference_enabled=False,
                    serial_enabled=True,
                    trigger_on=True,
                    controller_protocol="watchdog_v1_experimental",
                ),
                False,
            ),
            self.hooks(),
        )
        self.assertFalse(result.started)
        self.assertIn("watchdog negotiation failed", result.error or "")
        self.assertNotIn("capture-spawn", self.events)

    def test_invalid_qualification_binding_fails_before_run_creation(self) -> None:
        hooks = replace(
            self.hooks(),
            resolve_qualification_case=lambda _config, _device: (
                (_ for _ in ()).throw(ValueError("preview_enabled mismatch"))
            ),
        )

        result = startup.start_run(
            startup.StartupRequest(
                config=RunRequest(
                    task_cfg=self.task,
                    inference_enabled=False,
                    serial_enabled=False,
                ),
                already_running=False,
            ),
            hooks,
        )

        self.assertFalse(result.started)
        self.assertIn("preview_enabled mismatch", result.error or "")
        self.assertNotIn("create-run-dir", self.events)
        self.assertNotIn("serial-open", self.events)
        self.assertNotIn("capture-spawn", self.events)
        self.assertIn("lock-release", self.events)

    def test_metadata_failure_finalizes_before_serial_or_capture(self) -> None:
        result = startup.start_run(
            startup.StartupRequest(
                RunRequest(task_cfg=self.task, inference_enabled=False, serial_enabled=True),
                already_running=False,
            ),
            self.hooks(persist_error=OSError("ENOSPC")),
        )

        self.assertFalse(result.started)
        self.assertIn("pre-start run metadata persistence failed", result.error)
        self.assertTrue(any(event.startswith("finalize:False:") for event in self.events))
        self.assertNotIn("serial-create", self.events)
        self.assertNotIn("capture-spawn", self.events)

    def test_run_directory_failure_releases_lock_without_starting_devices(self) -> None:
        result = startup.start_run(
            startup.StartupRequest(
                RunRequest(task_cfg=self.task, inference_enabled=False, serial_enabled=True),
                already_running=False,
            ),
            self.hooks(create_error=OSError("read-only filesystem")),
        )

        self.assertFalse(result.started)
        self.assertIn("failed to create run directory", result.error)
        self.assertIn("lock-release", self.events)
        self.assertNotIn("establish", self.events)
        self.assertNotIn("serial-create", self.events)
        self.assertNotIn("capture-spawn", self.events)

    def test_initialization_exception_releases_lock_before_run_creation(self) -> None:
        hooks = replace(
            self.hooks(),
            device_context=lambda: (_ for _ in ()).throw(RuntimeError("device read")),
        )

        result = startup.start_run(
            startup.StartupRequest(
                RunRequest(task_cfg=self.task, inference_enabled=False), False
            ),
            hooks,
        )

        self.assertFalse(result.started)
        self.assertIn("startup initialization failed", result.error)
        self.assertIn("lock-release", self.events)
        self.assertNotIn("create-run-dir", self.events)

    def test_serial_open_exception_enters_fail_closed_finalization(self) -> None:
        self.serial.open = lambda _run: (_ for _ in ()).throw(OSError("USB gone"))

        result = startup.start_run(
            startup.StartupRequest(
                RunRequest(
                    task_cfg=self.task,
                    inference_enabled=False,
                    serial_enabled=True,
                ),
                False,
            ),
            self.hooks(),
        )

        self.assertFalse(result.started)
        self.assertIn("serial controller open failed", result.error)
        self.assertTrue(any(event.startswith("finalize:False:") for event in self.events))
        self.assertNotIn("capture-spawn", self.events)

    def test_watcher_exception_stops_spawned_capture(self) -> None:
        hooks = replace(
            self.hooks(),
            start_run_dir_watch=lambda: (_ for _ in ()).throw(
                RuntimeError("thread unavailable")
            ),
        )

        result = startup.start_run(
            startup.StartupRequest(
                RunRequest(task_cfg=self.task, inference_enabled=False), False
            ),
            hooks,
        )

        self.assertFalse(result.started)
        self.assertIn("run-directory watcher failed", result.error)
        self.assertTrue(any(event.startswith("finalize:True:") for event in self.events))

    def test_inference_model_without_engine_identity_is_rejected_before_storage(self) -> None:
        hooks = self.hooks()
        hooks = startup.StartupHooks(
            **{
                field: getattr(hooks, field)
                for field in hooks.__dataclass_fields__
                if field != "validate_model"
            },
            validate_model=lambda _path: startup.ModelSelection(
                name="legacy-model",
                snapshot={"manifest_schema_version": "2"},
                engine_build_identity=None,
            ),
        )

        result = startup.start_run(
            startup.StartupRequest(
                RunRequest(
                    task_cfg=self.task,
                    inference_enabled=True,
                    ds_cfg=self.root / "model" / "config.txt",
                    serial_enabled=False,
                ),
                already_running=False,
            ),
            hooks,
        )

        self.assertFalse(result.started)
        self.assertIn("Project Setup", result.error)
        self.assertNotIn("storage", self.events)
        self.assertNotIn("lock-acquire", self.events)

    def test_post_spawn_barrier_failure_terminates_capture(self) -> None:
        hooks = self.hooks()
        hooks = startup.StartupHooks(
            **{
                field: getattr(hooks, field)
                for field in hooks.__dataclass_fields__
                if field != "after_capture_spawn"
            },
            after_capture_spawn=lambda _run: (_ for _ in ()).throw(
                RuntimeError("invalid barrier")
            ),
        )

        result = startup.start_run(
            startup.StartupRequest(
                RunRequest(task_cfg=self.task, inference_enabled=False), False
            ),
            hooks,
        )

        self.assertFalse(result.started)
        self.assertIn("post-spawn qualification barrier failed", result.error)
        self.assertTrue(any(event.startswith("finalize:True:") for event in self.events))

    def test_unsafe_trigger_combinations_are_rejected_before_storage_or_run_creation(self) -> None:
        cases = (
            (
                RunRequest(
                    task_cfg=self.task,
                    inference_enabled=False,
                    serial_enabled=False,
                    trigger_on=True,
                    fps=30,
                    arduino_fps=30,
                ),
                "requires serial controller",
            ),
            (
                RunRequest(
                    task_cfg=self.task,
                    inference_enabled=False,
                    serial_enabled=True,
                    trigger_on=True,
                    fps=60,
                    arduino_fps=30,
                ),
                "Arduino FPS to match camera FPS",
            ),
        )
        for config, message in cases:
            with self.subTest(message=message):
                self.events.clear()
                result = startup.start_run(
                    startup.StartupRequest(config, already_running=False),
                    self.hooks(),
                )
                self.assertFalse(result.started)
                self.assertIn(message, result.error or "")
                self.assertNotIn("storage", self.events)
                self.assertNotIn("create-run-dir", self.events)
                self.assertNotIn("serial-create", self.events)

    def test_invalid_capture_shape_is_rejected_before_storage(self) -> None:
        for config, message in (
            (
                RunRequest(
                    task_cfg=self.task,
                    inference_enabled=False,
                    fps=None,
                    serial_enabled=False,
                ),
                "fps must be a positive integer",
            ),
            (
                RunRequest(
                    task_cfg=self.task,
                    inference_enabled=False,
                    pixel_format=" ",
                    serial_enabled=False,
                ),
                "pixel_format must be a non-empty string",
            ),
            (
                RunRequest(
                    task_cfg=self.task,
                    inference_enabled=False,
                    num_cameras=2,
                    camera_serials=("camera-a",),
                    serial_enabled=False,
                ),
                "camera serial count must match camera count",
            ),
        ):
            with self.subTest(message=message):
                self.events.clear()
                result = startup.start_run(
                    startup.StartupRequest(config, already_running=False), self.hooks()
                )
                self.assertFalse(result.started)
                self.assertIn(message, result.error or "")
                self.assertNotIn("storage", self.events)

    def test_duplicate_task_config_is_rejected_before_storage_or_run_creation(self) -> None:
        self.task.write_text("task_name: first\ntask_name: second\n")

        result = startup.start_run(
            startup.StartupRequest(
                RunRequest(
                    task_cfg=self.task,
                    inference_enabled=False,
                    serial_enabled=False,
                ),
                already_running=False,
            ),
            self.hooks(),
        )

        self.assertFalse(result.started)
        self.assertIn("duplicate key", result.error or "")
        self.assertNotIn("storage", self.events)
        self.assertNotIn("create-run-dir", self.events)

    def test_model_batch_mismatch_is_rejected_before_storage(self) -> None:
        hooks = replace(
            self.hooks(),
            validate_model=lambda _path: startup.ModelSelection(
                name="batch-one",
                snapshot={"manifest_schema_version": "3"},
                engine_build_identity={"engine": "verified"},
                batch_size=1,
            ),
        )

        result = startup.start_run(
            startup.StartupRequest(
                RunRequest(
                    task_cfg=self.task,
                    inference_enabled=True,
                    ds_cfg=self.root / "model" / "config.txt",
                    serial_enabled=False,
                    num_cameras=2,
                ),
                already_running=False,
            ),
            hooks,
        )

        self.assertFalse(result.started)
        self.assertIn("batch size (1) does not match camera count (2)", result.error or "")
        self.assertNotIn("storage", self.events)
        self.assertNotIn("create-run-dir", self.events)

    def test_prepared_contract_snapshots_mutable_context(self) -> None:
        storage: dict[str, Any] = {"free_bytes": 100}
        prepared = startup.PreparedRun(
            config=RunRequest(task_cfg=self.task),
            run_dir=self.run_dir,
            started_at="2026-09-08T12:00:00",
            storage=storage,
            model=None,
            failure_plan=None,
            device_context={"model": "Orin Nano"},
        )
        storage["free_bytes"] = 0

        self.assertEqual(prepared.storage["free_bytes"], 100)
        with self.assertRaises(TypeError):
            prepared.storage["free_bytes"] = 0  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            prepared.started_at = "changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
