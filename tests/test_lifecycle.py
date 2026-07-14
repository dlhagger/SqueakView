from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps.operator.backend import manager, process
from squeakview.common import run_context


class FakeProcessHandle:
    def __init__(self) -> None:
        self.running = True
        self.terminate_calls = 0

    def is_running(self) -> bool:
        return self.running

    def terminate_group_graceful(self, *_args, **_kwargs) -> None:
        self.terminate_calls += 1
        self.running = False

    def wait(self, timeout: float | None = None) -> None:
        del timeout


class FakeSerialHandle:
    instances: list["FakeSerialHandle"] = []

    def __init__(self, port: str, baud: int, emit_fn) -> None:
        self.port = port
        self.baud = baud
        self.emit_fn = emit_fn
        self.sent: list[str] = []
        self.markers: list[str] = []
        self.closed = False
        self.__class__.instances.append(self)

    def open(self, _run_dir: Path | None = None) -> bool:
        return True

    def log_marker(self, marker: str) -> None:
        self.markers.append(marker)

    def send_line(self, text: str) -> None:
        self.sent.append(text)

    def wait_for_ttl(self, timeout_s: float = 3.0) -> bool:
        del timeout_s
        return True

    def close(self) -> None:
        self.closed = True


class BackendLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.task_cfg = self.root / "task.yaml"
        self.task_cfg.write_text("task_name: test\n")
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.logs: list[str] = []
        self.started: list[bool] = []
        self.failures: list[str] = []
        self.backend = manager.OperatorBackend(
            self.logs.append,
            on_run_started=lambda: self.started.append(True),
            on_run_failed=self.failures.append,
        )
        self.handle = FakeProcessHandle()
        self.exit_callback = None

        def spawn(_cfg, emit, on_exit=None):
            del emit
            self.exit_callback = on_exit
            return self.handle

        self.patches = [
            mock.patch.object(manager.run_context, "assert_runs_dir_ready", return_value={"free_bytes": 10_000}),
            mock.patch.object(manager.run_context, "create_run_dir", return_value=(self.run_dir, self.run_dir.name)),
            mock.patch.object(manager.process, "spawn_inference", side_effect=spawn),
            mock.patch.object(self.backend, "_write_run_manifest"),
            mock.patch.object(self.backend, "_write_bottle_measurements", return_value={}),
            mock.patch.object(self.backend, "_ensure_metadata"),
            mock.patch.object(self.backend, "_run_post_run_alignment", return_value=None),
            mock.patch.object(self.backend, "_run_output_snapshot", return_value={}),
            mock.patch.object(manager.time, "sleep"),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self) -> None:
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

    def test_ready_marker_transitions_starting_to_recording(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))
        self.assertEqual(self.status()["state"], "starting")

        self.backend._inference_emit("[12:00:00] [READY] inference playing")

        status = self.status()
        self.assertEqual(status["state"], "recording")
        self.assertEqual(self.started, [True])
        self.assertIn("starting_at", status)
        self.assertIn("started_at", status)

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

    def test_spawn_failure_marks_created_run_failed(self) -> None:
        with mock.patch.object(manager.process, "spawn_inference", side_effect=OSError("spawn denied")):
            result = self.backend.start_run(self.config())

        self.assertFalse(result)
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("spawn denied", status["error"])
        self.assertEqual(len(self.failures), 1)

    def test_invalid_model_is_rejected_before_run_creation(self) -> None:
        missing_config = self.root / "models" / "missing" / "configs" / "missing.txt"

        result = self.backend.start_run(self.config(inference_enabled=True, ds_cfg=missing_config))

        self.assertFalse(result)
        self.assertFalse((self.run_dir / run_context.RUN_STATUS_FILENAME).exists())
        self.assertTrue(any("model package is invalid" in line for line in self.logs))

    def test_stop_finalizes_even_when_child_already_exited(self) -> None:
        self.assertTrue(self.backend.start_run(self.config()))
        self.handle.running = False

        self.backend.stop_run()
        self.backend.stop_run()

        self.assertEqual(self.status()["state"], "finalized")
        self.assertEqual(self.handle.terminate_calls, 0)

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
        self.assertIn("sudo usermod -aG dialout $USER", error)
        self.assertEqual(self.failures, [error])


class ProcessHandleTests(unittest.TestCase):
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
