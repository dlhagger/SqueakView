"""Behavior activation must be confirmed even when cameras free-run."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest import mock

from squeakview.apps.operator.backend.manager import OperatorBackend
from squeakview.project import AppPaths, Project, ProjectMetadata, ProjectPaths, RuntimeContext, UserPaths


class BehaviorOnlyRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        project_root = root / "project"
        for path in (project_root, project_root / "runs", root / "app"):
            path.mkdir(exist_ok=True)
        context = RuntimeContext(
            app=AppPaths.from_root(root / "app"),
            project=Project(paths=ProjectPaths.from_existing_root(project_root), metadata=ProjectMetadata.create("Test")),
            user=UserPaths(config=root / "config", state=root / "state", runtime=root / "runtime", projects_parent=root / "projects"),
        )
        self.backend = OperatorBackend(lambda _line: None, runtime_context=context)
        self.backend._run_finalized = False
        self.backend.launch_cfg = SimpleNamespace(serial_enabled=True, trigger_on=False, controller_protocol="v2")

    def test_trigger_off_uses_acknowledged_behavior_start(self):
        handle = mock.Mock()
        entered = threading.Event()
        release = threading.Event()
        result = []
        def acknowledged_behavior_start(*_args, **_kwargs):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("behavior acknowledgement missing")
        handle.send_behavior_start.side_effect = acknowledged_behavior_start
        worker = threading.Thread(target=lambda: result.append(self.backend._startup_hooks().start_controller(handle, 30)))
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertFalse(self.backend._controller_started)
            self.assertFalse(self.backend._recording_started)
        finally:
            release.set()
            worker.join(2)
        self.assertEqual(result, [True])
        handle.send_behavior_start.assert_called_once()
        handle.send_start.assert_not_called()
        self.assertTrue(self.backend._controller_started)

    def test_missing_behavior_ack_never_marks_controller_started(self):
        handle = mock.Mock()
        handle.send_behavior_start.side_effect = TimeoutError("behavior start not acknowledged")
        with self.assertRaisesRegex(TimeoutError, "not acknowledged"):
            self.backend._startup_hooks().start_controller(handle, 30)
        self.assertFalse(self.backend._controller_started)
        self.assertFalse(self.backend._recording_started)
        handle.send_start.assert_not_called()

    def test_inference_readiness_does_not_publish_recording_before_behavior_ack(self):
        with mock.patch.object(self.backend, "_confirm_recording") as confirm:
            self.backend._mark_inference_ready()
        self.assertTrue(self.backend._inference_ready.is_set())
        confirm.assert_not_called()
        self.assertFalse(self.backend._recording_started)

    def test_without_serial_inference_readiness_still_confirms_recording(self):
        self.backend.launch_cfg.serial_enabled = False
        with mock.patch.object(self.backend, "_confirm_recording") as confirm:
            self.backend._mark_inference_ready()
        confirm.assert_called_once()
        self.assertTrue(self.backend._inference_ready.is_set())

    def test_trigger_on_retains_camera_start(self):
        self.backend.launch_cfg.trigger_on = True
        handle = mock.Mock()
        self.assertTrue(self.backend._startup_hooks().start_controller(handle, 30))
        handle.send_start.assert_called_once_with(30)
        handle.send_behavior_start.assert_not_called()

    def test_canceled_start_never_activates_behavior(self):
        self.backend._stop_requested.set()
        handle = mock.Mock()
        self.assertFalse(self.backend._startup_hooks().start_controller(handle, 30))
        handle.send_behavior_start.assert_not_called()
        handle.send_start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
