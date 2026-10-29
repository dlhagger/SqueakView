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
            runtime_context = _runtime_context(Path(temp_dir))
            run_dir = runtime_context.project.paths.runs / "run"
            run_dir.mkdir(parents=True)
            original = {
                "schema_version": "3.0",
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
            backend = manager.OperatorBackend(
                lambda _message: None,
                runtime_context=runtime_context,
            )
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
