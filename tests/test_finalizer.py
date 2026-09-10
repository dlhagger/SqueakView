from __future__ import annotations

import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps.operator.backend import finalizer
from squeakview.common import run_context


class _CompletedWorker:
    returncode = 0

    def poll(self) -> int:
        return 0


class _HungWorker:
    pid = 4242
    returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        raise subprocess.TimeoutExpired("post-run", timeout)


class FinalizerServiceTests(unittest.TestCase):
    def test_progress_count_never_coerces_untrusted_metadata(self) -> None:
        for value in (None, True, -1, 1.5, "999", "nan", [1]):
            with self.subTest(value=value):
                self.assertEqual(finalizer._progress_count({"frames_processed": value}), 0)
        self.assertEqual(finalizer._progress_count({"frames_processed": 123}), 123)

    def test_timeout_environment_is_finite_and_bounded(self) -> None:
        for raw in ("nan", "inf", "-inf", "0", "86401", "invalid"):
            with self.subTest(raw=raw), mock.patch.dict(
                finalizer.os.environ,
                {"SQUEAKVIEW_POST_RUN_TIMEOUT_S": raw},
            ):
                self.assertEqual(
                    finalizer._bounded_timeout(
                        "SQUEAKVIEW_POST_RUN_TIMEOUT_S", 21_600.0, 86_400.0
                    ),
                    21_600.0,
                )

    def test_completed_worker_returns_its_concrete_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            finalizer.process, "spawn_post_run", return_value=_CompletedWorker()
        ) as spawn:
            result = finalizer.run_capture_finalizer(
                Path(temp_dir),
                camera_count=2,
                enable_infer=True,
                enable_align=False,
                emit=lambda _message: None,
            )

        self.assertEqual(result, 0)
        self.assertEqual(spawn.call_args.kwargs["camera_count"], 2)
        self.assertTrue(spawn.call_args.kwargs["enable_infer"])
        self.assertFalse(spawn.call_args.kwargs["enable_align"])

    def test_timeout_marks_failed_and_escalates_process_group(self) -> None:
        worker = _HungWorker()
        signals: list[signal.Signals] = []

        def kill_group(_pid: int, sent: signal.Signals) -> None:
            signals.append(sent)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            with (
                mock.patch.object(
                    finalizer.process, "spawn_post_run", return_value=worker
                ),
                mock.patch.object(
                    finalizer.time, "monotonic", side_effect=[0.0, 1.0]
                ),
                mock.patch.object(finalizer.os, "killpg", side_effect=kill_group),
                mock.patch.dict(
                    finalizer.os.environ,
                    {
                        "SQUEAKVIEW_POST_RUN_TIMEOUT_S": "0.1",
                        "SQUEAKVIEW_POST_RUN_TERMINATE_GRACE_S": "0.1",
                        "SQUEAKVIEW_POST_RUN_KILL_GRACE_S": "0.1",
                    },
                ),
            ):
                result = finalizer.run_capture_finalizer(
                    run_dir,
                    camera_count=1,
                    enable_infer=False,
                    enable_align=False,
                    emit=lambda _message: None,
                )

            status = run_context.read_json(run_dir / "run_status.json")

        self.assertEqual(result, 124)
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(status["state"], "finalization_failed")
        self.assertIn("timed out", status["error"])

    def test_qualification_timeout_forces_same_bounded_failure_path(self) -> None:
        worker = _HungWorker()
        signals: list[signal.Signals] = []
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.object(finalizer.process, "spawn_post_run", return_value=worker),
            mock.patch.object(
                finalizer.os,
                "killpg",
                side_effect=lambda _pid, sent: signals.append(sent),
            ),
        ):
            run_dir = Path(temp_dir)
            result = finalizer.run_capture_finalizer(
                run_dir,
                camera_count=1,
                enable_infer=False,
                enable_align=False,
                emit=lambda _message: None,
                force_timeout=True,
            )
            status = run_context.read_json(run_dir / "run_status.json")

        self.assertEqual(result, 124)
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertIn("qualification-injected", status["error"])


if __name__ == "__main__":
    unittest.main()
