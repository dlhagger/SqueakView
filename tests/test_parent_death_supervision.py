from __future__ import annotations

import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps.operator.backend import process, supervision
from squeakview.apps.operator.backend.contracts import RunRequest


class ParentDeathSupervisionTests(unittest.TestCase):
    def test_supervision_wraps_capture_argv_without_preexec(self) -> None:
        sentinel = object()
        popen = mock.Mock()
        with (
            mock.patch.object(supervision.os, "getpid", return_value=4321),
            mock.patch.object(supervision.subprocess, "Popen", return_value=popen) as spawn,
            mock.patch.object(supervision, "ProcessHandle", return_value=sentinel),
        ):
            result = supervision.spawn(
                "capture.module",
                ["--fps", "30"],
                lambda _line: None,
                "capture",
                workspace=Path("/tmp"),
                parent_death_signal=signal.SIGINT,
            )

        self.assertIs(result, sentinel)
        command = spawn.call_args.args[0]
        self.assertEqual(
            command[:8],
            [
                sys.executable,
                "-m",
                "squeakview.apps.operator.backend.parent_death_exec",
                "--expected-parent-pid",
                "4321",
                "--signal",
                str(int(signal.SIGINT)),
                "--",
            ],
        )
        self.assertEqual(command[8:], [sys.executable, "-m", "capture.module", "--fps", "30"])
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        self.assertNotIn("preexec_fn", spawn.call_args.kwargs)

    def test_only_acquisition_spawn_requests_parent_death_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            config = RunRequest(
                fps=30,
                ds_cfg=None,
                inference_enabled=False,
                serial_enabled=False,
                run_dir=run_dir,
            )
            with mock.patch.object(process, "_spawn", return_value=mock.sentinel.handle) as spawn:
                result = process.spawn_inference(config, lambda _line: None)

        self.assertIs(result, mock.sentinel.handle)
        self.assertEqual(spawn.call_args.kwargs["parent_death_signal"], signal.SIGINT)


if __name__ == "__main__":
    unittest.main()
