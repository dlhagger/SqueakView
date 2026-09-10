from __future__ import annotations

import os
import signal
import unittest
from unittest import mock

from squeakview.apps.operator.backend import parent_death_exec


class ParentDeathExecTests(unittest.TestCase):
    def test_arms_signal_then_verifies_expected_parent(self) -> None:
        calls = []

        parent_death_exec.arm_parent_death_signal(
            123,
            signal.SIGINT,
            prctl=lambda option, value: calls.append((option, value)),
            getppid=lambda: 123,
        )

        self.assertEqual(
            calls,
            [(parent_death_exec.PR_SET_PDEATHSIG, signal.SIGINT)],
        )

    def test_parent_change_after_prctl_fails_before_exec(self) -> None:
        with self.assertRaises(parent_death_exec.ParentChangedError):
            parent_death_exec.arm_parent_death_signal(
                123,
                signal.SIGINT,
                prctl=lambda *_args: None,
                getppid=lambda: 1,
            )

    def test_main_execs_exact_argv_after_successful_arm(self) -> None:
        with (
            mock.patch.object(parent_death_exec, "arm_parent_death_signal") as arm,
            mock.patch.object(
                parent_death_exec.os,
                "execvpe",
                side_effect=SystemExit(0),
            ) as execute,
        ):
            with self.assertRaises(SystemExit):
                parent_death_exec.main(
                    [
                        "--expected-parent-pid",
                        "123",
                        "--signal",
                        str(int(signal.SIGINT)),
                        "--",
                        "python3",
                        "-m",
                        "capture",
                    ]
                )

        arm.assert_called_once_with(123, int(signal.SIGINT))
        self.assertEqual(execute.call_args.args[:2], ("python3", ["python3", "-m", "capture"]))
        self.assertEqual(execute.call_args.args[2], os.environ)

    def test_missing_command_returns_reserved_failure_code(self) -> None:
        self.assertEqual(
            parent_death_exec.main(["--expected-parent-pid", "123"]),
            125,
        )


if __name__ == "__main__":
    unittest.main()
