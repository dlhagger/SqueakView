from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from scripts import inject_supervisor_failure as injection


class SupervisorFailureInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.proc = self.root / "proc"
        self.proc.mkdir()
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        (self.run_dir / "run_status.json").write_text(
            json.dumps(
                {
                    "state": "recording",
                    "run_directory": str(self.run_dir),
                    "production_eligible": False,
                    "production_disqualifiers": ["supervisor_failure_injection"],
                    "process_topology": {"acquisition_owner": "durable_supervisor"},
                }
            )
        )
        self._process(
            100,
            1,
            1000,
            ("python", "-m", "squeakview.apps.operator.backend.supervisor", "--gui-command"),
            environment={injection.GATE: "1"},
        )
        self._process(101, 100, 1001, ("python", "/workspace/squeakview_gui.py"))
        self._process(
            102,
            100,
            1002,
            (
                "python",
                "-m",
                "squeakview.apps.inference.main",
                "--run-dir",
                str(self.run_dir),
            ),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _process(
        self,
        pid: int,
        ppid: int,
        start: int,
        argv: tuple[str, ...],
        *,
        environment: dict[str, str] | None = None,
    ) -> None:
        directory = self.proc / str(pid)
        directory.mkdir()
        fields = ["S", str(ppid), *(["0"] * 17), str(start)]
        (directory / "stat").write_text(f"{pid} (fixture) " + " ".join(fields))
        (directory / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")
        (directory / "environ").write_bytes(
            b"\0".join(
                f"{key}={value}".encode() for key, value in (environment or {}).items()
            )
            + b"\0"
        )

    def _plan(self, **updates: object) -> injection.InjectionPlan:
        values: dict[str, object] = {
            "action": "gui-freeze",
            "supervisor_pid": 100,
            "run_dir": self.run_dir,
            "expected_phase": "recording",
            "expected_stage": None,
            "expected_supervisor_start": None,
            "expected_target_pid": None,
            "expected_target_start": None,
            "execute": False,
            "confirmation": None,
            "proc_root": self.proc,
        }
        values.update(updates)
        with mock.patch.object(injection.os, "getuid", return_value=self.proc.stat().st_uid):
            return injection.build_plan(**values)  # type: ignore[arg-type]

    def test_dry_run_identifies_exact_supervisor_gui_and_capture(self) -> None:
        plan = self._plan()

        self.assertTrue(plan.dry_run)
        self.assertEqual(plan.supervisor.pid, 100)
        self.assertEqual(plan.gui.pid, 101)
        self.assertEqual(plan.capture.pid, 102)
        self.assertIsNone(plan.finalizer)
        self.assertEqual(plan.target.pid, 101)

    def test_execute_requires_all_stale_pid_guards_and_confirmation(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit supervisor/target"):
            self._plan(execute=True, confirmation=injection.CONFIRMATION)
        with self.assertRaisesRegex(ValueError, "requires --confirm"):
            self._plan(
                execute=True,
                expected_supervisor_start=1000,
                expected_target_pid=101,
                expected_target_start=1001,
            )
        with self.assertRaisesRegex(ValueError, "start time changed"):
            self._plan(expected_supervisor_start=9999)

    def test_gate_and_nonproduction_status_are_both_required(self) -> None:
        (self.proc / "100" / "environ").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, injection.GATE):
            self._plan()

        (self.proc / "100" / "environ").write_bytes(
            f"{injection.GATE}=1\0".encode()
        )
        status_path = self.run_dir / "run_status.json"
        status = json.loads(status_path.read_text())
        status["production_eligible"] = True
        status_path.write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError, "not explicitly marked"):
            self._plan()

    def test_wrong_capture_run_directory_or_duplicate_gui_fails_closed(self) -> None:
        (self.proc / "102" / "cmdline").write_bytes(
            b"python\0-m\0squeakview.apps.inference.main\0--run-dir\0/wrong\0"
        )
        with self.assertRaisesRegex(ValueError, "does not name the exact run"):
            self._plan()

        (self.proc / "102" / "cmdline").write_bytes(
            b"python\0-m\0squeakview.apps.inference.main\0--run-dir\0"
            + str(self.run_dir).encode()
            + b"\0"
        )
        self._process(103, 100, 1003, ("python", "/other/squeakview_gui.py"))
        with self.assertRaisesRegex(ValueError, "GUI, found 2"):
            self._plan()

    def test_signal_rechecks_identity_and_never_signals_dry_run(self) -> None:
        dry = self._plan()
        sender = mock.Mock()
        with self.assertRaisesRegex(ValueError, "dry-run"):
            injection.send_verified_signal(dry, proc_root=self.proc, kill=sender)
        sender.assert_not_called()

        armed = self._plan(
            execute=True,
            confirmation=injection.CONFIRMATION,
            expected_supervisor_start=1000,
            expected_target_pid=101,
            expected_target_start=1001,
        )
        changed = replace(armed, target=replace(armed.target, start_ticks=9999))
        with self.assertRaisesRegex(ValueError, "identity changed"):
            injection.send_verified_signal(changed, proc_root=self.proc, kill=sender)
        sender.assert_not_called()

        injection.send_verified_signal(armed, proc_root=self.proc, kill=sender)
        sender.assert_called_once_with(101, int(injection.signal.SIGSTOP))

    def test_finalization_requires_exact_stage_and_post_run_child(self) -> None:
        (self.proc / "102" / "cmdline").write_bytes(
            b"python\0-m\0squeakview.apps.inference.post_run\0"
            + str(self.run_dir).encode()
            + b"\0--camera-count\0" + b"1\0"
        )
        status_path = self.run_dir / "run_status.json"
        status = json.loads(status_path.read_text())
        status.update({"state": "finalizing", "stage": "recording_validation"})
        status_path.write_text(json.dumps(status))

        plan = self._plan(
            expected_phase="finalizing", expected_stage="recording_validation"
        )

        self.assertIsNone(plan.capture)
        self.assertEqual(plan.finalizer.pid, 102)
        with self.assertRaisesRegex(ValueError, "requires --expected-stage"):
            self._plan(expected_phase="finalizing", expected_stage=None)
        with self.assertRaisesRegex(ValueError, "run stage changed"):
            self._plan(
                expected_phase="finalizing", expected_stage="capture_reconciliation"
            )

    def test_pre_capture_barrier_requires_no_workload_and_exact_waiting_state(self) -> None:
        (self.proc / "102" / "stat").unlink()
        status_path = self.run_dir / "run_status.json"
        status = json.loads(status_path.read_text())
        status.update(
            {
                "state": "starting",
                "supervisor_failure_barrier": {
                    "name": "pre_capture",
                    "state": "waiting",
                },
            }
        )
        status_path.write_text(json.dumps(status))

        plan = self._plan(
            expected_phase="starting",
            expected_barrier="pre_capture",
        )

        self.assertIsNone(plan.capture)
        self.assertEqual(plan.barrier, "pre_capture")

    def test_bounded_wait_requires_exact_barrier_state(self) -> None:
        statuses = iter(
            [
                {"state": "starting"},
                {
                    "state": "starting",
                    "supervisor_failure_barrier": {
                        "name": "after_spawn_before_ready",
                        "state": "waiting",
                    },
                },
            ]
        )
        clock = [0.0]

        with mock.patch.object(injection, "_status", side_effect=lambda _path: next(statuses)):
            injection.wait_for_state(
                self.run_dir,
                phase="starting",
                stage=None,
                barrier="after_spawn_before_ready",
                timeout_s=1.0,
                monotonic=lambda: clock[0],
                sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            )

        self.assertLess(clock[0], 1.0)

    def test_signal_recheck_rejects_phase_change(self) -> None:
        armed = self._plan(
            execute=True,
            confirmation=injection.CONFIRMATION,
            expected_supervisor_start=1000,
            expected_target_pid=101,
            expected_target_start=1001,
        )
        status_path = self.run_dir / "run_status.json"
        status = json.loads(status_path.read_text())
        status["state"] = "stopping"
        status_path.write_text(json.dumps(status))
        sender = mock.Mock()

        with self.assertRaisesRegex(ValueError, "phase changed"):
            injection.send_verified_signal(armed, proc_root=self.proc, kill=sender)
        sender.assert_not_called()

    def test_signal_recheck_rejects_vanished_or_replaced_workload(self) -> None:
        armed = self._plan(
            execute=True,
            confirmation=injection.CONFIRMATION,
            expected_supervisor_start=1000,
            expected_target_pid=101,
            expected_target_start=1001,
        )
        sender = mock.Mock()
        (self.proc / "102" / "stat").write_text(
            "102 (fixture) " + " ".join(["S", "1", *(["0"] * 17), "1002"])
        )
        with self.assertRaisesRegex(ValueError, "workload relationship changed"):
            injection.send_verified_signal(armed, proc_root=self.proc, kill=sender)
        sender.assert_not_called()

        (self.proc / "102" / "stat").write_text(
            "102 (fixture) " + " ".join(["S", "100", *(["0"] * 17), "9999"])
        )
        with self.assertRaisesRegex(ValueError, "workload relationship changed"):
            injection.send_verified_signal(armed, proc_root=self.proc, kill=sender)
        sender.assert_not_called()

    def test_signal_recheck_rejects_extra_or_opposite_workload(self) -> None:
        armed = self._plan(
            execute=True,
            confirmation=injection.CONFIRMATION,
            expected_supervisor_start=1000,
            expected_target_pid=101,
            expected_target_start=1001,
        )
        sender = mock.Mock()
        self._process(
            104,
            100,
            1004,
            (
                "python",
                "-m",
                "squeakview.apps.inference.main",
                "--run-dir",
                str(self.run_dir),
            ),
        )
        with self.assertRaisesRegex(ValueError, "workload relationship changed"):
            injection.send_verified_signal(armed, proc_root=self.proc, kill=sender)
        sender.assert_not_called()

        (self.proc / "104" / "cmdline").write_bytes(
            b"python\0-m\0squeakview.apps.inference.post_run\0"
            + str(self.run_dir).encode()
            + b"\0"
        )
        with self.assertRaisesRegex(ValueError, "workload relationship changed"):
            injection.send_verified_signal(armed, proc_root=self.proc, kill=sender)
        sender.assert_not_called()
