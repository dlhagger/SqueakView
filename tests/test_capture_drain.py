from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps.operator.backend.capture_drain import (
    CaptureDrainCoordinator,
    CaptureDrainHooks,
    CaptureDrainRequest,
    DrainResult,
    ledger_frame_counts,
    last_complete_ledger_line,
    wait_for_capture_drain,
)
from squeakview.apps.operator.backend.events import RunPhase


class CaptureDrainTests(unittest.TestCase):
    def test_ignores_partial_concurrent_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ledger.csv"
            path.write_bytes(b"header\n0,4,100\n0,5")

            self.assertEqual(last_complete_ledger_line(path), "0,4,100")

    def test_malformed_last_complete_row_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "capture_cam0.jsonl").write_text(
                '{"source_sequence_index":0}\nmalformed\n'
            )
            (run_dir / "record_admission.csv").write_text(
                "stream_id,record_frame_index,pts_ns\n0,0,0\n"
            )

            self.assertEqual(ledger_frame_counts(run_dir, 1), (None, 1))

    def test_deterministic_equal_count_quiet_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "capture_cam0.jsonl").write_text(
                '{"source_sequence_index":0}\n'
            )
            (run_dir / "record_admission.csv").write_text(
                "stream_id,record_frame_index,pts_ns\n0,0,0\n"
            )
            now = [0.0]

            def monotonic() -> float:
                return now[0]

            def sleep(duration: float) -> None:
                now[0] += duration

            result = wait_for_capture_drain(
                run_dir,
                1,
                expected_ttl_count=1,
                quiet_s=0.1,
                timeout_s=1.0,
                poll_s=0.05,
                monotonic=monotonic,
                sleep=sleep,
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.ledger_frame_counts, (1, 1))


class CaptureDrainCoordinatorTests(unittest.TestCase):
    def test_persists_draining_and_success_protocol_and_transitions(self) -> None:
        run_dir = Path("/tmp/unit-run")
        waiter = mock.Mock(
            return_value=DrainResult(True, (101, 202), (10, 10), 10)
        )
        write_status = mock.Mock()
        transition = mock.Mock()
        coordinator = CaptureDrainCoordinator(
            environ={
                "SQUEAKVIEW_CAPTURE_DRAIN_QUIET_S": "0.5",
                "SQUEAKVIEW_CAPTURE_DRAIN_TIMEOUT_S": "8",
            },
            waiter=waiter,
        )

        passed = coordinator.wait(
            CaptureDrainRequest(run_dir, 2, RunPhase.STOPPING, 10),
            CaptureDrainHooks(write_status, transition),
        )

        self.assertTrue(passed)
        transition.assert_called_once_with(RunPhase.DRAINING)
        waiter.assert_called_once_with(
            run_dir,
            2,
            expected_ttl_count=10,
            quiet_s=0.5,
            timeout_s=8.0,
        )
        self.assertEqual(write_status.call_count, 2)
        self.assertEqual(write_status.call_args_list[0].args, (run_dir, "capture_draining"))
        self.assertEqual(write_status.call_args_list[1].args, (run_dir, "capture_drained"))
        self.assertEqual(
            write_status.call_args_list[1].kwargs["ledger_frame_counts"], [10, 10]
        )

    def test_timeout_persists_evidence_without_advancing_unrelated_phase(self) -> None:
        run_dir = Path("/tmp/unit-run")
        waiter = mock.Mock(
            return_value=DrainResult(
                False,
                (101, 99),
                (10, 9),
                10,
                "ledger counts differ",
            )
        )
        write_status = mock.Mock()
        transition = mock.Mock()
        coordinator = CaptureDrainCoordinator(
            environ={"SQUEAKVIEW_CAPTURE_DRAIN_QUIET_S": "invalid"},
            waiter=waiter,
        )

        passed = coordinator.wait(
            CaptureDrainRequest(run_dir, 1, RunPhase.FAILED, 10),
            CaptureDrainHooks(write_status, transition),
        )

        self.assertFalse(passed)
        transition.assert_not_called()
        self.assertEqual(waiter.call_args.kwargs["quiet_s"], 0.35)
        self.assertEqual(waiter.call_args.kwargs["timeout_s"], 5.0)
        failure = write_status.call_args_list[-1]
        self.assertEqual(failure.args, (run_dir, "capture_drain_timeout"))
        self.assertEqual(failure.kwargs["ledger_sizes"], [101, 99])
        self.assertEqual(failure.kwargs["ledger_frame_counts"], [10, 9])
        self.assertEqual(failure.kwargs["drain_error"], "ledger counts differ")

    def test_nonfinite_or_unbounded_timing_uses_bounded_defaults(self) -> None:
        for quiet, timeout in (
            ("nan", "5"),
            ("0.35", "inf"),
            ("61", "62"),
            ("0.35", "601"),
        ):
            with self.subTest(quiet=quiet, timeout=timeout):
                coordinator = CaptureDrainCoordinator(
                    environ={
                        "SQUEAKVIEW_CAPTURE_DRAIN_QUIET_S": quiet,
                        "SQUEAKVIEW_CAPTURE_DRAIN_TIMEOUT_S": timeout,
                    }
                )
                self.assertEqual(coordinator._timings(), (0.35, 5.0))


if __name__ == "__main__":
    unittest.main()
