from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from squeakview.common import qualification_barrier, run_context


class QualificationBarrierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name)
        run_context.write_status(self.run_dir, "starting")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_barrier_is_inert_without_an_explicit_selection(self) -> None:
        self.assertFalse(
            qualification_barrier.wait_at_barrier(
                self.run_dir, "pre_capture", environ={}
            )
        )
        self.assertNotIn(
            "supervisor_failure_barrier",
            run_context.read_json(self.run_dir / "run_status.json"),
        )

    def test_selection_requires_gate_and_bounded_timeout(self) -> None:
        with self.assertRaisesRegex(RuntimeError, qualification_barrier.GATE_ENV):
            qualification_barrier.wait_at_barrier(
                self.run_dir,
                "pre_capture",
                environ={qualification_barrier.BARRIER_ENV: "pre_capture"},
            )
        with self.assertRaisesRegex(RuntimeError, "between"):
            qualification_barrier.wait_at_barrier(
                self.run_dir,
                "pre_capture",
                environ={
                    qualification_barrier.GATE_ENV: "1",
                    qualification_barrier.BARRIER_ENV: "pre_capture",
                    qualification_barrier.TIMEOUT_ENV: "600",
                },
            )
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value), self.assertRaisesRegex(
                RuntimeError, "between"
            ):
                qualification_barrier.wait_at_barrier(
                    self.run_dir,
                    "pre_capture",
                    environ={
                        qualification_barrier.GATE_ENV: "1",
                        qualification_barrier.BARRIER_ENV: "pre_capture",
                        qualification_barrier.TIMEOUT_ENV: value,
                    },
                )

    def test_barrier_self_releases_and_persists_reason(self) -> None:
        clock = [0.0]

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        waited = qualification_barrier.wait_at_barrier(
            self.run_dir,
            "after_spawn_before_ready",
            environ={
                qualification_barrier.GATE_ENV: "1",
                qualification_barrier.BARRIER_ENV: "after_spawn_before_ready",
                qualification_barrier.TIMEOUT_ENV: "0.2",
            },
            monotonic=lambda: clock[0],
            sleep=sleep,
        )

        self.assertTrue(waited)
        barrier = run_context.read_json(self.run_dir / "run_status.json")[
            "supervisor_failure_barrier"
        ]
        self.assertEqual(barrier["state"], "released")
        self.assertEqual(barrier["release_reason"], "timeout")
        self.assertLessEqual(clock[0], 0.2)

    def test_stop_request_releases_without_waiting_for_deadline(self) -> None:
        clock = [0.0]
        qualification_barrier.wait_at_barrier(
            self.run_dir,
            "pre_capture",
            stop_requested=lambda: True,
            environ={
                qualification_barrier.GATE_ENV: "1",
                qualification_barrier.BARRIER_ENV: "pre_capture",
            },
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        )
        barrier = run_context.read_json(self.run_dir / "run_status.json")[
            "supervisor_failure_barrier"
        ]
        self.assertEqual(barrier["release_reason"], "stop_requested")
        self.assertEqual(clock[0], 0.0)


if __name__ == "__main__":
    unittest.main()
