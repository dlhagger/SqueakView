from __future__ import annotations

import unittest
from types import SimpleNamespace

from squeakview.apps.inference.recording import RecordingActivity
from squeakview.apps.inference.recording_liveness import (
    RecordingLivenessMonitor,
    RecordingLivenessPolicy,
    resolve_recording_liveness_policy,
)


class OneCycleEvent:
    def __init__(self) -> None:
        self.calls = 0

    def wait(self, _timeout: float) -> bool:
        self.calls += 1
        return self.calls > 1


class RecordingLivenessTests(unittest.TestCase):
    def test_policy_preserves_existing_defaults_and_environment_override(self) -> None:
        normal = resolve_recording_liveness_policy(30, failure_injection=False, environ={})
        fault = resolve_recording_liveness_policy(30, failure_injection=True, environ={})
        override = resolve_recording_liveness_policy(
            30,
            failure_injection=False,
            environ={"SQUEAKVIEW_RECORDING_STALL_TIMEOUT_S": "3.5"},
        )

        self.assertEqual(normal.configured_timeout_s, 8.0)
        self.assertEqual(fault.configured_timeout_s, 2.0)
        self.assertEqual(override.effective_timeout_ns, 3_500_000_000)
        self.assertEqual(
            resolve_recording_liveness_policy(
                30,
                failure_injection=False,
                environ={"SQUEAKVIEW_RECORDING_STALL_TIMEOUT_S": "inf"},
            ).configured_timeout_s,
            8.0,
        )

    def test_source_with_no_frames_is_not_reported_as_stalled(self) -> None:
        source = SimpleNamespace(
            stream_id=0,
            activity=lambda: RecordingActivity(0, 0, 0, None, None, None),
        )
        failures: list[str] = []
        monitor = RecordingLivenessMonitor(
            [source],
            OneCycleEvent(),
            failures.append,
            RecordingLivenessPolicy(1.0, 1_000_000_000, 0.0),
            monotonic_ns=lambda: 2_000_000_000,
        )

        monitor.run()
        self.assertEqual(failures, [])

    def test_stalled_source_reports_counts_and_stream(self) -> None:
        source = SimpleNamespace(
            stream_id=4,
            activity=lambda: RecordingActivity(10, 9, 8, 1, 2, 3),
        )
        failures: list[str] = []
        monitor = RecordingLivenessMonitor(
            [source],
            OneCycleEvent(),
            failures.append,
            RecordingLivenessPolicy(2.0, 2_000_000_000, 0.0),
            monotonic_ns=lambda: 2_000_000_001,
        )
        monitor.run()

        self.assertEqual(len(failures), 1)
        self.assertIn("stream 4", failures[0])
        self.assertIn("source=10 admitted=9 encoded=8", failures[0])

    def test_activity_failure_is_fatal_instead_of_silently_killing_monitor(self) -> None:
        def fail():
            raise OSError("snapshot unavailable")

        source = SimpleNamespace(stream_id=2, activity=fail)
        failures: list[str] = []
        monitor = RecordingLivenessMonitor(
            [source],
            OneCycleEvent(),
            failures.append,
            RecordingLivenessPolicy(8.0, 8_000_000_000, 0.0),
        )
        monitor.run()

        self.assertEqual(len(failures), 1)
        self.assertIn("telemetry failed on stream 2", failures[0])


if __name__ == "__main__":
    unittest.main()
