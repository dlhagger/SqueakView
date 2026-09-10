from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from squeakview.apps.operator.backend.capture_events import interpret_capture_output
from squeakview.common.child_events import EVENT_PREFIX, encode_child_event


class CaptureOutputInterpretationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name) / "prepared-run"
        self.run_dir.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def interpret(self, message: str, *, recording_started: bool = False):
        return interpret_capture_output(
            message,
            expected_run_dir=self.run_dir,
            recording_started=recording_started,
        )

    def test_pipeline_ready_requires_the_prepared_existing_run_directory(self) -> None:
        action = self.interpret(
            encode_child_event("pipeline_ready", run_dir=str(self.run_dir))
        )

        self.assertEqual(action.kind, "ready")
        self.assertEqual(action.run_dir, self.run_dir)

    def test_malformed_structured_record_fails_closed(self) -> None:
        action = self.interpret(EVENT_PREFIX + "not-json [READY] inference playing")

        self.assertEqual(action.kind, "fatal")
        self.assertIn("invalid or unsupported", action.error or "")

    def test_ready_with_missing_or_mismatched_run_directory_fails_closed(self) -> None:
        other = self.run_dir.parent / "other-run"
        other.mkdir()
        for line in (
            encode_child_event("pipeline_ready"),
            encode_child_event("pipeline_ready", run_dir=str(other)),
        ):
            with self.subTest(line=line):
                action = self.interpret(line)
                self.assertEqual(action.kind, "fatal")
                self.assertFalse(action.discover_run_dir)

    def test_structured_fatal_preserves_child_detail(self) -> None:
        action = self.interpret(
            encode_child_event(
                "fatal", run_dir=str(self.run_dir), error="recording sink stalled"
            )
        )

        self.assertEqual(action.kind, "fatal")
        self.assertEqual(action.error, "recording sink stalled")

    def test_capture_closed_is_distinct_from_readiness(self) -> None:
        action = self.interpret(
            encode_child_event("capture_closed", run_dir=str(self.run_dir), exit_code=0)
        )

        self.assertEqual(action.kind, "capture_closed")
        self.assertEqual(action.run_dir, self.run_dir)

    def test_legacy_readiness_cannot_arm_capture(self) -> None:
        action = self.interpret("[12:00:00] [READY] inference playing")

        self.assertEqual(action.kind, "none")
        self.assertIsNotNone(action.warning)

    def test_ordinary_output_cannot_change_the_prepared_run_directory(self) -> None:
        action = self.interpret("[INFO] building pipeline")

        self.assertEqual(action.kind, "none")
        self.assertFalse(action.discover_run_dir)


if __name__ == "__main__":
    unittest.main()
