from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from squeakview.apps.operator.backend.events import RunPhase
from squeakview.apps.operator.gui import run_presentation, run_presenter


class RunPresentationPolicyTests(unittest.TestCase):
    def test_legacy_presenter_api_reexports_extracted_policy(self) -> None:
        names = (
            "CaptureHealthPresentation",
            "CaptureHealthSnapshot",
            "FinalizationPresentation",
            "RunPresentation",
            "present_capture_health",
            "present_finalization",
            "present_run_phase",
            "scan_capture_health",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIs(getattr(run_presenter, name), getattr(run_presentation, name))

    def test_phase_and_terminal_policy_remain_fail_closed(self) -> None:
        self.assertEqual(
            run_presentation.present_run_phase(RunPhase.DRAINING).badge_state,
            "finalizing",
        )
        self.assertFalse(
            run_presentation.present_finalization(
                {"state": "finalized", "recording_validation": {"passed": False}}
            ).passed
        )
        self.assertFalse(
            run_presentation.present_finalization(
                {"state": "analysis_failed", "recording_validation": {"passed": True}}
            ).passed
        )
        self.assertFalse(
            run_presentation.present_finalization(
                {"state": "recording", "recording_validation": {"passed": True}}
            ).passed
        )
        self.assertTrue(
            run_presentation.present_finalization(
                {"state": "finalized", "recording_validation": {"passed": True}}
            ).passed
        )

    def test_scans_latest_health_artifacts_and_presents_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "capture_cam0.jsonl").write_text(
                json.dumps({"source_sequence_index": 4})
                + "\n"
                + json.dumps(
                    {
                        "source_sequence_index": 10,
                        "total_frame_gap_events": 0,
                        "total_incomplete": 0,
                        "stream_dropped_frames": 0,
                        "stream_lost_frames": 0,
                        "sensor_temperature_c": 41.25,
                    }
                )
                + "\n"
            )
            diagnostics = run_dir / "diagnostics"
            diagnostics.mkdir()
            with (diagnostics / "recording.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["waiting_for_record_admission", "encoder_in_flight"],
                )
                writer.writeheader()
                writer.writerow(
                    {"waiting_for_record_admission": 2, "encoder_in_flight": 1}
                )
                writer.writerow(
                    {"waiting_for_record_admission": 30, "encoder_in_flight": 4}
                )

            snapshot = run_presentation.scan_capture_health(run_dir)
            presentation = run_presentation.present_capture_health(
                snapshot, disk_text="123 GB"
            )

        self.assertEqual(snapshot.frames, 11)
        self.assertEqual(snapshot.waiting, 30)
        self.assertEqual(snapshot.in_flight, 4)
        self.assertEqual(snapshot.sensor_temperature_c, 41.25)
        self.assertEqual(presentation.level, "warning")
        self.assertIn("Frames 11", presentation.text)
        self.assertIn("Queue 30/4", presentation.text)
        self.assertIn("Cam 41.2°C", presentation.text)
        self.assertIn("Disk 123 GB", presentation.text)

    def test_corrupt_live_values_do_not_crash_poll_or_hide_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "capture_cam0.jsonl").write_text(
                json.dumps(
                    {
                        "source_sequence_index": "bad",
                        "total_frame_gap_events": 2,
                        "total_incomplete": "bad",
                        "stream_dropped_frames": -4,
                        "stream_lost_frames": 1,
                        "sensor_temperature_c": "bad",
                    }
                )
                + "\n"
            )
            snapshot = run_presentation.scan_capture_health(run_dir)
            presentation = run_presentation.present_capture_health(
                snapshot, disk_text="--"
            )

        self.assertEqual(snapshot.frames, 0)
        self.assertEqual(snapshot.gaps, 2)
        self.assertEqual(snapshot.lost, 1)
        self.assertEqual(snapshot.dropped, 0)
        self.assertIsNone(snapshot.sensor_temperature_c)
        self.assertEqual(presentation.level, "error")
        self.assertIn("Frames --", presentation.text)

    def test_concurrent_partial_rows_fall_back_to_latest_complete_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "capture_cam0.jsonl").write_text(
                '{"source_sequence_index": 19}\n{"source_sequence_index":'
            )
            diagnostics = run_dir / "diagnostics"
            diagnostics.mkdir()
            (diagnostics / "recording.csv").write_text(
                "waiting_for_record_admission,encoder_in_flight\n3,2\n99,"
            )

            snapshot = run_presentation.scan_capture_health(run_dir)

        self.assertEqual(snapshot.frames, 20)
        self.assertEqual(snapshot.waiting, 3)
        self.assertEqual(snapshot.in_flight, 2)


if __name__ == "__main__":
    unittest.main()
