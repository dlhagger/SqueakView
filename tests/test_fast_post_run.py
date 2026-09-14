from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps.inference import post_run
from squeakview.common import run_context


class FastPostRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        (self.root / "diagnostics").mkdir()
        (self.root / "capture_cam0.jsonl").write_text(
            json.dumps({"source_sequence_index": 41}) + "\n"
        )
        (self.root / "record_admission.csv").write_text(
            "stream_id,record_frame_index,pts_ns\n0,41,100\n"
        )
        (self.root / "raw.mp4").write_bytes(b"closed-mp4")
        run_context.write_status(
            self.root,
            "capture_closed",
            expected_ttl_count=42,
            ledger_frame_counts=[42, 42],
        )

    def tearDown(self) -> None:
        self._temp.cleanup()

    @staticmethod
    def _integrity(_run_dir: Path, _camera_count: int) -> tuple[dict, bool]:
        return {"passed": True}, True

    def test_fast_finalizer_compares_counts_without_scanning_or_hashing(self) -> None:
        with (
            mock.patch.object(
                post_run,
                "probe_video_frames",
                return_value={
                    "count": 42,
                    "method": "mp4_sample_table",
                    "error": None,
                },
            ) as probe,
            mock.patch.object(
                post_run,
                "_validate_acquisition_integrity",
                side_effect=self._integrity,
            ),
            mock.patch.object(
                post_run,
                "capture_recording_evidence",
                side_effect=AssertionError("fast shutdown must not hash evidence"),
            ),
        ):
            result = post_run.fast_finalize_run(self.root, camera_count=1)

        self.assertTrue(result.validation_passed)
        self.assertEqual(result.recorded_total, 42)
        probe.assert_called_once_with(self.root / "raw.mp4", expected_frames=42)
        status = run_context.read_json(self.root / "run_status.json")
        self.assertTrue(status["recording_validation_passed"])
        self.assertEqual(
            status["recording_validation"]["validation_tier"],
            "shutdown_fast_count",
        )
        self.assertFalse(status["analysis_complete"])
        self.assertFalse(status["alignment_validated"])

    def test_fast_finalizer_fails_when_video_and_metadata_counts_differ(self) -> None:
        with (
            mock.patch.object(
                post_run,
                "probe_video_frames",
                return_value={
                    "count": 41,
                    "method": "mp4_sample_table",
                    "error": "MP4 sample-count mismatch: recording=41, expected=42",
                },
            ),
            mock.patch.object(
                post_run,
                "_validate_acquisition_integrity",
                side_effect=self._integrity,
            ),
        ):
            result = post_run.fast_finalize_run(self.root, camera_count=1)

        self.assertFalse(result.validation_passed)
        status = run_context.read_json(self.root / "run_status.json")
        self.assertFalse(status["recording_validation_passed"])
        camera = status["recording_validation"]["cameras"][0]
        self.assertEqual(camera["source_frames"], 42)
        self.assertEqual(camera["record_admitted_frames"], 42)
        self.assertEqual(camera["video_frames"], 41)
        self.assertFalse(camera["frame_count_matches"])

    def test_cli_defaults_to_fast_validation(self) -> None:
        result = post_run.FinalizationResult(
            {0: 42},
            {0: 42},
            42,
            True,
            {"count": 42, "method": "mp4_sample_table", "error": None},
        )
        arguments = argparse.Namespace(
            run_dir=self.root,
            camera_count=1,
            full_analysis=False,
            enable_infer=True,
            align=True,
        )
        with (
            mock.patch.object(post_run, "parse_args", return_value=arguments),
            mock.patch.object(
                post_run, "fast_finalize_run", return_value=result
            ) as fast,
            mock.patch.object(post_run, "finalize_run") as full,
            mock.patch.object(post_run, "cleanup_successful_run"),
        ):
            returncode = post_run.main()

        self.assertEqual(returncode, 0)
        fast.assert_called_once_with(self.root, camera_count=1)
        full.assert_not_called()


if __name__ == "__main__":
    unittest.main()
