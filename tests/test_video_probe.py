from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps.inference import gstreamer_video_probe, video_probe


class VideoProbeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video = Path(self.temp_dir.name) / "raw.mp4"
        self.video.write_bytes(b"mp4")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_structural_validation_is_routine_and_decoders_are_not_started(self) -> None:
        with (
            mock.patch.object(
                video_probe,
                "_structural_probe",
                return_value={
                    "count": 20_734,
                    "method": video_probe.STRUCTURAL_METHOD,
                    "error": None,
                },
            ),
            mock.patch.object(video_probe, "_full_decode_gstreamer") as gstreamer_decode,
            mock.patch.object(video_probe, "_full_decode") as ffmpeg,
            mock.patch.dict(video_probe.os.environ, {}, clear=True),
        ):
            result = video_probe.probe_video_frames(
                self.video, expected_frames=20_734
            )

        self.assertEqual(result["count"], 20_734)
        self.assertEqual(result["method"], video_probe.STRUCTURAL_METHOD)
        gstreamer_decode.assert_not_called()
        ffmpeg.assert_not_called()

    def test_structural_failure_does_not_start_a_decoder(self) -> None:
        with (
            mock.patch.object(
                video_probe,
                "_structural_probe",
                return_value={
                    "count": None,
                    "method": video_probe.STRUCTURAL_METHOD,
                    "error": "parser error",
                },
            ),
            mock.patch.object(video_probe, "_full_decode_gstreamer") as decode,
            mock.patch.object(video_probe, "_full_decode") as ffmpeg,
            mock.patch.dict(video_probe.os.environ, {}, clear=True),
        ):
            result = video_probe.probe_video_frames(self.video)

        self.assertIsNone(result["count"])
        self.assertEqual(result["error"], "parser error")
        decode.assert_not_called()
        ffmpeg.assert_not_called()

    def test_opt_in_ab_verification_requires_identical_counts(self) -> None:
        with (
            mock.patch.object(
                video_probe,
                "_structural_probe",
                return_value={
                    "count": 42,
                    "method": video_probe.STRUCTURAL_METHOD,
                    "error": None,
                },
            ),
            mock.patch.object(
                video_probe,
                "_full_decode_gstreamer",
                return_value={"count": 42, "method": video_probe.GSTREAMER_METHOD, "error": None},
            ),
            mock.patch.object(
                video_probe,
                "_full_decode",
                return_value={"count": 41, "method": "full_decode_ffmpeg", "error": None},
            ),
            mock.patch.object(video_probe.shutil, "which", return_value="/usr/bin/ffmpeg"),
            mock.patch.dict(
                video_probe.os.environ,
                {"SQUEAKVIEW_VIDEO_VALIDATION_AB_VERIFY": "1"},
            ),
        ):
            result = video_probe.probe_video_frames(self.video)

        self.assertIsNone(result["count"])
        self.assertIn("GStreamer=42, FFmpeg=41", result["error"])

    def test_structural_count_mismatch_fails_without_full_decode(self) -> None:
        with (
            mock.patch.object(
                video_probe,
                "_structural_probe",
                return_value={
                    "count": 41,
                    "method": video_probe.STRUCTURAL_METHOD,
                    "error": None,
                },
            ),
            mock.patch.object(video_probe, "_full_decode_gstreamer") as decode,
            mock.patch.object(video_probe, "_full_decode") as ffmpeg,
            mock.patch.dict(video_probe.os.environ, {}, clear=True),
        ):
            result = video_probe.probe_video_frames(
                self.video, expected_frames=42
            )

        decode.assert_not_called()
        ffmpeg.assert_not_called()
        self.assertEqual(result["count"], 41)
        self.assertIn("MP4 sample-count mismatch", result["error"])

    def test_gstreamer_worker_pins_complete_hardware_pipeline(self) -> None:
        self.assertEqual(
            gstreamer_video_probe.ELEMENT_FACTORIES,
            (
                "filesrc",
                "qtdemux",
                "h264parse",
                "nvv4l2decoder",
                "identity",
                "fakesink",
            ),
        )
        command = video_probe._gstreamer_command(
            self.video, 7, parse_only=True
        )
        self.assertIn("--parse-only", command)


if __name__ == "__main__":
    unittest.main()
