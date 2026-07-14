from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import align_run_outputs
from squeakview.apps.operator.backend import manager
from squeakview.common.serial import SERIAL_HEADER


FRAME_HEADER = [
    "stream_id",
    "source",
    "raw_frame_index",
    "camera_frame_id",
    "pts_ns",
    "dts_ns",
    "duration_ns",
    "host_monotonic_ns",
    "host_unix_ns",
    "record_segment_index",
    "record_segment_file",
    "segment_local_frame_index",
    "segment_start_raw_frame_index",
    "segment_mapping_source",
    "status",
]

DETECTION_HEADER = [
    "frame",
    "raw_frame_num",
    "ts_us",
    "raw_frame_mapping_method",
    "raw_frame_mapping_ok",
    "raw_frame_mapping_pts_ns",
    "raw_frame_mapping_delta",
    "stream_id",
    "source",
    "obj_id",
    "class_id",
    "class_label",
    "conf",
    "x",
    "y",
    "w",
    "h",
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg tools unavailable")
class AlignmentIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name) / "run"
        self.out_dir = self.run_dir / "analysis"
        self.run_dir.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_three_frame_run_aligns_and_validates_video(self) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x48:r=30",
                "-frames:v",
                "3",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(self.run_dir / "raw.mp4"),
            ],
            check=True,
            timeout=15,
        )
        write_csv(
            self.run_dir / "frames.csv",
            FRAME_HEADER,
            [
                {
                    "stream_id": 0,
                    "source": "cam0",
                    "raw_frame_index": index,
                    "camera_frame_id": index,
                    "pts_ns": index * 33_333_333,
                    "dts_ns": index * 33_333_333,
                    "duration_ns": 33_333_333,
                    "host_monotonic_ns": 1_000 + index,
                    "host_unix_ns": 2_000 + index,
                    "status": "recorded",
                }
                for index in range(3)
            ],
        )
        serial_rows = []
        for index, rp2040_time in enumerate((100_000, 133_333, 166_666), start=1):
            serial_rows.append(
                {
                    "eventType": "CAMERA_HIGH",
                    "unixTime": 1_000_000 + rp2040_time,
                    "rp2040Time": rp2040_time,
                    "count": index,
                    "hostUnixNs": 3_000 + index,
                    "hostMonotonicNs": 4_000 + index,
                    "rawLine": f"CAMERA_HIGH,{index}",
                }
            )
        write_csv(self.run_dir / "serial.csv", SERIAL_HEADER, serial_rows)
        write_csv(
            self.run_dir / "detections.csv",
            DETECTION_HEADER,
            [
                {
                    "frame": 1,
                    "raw_frame_num": 1,
                    "ts_us": 33_333,
                    "raw_frame_mapping_method": "pts_match",
                    "raw_frame_mapping_ok": 1,
                    "raw_frame_mapping_pts_ns": 33_333_333,
                    "raw_frame_mapping_delta": 0,
                    "stream_id": 0,
                    "source": "cam0",
                    "obj_id": 1,
                    "class_id": 0,
                    "class_label": "mouse",
                    "conf": 0.9,
                    "x": 1,
                    "y": 2,
                    "w": 3,
                    "h": 4,
                }
            ],
        )

        summary = align_run_outputs.build_alignment(self.run_dir, self.out_dir)

        self.assertEqual(summary["counts"]["recorded_frames"], 3)
        self.assertEqual(summary["counts"]["frames_missing_ttl"], 0)
        self.assertEqual(summary["counts"]["detections"], 1)
        self.assertEqual(summary["validation"]["video_total_nb_frames"], 3)
        self.assertTrue(summary["validation"]["video_frame_count_matches_frames_csv"])
        self.assertEqual(summary["validation"]["detections_missing_frame_count"], 0)
        self.assertEqual(summary["validation"]["detection_ts_mismatch_count"], 0)
        self.assertEqual(summary["validation"]["detection_pts_mismatch_count"], 0)
        self.assertEqual(summary["validation"]["detection_mapping_method_counts"], {"pts_match": 1})
        self.assertEqual(summary["validation"]["segment_mapping_source_counts"], {"single_file": 3})
        self.assertEqual(
            json.loads((self.out_dir / "alignment_summary.json").read_text()),
            summary,
        )


class AlignmentFailureTests(unittest.TestCase):
    def test_missing_camera_high_rows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            (run_dir / "raw.mp4").touch()
            write_csv(
                run_dir / "frames.csv",
                FRAME_HEADER,
                [{"stream_id": 0, "raw_frame_index": 0, "camera_frame_id": 0}],
            )
            write_csv(
                run_dir / "serial.csv",
                SERIAL_HEADER,
                [{"eventType": "SYSTEM_START", "rp2040Time": 1}],
            )

            with self.assertRaisesRegex(RuntimeError, "no CAMERA_HIGH rows"):
                align_run_outputs.build_alignment(run_dir, run_dir / "analysis")

    def test_summary_outputs_follow_atomic_directory_rename(self) -> None:
        summary = {
            "outputs": {
                "aligned_frames": "/tmp/run/analysis.tmp/aligned_frames.csv",
                "external": "/tmp/elsewhere/report.csv",
            }
        }

        relocated = manager._relocate_summary_outputs(
            summary,
            Path("/tmp/run/analysis.tmp"),
            Path("/tmp/run/analysis"),
        )

        self.assertEqual(relocated["outputs"]["aligned_frames"], "/tmp/run/analysis/aligned_frames.csv")
        self.assertEqual(relocated["outputs"]["external"], "/tmp/elsewhere/report.csv")


if __name__ == "__main__":
    unittest.main()
