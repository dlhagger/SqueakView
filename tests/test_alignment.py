from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import align_run_outputs, align_run_outputs_streaming as streaming
from squeakview.common.serial import SERIAL_HEADER
from squeakview.common.diagnostics import qualification


FRAME_HEADER = [
    "stream_id",
    "source",
    "raw_frame_index",
    "camera_frame_id",
    "camera_timestamp_ns",
    "pts_ns",
    "dts_ns",
    "duration_ns",
    "host_monotonic_ns",
    "host_unix_ns",
    "status",
]

OBJECT_HEADER = [
    "observation_id",
    "stream_id",
    "deepstream_frame_number",
    "source_sequence_index",
    "camera_frame_id",
    "gst_pts_ns",
    "class_id",
    "class_label",
    "detector_confidence",
    "track_x",
    "track_y",
    "track_w",
    "track_h",
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_minimal_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw.mp4").touch()
    write_csv(
        run_dir / "frames.csv",
        FRAME_HEADER,
        [
            {
                "stream_id": 0,
                "raw_frame_index": 0,
                "camera_frame_id": 10,
                "camera_timestamp_ns": 1_000_000,
                "pts_ns": 0,
            }
        ],
    )
    write_csv(
        run_dir / "serial.csv",
        SERIAL_HEADER,
        [
            {"eventType": "MARKER", "reason": "START_SENT"},
            {"eventType": "CAMERA_HIGH", "rp2040Time": 1000, "count": 1},
            {"eventType": "MARKER", "reason": "CAPTURE_STOP_REQUESTED"},
            {"eventType": "MARKER", "reason": "STOP_SENT"},
            {"eventType": "MARKER", "reason": "CAPTURE_STOP_DONE"},
        ],
    )


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
class AlignmentIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name) / "run"
        self.out_dir = self.run_dir / "analysis"
        self.run_dir.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_three_frame_run_aligns_and_validates_video(self) -> None:
        (self.run_dir / "raw.mp4").write_bytes(b"test-video")
        validated_video = {
            "count": 3,
            "method": "test_full_decode",
            "error": None,
        }
        write_csv(
            self.run_dir / "frames.csv",
            FRAME_HEADER,
            [
                {
                    "stream_id": 0,
                    "source": "cam0",
                    "raw_frame_index": index,
                    "camera_frame_id": 18_301 + index,
                    "camera_timestamp_ns": 500_000_000 + index * 33_333_000,
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
        serial_rows = [
            {
                "eventType": "MARKER",
                "context": "HOST",
                "reason": "START_SENT",
                "hostUnixNs": 2_500,
                "hostMonotonicNs": 3_500,
                "rawLine": "MARKER,START_SENT",
            }
        ]
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
        serial_rows.extend(
            [
                {"eventType": "MARKER", "reason": "CAPTURE_STOP_REQUESTED"},
                {"eventType": "MARKER", "reason": "STOP_SENT"},
                {"eventType": "MARKER", "reason": "CAPTURE_STOP_DONE"},
            ]
        )
        write_csv(self.run_dir / "serial.csv", SERIAL_HEADER, serial_rows)
        write_csv(
            self.run_dir / "objects.csv",
            OBJECT_HEADER,
            [
                {
                    "observation_id": "s0:f1:o0",
                    "stream_id": 0,
                    "deepstream_frame_number": 1,
                    "source_sequence_index": 1,
                    "camera_frame_id": 18_302,
                    "gst_pts_ns": 33_333_333,
                    "class_id": 0,
                    "class_label": "mouse",
                    "detector_confidence": 0.9,
                    "track_x": 1,
                    "track_y": 2,
                    "track_w": 3,
                    "track_h": 4,
                }
            ],
        )

        summary = align_run_outputs.build_alignment(
            self.run_dir,
            self.out_dir,
            video_validation=validated_video,
        )

        self.assertEqual(summary["counts"]["recorded_frames"], 3)
        self.assertEqual(summary["counts"]["frames_missing_ttl"], 0)
        self.assertEqual(
            summary["frame_alignment"]["method"],
            "first_recorded_frame_to_first_camera_high_after_start_sent",
        )
        self.assertEqual(summary["frame_alignment"]["first_camera_frame_id"], 18_301)
        self.assertEqual(summary["frame_alignment"]["first_ttl_count"], 1)
        self.assertEqual(summary["frame_alignment"]["camera_frame_id_offset"], 18_300)
        self.assertTrue(summary["frame_alignment"]["validated"])
        self.assertEqual(summary["frame_alignment"]["validated_pairs"], 3)
        self.assertEqual(summary["frame_alignment"]["missing_ttl_pairs"], 0)
        self.assertEqual(summary["frame_alignment"]["source_sequence_mismatch_count"], 0)
        self.assertEqual(summary["frame_alignment"]["clock_validation_pairs"], 3)
        self.assertEqual(summary["frame_alignment"]["clock_elapsed_error_us_max_abs"], 0.0)
        self.assertTrue(summary["frame_alignment"]["clock_within_tolerance"])
        self.assertEqual(summary["frame_alignment"]["clock_tolerance_us"], 16_666.5)
        self.assertTrue(summary["start_marker_seen"])
        self.assertEqual(
            list(summary["frame_alignment"]["marker_indices"]),
            ["START_SENT", "CAPTURE_STOP_REQUESTED", "STOP_SENT", "CAPTURE_STOP_DONE"],
        )
        gates = qualification._integrity_gates(
            {"alignment_validated": True},
            {"serial": {"enabled": True}},
            self.run_dir,
            alignment_summary=summary,
        )
        self.assertTrue(gates["alignment_validated_when_required"])
        self.assertEqual(summary["counts"]["object_observations"], 1)
        self.assertEqual(summary["validation"]["video_total_nb_frames"], 3)
        self.assertTrue(summary["validation"]["video_frame_count_matches_frames_csv"])
        self.assertEqual(summary["validation"]["objects_missing_frame_count"], 0)
        self.assertEqual(summary["validation"]["object_ts_mismatch_count"], 0)
        self.assertEqual(summary["validation"]["object_pts_mismatch_count"], 0)
        self.assertEqual(summary["validation"]["object_mapping_method_counts"], {"flir_user_meta": 1})
        self.assertEqual(summary["validation"]["video_mapping_source_counts"], {"single_file_frames_csv": 3})
        self.assertNotIn("outputs", summary)
        self.assertFalse((self.out_dir / "aligned_frames.csv").exists())
        self.assertFalse((self.out_dir / "aligned_detections.csv").exists())
        self.assertEqual(
            json.loads((self.out_dir / "alignment_summary.json").read_text()),
            summary,
        )

        offline_objects = self.run_dir / "offline_objects.csv"
        write_csv(
            offline_objects, OBJECT_HEADER,
            [dict(
                observation_id="s0:f0:o0", stream_id=0, deepstream_frame_number=0,
                source_sequence_index=0, camera_frame_id=18301, gst_pts_ns=0,
                class_id=0, class_label="mouse", detector_confidence=.8,
                track_x=1, track_y=2, track_w=3, track_h=4,
            )],
        )
        offline_out = self.run_dir / "offline_analysis"
        offline_summary = align_run_outputs.build_alignment(
            self.run_dir,
            offline_out,
            objects_path=offline_objects,
            video_validation=validated_video,
        )
        self.assertEqual(offline_summary["counts"]["object_observations"], 1)
        self.assertEqual(
            offline_summary["validation"]["object_mapping_method_counts"],
            {"flir_user_meta": 1},
        )

        cli_out = self.run_dir / "cli_analysis"
        fake_bin = self.run_dir / "fake-bin"
        fake_bin.mkdir()
        fake_ffmpeg = fake_bin / "ffmpeg"
        fake_ffmpeg.write_text(
            "#!/bin/sh\nprintf 'frame=3\\nprogress=end\\n'\n"
        )
        fake_ffmpeg.chmod(0o755)
        cli_env = dict(os.environ)
        cli_env["PATH"] = f"{fake_bin}:{cli_env.get('PATH', '')}"
        result = subprocess.run(
            [
                "python3",
                str(Path(__file__).resolve().parents[1] / "scripts" / "align_run_outputs.py"),
                str(self.run_dir),
                "--out-dir",
                str(cli_out),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
            env=cli_env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads((cli_out / "alignment_summary.json").read_text())["counts"]["recorded_frames"],
            3,
        )


class AlignmentFailureTests(unittest.TestCase):
    def test_supplied_video_validation_requires_nonnegative_integer_count(self) -> None:
        path = Path("raw.mp4")
        for value in (None, -1, 1.5, True):
            with self.subTest(value=value):
                result = streaming._video_validation(
                    path,
                    {"count": value, "method": "test", "error": None},
                )
                self.assertIsNone(result["count"])

    def test_multiline_csv_record_is_rejected_without_unbounded_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_minimal_run(run_dir)
            with (run_dir / "frames.csv").open("a") as handle:
                handle.write('0,cam,1,11,2,3,3,3,4,5,"unterminated\n')
                handle.write('continued"\n')

            with self.assertRaisesRegex(ValueError, "spans physical lines"):
                streaming.build_alignment(run_dir, run_dir / "analysis")

    def test_symlink_and_special_file_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            write_minimal_run(run_dir)
            serial = run_dir / "serial.csv"
            real_serial = root / "serial-real.csv"
            serial.replace(real_serial)
            serial.symlink_to(real_serial)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                streaming.build_alignment(run_dir, run_dir / "analysis")

            serial.unlink()
            shutil.copyfile(real_serial, serial)
            (run_dir / "frames.csv").unlink()
            os.mkfifo(run_dir / "frames.csv")
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                streaming.build_alignment(run_dir, run_dir / "analysis-special")

    def test_input_mutation_fails_without_partial_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_minimal_run(run_dir)
            out_dir = run_dir / "analysis"
            out_dir.mkdir()
            summary_path = out_dir / "alignment_summary.json"
            summary_path.write_text("old-summary")

            def mutate_after_reads(_path: Path) -> dict[str, object]:
                with (run_dir / "serial.csv").open("a") as handle:
                    handle.write("\n")
                return {"count": 1, "method": "test", "error": None}

            with (
                mock.patch.object(
                    streaming,
                    "probe_video_frames",
                    side_effect=mutate_after_reads,
                ),
                self.assertRaisesRegex(RuntimeError, "inputs changed"),
            ):
                streaming.build_alignment(run_dir, out_dir)

            self.assertEqual(summary_path.read_text(), "old-summary")
            self.assertEqual(list(out_dir.glob(".alignment_index.*.sqlite")), [])

    def test_private_temporary_index_does_not_delete_existing_user_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_minimal_run(run_dir)
            out_dir = run_dir / "analysis"
            out_dir.mkdir()
            user_file = out_dir / ".alignment_index.sqlite"
            user_file.write_text("keep")

            streaming.build_alignment(run_dir, out_dir)

            self.assertEqual(user_file.read_text(), "keep")

    def test_fractional_or_duplicate_controller_counts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for count, message in (("1.5", "unsigned decimal"), ("1", "duplicate or out of order")):
                run_dir = root / message.replace(" ", "-")
                write_minimal_run(run_dir)
                rows = [
                    {"eventType": "CAMERA_HIGH", "rp2040Time": 1000, "count": count},
                ]
                if count == "1":
                    rows.append(
                        {"eventType": "CAMERA_HIGH", "rp2040Time": 2000, "count": 1}
                    )
                write_csv(run_dir / "serial.csv", SERIAL_HEADER, rows)
                with self.subTest(count=count), self.assertRaisesRegex(
                    (ValueError, RuntimeError), message
                ):
                    streaming.build_alignment(run_dir, run_dir / "analysis")

    def test_source_sequence_never_falls_back_to_camera_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_minimal_run(run_dir)
            write_csv(
                run_dir / "objects.csv",
                OBJECT_HEADER,
                [
                    {
                        "source_sequence_index": 10,
                        "gst_pts_ns": 0,
                    }
                ],
            )

            summary = streaming.build_alignment(run_dir, run_dir / "analysis")

            self.assertEqual(summary["validation"]["objects_missing_frame_count"], 1)
            self.assertEqual(summary["validation"]["object_mapping_failed_rows"], 1)

    def test_duplicate_lifecycle_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_minimal_run(run_dir)
            write_csv(
                run_dir / "serial.csv",
                SERIAL_HEADER,
                [
                    {"eventType": "MARKER", "reason": "START_SENT"},
                    {"eventType": "MARKER", "reason": "START_SENT"},
                    {"eventType": "CAMERA_HIGH", "rp2040Time": 1000, "count": 1},
                ],
            )

            with self.assertRaisesRegex(RuntimeError, "duplicate marker"):
                streaming.build_alignment(run_dir, run_dir / "analysis")

    def test_markerless_serial_log_cannot_validate_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_minimal_run(run_dir)
            write_csv(
                run_dir / "serial.csv",
                SERIAL_HEADER,
                [{"eventType": "CAMERA_HIGH", "rp2040Time": 1000, "count": 1}],
            )

            with self.assertRaisesRegex(RuntimeError, "scientific trigger epoch"):
                streaming.build_alignment(run_dir, run_dir / "analysis")

    def test_unrecorded_high_inside_epoch_invalidates_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_minimal_run(run_dir)
            write_csv(
                run_dir / "serial.csv",
                SERIAL_HEADER,
                [
                    {"eventType": "MARKER", "reason": "START_SENT"},
                    {"eventType": "CAMERA_HIGH", "rp2040Time": 1000, "count": 1},
                    {"eventType": "CAMERA_HIGH", "rp2040Time": 2000, "count": 2},
                    {"eventType": "MARKER", "reason": "CAPTURE_STOP_REQUESTED"},
                    {"eventType": "MARKER", "reason": "STOP_SENT"},
                    {"eventType": "MARKER", "reason": "CAPTURE_STOP_DONE"},
                ],
            )

            summary = streaming.build_alignment(run_dir, run_dir / "analysis")

            self.assertFalse(summary["frame_alignment"]["validated"])
            self.assertEqual(
                summary["frame_alignment"]["controller_high_events_unmatched"], 1
            )

    def test_unrecorded_shutdown_tail_high_invalidates_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_minimal_run(run_dir)
            write_csv(
                run_dir / "serial.csv",
                SERIAL_HEADER,
                [
                    {"eventType": "MARKER", "reason": "START_SENT"},
                    {"eventType": "CAMERA_HIGH", "rp2040Time": 1000, "count": 1},
                    {"eventType": "MARKER", "reason": "CAPTURE_STOP_REQUESTED"},
                    {"eventType": "MARKER", "reason": "STOP_SENT"},
                    {"eventType": "CAMERA_HIGH", "rp2040Time": 2000, "count": 2},
                    {"eventType": "MARKER", "reason": "CAPTURE_STOP_DONE"},
                ],
            )

            summary = streaming.build_alignment(run_dir, run_dir / "analysis")

            alignment = summary["frame_alignment"]
            self.assertFalse(alignment["validated"])
            self.assertEqual(alignment["shutdown_tail_high_events"], 1)
            self.assertEqual(alignment["shutdown_tail_high_events_unmatched"], 1)

    def test_camera_gap_is_counted_without_false_source_sequence_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            (run_dir / "raw.mp4").touch()
            write_csv(
                run_dir / "frames.csv",
                FRAME_HEADER,
                [
                    {
                        "stream_id": 0,
                        "raw_frame_index": 0,
                        "camera_frame_id": 100,
                        "camera_timestamp_ns": 1_000_000_000,
                        "pts_ns": 0,
                    },
                    {
                        "stream_id": 0,
                        "raw_frame_index": 1,
                        "camera_frame_id": 102,
                        "camera_timestamp_ns": 1_066_666_000,
                        "pts_ns": 33_333_333,
                    },
                ],
            )
            write_csv(
                run_dir / "serial.csv",
                SERIAL_HEADER,
                [
                    {"eventType": "MARKER", "reason": "START_SENT"},
                    {"eventType": "CAMERA_HIGH", "rp2040Time": 100_000, "count": 1},
                    {"eventType": "CAMERA_HIGH", "rp2040Time": 166_666, "count": 3},
                    {"eventType": "MARKER", "reason": "CAPTURE_STOP_REQUESTED"},
                    {"eventType": "MARKER", "reason": "STOP_SENT"},
                    {"eventType": "MARKER", "reason": "CAPTURE_STOP_DONE"},
                ],
            )

            summary = align_run_outputs.build_alignment(run_dir, run_dir / "analysis")

            self.assertEqual(summary["counts"]["frame_gaps_detected"], 1)
            self.assertEqual(summary["counts"]["camera_frames_missing"], 1)
            self.assertEqual(
                summary["frame_alignment"]["source_sequence_mismatch_count"], 0
            )
            self.assertFalse(summary["frame_alignment"]["validated"])

            cli_result = subprocess.run(
                [
                    "python3",
                    str(
                        Path(__file__).resolve().parents[1]
                        / "scripts"
                        / "align_run_outputs.py"
                    ),
                    str(run_dir),
                    "--out-dir",
                    str(run_dir / "cli_analysis"),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            self.assertEqual(cli_result.returncode, 1, cli_result.stderr)

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
                [
                    {"eventType": "MARKER", "reason": "START_SENT"},
                    {"eventType": "MARKER", "reason": "CAPTURE_STOP_REQUESTED"},
                    {"eventType": "MARKER", "reason": "STOP_SENT"},
                    {"eventType": "MARKER", "reason": "CAPTURE_STOP_DONE"},
                ],
            )

            with self.assertRaisesRegex(RuntimeError, "no CAMERA_HIGH rows"):
                align_run_outputs.build_alignment(run_dir, run_dir / "analysis")

if __name__ == "__main__":
    unittest.main()
