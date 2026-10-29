from __future__ import annotations

import csv
import json
import tempfile
import threading
import tracemalloc
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from squeakview.apps.inference import service_maker_runner as runner
from squeakview.apps.inference import (
    contracts,
    debug_instrumentation,
    frame_audit,
    main as inference_main,
    pipeline_builder,
    post_run,
    video_probe,
)
from squeakview.apps.inference.pose_pipeline import PoseClass, PoseSchema, decode_yolo26_rows, load_pose_schema
from squeakview.common import run_context
from squeakview.common.capture_policy import capture_buffer_policy
from squeakview.common.child_events import decode_child_event
from squeakview.common.failure_injection import FailurePlan


from tests.support.service_maker_runner import ServiceMakerCase, FakePipeline

class CaptureValidationTests(ServiceMakerCase):
    def test_video_validation_timeout_rejects_nonfinite_environment(self) -> None:
        with mock.patch.dict(
            video_probe.os.environ,
            {"SQUEAKVIEW_VIDEO_VALIDATION_TIMEOUT_S": "nan"},
        ):
            self.assertEqual(
                video_probe._timeout_seconds(),
                float(video_probe.DEFAULT_VIDEO_VALIDATION_TIMEOUT_S),
            )


    def test_video_frame_probe_requires_successful_full_decode(self) -> None:
        video = self.root / "raw.mp4"
        video.write_bytes(b"mp4")
        ffmpeg = self.root / "fake-ffmpeg"
        ffmpeg.write_text("#!/bin/sh\nprintf 'frame=3389\\nprogress=end\\n'\n")
        ffmpeg.chmod(0o755)

        with (
            mock.patch.object(video_probe.shutil, "which", return_value=str(ffmpeg)),
            mock.patch.dict(
                video_probe.os.environ,
                {"SQUEAKVIEW_VIDEO_VALIDATION_FULL_DECODE": "1"},
                clear=True,
            ),
        ):
            probe = video_probe.probe_video_frames(video)

        self.assertEqual(probe["count"], 3389)
        self.assertEqual(probe["method"], "full_decode_ffmpeg_fallback")
        self.assertIn("GStreamer primary decode failed", probe["warning"])
        self.assertIsNone(probe["error"])


    def test_video_frame_probe_reports_bounded_decoder_progress(self) -> None:
        video = self.root / "raw.mp4"
        video.write_bytes(b"mp4")
        ffmpeg = self.root / "fake-ffmpeg-progress"
        ffmpeg.write_text(
            "#!/bin/sh\n"
            "printf 'frame=10\\nprogress=continue\\n'\n"
            "printf 'frame=20\\nprogress=end\\n'\n"
        )
        ffmpeg.chmod(0o755)
        updates: list[int] = []

        with (
            mock.patch.object(video_probe.shutil, "which", return_value=str(ffmpeg)),
            mock.patch.dict(
                video_probe.os.environ,
                {"SQUEAKVIEW_VIDEO_VALIDATION_FULL_DECODE": "1"},
                clear=True,
            ),
        ):
            probe = video_probe.probe_video_frames(
                video, progress_callback=updates.append
            )

        self.assertEqual(probe["count"], 20)
        self.assertEqual(updates, [10, 20])


    def test_video_progress_payload_calculates_rate_and_eta(self) -> None:
        payload = post_run._video_progress_payload(
            5_000,
            20_000,
            100.0,
            now_monotonic=200.0,
        )

        self.assertEqual(payload["video_validation_percent"], 25.0)
        self.assertEqual(payload["video_validation_rate_fps"], 50.0)
        self.assertEqual(payload["video_validation_eta_s"], 300.0)


    def test_video_probe_pins_jetson_decoder_encoder_and_synthetic_timestamps(self) -> None:
        command = video_probe._decode_command("/usr/bin/ffmpeg", Path("raw.mp4"))

        input_index = command.index("-i")
        decoder_index = command.index(video_probe.JETSON_H264_DECODER)
        raw_encoder_index = command.index("rawvideo")
        self.assertLess(decoder_index, input_index)
        self.assertGreater(raw_encoder_index, input_index)
        self.assertNotIn("h264_cuvid", command)
        self.assertIn(video_probe.VALIDATION_OUTPUT_FILTER, command)
        self.assertEqual(command[-3:], ["-f", "null", "-"])


    def test_restricted_nvidia_ffmpeg_registration_order_cannot_select_cuvid(self) -> None:
        video = self.root / "raw.mp4"
        video.write_bytes(b"mp4")
        ffmpeg = self.root / "restricted-nvidia-ffmpeg"
        ffmpeg.write_text(
            "#!/bin/sh\n"
            "case \" $* \" in\n"
            "  *\" -c:v h264_nvv4l2dec \"*\" -c:v rawvideo \"*)\n"
            "    printf 'frame=7\\nprogress=end\\n' ;;\n"
            "  *) printf 'automatic decoder selection reached h264_cuvid\\n'; exit 9 ;;\n"
            "esac\n"
        )
        ffmpeg.chmod(0o755)

        with (
            mock.patch.object(video_probe.shutil, "which", return_value=str(ffmpeg)),
            mock.patch.dict(
                video_probe.os.environ,
                {"SQUEAKVIEW_VIDEO_VALIDATION_FULL_DECODE": "1"},
                clear=True,
            ),
        ):
            probe = video_probe.probe_video_frames(video)

        self.assertEqual(probe["count"], 7)
        self.assertIsNone(probe["error"])


    def test_video_frame_probe_rejects_decode_error_even_with_frame_progress(self) -> None:
        video = self.root / "raw.mp4"
        video.write_bytes(b"mp4")
        ffmpeg = self.root / "fake-ffmpeg-error"
        ffmpeg.write_text(
            "#!/bin/sh\nprintf 'frame=2\\ndecode corruption\\n'\nexit 1\n"
        )
        ffmpeg.chmod(0o755)

        with (
            mock.patch.object(video_probe.shutil, "which", return_value=str(ffmpeg)),
            mock.patch.dict(
                video_probe.os.environ,
                {"SQUEAKVIEW_VIDEO_VALIDATION_FULL_DECODE": "1"},
                clear=True,
            ),
        ):
            probe = video_probe.probe_video_frames(video)

        self.assertIsNone(probe["count"])
        self.assertEqual(probe["method"], "full_decode_gstreamer_then_ffmpeg")
        self.assertIn("GStreamer decode failed", probe["error"])
        self.assertIn("decode corruption", probe["error"])


    def test_video_frame_probe_times_out_and_terminates_decoder_group(self) -> None:
        video = self.root / "raw.mp4"
        video.write_bytes(b"mp4")
        ffmpeg = self.root / "fake-ffmpeg-hang"
        ffmpeg.write_text("#!/bin/sh\nsleep 5\n")
        ffmpeg.chmod(0o755)

        probe = video_probe._full_decode(str(ffmpeg), video, 0.01)

        self.assertIsNone(probe["count"])
        self.assertIn("timed out", probe["error"])


    def test_recording_validation_rejects_unadmitted_source_frame(self) -> None:
        app, _pipeline = self.build(self.config())
        app.artifacts.raw_video.write_bytes(b"mp4")
        with mock.patch.object(
            post_run,
            "probe_video_frames",
            return_value={"count": 2, "method": "container_nb_frames", "error": None},
        ):
            report, passed = post_run._validate_recordings(
                app.run_dir,
                1,
                {0: 3},
                {0: 2},
            )

        self.assertFalse(passed)
        camera = report["cameras"][0]
        self.assertFalse(report["passed"])
        self.assertEqual(camera["source_frames"], 3)
        self.assertEqual(camera["record_admitted_frames"], 2)
        self.assertFalse(camera["source_count_matches"])
        self.assertTrue(camera["frame_count_matches"])
        app.observations.close()
        for admission in app.record_admissions:
            admission.close()
        for telemetry in app.record_telemetry:
            telemetry.close()
        app.frames.close()
        app._stopped = True


    def test_recording_validation_rejects_zero_frame_run(self) -> None:
        video = self.root / "raw.mp4"
        video.write_bytes(b"mp4-header")
        with mock.patch.object(
            post_run,
            "probe_video_frames",
            return_value={"count": 0, "method": "container_nb_frames", "error": None},
        ):
            report, passed = post_run._validate_recordings(
                self.root, 1, {0: 0}, {0: 0}
            )

        self.assertFalse(passed)
        self.assertFalse(report["cameras"][0]["nonzero_frame_count"])


    def test_acquisition_integrity_rejects_camera_event_and_transport_loss(self) -> None:
        diagnostics = self.root / "diagnostics"
        diagnostics.mkdir(exist_ok=True)
        (diagnostics / "errors.csv").write_text(
            "host_unix_ns,host_monotonic_ns,event_type,stream_id,"
            "expected_frame_id,actual_frame_id,details\n"
            "1,2,payload_crc_failure,0,,3,{}\n"
        )
        (diagnostics / "camera.csv").write_text(
            "stream_id,stream_lost_frames,stream_dropped_frames,stream_incomplete_frames\n"
            "0,1,0,0\n"
        )

        report, passed = post_run._validate_acquisition_integrity(self.root)

        self.assertFalse(passed)
        self.assertEqual(report["event_counts"], {"payload_crc_failure": 1})
        self.assertEqual(report["transport_counter_maxima"]["stream_lost_frames"], 1)


    def test_acquisition_integrity_rejects_header_only_or_malformed_telemetry(self) -> None:
        diagnostics = self.root / "diagnostics"
        diagnostics.mkdir(exist_ok=True)
        (diagnostics / "errors.csv").write_text("event_type\n")
        (diagnostics / "camera.csv").write_text(
            "stream_id,stream_lost_frames,stream_dropped_frames,stream_incomplete_frames\n"
        )

        report, passed = post_run._validate_acquisition_integrity(self.root)

        self.assertFalse(passed)
        self.assertTrue(report["camera_telemetry_schema_valid"])
        self.assertEqual(report["camera_telemetry_sample_rows"], 0)

        (diagnostics / "camera.csv").write_text(
            "stream_id,stream_lost_frames,stream_dropped_frames,stream_incomplete_frames\n"
            "0,-1,,0\n"
        )
        report, passed = post_run._validate_acquisition_integrity(self.root)
        self.assertFalse(passed)
        self.assertGreater(report["invalid_diagnostic_rows"], 0)
        self.assertEqual(report["transport_counter_samples"]["stream_dropped_frames"], 0)


    def test_capture_ledger_must_exist_and_have_contiguous_source_sequence(self) -> None:
        missing = self.root / "capture_cam0.jsonl"
        with self.assertRaisesRegex(RuntimeError, "capture ledger is missing"):
            list(post_run._iter_capture_payloads(missing, 0))

        missing.write_text(
            json.dumps(
                {
                    "camera_index": 0,
                    "source_sequence_index": 1,
                    "gst_pts_ns": 0,
                }
            )
            + "\n"
        )
        with self.assertRaisesRegex(RuntimeError, "source sequence is not contiguous"):
            list(post_run._iter_capture_payloads(missing, 0))


    def test_capture_ledger_rejects_missing_camera_frame_identity(self) -> None:
        ledger = self.root / "capture_cam0.jsonl"
        ledger.write_text(
            json.dumps(
                {
                    "camera_index": 0,
                    "source_sequence_index": 0,
                    "gst_pts_ns": 0,
                }
            )
            + "\n"
        )

        with self.assertRaisesRegex(RuntimeError, "frame identity is unavailable"):
            list(post_run._iter_capture_payloads(ledger, 0))


    def test_capture_ledger_rejects_noncontiguous_camera_frame_ids(self) -> None:
        first = {
            "camera_index": 0,
            "source_sequence_index": 0,
            "camera_frame_id": 100,
            "gst_pts_ns": 0,
        }
        cases = {
            "duplicate": 100,
            "regressed": 99,
            "gap": 102,
        }
        for name, second_frame_id in cases.items():
            with self.subTest(name=name):
                ledger = self.root / f"capture_{name}.jsonl"
                second = dict(
                    first,
                    source_sequence_index=1,
                    camera_frame_id=second_frame_id,
                    gst_pts_ns=1,
                )
                ledger.write_text(
                    json.dumps(first) + "\n" + json.dumps(second) + "\n"
                )

                with self.assertRaisesRegex(
                    RuntimeError, "camera frame IDs are not contiguous"
                ):
                    list(post_run._iter_capture_payloads(ledger, 0))


    def test_inference_ledger_rejects_duplicate_and_out_of_order_rows(self) -> None:
        cases = {
            "duplicate": [0, 0],
            "out_of_order": [1, 0],
        }
        for name, sequences in cases.items():
            with self.subTest(name=name):
                database = self.root / f"inference_{name}.sqlite"
                ledger = self.root / f"inference_{name}.csv"
                ledger.write_text(
                    "stream_id,source_sequence_index\n"
                    + "".join(f"0,{sequence}\n" for sequence in sequences)
                )
                connection = post_run._open_index(database)
                try:
                    with self.assertRaisesRegex(
                        RuntimeError, "duplicate or out of order"
                    ):
                        post_run._index_inference_frames(connection, ledger)
                finally:
                    connection.close()


    def test_inference_ledger_rejects_negative_identity(self) -> None:
        cases = ((-1, 0), (0, -1))
        for stream_id, sequence in cases:
            with self.subTest(stream_id=stream_id, sequence=sequence):
                database = self.root / f"negative_{stream_id}_{sequence}.sqlite"
                ledger = self.root / f"negative_{stream_id}_{sequence}.csv"
                ledger.write_text(
                    "stream_id,source_sequence_index\n"
                    f"{stream_id},{sequence}\n"
                )
                connection = post_run._open_index(database)
                try:
                    with self.assertRaisesRegex(
                        RuntimeError, "negative inference frame identity"
                    ):
                        post_run._index_inference_frames(connection, ledger)
                finally:
                    connection.close()


    def test_orphan_inference_frame_fails_admission_summary(self) -> None:
        database = self.root / "orphan.sqlite"
        ledger = self.root / "orphan.csv"
        ledger.write_text(
            "stream_id,source_sequence_index\n"
            "0,0\n"
            "0,2\n"
        )
        connection = post_run._open_index(database)
        try:
            connection.execute(
                "INSERT INTO recorded VALUES (?, ?, ?, ?)",
                (0, 0, 0, "100"),
            )
            post_run._index_inference_frames(connection, ledger)

            summary = post_run._summarize_inference_admission(
                connection, {0: 1}
            )
        finally:
            connection.close()

        self.assertEqual(summary["orphan_inference_frames"], 1)
        self.assertEqual(summary["inference_admitted_frames"], {0: 1})
        self.assertFalse(summary["passed"])


    def test_recording_admission_validates_stream_and_contiguous_index(self) -> None:
        admission = self.root / "record_admission.csv"
        admission.write_text(
            "stream_id,record_frame_index,pts_ns,observer_monotonic_ns\n"
            "1,0,0,10\n"
        )
        with self.assertRaisesRegex(RuntimeError, "stream mismatch"):
            list(post_run._iter_admission_pts(admission, 0))

        admission.write_text(
            "stream_id,record_frame_index,pts_ns,observer_monotonic_ns\n"
            "0,1,0,10\n"
        )
        with self.assertRaisesRegex(RuntimeError, "index is not contiguous"):
            list(post_run._iter_admission_pts(admission, 0))


    def test_yolo26_decoder_undoes_symmetric_letterbox_and_filters_confidence(self) -> None:
        schema = PoseSchema(

            version=2,
            input_width=640,
            input_height=640,
            output_layer="output0",
            keypoint_names=("nose", "tail"),
            classes=(PoseClass(0, "mouse", 0.25, True, (0, 1)),),
            keypoint_threshold=0.5,
        )
        accepted = [
            0, 80, 640, 560, 0.9, 0,
            320, 320, 0.8,
            640, 560, 0.4,
        ]
        rejected = accepted.copy()
        rejected[4] = 0.1
        rows = decode_yolo26_rows(
            [accepted, rejected], schema, source_width=1440, source_height=1080
        )

        self.assertEqual(len(rows), 1)
        bbox = rows[0]["detector_bbox"]
        self.assertAlmostEqual(bbox["x"], 0.0)
        self.assertAlmostEqual(bbox["y"], 0.0)
        self.assertAlmostEqual(bbox["w"], 1440.0)
        self.assertAlmostEqual(bbox["h"], 1080.0)
        self.assertEqual(rows[0]["keypoints"][0]["name"], "nose")
        self.assertTrue(rows[0]["keypoints"][0]["visible"])
        self.assertFalse(rows[0]["keypoints"][1]["visible"])


    def test_successful_cleanup_preserves_scientific_provenance(self) -> None:
        for name in (
            "capture_cam0.jsonl",
            "capture_cam1.jsonl",
            "record_admission.csv",
            "record_admission_cam1.csv",
            post_run.PROGRESS_FILENAME,
        ):
            (self.root / name).write_text("temporary\n")
        inference = self.root / "inference"
        inference.mkdir()
        (inference / "frames.csv").write_text("temporary\n")
        for name in ("frames.csv", "objects.csv", "keypoints.csv", "alignment_summary.json"):
            (self.root / name).write_text("canonical\n")

        post_run.cleanup_successful_run(self.root, 2)

        self.assertEqual((self.root / "capture_cam0.jsonl").read_text(), "temporary\n")
        self.assertEqual(
            (self.root / "record_admission_cam1.csv").read_text(), "temporary\n"
        )
        self.assertEqual((inference / "frames.csv").read_text(), "temporary\n")
        self.assertFalse((self.root / post_run.PROGRESS_FILENAME).exists())
        for name in ("frames.csv", "objects.csv", "keypoints.csv", "alignment_summary.json"):
            self.assertEqual((self.root / name).read_text(), "canonical\n")


    def test_failed_alignment_sets_overall_progress_failed(self) -> None:
        summary = {
            "frame_alignment": {"validated": False},
            "counts": {"frame_gaps_detected": 1},
            "validation": {"video_frame_count_matches_frames_csv": True},
            "outputs": {},
        }
        video_validation = {
            "count": 12,
            "method": "full_decode_ffmpeg",
            "error": None,
        }
        with mock.patch(
            "squeakview.apps.inference.legacy_alignment.build_alignment",
            return_value=summary,
        ) as build:
            with self.assertRaisesRegex(RuntimeError, "alignment validation failed"):
                post_run.align_run(
                    self.root,
                    video_validation=video_validation,
                )

        self.assertEqual(
            build.call_args.kwargs["video_validation"],
            video_validation,
        )

        progress = run_context.read_json(self.root / post_run.PROGRESS_FILENAME)
        self.assertEqual(progress["stage"], "failed")
        self.assertFalse(progress["alignment_validation_passed"])
        self.assertFalse(progress["overall_validation_passed"])
        self.assertEqual(run_context.read_json(self.root / "run_status.json")["state"], "analysis_failed")


    def test_successful_alignment_retry_clears_prior_failure(self) -> None:
        run_context.write_status(
            self.root,
            "analysis_failed",
            stage="streaming_alignment",
            error="old recoverable failure",
        )
        summary = {
            "schema_version": "2.0",
            "frame_alignment": {
                "validated": True,
                "epoch_markers_complete": True,
                "controller_high_events_unmatched": 0,
                "shutdown_tail_high_events_unmatched": 0,
                "controller_high_events_in_epoch": 2,
            },
            "start_marker_seen": True,
            "counts": {"recorded_frames": 2, "frame_gaps_detected": 0},
            "validation": {
                "video_frame_count_matches_frames_csv": True,
                "objects_missing_frame_count": 0,
                "object_mapping_failed_rows": 0,
                "object_ts_mismatch_count": 0,
                "object_pts_mismatch_count": 0,
            },
        }

        with mock.patch(
            "squeakview.apps.inference.legacy_alignment.build_alignment",
            return_value=summary,
        ):
            post_run.align_run(self.root)

        status = run_context.read_json(self.root / "run_status.json")
        self.assertEqual(status["state"], "analysis_complete")
        self.assertEqual(status["stage"], "complete")
        self.assertIsNone(status["error"])
        self.assertTrue(status["analysis_complete"])


