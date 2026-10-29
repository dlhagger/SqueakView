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

class CaptureObservationsTests(ServiceMakerCase):
    def test_recording_telemetry_schema_matches_every_row(self) -> None:
        output = self.root / "recording_schema.csv"
        telemetry = runner.RecordingPathTelemetry(output, stream_id=0)
        telemetry.source(10)
        telemetry.admit(10)
        telemetry.egress(10)
        telemetry.close()

        with output.open(newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertTrue(rows)
        self.assertTrue(all(len(row) == len(rows[0]) for row in rows[1:]))


    def test_recording_admission_operator_writes_pts_at_recording_boundary(self) -> None:
        output = self.root / "record_admission.csv"
        operator = runner.RecordingAdmissionOperator(output, stream_id=2)

        self.assertTrue(operator.handle_buffer(SimpleNamespace(timestamp=123_456_789)))
        operator.close()

        with output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["stream_id"], "2")
        self.assertEqual(rows[0]["record_frame_index"], "0")
        self.assertEqual(rows[0]["pts_ns"], "123456789")


    def test_recording_admission_row_is_visible_before_capture_close(self) -> None:
        output = self.root / "live_record_admission.csv"
        operator = runner.RecordingAdmissionOperator(output, stream_id=0)

        operator.handle_buffer(SimpleNamespace(timestamp=42))
        with output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        operator.close()

        self.assertEqual(rows[0]["pts_ns"], "42")


    def test_recording_health_sample_is_visible_before_capture_close(self) -> None:
        output = self.root / "live_recording_health.csv"
        telemetry = runner.RecordingPathTelemetry(
            output, stream_id=0, sample_interval_s=0
        )

        telemetry.source(42)
        with output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        telemetry.close()

        self.assertTrue(any(row["event"] == "sample" for row in rows))


    def test_recording_activity_snapshot_is_constant_memory(self) -> None:
        telemetry = runner.RecordingPathTelemetry(
            self.root / "recording_activity.csv",
            stream_id=0,
        )
        telemetry.source(100)
        telemetry.admit(100)
        telemetry.egress(100)

        activity = telemetry.activity()

        self.assertEqual(activity.source_count, 1)
        self.assertEqual(activity.admission_count, 1)
        self.assertEqual(activity.egress_count, 1)
        self.assertIsNotNone(activity.last_source_monotonic_ns)
        telemetry.close()


    def test_recording_path_telemetry_is_sampled_and_bounded(self) -> None:
        output = self.root / "recording_path_telemetry.csv"
        telemetry = runner.RecordingPathTelemetry(
            output,
            stream_id=0,
            sample_interval_s=3600,
            warning_depth=2,
            max_pending=3,
        )
        for pts_ns in range(10):
            telemetry.admit(pts_ns)
        self.assertEqual(len(telemetry._pending), 3)
        telemetry.egress(9)
        telemetry.close()

        with output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(any(row["event"] == "backpressure_enter" for row in rows))
        self.assertEqual(rows[-1]["event"], "closed")
        self.assertEqual(rows[-1]["pending_evictions"], "7")


    def test_recording_path_telemetry_uses_fifo_when_egress_timestamp_is_rewritten(self) -> None:
        output = self.root / "recording_path_fifo.csv"
        telemetry = runner.RecordingPathTelemetry(
            output,
            stream_id=0,
            sample_interval_s=0,
            warning_depth=10,
            max_pending=10,
        )
        telemetry.source(100)
        telemetry.admit(100)
        telemetry.egress(999)
        telemetry.source(200)
        telemetry.admit(200)
        telemetry.egress(200)
        self.assertEqual(len(telemetry._pending), 0)
        telemetry.close()

        with output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        fifo = next(row for row in rows if row["encoder_correlation"] == "fifo")
        exact = next(row for row in rows if row["encoder_correlation"] == "pts")
        self.assertEqual(fifo["pts_ns"], "100")
        self.assertEqual(fifo["egress_timestamp_ns"], "999")
        self.assertNotEqual(fifo["encoder_latency_ms"], "")
        self.assertEqual(exact["pts_ns"], "200")
        self.assertEqual(exact["egress_timestamp_ns"], "200")
        self.assertEqual(rows[-1]["pending_evictions"], "0")


    def test_recording_backpressure_fails_before_non_leaky_queue_is_full(self) -> None:
        output = self.root / "recording_path_fatal.csv"
        faults: list[str] = []
        telemetry = runner.RecordingPathTelemetry(
            output,
            stream_id=0,
            sample_interval_s=3600,
            warning_depth=1,
            fatal_depth=3,
            max_pending=10,
            on_fatal=faults.append,
        )

        for pts_ns in range(4):
            telemetry.source(pts_ns)
        telemetry.close()

        self.assertEqual(len(faults), 1)
        self.assertIn("reached 3 frames", faults[0])
        with output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(sum(row["event"] == "backpressure_fatal" for row in rows), 1)


    def test_capture_ledger_is_authoritative_and_inference_skips_are_audited(self) -> None:
        app, _pipeline = self.build(self.config())
        first = {
            "camera_index": 0,
            "source_sequence_index": 0,
            "camera_frame_id": 100,
            "gst_pts_ns": 0,
            "host_received_monotonic_ns": 1,
            "host_received_unix_ns": 2,
            "actual_fps": 30.0,
            "camera_serial": "25187166",
            "telemetry_sample": True,
            "stream_incomplete_frames": 0,
            "stream_lost_frames": 0,
            "stream_dropped_frames": 0,
        }
        second = dict(first, source_sequence_index=1, camera_frame_id=101, gst_pts_ns=33_333_333)
        third = dict(first, source_sequence_index=2, camera_frame_id=102, gst_pts_ns=66_666_666)
        (app.run_dir / "capture_cam0.jsonl").write_text(
            "\n".join(json.dumps(item) for item in (first, second, third)) + "\n"
        )
        for operator in app.record_admissions:
            operator.close()
        (app.run_dir / "record_admission.csv").write_text(
            "stream_id,record_frame_index,pts_ns,observer_monotonic_ns\n"
            "0,0,0,10\n0,1,33333333,11\n"
        )
        app.frames.close()
        (app.run_dir / "inference" / "frames.csv").write_text(
            "stream_id,source_sequence_index\n0,0\n"
        )
        app.artifacts.raw_video.write_bytes(b"mp4")
        with mock.patch.object(
            post_run,
            "probe_video_frames",
            return_value={"count": 2, "method": "container_nb_frames", "error": None},
        ):
            result = post_run.finalize_run(
                app.run_dir, camera_count=1, enable_infer=True
            )
        self.assertEqual(result.recorded_total, 2)
        self.assertEqual(
            result.video_validation,
            {"count": 2, "method": "container_nb_frames", "error": None},
        )
        with app.artifacts.frames_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["raw_frame_index"] for row in rows], ["0", "1"])
        self.assertEqual([row["inference_admitted"] for row in rows], ["1", "0"])
        status = run_context.read_json(app.run_dir / "run_status.json")
        summary = status["inference_admission"]
        self.assertEqual(summary["inference_skipped_frames"]["0"], 1)

        app.observations.close()
        for telemetry in app.record_telemetry:
            telemetry.close()
        reconciliation = status["capture_reconciliation"]
        self.assertEqual(reconciliation["source_not_recorded_frames"]["0"], 1)
        app._stopped = True


    def test_post_run_finalizer_memory_does_not_scale_with_rows(self) -> None:
        run_dir = self.root / "scale_run"
        run_dir.mkdir()
        frame_count = 20_000
        with (
            (run_dir / "capture_cam0.jsonl").open("w") as capture,
            (run_dir / "record_admission.csv").open("w", newline="") as admission,
        ):
            admission_writer = csv.writer(admission)
            admission_writer.writerow(runner.RecordingAdmissionOperator.HEADERS)
            for index in range(frame_count):
                pts_ns = index * 33_333_333
                capture.write(
                    json.dumps(
                        {
                            "camera_index": 0,
                            "source_sequence_index": index,
                            "camera_frame_id": 1000 + index,
                            "gst_pts_ns": pts_ns,
                            "host_received_monotonic_ns": index,
                            "host_received_unix_ns": index,
                            "actual_fps": 30.0,
                            "telemetry_sample": index == 0,
                            "stream_incomplete_frames": 0,
                            "stream_lost_frames": 0,
                            "stream_dropped_frames": 0,
                        }
                    )
                    + "\n"
                )
                admission_writer.writerow([0, index, pts_ns, index])
        (run_dir / "raw.mp4").write_bytes(b"mp4")

        tracemalloc.start()
        try:
            with mock.patch.object(
                post_run,
                "probe_video_frames",
                return_value={
                    "count": frame_count,
                    "method": "container_nb_frames",
                    "error": None,
                },
            ):
                result = post_run.finalize_run(
                    run_dir, camera_count=1, enable_infer=False
                )
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        self.assertTrue(result.validation_passed)
        self.assertEqual(result.recorded_total, frame_count)
        self.assertLess(peak, 32 * 1024 * 1024)


    def test_artifact_promotion_is_restart_safe_after_partial_failure(self) -> None:
        first_temp = self.root / "promote-first"
        second_temp = self.root / "promote-second"
        run_dir = self.root / "promote-run"
        for directory in (first_temp, second_temp, run_dir):
            (directory / "diagnostics").mkdir(parents=True)
        for temp_dir, suffix in ((first_temp, "first"), (second_temp, "retry")):
            (temp_dir / "frames.csv").write_text(f"frames-{suffix}\n")
            (temp_dir / "diagnostics" / "camera.csv").write_text(
                f"camera-{suffix}\n"
            )
        paths = {
            "frames.csv": "frames.csv",
            "diagnostics/camera.csv": "diagnostics/camera.csv",
        }
        original = post_run._durable_replace
        calls = 0

        def fail_second(source: Path, destination: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected promotion interruption")
            original(source, destination)

        with mock.patch.object(post_run, "_durable_replace", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "injected promotion interruption"):
                post_run._promote(first_temp, run_dir, paths)

        self.assertEqual((run_dir / "frames.csv").read_text(), "frames-first\n")
        self.assertFalse((run_dir / "diagnostics" / "camera.csv").exists())

        post_run._promote(second_temp, run_dir, paths)

        self.assertEqual((run_dir / "frames.csv").read_text(), "frames-retry\n")
        self.assertEqual(
            (run_dir / "diagnostics" / "camera.csv").read_text(),
            "camera-retry\n",
        )


    def test_frame_operator_preserves_camera_id_and_source_sequence(self) -> None:
        output = self.root / "frames.csv"
        operator = frame_audit.FrameCsvOperator(output, meta_type=4242)
        payload = {
            "camera_serial": "25187166",
            "camera_index": 0,
            "source_sequence_index": 7,
            "camera_frame_id": 109,
            "stream_frame_id": 7,
            "chunk_frame_id": 109,
            "frame_id_delta_consistent": True,
            "missing_frames_before": 2,
            "transport_timestamp_ns": 5_000_000,
            "gst_pts_ns": 2_000_000,
            "timestamp_origin": "flir_transport",
            "host_received_monotonic_ns": 1_234_567_890,
            "host_received_unix_ns": 1_784_827_864_529_828_000,
            "image_status": "No Error",
            "timestamp_latch_available": True,
            "timestamp_latch_raw": 123456,
            "telemetry_sample": True,
            "sensor_temperature_c": 48.5,
            "stream_started_frames": 8,
            "stream_delivered_frames": 8,
            "stream_incomplete_frames": 0,
            "stream_lost_frames": 0,
            "stream_dropped_frames": 0,
            "stream_input_buffers": 8,
            "stream_output_buffers": 0,
        }
        user_meta = SimpleNamespace(get_user_data_json=lambda: payload)
        frame_meta = SimpleNamespace(
            frame_number=15,
            source_id=0,
            pad_index=0,
            buffer_pts=2_000_000,
            user_meta_items=lambda meta_type: iter([user_meta]) if meta_type == 4242 else iter([]),
        )

        with mock.patch.object(
            frame_audit.run_context,
            "atomic_write_json",
            wraps=frame_audit.run_context.atomic_write_json,
        ) as atomic_write:
            operator.handle_metadata(SimpleNamespace(frame_items=[frame_meta]))
            atomic_write.assert_not_called()
            operator.close()
            atomic_write.assert_called_once()

        with output.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["deepstream_frame_number"], "15")
        self.assertEqual(row["source_sequence_index"], "7")
        self.assertEqual(row["camera_frame_id"], "109")
        self.assertEqual(row["camera_frame_id_available"], "1")
        self.assertEqual(row["missing_frames_before"], "2")
        self.assertEqual(row["pipeline_missing_frames_before"], "0")
        self.assertEqual(row["metadata_status"], "ok")
        with (self.root / "camera.csv").open(newline="") as handle:
            telemetry = next(csv.DictReader(handle))
        self.assertEqual(telemetry["host_monotonic_ns"], "1234567890")
        self.assertEqual(telemetry["host_unix_ns"], "1784827864529828000")
        self.assertEqual(telemetry["sensor_temperature_c"], "48.5")
        self.assertEqual(telemetry["stream_delivered_frames"], "8")
        with (self.root / "errors.csv").open(newline="") as handle:
            event = next(csv.DictReader(handle))
        self.assertEqual(event["event_type"], "camera_frame_gap")
        self.assertEqual(event["expected_frame_id"], "107")
        self.assertEqual(event["actual_frame_id"], "109")
        self.assertEqual(event["host_unix_ns"], "1784827864529828000")
        self.assertEqual(event["host_monotonic_ns"], "1234567890")
        runtime = json.loads((self.root / "camera_runtime.json").read_text())
        self.assertTrue(runtime["cameras"][0]["timestamp_latch_available"])
        self.assertEqual(runtime["cameras"][0]["timestamp_latch_raw"], 123456)


    def test_frame_operator_does_not_substitute_deepstream_counter_when_metadata_missing(self) -> None:
        output = self.root / "frames.csv"
        operator = frame_audit.FrameCsvOperator(output, meta_type=4242)
        frame_meta = SimpleNamespace(
            frame_number=15,
            source_id=0,
            pad_index=0,
            buffer_pts=2_000_000,
            user_meta_items=lambda _meta_type: iter([]),
        )

        operator.handle_metadata(SimpleNamespace(frame_items=[frame_meta]))
        operator.close()

        with output.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["deepstream_frame_number"], "15")
        self.assertEqual(row["source_sequence_index"], "")
        self.assertEqual(row["camera_frame_id"], "")
        self.assertEqual(row["camera_frame_id_available"], "0")
        self.assertEqual(row["metadata_status"], "missing")
        with (self.root / "errors.csv").open(newline="") as handle:
            event = next(csv.DictReader(handle))
        self.assertEqual(event["event_type"], "frame_metadata_missing")


    def test_frame_operator_bounds_native_user_metadata_scan(self) -> None:
        output = self.root / "frames.csv"
        operator = frame_audit.FrameCsvOperator(
            output,
            meta_type=4242,
            write_audit_sidecars=False,
        )
        yielded: list[int] = []

        def user_meta_items(_meta_type):
            index = 0
            while True:
                yielded.append(index)
                yield SimpleNamespace(
                    get_user_data_json=lambda index=index: {
                        "camera_index": 0,
                        "source_sequence_index": index,
                        "camera_frame_id": index,
                    }
                )
                index += 1

        frame_meta = SimpleNamespace(
            frame_number=0,
            source_id=0,
            pad_index=0,
            buffer_pts=0,
            user_meta_items=user_meta_items,
        )
        operator.handle_metadata(SimpleNamespace(frame_items=[frame_meta]))
        operator.close()

        self.assertEqual(yielded, [0, 1])
        with output.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["metadata_status"], "ok_multiple")


    def test_camera_runtime_identity_is_bounded_and_stable(self) -> None:
        operator = frame_audit.FrameCsvOperator(
            self.root / "frames.csv",
            meta_type=4242,
            max_cameras=1,
        )
        operator._remember_camera({"camera_index": 0, "camera_serial": "A"})

        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            operator._remember_camera(
                {"camera_index": 0, "camera_serial": "B"}
            )
        with self.assertRaisesRegex(RuntimeError, "bounded capacity"):
            operator._remember_camera(
                {"camera_index": 1, "camera_serial": "C"}
            )
        operator.close()


