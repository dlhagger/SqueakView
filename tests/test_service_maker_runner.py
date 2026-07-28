from __future__ import annotations

import csv
import json
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from squeakview.apps.inference import service_maker_runner as runner
from squeakview.apps.inference import post_run
from squeakview.apps.inference.pose_pipeline import PoseClass, PoseSchema, decode_yolo26_rows, load_pose_schema
from squeakview.common import run_context


class FakePipeline:
    def __init__(self, name: str):
        self.name = name
        self.nodes: dict[str, tuple[str, dict]] = {}
        self.links: list[tuple] = []
        self.attachments: list[tuple] = []
        self.stopped = False
        self.waited = False

    def add(self, type_name: str, name: str, properties: dict | None = None):
        self.nodes[name] = (type_name, properties or {})
        return self

    def link(self, *args):
        self.links.append(args)
        return self

    def attach(self, target: str, what):
        self.attachments.append((target, what))
        return self

    def stop(self):
        self.stopped = True
        return self

    def wait(self):
        self.waited = True
        return self


class ServiceMakerRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        labels = self.root / "classes.txt"
        labels.write_text("mouse\nbottle\n")
        self.infer_config = self.root / "infer.txt"
        self.infer_config.write_text(
            "[property]\n"
            "batch-size=1\n"
            "labelfile-path=classes.txt\n"
        )
        keypoints = self.root / "keypoints.txt"
        keypoints.write_text("nose\nbottle_tip\n")
        self.pose_sidecar = self.root / "infer.pose.json"
        self.pose_sidecar.write_text(json.dumps({
            "schema_version": 2,
            "task": "pose",
            "postprocess": "pyservicemaker_yolo26_pose_v1",
            "output_layer": "output0",
            "input_width": 640,
            "input_height": 640,
            "letterbox": "symmetric",
            "end2end": True,
            "keypoint_labels_path": str(keypoints),
            "keypoint_count": 2,
            "keypoint_dims": 3,
            "keypoint_threshold": 0.5,
            "classes": [
                {"id": 0, "name": "mouse", "threshold": 0.25, "track": True, "keypoint_indices": [0]},
                {"id": 1, "name": "bottle", "threshold": 0.25, "track": False, "keypoint_indices": [1]},
            ],
        }, indent=2) + "\n")


    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_runtime_rejects_legacy_pose_schema(self) -> None:
        payload = json.loads(self.pose_sidecar.read_text())
        payload["schema_version"] = 1
        self.pose_sidecar.write_text(json.dumps(payload))

        with self.assertRaisesRegex(ValueError, "version 2 is required"):
            load_pose_schema(self.infer_config, ["mouse", "bottle"])

    def test_orin_nano_tracker_uses_stable_cuda_nvdcf(self) -> None:
        tracker_config = Path(runner.__file__).resolve().parents[3] / "configs/tracker_mouse_nvdcf.yml"
        text = tracker_config.read_text()

        self.assertIn("visualTrackerType: 1", text)
        self.assertNotIn("visualTrackerType: 2", text)

    def config(self, **overrides) -> runner.InferenceConfig:
        values = {
            "cfg_path": self.infer_config,
            "run_dir": self.root / "run",
            "width": 640,
            "height": 480,
            "fps": 30,
            "bitrate": 4000,
        }
        values.update(overrides)
        return runner.InferenceConfig(**values)

    def build(self, config: runner.InferenceConfig) -> tuple[runner.ServiceMakerApp, FakePipeline]:
        app = runner.ServiceMakerApp(
            config,
            pipeline_factory=FakePipeline,
            probe_factory=lambda name, operator: (name, operator),
        )
        pipeline = app.build()
        return app, pipeline

    def test_builds_flir_recording_inference_and_headless_display_paths(self) -> None:
        app, pipeline = self.build(self.config())

        self.assertEqual(pipeline.nodes["flirsrc0"][0], "flirspinsrc")
        self.assertEqual(
            pipeline.nodes["flirsrc0"][1]["capture-log-path"],
            str(app.run_dir / "capture_cam0.jsonl"),
        )
        self.assertNotIn("leaky", pipeline.nodes["record_queue0"][1])
        self.assertEqual(pipeline.nodes["record_queue0"][1]["max-size-buffers"], 120)
        self.assertEqual(pipeline.nodes["flirsrc0"][1]["stream-buffer-count"], 64)
        self.assertNotIn("record_convert0", pipeline.nodes)
        self.assertNotIn("record_caps0", pipeline.nodes)
        self.assertEqual(pipeline.nodes["record_encoder0"][0], "x264enc")
        self.assertFalse(pipeline.nodes["record_encoder0"][1]["sliced-threads"])
        self.assertEqual(pipeline.nodes["infer_convert0"][1]["compute-hw"], 2)
        self.assertEqual(pipeline.nodes["infer_convert0"][1]["copy-hw"], 2)
        self.assertEqual(pipeline.nodes["infer_queue0"][1]["leaky"], 2)
        self.assertEqual(pipeline.nodes["infer_queue0"][1]["max-size-buffers"], 32)
        self.assertEqual(pipeline.nodes["mux"][0], "nvstreammux")
        self.assertEqual(pipeline.nodes["infer"][0], "nvinfer")
        self.assertEqual(pipeline.nodes["tracker"][0], "nvtracker")
        self.assertEqual(pipeline.nodes["tracker"][1]["operate-on-class-ids"], "0")
        self.assertEqual(pipeline.nodes["tracker"][1]["tracking-id-reset-mode"], 3)
        self.assertEqual(pipeline.nodes["record_sink0"][1]["location"], str(app.artifacts.raw_video))
        self.assertEqual(pipeline.nodes["sink"][0], "fakesink")
        self.assertNotIn("osd", pipeline.nodes)
        self.assertEqual(app.frames.path, app.run_dir / "inference" / "frames.csv")
        self.assertTrue(
            any(link == (("infer_caps0", "mux"), ("", "sink_%u")) for link in pipeline.links)
        )
        self.assertTrue(any(target == "mux" for target, _probe in pipeline.attachments))
        self.assertTrue(any(target == "infer" for target, _probe in pipeline.attachments))
        self.assertTrue(any(target == "tracker" for target, _probe in pipeline.attachments))
        self.assertTrue(any(target == "record_queue0" for target, _probe in pipeline.attachments))
        app.stop()
        self.assertTrue(pipeline.stopped)
        self.assertTrue(pipeline.waited)

    def test_builds_non_blocking_ipc_preview_for_each_camera(self) -> None:
        socket_path = self.root / "preview.sock"
        app, pipeline = self.build(self.config(preview_sockets=(socket_path,)))

        self.assertEqual(pipeline.nodes["preview_demux"][0], "nvstreamdemux")
        self.assertEqual(pipeline.nodes["osd"][0], "nvosdbin")
        self.assertEqual(pipeline.nodes["preview_queue0"][0], "queue")
        self.assertEqual(pipeline.nodes["preview_queue0"][1]["leaky"], 2)
        self.assertEqual(pipeline.nodes["preview_queue0"][1]["max-size-buffers"], 1)
        self.assertEqual(pipeline.nodes["preview_sink0"][0], "nvunixfdsink")
        self.assertEqual(pipeline.nodes["preview_sink0"][1]["socket-path"], str(socket_path))
        self.assertFalse(pipeline.nodes["preview_sink0"][1]["async"])
        self.assertFalse(pipeline.nodes["preview_sink0"][1]["sync"])
        self.assertTrue(pipeline.nodes["preview_sink0"][1]["buffer-timestamp-copy"])
        self.assertNotIn("nveglglessink", [kind for kind, _props in pipeline.nodes.values()])
        self.assertIn(
            (("preview_demux", "preview_queue0"), ("src_0", "")),
            pipeline.links,
        )
        app.stop()

    def test_disable_infer_omits_infer_and_metadata_writer(self) -> None:
        app, pipeline = self.build(self.config(cfg_path=None, enable_infer=False))

        self.assertNotIn("infer", pipeline.nodes)
        self.assertIsNotNone(app.frames)
        self.assertTrue(any(target == "mux" for target, _probe in pipeline.attachments))
        app.stop()

    def test_stable_camera_serial_is_validated_and_passed_to_source(self) -> None:
        app, pipeline = self.build(self.config(camera_serials=("25187166",)))
        self.assertEqual(pipeline.nodes["flirsrc0"][1]["camera-serial"], "25187166")
        app.stop()

        with self.assertRaisesRegex(ValueError, "serial count"):
            runner.ServiceMakerApp(self.config(camera_serials=("one", "two")))
        with self.assertRaisesRegex(ValueError, "unique"):
            runner.ServiceMakerApp(
                self.config(num_cameras=2, camera_serials=("same", "same"), enable_infer=False)
            )

    def test_inference_requires_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "config"):
            runner.ServiceMakerApp(self.config(cfg_path=None))

    def test_batch_size_must_match_camera_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            runner.ServiceMakerApp(self.config(num_cameras=2))

    def test_preview_socket_count_must_match_camera_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "preview socket count"):
            runner.ServiceMakerApp(
                self.config(
                    num_cameras=1,
                    preview_sockets=(self.root / "one.sock", self.root / "two.sock"),
                )
            )

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
            runner.ServiceMakerApp,
            "_video_frame_probe",
            return_value={"count": 2, "method": "container_nb_frames", "error": None},
        ):
            result = post_run.finalize_run(
                app.run_dir, camera_count=1, enable_infer=True
            )
        self.assertEqual(result.recorded_total, 2)
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
                        }
                    )
                    + "\n"
                )
                admission_writer.writerow([0, index, pts_ns, index])
        (run_dir / "raw.mp4").write_bytes(b"mp4")

        tracemalloc.start()
        try:
            with mock.patch.object(
                runner.ServiceMakerApp,
                "_video_frame_probe",
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

    def test_video_frame_probe_prefers_fast_container_count(self) -> None:
        video = self.root / "raw.mp4"
        video.write_bytes(b"mp4")
        result = SimpleNamespace(returncode=0, stdout="3389\n", stderr="")

        with (
            mock.patch.object(runner.shutil, "which", return_value="/usr/bin/ffprobe"),
            mock.patch.object(runner.subprocess, "run", return_value=result) as run,
        ):
            probe = runner.ServiceMakerApp._video_frame_probe(video)

        self.assertEqual(probe["count"], 3389)
        self.assertEqual(probe["method"], "container_nb_frames")
        self.assertNotIn("-count_frames", run.call_args.args[0])

    def test_video_frame_probe_falls_back_to_decoded_count(self) -> None:
        video = self.root / "raw.mp4"
        video.write_bytes(b"mp4")
        no_metadata = SimpleNamespace(returncode=0, stdout="N/A\n", stderr="")
        decoded = SimpleNamespace(returncode=0, stdout="3389\n", stderr="")

        with (
            mock.patch.object(runner.shutil, "which", return_value="/usr/bin/ffprobe"),
            mock.patch.object(
                runner.subprocess, "run", side_effect=[no_metadata, decoded]
            ) as run,
        ):
            probe = runner.ServiceMakerApp._video_frame_probe(video)

        self.assertEqual(probe["count"], 3389)
        self.assertEqual(probe["method"], "decoded_nb_read_frames")
        self.assertEqual(run.call_count, 2)
        self.assertIn("-count_frames", run.call_args.args[0])

    def test_recording_validation_rejects_unadmitted_source_frame(self) -> None:
        app, _pipeline = self.build(self.config())
        app.artifacts.raw_video.write_bytes(b"mp4")
        with mock.patch.object(
            runner.ServiceMakerApp,
            "_video_frame_probe",
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

    def test_frame_operator_preserves_camera_id_and_source_sequence(self) -> None:
        output = self.root / "frames.csv"
        operator = runner.FrameCsvOperator(output, meta_type=4242)
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

        operator.handle_metadata(SimpleNamespace(frame_items=[frame_meta]))
        operator.close()

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
        operator = runner.FrameCsvOperator(output, meta_type=4242)
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

    def test_successful_cleanup_removes_only_recovery_artifacts(self) -> None:
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

        self.assertFalse((self.root / "capture_cam0.jsonl").exists())
        self.assertFalse((self.root / "record_admission_cam1.csv").exists())
        self.assertFalse(inference.exists())
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
        with mock.patch(
            "scripts.align_run_outputs_streaming.build_alignment",
            return_value=summary,
        ):
            with self.assertRaisesRegex(RuntimeError, "alignment validation failed"):
                post_run.align_run(self.root)

        progress = run_context.read_json(self.root / post_run.PROGRESS_FILENAME)
        self.assertEqual(progress["stage"], "failed")
        self.assertFalse(progress["alignment_validation_passed"])
        self.assertFalse(progress["overall_validation_passed"])
        self.assertEqual(run_context.read_json(self.root / "run_status.json")["state"], "analysis_failed")


if __name__ == "__main__":
    unittest.main()
