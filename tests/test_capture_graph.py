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

class CaptureGraphTests(ServiceMakerCase):
    def test_runner_preserves_frame_audit_compatibility_exports(self) -> None:
        self.assertIs(runner.FrameCsvOperator, frame_audit.FrameCsvOperator)
        self.assertIs(
            runner.FLIR_FRAME_META_DESCRIPTOR,
            frame_audit.FLIR_FRAME_META_DESCRIPTOR,
        )


    def test_runner_preserves_immutable_config_compatibility_export(self) -> None:
        config = contracts.InferenceConfig(enable_infer=False)

        self.assertIs(runner.InferenceConfig, contracts.InferenceConfig)
        self.assertIs(runner._read_config_value, contracts._read_config_value)
        self.assertIs(runner._load_class_names, contracts._load_class_names)
        self.assertIs(runner._validate_config, contracts._validate_config)
        self.assertEqual(contracts.load_class_names(self.infer_config), ["mouse", "bottle"])
        with self.assertRaises(FrozenInstanceError):
            config.fps = 60


    def test_inference_cli_constructs_the_extracted_config_contract(self) -> None:
        args = SimpleNamespace(
            cfg=self.infer_config,
            capture_backend="flir_direct",
            num_cameras=1,
            camera_serial=["25187166"],
            pixel_format="Mono8",
            trigger="on",
            trigger_activation="falling",
            exposure_us=9000.0,
            gain=1.5,
            width=640,
            height=480,
            fps=30,
            bitrate=4000,
            preview_socket=[self.root / "preview.sock"],
            disable_infer=False,
            run_dir=self.root / "cli-run",
        )
        with (
            mock.patch.object(inference_main, "parse_args", return_value=args),
            mock.patch.object(inference_main.runner, "run", return_value=7) as run,
        ):
            returncode = inference_main.main()

        self.assertEqual(returncode, 7)
        config = run.call_args.args[0]
        self.assertIsInstance(config, contracts.InferenceConfig)
        self.assertEqual(config.camera_serials, ("25187166",))
        self.assertEqual(config.preview_sockets, (self.root / "preview.sock",))
        self.assertTrue(config.trigger_on)
        self.assertEqual(config.trigger_activation, "falling")


    def test_runtime_rejects_legacy_pose_schema(self) -> None:
        payload = json.loads(self.pose_sidecar.read_text())
        payload["schema_version"] = 1
        self.pose_sidecar.write_text(json.dumps(payload))

        with self.assertRaisesRegex(ValueError, "version 2 is required"):
            load_pose_schema(self.infer_config, ["mouse", "bottle"])


    def test_runtime_rejects_duplicate_or_oversized_pose_schema(self) -> None:
        self.pose_sidecar.write_text('{"schema_version":2,"schema_version":2}')
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            load_pose_schema(self.infer_config, ["mouse", "bottle"])

        self.pose_sidecar.write_bytes(b" " * (4 * 1024 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            load_pose_schema(self.infer_config, ["mouse", "bottle"])


    def test_orin_nano_tracker_uses_stable_cuda_nvdcf(self) -> None:
        tracker_config = Path(runner.__file__).resolve().parents[3] / "configs/tracker_mouse_nvdcf.yml"
        text = tracker_config.read_text()

        self.assertIn("visualTrackerType: 1", text)
        self.assertNotIn("visualTrackerType: 2", text)


    def test_builds_flir_recording_inference_and_headless_display_paths(self) -> None:
        app, pipeline = self.build(self.config())

        self.assertEqual(pipeline.nodes["flirsrc0"][0], "flirspinsrc")
        self.assertEqual(
            pipeline.nodes["flirsrc0"][1]["capture-health-path"],
            str(app.run_dir / "capture_health_cam0.json"),
        )
        self.assertEqual(
            pipeline.nodes["flirsrc0"][1]["frame-manifest-path"],
            str(app.run_dir / "frames.csv"),
        )
        self.assertEqual(
            pipeline.nodes["flirsrc0"][1]["camera-telemetry-path"],
            str(app.run_dir / "diagnostics/camera.csv"),
        )
        self.assertEqual(
            pipeline.nodes["flirsrc0"][1]["error-log-path"],
            str(app.run_dir / "diagnostics/errors.csv"),
        )
        self.assertEqual(
            pipeline.nodes["flirsrc0"][1]["camera-runtime-path"],
            str(app.run_dir / "diagnostics/camera_runtime.json"),
        )
        self.assertEqual(pipeline.nodes["record_queue0"][1]["leaky"], 0)
        self.assertFalse(pipeline.nodes["record_queue0"][1]["flush-on-eos"])
        self.assertEqual(pipeline.nodes["record_queue0"][1]["max-size-buffers"], 120)
        self.assertEqual(pipeline.nodes["flirsrc0"][1]["stream-buffer-count"], 64)
        self.assertNotIn("record_convert0", pipeline.nodes)
        self.assertNotIn("record_caps0", pipeline.nodes)
        self.assertEqual(pipeline.nodes["record_encoder0"][0], "x264enc")
        self.assertFalse(pipeline.nodes["record_encoder0"][1]["sliced-threads"])
        self.assertFalse(pipeline.nodes["record_encoder0"][1]["qos"])
        self.assertEqual(pipeline.nodes["record_encoder0"][1]["ref"], 1)
        self.assertEqual(
            pipeline.nodes["record_encoder0"][1]["option-string"], "aq-mode=0"
        )
        self.assertEqual(pipeline.nodes["infer_convert0"][1]["compute-hw"], 2)
        self.assertEqual(pipeline.nodes["infer_convert0"][1]["copy-hw"], 2)
        self.assertEqual(
            pipeline.nodes["infer_convert0"][1]["nvbuf-memory-type"], 4
        )
        self.assertEqual(pipeline.nodes["infer_queue0"][1]["leaky"], 2)
        self.assertEqual(pipeline.nodes["infer_queue0"][1]["max-size-buffers"], 32)
        self.assertEqual(pipeline.nodes["mux"][0], "nvstreammux")
        self.assertNotIn("width", pipeline.nodes["mux"][1])
        self.assertNotIn("height", pipeline.nodes["mux"][1])
        self.assertNotIn("live-source", pipeline.nodes["mux"][1])
        self.assertEqual(pipeline.nodes["mux"][1]["max-latency"], 0)
        self.assertEqual(pipeline.nodes["infer"][0], "nvinfer")
        self.assertEqual(pipeline.nodes["tracker"][0], "nvtracker")
        self.assertEqual(pipeline.nodes["tracker"][1]["operate-on-class-ids"], "0")
        self.assertEqual(pipeline.nodes["tracker"][1]["tracking-id-reset-mode"], 3)
        self.assertEqual(pipeline.nodes["record_sink0"][1]["location"], str(app.artifacts.raw_video))
        self.assertFalse(pipeline.nodes["record_sink0"][1]["qos"])
        self.assertEqual(pipeline.nodes["sink"][0], "fakesink")
        self.assertNotIn("osd", pipeline.nodes)
        self.assertEqual(app.frames.path, app.run_dir / "inference" / "frames.csv")
        self.assertTrue(
            any(link == (("infer_caps0", "mux"), ("", "sink_%u")) for link in pipeline.links)
        )
        self.assertTrue(any(item[0] == "mux" for item in pipeline.attachments))
        self.assertTrue(any(item[0] == "infer" for item in pipeline.attachments))
        self.assertTrue(any(item[0] == "tracker" for item in pipeline.attachments))
        self.assertTrue(any(item[0] == "record_queue0" for item in pipeline.attachments))
        app.stop()
        self.assertTrue(pipeline.stopped)
        self.assertTrue(pipeline.waited)


    def test_debug_profile_attaches_shipped_probes_only_after_leaky_work(self) -> None:
        for enable_infer, target in ((True, "tracker"), (False, "mux")):
            with (
                self.subTest(enable_infer=enable_infer),
                mock.patch.dict(
                    debug_instrumentation.os.environ,
                    {debug_instrumentation.DEBUG_PROFILE_ENV: "1"},
                ),
                mock.patch.object(
                    debug_instrumentation, "validate_probe_modules", return_value={}
                ),
            ):
                app, pipeline = self.build(
                    self.config(enable_infer=enable_infer, cfg_path=(
                        self.infer_config if enable_infer else None
                    ))
                )
                self.assertIn(
                    (
                        target,
                        "measure_latency_probe",
                        "squeakview_latency",
                        "",
                        None,
                    ),
                    pipeline.attachments,
                )
                self.assertIn(
                    (
                        target,
                        "measure_fps_probe",
                        "squeakview_fps",
                        "",
                        {"interval": debug_instrumentation.FPS_INTERVAL_SECONDS},
                    ),
                    pipeline.attachments,
                )
                self.assertEqual(pipeline.nodes[f"infer_queue0"][1]["leaky"], 2)
                app.stop()


    def test_recording_graph_is_golden_for_one_and_two_cameras(self) -> None:
        for camera_count in (1, 2):
            with self.subTest(camera_count=camera_count):
                run_dir = self.root / f"golden-{camera_count}"
                app, pipeline = self.build(
                    self.config(
                        cfg_path=None,
                        enable_infer=False,
                        num_cameras=camera_count,
                        run_dir=run_dir,
                    )
                )

                self.assertEqual(len(app.record_admissions), camera_count)
                self.assertEqual(len(app.record_telemetry), camera_count)
                for index in range(camera_count):
                    expected_video = (
                        app.artifacts.raw_video
                        if index == 0
                        else run_dir / f"raw_cam{index}.mp4"
                    )
                    self.assertEqual(
                        pipeline.nodes[f"flirsrc{index}"],
                        (
                            "flirspinsrc",
                            {
                                "camera-index": index,
                                "width": 640,
                                "height": 480,
                                "fps": 30,
                                "pixel-format": "Mono8",
                                "trigger": False,
                                "trigger-activation": "rising",
                                "exposure-us": 10000.0,
                                "gain": -1.0,
                                "drop-incomplete": False,
                                "buffer-handling": "OldestFirst",
                                "stream-buffer-count": 64,
                                "capture-health-path": str(
                                    run_dir / f"capture_health_cam{index}.json"
                                ),
                                "frame-manifest-path": str(
                                    run_dir
                                    / (
                                        "frames.csv"
                                        if index == 0
                                        else f"frames_cam{index}.csv"
                                    )
                                ),
                                "camera-telemetry-path": str(
                                    run_dir
                                    / "diagnostics"
                                    / (
                                        "camera.csv"
                                        if index == 0
                                        else f"camera_cam{index}.csv"
                                    )
                                ),
                                "error-log-path": str(
                                    run_dir
                                    / "diagnostics"
                                    / (
                                        "errors.csv"
                                        if index == 0
                                        else f"errors_cam{index}.csv"
                                    )
                                ),
                                "camera-runtime-path": str(
                                    run_dir
                                    / "diagnostics"
                                    / (
                                        "camera_runtime.json"
                                        if index == 0
                                        else f"camera_runtime_cam{index}.json"
                                    )
                                ),
                                "metadata-profile": "scientific",
                                "max-consecutive-timeouts": 10,
                            },
                        ),
                    )
                    self.assertEqual(
                        pipeline.nodes[f"source_caps{index}"],
                        (
                            "capsfilter",
                            {
                                "caps": (
                                    "video/x-raw,format=GRAY8,width=640,height=480,"
                                    "framerate=30/1"
                                )
                            },
                        ),
                    )
                    self.assertEqual(pipeline.nodes[f"camera_tee{index}"], ("tee", {}))
                    self.assertEqual(
                        pipeline.nodes[f"record_queue{index}"],
                        (
                            "queue",
                            {
                                "max-size-buffers": 120,
                                "max-size-bytes": 0,
                                "max-size-time": 0,
                                "leaky": 0,
                                "flush-on-eos": False,
                            },
                        ),
                    )
                    self.assertEqual(
                        pipeline.nodes[f"record_encoder{index}"],
                        (
                            "x264enc",
                            {
                                "tune": 4,
                                "speed-preset": 1,
                                "bitrate": 4000,
                                "key-int-max": 30,
                                "ref": 1,
                                "option-string": "aq-mode=0",
                                "bframes": 0,
                                "rc-lookahead": 0,
                                "sync-lookahead": 0,
                                "sliced-threads": False,
                                "vbv-buf-capacity": 100,
                                "qos": False,
                            },
                        ),
                    )
                    self.assertEqual(
                        pipeline.nodes[f"record_parser{index}"], ("h264parse", {})
                    )
                    self.assertEqual(
                        pipeline.nodes[f"record_muxer{index}"], ("mp4mux", {})
                    )
                    self.assertEqual(
                        pipeline.nodes[f"record_sink{index}"],
                        (
                            "filesink",
                            {
                                "location": str(expected_video),
                                "qos": False,
                                "sync": False,
                            },
                        ),
                    )
                    self.assertIn(
                        (f"flirsrc{index}", f"source_caps{index}", f"camera_tee{index}"),
                        pipeline.links,
                    )
                    self.assertIn(
                        (
                            f"camera_tee{index}",
                            f"record_queue{index}",
                            f"record_encoder{index}",
                            f"record_parser{index}",
                            f"record_muxer{index}",
                            f"record_sink{index}",
                        ),
                        pipeline.links,
                    )
                    attached = {
                        item[0]: item[1][0]
                        for item in pipeline.attachments
                        if item[0] in {
                            f"source_caps{index}",
                            f"record_queue{index}",
                            f"record_parser{index}",
                        }
                    }
                    self.assertEqual(
                        attached,
                        {
                            f"source_caps{index}": f"record_ingress{index}",
                            f"record_queue{index}": f"record_admission{index}",
                            f"record_parser{index}": f"record_egress{index}",
                        },
                    )
                app.stop()


    def test_camera_property_compatibility_wrapper_delegates_to_builder(self) -> None:
        app = runner.ServiceMakerApp(self.config(enable_infer=False))

        self.assertEqual(
            app._camera_properties(0),
            pipeline_builder.camera_source_properties(app.config, app.run_dir, 0),
        )
        self.assertEqual(runner._flir_pixel_format("GRAY8"), "Mono8")
        app.stop()


    def test_scientific_capture_policy_fails_before_non_leaky_queue_fills(self) -> None:
        for fps in (1, 30, 60, 120):
            policy = capture_buffer_policy(fps)
            self.assertLess(policy.record_warning_frames, policy.record_failure_frames)
            self.assertLess(policy.record_failure_frames, policy.record_queue_frames)

        _app, pipeline = self.build(self.config(fps=60))
        record_queue = pipeline.nodes["record_queue0"][1]
        policy = capture_buffer_policy(60)
        self.assertEqual(record_queue["leaky"], 0)
        self.assertEqual(record_queue["max-size-buffers"], policy.record_queue_frames)


    def test_failure_plan_is_gated_applied_to_one_source_and_marks_nonproduction(self) -> None:
        plan = FailurePlan(
            schema_version="1.0",
            target="flir_source",
            kind="source_read",
            after_frames=12,
            stream_id=1,
        )
        with mock.patch.dict(
            "os.environ", {"SQUEAKVIEW_ENABLE_FAILURE_INJECTION": "1"}
        ):
            app, pipeline = self.build(
                self.config(
                    num_cameras=2,
                    enable_infer=False,
                    failure_plan=plan,
                )
            )

        self.assertNotIn("fault-after-frames", pipeline.nodes["flirsrc0"][1])
        self.assertEqual(pipeline.nodes["flirsrc1"][1]["fault-after-frames"], 12)
        self.assertEqual(pipeline.nodes["flirsrc1"][1]["fault-kind"], "source_read")
        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertFalse(status["production_eligible"])
        self.assertEqual(status["failure_injection"]["after_frames"], 12)
        app.stop()


    def test_recording_failure_plans_insert_only_the_requested_fault(self) -> None:
        cases = (
            ("record_queue", "stall", "fault_record_queue0", None, None),
            ("encoder", "error", "fault_encoder0", "error-after", 7),
            ("muxer", "error", "fault_muxer0", "error-after", 7),
            ("filesink", "error", "fault_filesink0", "error-after", 7),
        )
        with mock.patch.dict(
            "os.environ", {"SQUEAKVIEW_ENABLE_FAILURE_INJECTION": "1"}
        ):
            for target, kind, node, prop, expected in cases:
                with self.subTest(target=target):
                    plan = FailurePlan(
                        schema_version="1.0",
                        target=target,
                        kind=kind,
                        after_frames=7,
                        stream_id=0,
                        delay_us=(250_000 if target == "record_queue" else 0),
                    )
                    app, pipeline = self.build(
                        self.config(
                            run_dir=self.root / f"fault-{target}",
                            enable_infer=False,
                            failure_plan=plan,
                        )
                    )
                    if prop is None:
                        stall_probe = next(
                            item[1]
                            for item in pipeline.attachments
                            if item[0] == node
                        )
                        self.assertEqual(stall_probe[1].after_frames, 7)
                        self.assertEqual(stall_probe[1].delay_s, 0.25)
                    else:
                        self.assertEqual(pipeline.nodes[node][1][prop], expected)
                    self.assertEqual(
                        pipeline.nodes["record_sink0"][1]["location"],
                        str(app.artifacts.raw_video),
                    )
                    app.stop()


    def test_disk_full_plan_redirects_only_selected_filesink(self) -> None:
        plan = FailurePlan(
            schema_version="1.0",
            target="filesink",
            kind="disk_full",
            after_frames=1,
            stream_id=1,
        )
        with mock.patch.dict(
            "os.environ", {"SQUEAKVIEW_ENABLE_FAILURE_INJECTION": "1"}
        ):
            app, pipeline = self.build(
                self.config(
                    run_dir=self.root / "fault-disk",
                    num_cameras=2,
                    enable_infer=False,
                    failure_plan=plan,
                )
            )

        self.assertEqual(
            pipeline.nodes["record_sink0"][1]["location"],
            str(app.artifacts.raw_video),
        )
        self.assertEqual(pipeline.nodes["record_sink1"][1]["location"], "/dev/full")
        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertFalse(status["production_eligible"])
        self.assertEqual(
            status["failure_injection"],
            {
                "schema_version": "1.0",
                "target": "filesink",
                "kind": "disk_full",
                "after_frames": 1,
                "stream_id": 1,
                "delay_us": 0,
            },
        )
        app.stop()


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
        self.assertEqual(pipeline.nodes["preview_admission0"], ("identity", {}))
        self.assertIn(
            (("preview_demux", "preview_admission0"), ("src_0", "")),
            pipeline.links,
        )
        self.assertIn(
            ("preview_admission0", "preview_queue0", "preview_sink0"),
            pipeline.links,
        )
        attached = {item[0]: item[1][0] for item in pipeline.attachments}
        self.assertEqual(attached["preview_admission0"], "preview_admission_probe0")
        self.assertEqual(attached["preview_queue0"], "preview_delivery_probe0")
        self.assertEqual(len(app.preview_boundaries), 2)
        app.stop()


    def test_disable_infer_omits_infer_and_metadata_writer(self) -> None:
        app, pipeline = self.build(self.config(cfg_path=None, enable_infer=False))

        self.assertNotIn("infer", pipeline.nodes)
        self.assertIsNotNone(app.frames)
        self.assertTrue(any(item[0] == "mux" for item in pipeline.attachments))
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


    def test_duplicate_batch_size_is_rejected(self) -> None:
        self.infer_config.write_text("[property]\nbatch-size=1\nbatch-size=1\n")

        with self.assertRaisesRegex(ValueError, "duplicate key 'batch-size'"):
            runner.ServiceMakerApp(self.config())


    def test_preview_socket_count_must_match_camera_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "preview socket count"):
            runner.ServiceMakerApp(
                self.config(
                    num_cameras=1,
                    preview_sockets=(self.root / "one.sock", self.root / "two.sock"),
                )
            )


    def test_direct_runtime_rejects_unsafe_camera_scalars(self) -> None:
        for updates, message in (
            ({"pixel_format": " "}, "pixel_format"),
            ({"trigger_activation": "sideways"}, "trigger_activation"),
            ({"exposure_us": float("nan")}, "exposure_us"),
            ({"gain": float("nan")}, "gain"),
            ({"gain": -2.0}, "gain"),
            ({"gain": -0.5}, "gain"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    runner.ServiceMakerApp(self.config(**updates))


