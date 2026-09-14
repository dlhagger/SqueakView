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


class FakePipeline:
    def __init__(self, name: str):
        self.name = name
        self.nodes: dict[str, tuple[str, dict]] = {}
        self.links: list[tuple] = []
        self.attachments: list[tuple] = []
        self.stopped = False
        self.waited = False
        self.started = False

    def add(self, type_name: str, name: str, properties: dict | None = None):
        self.nodes[name] = (type_name, properties or {})
        return self

    def link(self, *args):
        self.links.append(args)
        return self

    def attach(self, target: str, what, name="", tips="", properties=None):
        self.attachments.append((target, what, name, tips, properties))
        return self

    def stop(self):
        self.stopped = True
        return self

    def start(self, _callback):
        self.started = True
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
            mock.patch.object(
                inference_main.squeakview_config,
                "resolve_workspace_path",
                return_value=self.infer_config,
            ),
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

    def test_safe_print_cannot_break_shutdown_when_parent_pipe_is_closed(self) -> None:
        with mock.patch("builtins.print", side_effect=BrokenPipeError):
            runner._safe_print("shutdown")

    def test_ready_message_emits_versioned_event(self) -> None:
        app = runner.ServiceMakerApp(self.config(enable_infer=False))
        app._ready_origin = "sink"

        class ReadyMessage:
            new_state = runner.PipelineState.PLAYING
            origin = "sink"

        with (
            mock.patch.object(runner, "StateTransitionMessage", ReadyMessage),
            mock.patch.object(runner, "_safe_print") as safe_print,
        ):
            app._on_message(ReadyMessage())

        events = [
            decode_child_event(call.args[0])
            for call in safe_print.call_args_list
        ]
        ready = [event for event in events if event is not None]
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].type, "pipeline_ready")
        self.assertEqual(ready[0].payload["run_dir"], str(app.run_dir))

    def test_unsolicited_live_eos_is_fatal_but_requested_eos_is_clean(self) -> None:
        class EndMessage:
            pass

        with mock.patch.object(runner, "EOSMessage", EndMessage):
            unexpected = runner.ServiceMakerApp(self.config(enable_infer=False))
            unexpected._on_message(EndMessage())
            requested = runner.ServiceMakerApp(self.config(enable_infer=False))
            requested.request_stop()
            requested._on_message(EndMessage())

        self.assertEqual(unexpected.exit_code, 1)
        self.assertEqual(requested.exit_code, 0)

    def test_stop_attempts_every_closer_and_persists_all_errors(self) -> None:
        app = runner.ServiceMakerApp(self.config(enable_infer=False))
        pipeline = mock.Mock()
        pipeline.stop.side_effect = OSError("pipeline stop failed")
        observations = mock.Mock()
        observations.close.side_effect = OSError("observation flush failed")
        admission_failed = mock.Mock()
        admission_failed.close.side_effect = OSError("admission flush failed")
        admission_ok = mock.Mock()
        telemetry = mock.Mock()
        frames = mock.Mock()
        app.pipeline = pipeline
        app.observations = observations
        app.record_admissions = [admission_failed, admission_ok]
        app.record_telemetry = [telemetry]
        app.frames = frames

        app.stop()

        pipeline.stop.assert_called_once_with()
        pipeline.wait.assert_called_once_with()
        observations.close.assert_called_once_with()
        admission_failed.close.assert_called_once_with()
        admission_ok.close.assert_called_once_with()
        telemetry.close.assert_called_once_with()
        frames.close.assert_called_once_with()
        self.assertEqual(app.exit_code, 1)
        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertEqual(status["state"], "capture_closed")
        self.assertEqual(len(status["capture_close_errors"]), 3)
        self.assertIn("pipeline stop failed", status["capture_close_error"])

    def test_pipeline_shutdown_timeout_returns_and_marks_capture_invalid(self) -> None:
        release = threading.Event()
        app = runner.ServiceMakerApp(self.config(enable_infer=False))
        pipeline = mock.Mock()
        pipeline.stop.side_effect = release.wait
        app.pipeline = pipeline

        with mock.patch.dict(
            "os.environ", {"SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S": "0.1"}
        ):
            app.stop()
        release.set()

        self.assertEqual(app.exit_code, 1)
        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertIn("timed out", status["capture_close_error"])

    def test_recording_integrity_fault_bypasses_native_pipeline_shutdown(self) -> None:
        app = runner.ServiceMakerApp(self.config(enable_infer=False))
        pipeline = mock.Mock()
        app.pipeline = pipeline

        app._recording_fault("recording queue backlog reached fatal threshold")
        app.stop()

        pipeline.stop.assert_not_called()
        pipeline.wait.assert_not_called()
        self.assertEqual(app.exit_code, 4)
        self.assertTrue(app._force_process_exit)
        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertEqual(status["capture_exit_code"], 4)
        self.assertIn("native shutdown bypassed", status["capture_close_error"])

    def test_storage_reserve_fault_uses_normal_pipeline_drain(self) -> None:
        app = runner.ServiceMakerApp(self.config(enable_infer=False))
        pipeline = mock.Mock()
        app.pipeline = pipeline

        with mock.patch.object(runner, "_safe_print") as safe_print:
            app._storage_reserve_fault("reserve exhausted")
            app.stop()

        pipeline.stop.assert_called_once_with()
        pipeline.wait.assert_called_once_with()
        self.assertEqual(app.exit_code, 5)
        self.assertFalse(app._force_process_exit)
        events = [
            decode_child_event(call.args[0]) for call in safe_print.call_args_list
        ]
        fatal = [event for event in events if event is not None and event.type == "fatal"]
        self.assertEqual(len(fatal), 1)
        self.assertIn("reserve exhausted", fatal[0].payload["error"])

    def test_capture_process_owns_system_telemetry_lifecycle(self) -> None:
        order: list[str] = []
        telemetry_kwargs: dict[str, object] = {}

        class OrderedPipeline(FakePipeline):
            def start(self, callback):
                del callback
                order.append("pipeline_start")
                return super().start(None)

            def stop(self):
                order.append("pipeline_stop")
                return super().stop()

            def wait(self):
                order.append("pipeline_wait")
                return super().wait()

        class FakeSystemTelemetry:
            def __init__(self, path, **kwargs):
                self.path = path
                self.sample_count = 2
                self.last_error = None
                telemetry_kwargs.update(kwargs)

            def start(self):
                order.append("telemetry_start")
                return True

            def stop(self):
                order.append("telemetry_stop")

        app = runner.ServiceMakerApp(
            self.config(enable_infer=False),
            pipeline_factory=OrderedPipeline,
            probe_factory=lambda name, operator: (name, operator),
            system_telemetry_factory=FakeSystemTelemetry,
        )
        app.build()
        app.request_stop()

        self.assertEqual(app.run(), 0)
        self.assertLess(order.index("telemetry_start"), order.index("pipeline_start"))
        self.assertLess(order.index("pipeline_wait"), order.index("telemetry_stop"))
        shutdown_requested = telemetry_kwargs["shutdown_requested"]
        self.assertTrue(callable(shutdown_requested))
        self.assertTrue(shutdown_requested())
        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertEqual(status["system_telemetry"]["sample_count"], 2)

    def test_system_telemetry_loss_is_persisted_without_aborting_recording(self) -> None:
        app = runner.ServiceMakerApp(self.config(enable_infer=False))

        app._system_telemetry_fault("tegrastats reader stopped")

        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertTrue(status["system_telemetry_degraded"])
        self.assertEqual(
            status["system_telemetry_error"], "tegrastats reader stopped"
        )
        self.assertFalse(app._stop_event.is_set())

    def test_stop_wakes_extracted_recording_liveness_monitor_before_join(self) -> None:
        app = runner.ServiceMakerApp(self.config(enable_infer=False))
        app._recording_liveness_monitor = runner.RecordingLivenessMonitor(
            [],
            app._stop_event,
            app._recording_fault,
            runner.resolve_recording_liveness_policy(
                app.config.fps,
                failure_injection=False,
            ),
        )
        app._recording_liveness_thread = threading.Thread(
            target=app._recording_liveness_monitor.run,
            daemon=True,
        )
        app._recording_liveness_thread.start()

        app.stop()

        self.assertTrue(app._stop_event.is_set())
        self.assertFalse(app._recording_liveness_thread.is_alive())

    def test_unexpected_pipeline_end_wakes_runner_and_fails_capture(self) -> None:
        app = runner.ServiceMakerApp(
            self.config(enable_infer=False),
            pipeline_factory=FakePipeline,
            probe_factory=lambda name, operator: (name, operator),
        )
        app.build()

        result = app.run()

        self.assertEqual(result, 1)
        status = run_context.read_json(app.run_dir / "run_status.json")
        self.assertEqual(status["capture_exit_code"], 1)

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
                                "capture-log-path": str(
                                    run_dir / f"capture_cam{index}.jsonl"
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
            "scripts.align_run_outputs_streaming.build_alignment",
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
            "scripts.align_run_outputs_streaming.build_alignment",
            return_value=summary,
        ):
            post_run.align_run(self.root)

        status = run_context.read_json(self.root / "run_status.json")
        self.assertEqual(status["state"], "analysis_complete")
        self.assertEqual(status["stage"], "complete")
        self.assertIsNone(status["error"])
        self.assertTrue(status["analysis_complete"])


if __name__ == "__main__":
    unittest.main()
