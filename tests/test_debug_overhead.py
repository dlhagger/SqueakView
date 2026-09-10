from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from squeakview.common.diagnostics import debug_overhead
from squeakview.common.diagnostics.debug_overhead import (
    MAX_THRESHOLD_BYTES,
    compare_debug_overhead,
    load_debug_thresholds,
    scan_deepstream_latency_log,
)
from squeakview.common.diagnostics.evidence_identity import capture_source_evidence


_METRICS = (
    "recording.queue_wait_ms",
    "recording.encoder_latency_ms",
    "recording.encoder_in_flight",
    "system.cpu_util_mean_pct",
    "system.gpu_util_pct",
    "system.temp_max_c",
    "system.vdd_in_current_mw",
)


class DebugOverheadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.limits_path = self.root / "limits.yaml"
        self.limits_path.write_text("schema_version: '1.0'\nprofile_id: run-qualification-v1\n")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_checked_in_threshold_template_is_strict_and_unapproved(self) -> None:
        template = load_debug_thresholds(
            Path("qualification/debug_overhead_thresholds.v1.yaml")
        )

        self.assertIs(template["approved"], False)
        self.assertEqual(set(template["metrics"]), set(_METRICS))
        self.assertTrue(
            all(
                policy == {
                    "max_increase": None,
                    "max_percent_increase": None,
                }
                for policy in template["metrics"].values()
            )
        )

    def _run(
        self,
        name: str,
        *,
        debug: bool,
        cpu: float = 40,
        duration: float = 3600,
    ) -> Path:
        run_dir = self.root / name
        run_dir.mkdir()
        task_bytes = b"task_name: overhead\n"
        (run_dir / "config").mkdir()
        (run_dir / "config/task.yaml").write_bytes(task_bytes)
        factors = {
            "width": 10,
            "height": 20,
            "fps": 30,
            "camera_count": 1,
            "pixel_format": "Mono8",
            "trigger_enabled": True,
            "inference_enabled": True,
            "preview_enabled": True,
            "model_name": "model-a",
            "nvpmodel": "25W",
            "git_commit": "abc",
        }
        summary = {
            "result": "passed",
            "qualification_exceptions": (
                ["deepstream_debug_profile"] if debug else []
            ),
            "limits": {
                "path": str(self.limits_path),
                "schema_version": "1.0",
                "profile_id": "run-qualification-v1",
                "validated": True,
            },
            "capture_duration_s": duration,
            "factors": factors,
            "recording_telemetry": {
                "maxima": {
                    "queue_wait_ms": 1,
                    "encoder_latency_ms": 2,
                    "encoder_in_flight": 3,
                }
            },
            "system_telemetry": {
                "resource_maxima": {
                    "cpu_util_mean_pct": cpu,
                    "gpu_util_pct": 50,
                    "temp_max_c": 55,
                    "vdd_in_current_mw": 8000,
                }
            },
        }
        (run_dir / "qualification_summary.json").write_text(json.dumps(summary))
        model = {
            "model_manifest_sha256": "a" * 64,
            "pose_sidecar_sha256": "b" * 64,
            "onnx_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "engine_sha256": "e" * 64,
            "engine_build_identity": {"tensorrt_version": "10.16.2"},
        }
        qualification_binding = {
            "matrix_id": "debug-overhead-test-matrix",
            "matrix_path": str(self.root / "matrix.yaml"),
            "matrix_sha256": "8" * 64,
            "case_id": "capture--one-hour--infer-on--preview-on--25w",
            "expected_factors": {
                "width": 10,
                "height": 20,
                "fps": 30,
                "camera_count": 1,
                "pixel_format": "Mono8",
                "inference_enabled": True,
                "preview_enabled": True,
                "power_mode": "25W",
            },
        }
        (run_dir / "run_manifest.json").write_text(json.dumps({
            "process_topology": {"acquisition_owner": "durable_supervisor"},
            "qualification": qualification_binding,
            "capture": {
                "backend": "flir_direct",
                "num_cameras": 1,
                "camera_serials": ["camera-a"],
                "width": 10,
                "height": 20,
                "fps": 30,
                "pixel_format": "Mono8",
                "trigger_on": True,
                "trigger_activation": "rising",
                "arduino_fps": 30,
                "exposure_us": 10000.0,
            },
            "serial": {
                "enabled": True,
                "port": "/dev/ttyACM0",
                "baud": 115200,
                "controller_protocol": "legacy",
            },
            "task_config": {
                "snapshot_path": "config/task.yaml",
                "size_bytes": len(task_bytes),
                "sha256": hashlib.sha256(task_bytes).hexdigest(),
            },
            "observability": {
                "deepstream_debug_profile": debug,
                "deepstream_log": "diagnostics/deepstream.log",
                "deepstream_log_max_bytes": 64 * 1024 * 1024,
                "deepstream_debug_probes": ({
                    "measure_latency_probe": {
                        "available": True, "size_bytes": 10, "sha256": "a" * 64,
                    },
                    "measure_fps_probe": {
                        "available": True, "size_bytes": 11, "sha256": "b" * 64,
                    },
                } if debug else None),
            },
            "inference": {
                "enabled": True,
                "bitrate_kbps": 4000,
                "preview_enabled": True,
                "model_package": model,
            },
            "platform": {
                "device_model": "NVIDIA Jetson Orin Nano Super",
                "machine": "aarch64",
                "kernel": "6.8.12-tegra",
                "python": "3.12.3",
                "python_executable": "/usr/bin/python3",
                "jetson_linux_release": "R39.2.1",
                "deepstream_build": "DeepStreamSDK 9.1",
                "nvpmodel": "NV Power Mode: 25W",
                "packages": {
                    "nvidia-l4t-core": "39.2.1",
                    "deepstream-9.1": "9.1.0",
                    "cuda-toolkit-13-2": "13.2.0",
                    "libcudnn9-cuda-13": "9.20.0",
                    "libnvinfer10": "10.16.2",
                    "libgstreamer1.0-0": "1.24.2",
                    "libspinnaker": "4.2.0",
                    "ffmpeg": "8.0.1-nvidia1",
                },
            },
            "native_plugins": {
                "flir_gstreamer_source": {
                    "available": True, "size_bytes": 100, "sha256": "f" * 64,
                },
                "deepstream_yolo_parser": {
                    "available": True, "size_bytes": 200, "sha256": "9" * 64,
                },
            },
        }))
        diagnostics = run_dir / "diagnostics"
        diagnostics.mkdir()
        (run_dir / "raw.mp4").write_bytes(b"video")
        (run_dir / "capture_cam0.jsonl").write_bytes(b"capture\n")
        (run_dir / "record_admission.csv").write_bytes(b"admission\n")
        (diagnostics / "camera_runtime.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "metadata_type": "SQUEAKVIEW.FLIR.FRAME_META.v1",
                    "cameras": [
                        {
                            "camera_index": 0,
                            "camera_serial": "camera-a",
                            "device_model": "Blackfly",
                            "firmware_version": "1",
                            "source_width": 10,
                            "source_height": 20,
                            "source_pixel_format": "Mono8",
                            "actual_fps": 30,
                            "configured_exposure_us": 10000.0,
                            "configured_gain_db": -1.0,
                            "configured_stream_buffer_count": 120,
                            "timestamp_increment_ns": 33333333,
                            "timestamp_latch_available": True,
                            "enabled_chunks": ["FrameID", "Timestamp"],
                        }
                    ],
                }
            )
        )
        (run_dir / "run_status.json").write_text(
            json.dumps(
                {"state": "finalized", "qualification": qualification_binding}
            )
        )
        (diagnostics / "system.csv").write_text("sample\n1\n")
        (diagnostics / "recording.csv").write_text("sample\n1\n")
        (diagnostics / "deepstream.log").write_text(
            "Source id = 0 Frame_num = 12 Frame latency = 4.25 (ms)\n"
            "Comp name = nvinfer0 Component latency = 1.25\n"
            "**FPS:  30.00 (30.00)\n"
            if debug
            else "ordinary capture output\n"
        )
        summary["source_evidence"] = capture_source_evidence(
            run_dir, self.limits_path
        )
        (run_dir / "qualification_summary.json").write_text(json.dumps(summary))
        return run_dir

    @staticmethod
    def _thresholds(limit: float = 100) -> dict:
        return {
            "schema_version": "1.0",
            "profile_id": "approved-test-policy",
            "approved": True,
            "metrics": {
                name: {"max_increase": limit} for name in _METRICS
            },
        }

    def test_matched_pair_reports_metric_deltas(self) -> None:
        baseline = self._run("baseline", debug=False, cpu=40)
        debug = self._run("debug", debug=True, cpu=50)
        baseline_summary_bytes = (baseline / "qualification_summary.json").read_bytes()

        report = compare_debug_overhead(baseline, debug)

        self.assertTrue(report["comparable"])
        self.assertEqual(report["result"], "incomplete")
        cpu = report["metric_deltas"]["system.cpu_util_mean_pct"]
        self.assertEqual(cpu["delta"], 10)
        self.assertEqual(cpu["percent_delta"], 25)
        self.assertEqual(
            report["qualification_case"]["case_id"],
            "capture--one-hour--infer-on--preview-on--25w",
        )
        self.assertTrue(
            report["artifact_references"]["baseline"]["qualification_summary"][
                "available"
            ]
        )
        self.assertEqual(
            report["artifact_references"]["baseline"]["qualification_summary"][
                "sha256"
            ],
            hashlib.sha256(baseline_summary_bytes).hexdigest(),
        )

    def test_pair_fails_if_parsed_summary_changes_during_evaluation(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)
        original = debug_overhead._threshold_checks

        def mutate_then_check(*args, **kwargs):
            (baseline / "qualification_summary.json").write_text('{"result":"failed"}')
            return original(*args, **kwargs)

        with mock.patch.object(
            debug_overhead, "_threshold_checks", side_effect=mutate_then_check
        ):
            report = compare_debug_overhead(baseline, debug)

        self.assertFalse(report["comparable"])
        self.assertIn(
            "baseline qualification_summary changed while debug evidence was evaluated",
            report["mismatches"],
        )

    def test_pair_requires_durable_matching_case_binding_and_terminal_status(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)
        baseline_manifest_path = baseline / "run_manifest.json"
        baseline_manifest = json.loads(baseline_manifest_path.read_text())
        baseline_manifest["process_topology"]["acquisition_owner"] = "in_process_dev"
        baseline_manifest_path.write_text(json.dumps(baseline_manifest))
        debug_status_path = debug / "run_status.json"
        debug_status = json.loads(debug_status_path.read_text())
        debug_status["state"] = "recording"
        debug_status["qualification"]["case_id"] = "different-case"
        debug_status_path.write_text(json.dumps(debug_status))

        report = compare_debug_overhead(baseline, debug)

        self.assertFalse(report["comparable"])
        self.assertTrue(any("acquisition owner" in item for item in report["mismatches"]))
        self.assertTrue(any("not successfully terminal" in item for item in report["mismatches"]))
        self.assertTrue(any("manifest/status" in item for item in report["mismatches"]))

    def test_pair_can_require_an_explicit_case_id(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)

        report = compare_debug_overhead(
            baseline, debug, expected_case_id="different-case"
        )

        self.assertFalse(report["comparable"])
        self.assertTrue(any("requested case" in item for item in report["mismatches"]))

    def test_pair_requires_exact_qualification_exceptions(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)
        baseline_summary = json.loads(
            (baseline / "qualification_summary.json").read_text()
        )
        baseline_summary["qualification_exceptions"] = ["unexpected"]
        (baseline / "qualification_summary.json").write_text(
            json.dumps(baseline_summary)
        )
        debug_summary = json.loads((debug / "qualification_summary.json").read_text())
        debug_summary["qualification_exceptions"] = []
        (debug / "qualification_summary.json").write_text(json.dumps(debug_summary))

        report = compare_debug_overhead(baseline, debug)

        self.assertFalse(report["comparable"])
        self.assertTrue(any("baseline qualification_exceptions" in value for value in report["mismatches"]))
        self.assertTrue(any("debug qualification_exceptions" in value for value in report["mismatches"]))

    def test_pair_requires_same_validated_limits_identity(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)
        debug_summary = json.loads((debug / "qualification_summary.json").read_text())
        debug_summary["limits"]["profile_id"] = "different-policy"
        (debug / "qualification_summary.json").write_text(json.dumps(debug_summary))

        mismatch = compare_debug_overhead(baseline, debug)
        self.assertFalse(mismatch["comparable"])
        self.assertIn("qualification limits identity differs", mismatch["mismatches"])

        debug_summary["limits"]["profile_id"] = "run-qualification-v1"
        debug_summary["limits"]["validated"] = False
        (debug / "qualification_summary.json").write_text(json.dumps(debug_summary))
        invalid = compare_debug_overhead(baseline, debug)
        self.assertFalse(invalid["comparable"])
        self.assertTrue(any("debug validated qualification limits identity" in value for value in invalid["mismatches"]))

    def test_threshold_loader_is_bounded_strict_and_rejects_duplicates(self) -> None:
        path = self.root / "thresholds.yaml"
        path.write_bytes(b" " * (MAX_THRESHOLD_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            load_debug_thresholds(path)

        path.write_bytes(b"schema_version: \xff")
        with self.assertRaisesRegex(ValueError, "strict UTF-8"):
            load_debug_thresholds(path)

        path.write_text(
            "schema_version: '1.0'\n"
            "profile_id: test\n"
            "approved: false\n"
            "metrics:\n"
            f"  {_METRICS[0]}:\n"
            "    max_increase: 1\n"
            "    max_increase: 2\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            load_debug_thresholds(path)

    def test_threshold_loader_rejects_unknown_keys_and_inexact_types(self) -> None:
        invalid_documents = (
            ({**self._thresholds(), "schema_version": 1.0}, "schema_version"),
            ({**self._thresholds(), "approved": "true"}, "approved must be boolean"),
            ({**self._thresholds(), "unknown": 1}, "unknown keys"),
            (
                {
                    **self._thresholds(),
                    "metrics": {
                        **self._thresholds()["metrics"],
                        "unknown.metric": {"max_increase": 1},
                    },
                },
                "metrics has unknown keys",
            ),
        )
        path = self.root / "thresholds.yaml"
        for document, expected in invalid_documents:
            with self.subTest(expected=expected):
                path.write_text(yaml.safe_dump(document))
                with self.assertRaisesRegex(ValueError, expected):
                    load_debug_thresholds(path)

        document = self._thresholds()
        document["metrics"][_METRICS[0]]["max_increase"] = "1"
        path.write_text(yaml.safe_dump(document))
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            load_debug_thresholds(path)

    def test_threshold_loader_preserves_measurement_only_documents(self) -> None:
        path = self.root / "measurement-only.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "1.0",
                    "profile_id": "measurement-v1",
                    "approved": False,
                    "metrics": {
                        metric: {
                            "max_increase": None,
                            "max_percent_increase": None,
                        }
                        for metric in _METRICS
                    },
                }
            )
        )

        loaded = load_debug_thresholds(path)
        self.assertIs(loaded["approved"], False)

    def test_wrong_profile_or_factor_fails_closed(self) -> None:
        baseline = self._run("baseline", debug=True)
        debug = self._run("debug", debug=True)
        payload = json.loads((debug / "qualification_summary.json").read_text())
        payload["factors"]["fps"] = 60
        (debug / "qualification_summary.json").write_text(json.dumps(payload))

        report = compare_debug_overhead(baseline, debug)

        self.assertFalse(report["comparable"])
        self.assertTrue(any("debug-profile off" in item for item in report["mismatches"]))
        self.assertTrue(any("factor fps differs" in item for item in report["mismatches"]))

    def test_full_model_hash_identity_must_match(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)
        manifest = json.loads((debug / "run_manifest.json").read_text())
        manifest["inference"]["model_package"]["onnx_sha256"] = "f" * 64
        (debug / "run_manifest.json").write_text(json.dumps(manifest))

        report = compare_debug_overhead(baseline, debug)

        self.assertFalse(report["comparable"])
        self.assertIn("model identity onnx_sha256 differs", report["mismatches"])

    def test_changed_source_artifact_invalidates_cached_summary(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)
        (baseline / "diagnostics/system.csv").write_text("sample\nchanged\n")

        report = compare_debug_overhead(baseline, debug)

        self.assertFalse(report["comparable"])
        self.assertTrue(
            any("system_telemetry is stale" in item for item in report["mismatches"])
        )

    def test_changed_task_config_invalidates_cached_summary(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)
        (baseline / "config/task.yaml").write_text("task_name: changed\n")

        report = compare_debug_overhead(baseline, debug)

        self.assertFalse(report["comparable"])
        self.assertTrue(
            any("task_config is stale" in item for item in report["mismatches"])
        )

    def test_changed_valid_debug_log_invalidates_cached_summary(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)
        (debug / "diagnostics/deepstream.log").write_text(
            "Source id = 0 Frame_num = 99 Frame latency = 8.0 (ms)\n"
            "Comp name = nvinfer0 Component latency = 3.0\n"
        )

        report = compare_debug_overhead(baseline, debug)

        self.assertFalse(report["comparable"])
        self.assertTrue(
            any("deepstream_log is stale" in item for item in report["mismatches"])
        )

    def test_source_identity_rejects_non_regular_limits_input(self) -> None:
        run_dir = self._run("baseline", debug=False)

        evidence = capture_source_evidence(run_dir, Path("/dev/null"))

        self.assertFalse(evidence["limits"]["available"])
        self.assertIn("regular file", evidence["limits"]["error"])

    def test_runtime_identity_must_match(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)
        manifest_path = debug / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["platform"]["kernel"] = "different-tegra-kernel"
        manifest_path.write_text(json.dumps(manifest))
        summary = json.loads((debug / "qualification_summary.json").read_text())
        summary["source_evidence"] = capture_source_evidence(
            debug, self.limits_path
        )
        (debug / "qualification_summary.json").write_text(json.dumps(summary))

        report = compare_debug_overhead(baseline, debug)

        self.assertFalse(report["comparable"])
        self.assertIn("runtime identity differs", report["mismatches"])

    def test_task_and_acquisition_configuration_identity_must_match(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)
        manifest_path = debug / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["task_config"]["sha256"] = "6" * 64
        manifest["capture"]["trigger_activation"] = "falling"
        manifest_path.write_text(json.dumps(manifest))
        summary = json.loads((debug / "qualification_summary.json").read_text())
        summary["source_evidence"] = capture_source_evidence(debug, self.limits_path)
        (debug / "qualification_summary.json").write_text(json.dumps(summary))

        report = compare_debug_overhead(baseline, debug)

        self.assertFalse(report["comparable"])
        self.assertIn(
            "acquisition/task configuration identity differs", report["mismatches"]
        )

    def test_capture_durations_must_be_within_reported_policy(self) -> None:
        baseline = self._run("baseline", debug=False, duration=1000)
        debug = self._run("debug", debug=True, duration=1020)

        report = compare_debug_overhead(baseline, debug)

        self.assertFalse(report["comparable"])
        self.assertAlmostEqual(
            report["duration_comparison"]["allowed_delta_seconds"], 10.2
        )
        self.assertTrue(any("capture duration differs" in item for item in report["mismatches"]))

    def test_debug_log_requires_recognized_complete_latency_evidence(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)
        log = debug / "diagnostics" / "deepstream.log"
        log.write_text("no recognizable NVIDIA latency label\n")

        missing = compare_debug_overhead(baseline, debug)
        self.assertFalse(missing["comparable"])
        self.assertTrue(any("lacks complete" in item for item in missing["mismatches"]))

        log.write_text(
            "Source id = 0 Frame_num = 1 Frame latency = 1.0 (ms)\n"
            "Comp name = nvinfer0 Component latency = 1.0\n"
            "**FPS:  30.00 (30.00)\n"
            "[SQUEAKVIEW] diagnostic log size limit reached; later output omitted\n"
        )
        truncated = compare_debug_overhead(baseline, debug)
        self.assertFalse(truncated["comparable"])
        self.assertTrue(any("is truncated" in item for item in truncated["mismatches"]))

    def test_explicit_approved_thresholds_produce_pass_or_fail(self) -> None:
        baseline = self._run("baseline", debug=False, cpu=40)
        debug = self._run("debug", debug=True, cpu=50)

        passed = compare_debug_overhead(
            baseline, debug, thresholds=self._thresholds(limit=10)
        )
        failed = compare_debug_overhead(
            baseline, debug, thresholds=self._thresholds(limit=9)
        )

        self.assertEqual(passed["result"], "passed")
        self.assertTrue(passed["thresholds"]["passed"])
        self.assertEqual(failed["result"], "failed")
        self.assertFalse(failed["thresholds"]["passed"])

    def test_latency_log_scan_is_bounded_and_marks_truncation(self) -> None:
        path = self.root / "deepstream.log"
        path.write_bytes(
            b"Source id = 0 Frame_num = 1 Frame latency = 1 (ms)\n"
            b"Comp name = mux Component latency = 2\n"
            b"**FPS:  30.00 (30.00)\n"
            + b"x" * 100
        )

        evidence = scan_deepstream_latency_log(path, max_bytes=150)

        self.assertLessEqual(evidence["bytes_scanned"], 150)
        self.assertTrue(evidence["recognized"])
        self.assertTrue(evidence["truncated"])

    def test_latency_scanner_rejects_generic_and_nonfinite_labels(self) -> None:
        path = self.root / "deepstream.log"
        path.write_text(
            "Frame latency = 1.0 (ms)\n"
            "Source id = 0 Frame_num = 1 Frame latency = nan (ms)\n"
            "Comp name = nvinfer0 Component latency = inf\n"
        )

        evidence = scan_deepstream_latency_log(path)

        self.assertFalse(evidence["frame_latency_recognized"])
        self.assertFalse(evidence["component_latency_recognized"])
        self.assertFalse(evidence["fps_recognized"])
        self.assertFalse(evidence["recognized"])

    def test_latency_scanner_requires_finite_fps_evidence(self) -> None:
        path = self.root / "deepstream.log"
        path.write_text(
            "Source id = 0 Frame_num = 1 Frame latency = 1.0 (ms)\n"
            "Comp name = nvinfer0 Component latency = 1.0\n"
        )
        missing = scan_deepstream_latency_log(path)
        path.write_text(
            "Source id = 0 Frame_num = 1 Frame latency = 1.0 (ms)\n"
            "Comp name = nvinfer0 Component latency = 1.0\n"
            "**FPS: nan (nan)\n"
        )
        invalid = scan_deepstream_latency_log(path)

        self.assertFalse(missing["fps_recognized"])
        self.assertFalse(missing["recognized"])
        self.assertEqual(invalid["invalid_fps_records"], 1)
        self.assertFalse(invalid["recognized"])

    def test_run_evidence_requires_hashed_probe_provenance(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)
        manifest_path = debug / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["observability"].pop("deepstream_debug_probes")
        manifest_path.write_text(json.dumps(manifest))
        summary = json.loads((debug / "qualification_summary.json").read_text())
        summary["source_evidence"] = capture_source_evidence(
            debug, self.limits_path
        )
        (debug / "qualification_summary.json").write_text(json.dumps(summary))

        report = compare_debug_overhead(baseline, debug)

        self.assertFalse(report["comparable"])
        self.assertFalse(report["debug_latency_evidence"]["probe_modules_complete"])

    def test_latency_scanner_rejects_negative_values(self) -> None:
        path = self.root / "deepstream.log"
        path.write_text(
            "Source id = 0 Frame_num = 1 Frame latency = -1.0 (ms)\n"
            "Comp name = nvinfer0 Component latency = 1.0\n"
            "**FPS:  30.00 (30.00)\n"
        )

        evidence = scan_deepstream_latency_log(path)

        self.assertEqual(evidence["invalid_latency_records"], 1)
        self.assertFalse(evidence["recognized"])

    def test_invalid_duration_policy_and_log_path_fail_closed(self) -> None:
        baseline = self._run("baseline", debug=False)
        debug = self._run("debug", debug=True)
        manifest = json.loads((debug / "run_manifest.json").read_text())
        manifest["observability"]["deepstream_log"] = "../outside.log"
        (debug / "run_manifest.json").write_text(json.dumps(manifest))

        report = compare_debug_overhead(
            baseline,
            debug,
            duration_tolerance_seconds=float("nan"),
            duration_tolerance_percent=-1,
        )

        self.assertFalse(report["comparable"])
        self.assertTrue(any("finite and nonnegative" in item for item in report["mismatches"]))
        self.assertIn("escapes the run directory", report["debug_latency_evidence"]["error"])


if __name__ == "__main__":
    unittest.main()
