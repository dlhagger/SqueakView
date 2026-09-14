from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

import yaml

from squeakview.common.diagnostics import qualification
from squeakview.common.diagnostics.qualification import (
    MAX_LIMITS_BYTES,
    _read_yaml,
    _scan_recording,
    qualify_run,
)
from squeakview.common.recording_evidence import capture_recording_evidence
from squeakview.apps.inference.preview_attribution import HEADERS as PREVIEW_HEADERS
from squeakview.apps.inference.preview_attribution import reconcile_preview


SYSTEM_HEADERS = [
    "schema_version",
    "host_monotonic_ns",
    "parse_status",
    "thermal_throttled",
    "ram_pct",
    "swap_used_mb",
    "swap_total_mb",
    "cpu_util_mean_pct",
    "cpu_util_max_pct",
    "gpu_util_pct",
    "emc_util_pct",
    "emc_clock_pct_of_max",
    "temp_max_c",
    "vdd_in_current_mw",
]
RECORDING_HEADERS = [
    "host_unix_ns",
    "host_monotonic_ns",
    "stream_id",
    "event",
    "pts_ns",
    "egress_timestamp_ns",
    "encoder_correlation",
    "queue_wait_ms",
    "encoder_latency_ms",
    "waiting_for_record_admission",
    "encoder_in_flight",
    "max_waiting_since_sample",
    "max_encoder_in_flight_since_sample",
    "pending_evictions",
]


class QualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name) / "run"
        (self.run_dir / "diagnostics").mkdir(parents=True)
        preflight_output = b"verified preflight"
        self.preflight_evidence = {
            "schema_version": "3.0",
            "passed": True,
            "skipped": False,
            "ffprobe_available": True,
            "video_decode_validated": True,
            "new_streammux_validated": True,
            "automatic_suspend_disabled": True,
            "output_size_bytes": len(preflight_output),
            "output_sha256": hashlib.sha256(preflight_output).hexdigest(),
        }
        self.status_path = self.run_dir / "run_status.json"
        self.status = {
            "run_id": "run-1",
            "state": "finalized",
            "started_at": "2026-09-01T10:00:00",
            "capture_closed_at": "2026-09-01T10:00:04",
            "capture_exit_code": 0,
            "recording_validation": {"passed": True},
            "acquisition_integrity": {"passed": True},
            "capture_reconciliation": {
                "source_frames": {"0": 1},
                "record_admitted_frames": {"0": 1},
                "source_not_recorded_frames": {"0": 0},
            },
            "overall_validation_passed": True,
            "production_eligible": True,
            "process_topology": {"acquisition_owner": "durable_supervisor"},
            "preflight": self.preflight_evidence,
        }
        self.manifest = {
            "schema_version": "2.0",
            "run_id": "run-1",
            "production_eligible": True,
            "process_topology": {"acquisition_owner": "durable_supervisor"},
            "preflight": self.preflight_evidence,
            "task_config": {
                "original_path": "/selected/task.yaml",
                "snapshot_path": "config/task.yaml",
                "size_bytes": len(b"task_name: test\n"),
                "sha256": hashlib.sha256(b"task_name: test\n").hexdigest(),
            },
            "capture": {
                "width": 1440,
                "height": 1080,
                "fps": 30,
                "num_cameras": 1,
                "pixel_format": "Mono8",
                "trigger_on": True,
            },
            "inference": {
                "enabled": True,
                "preview_enabled": False,
                "preview_sockets": ["/tmp/preview"],
                "model_package": {
                    "name": "mouse-pose",
                    "model_manifest_schema": 3,
                    "engine_build_identity": {"tensorrt_version": "10.16.2"},
                    "model_manifest_sha256": "1" * 64,
                    "pose_sidecar_sha256": "2" * 64,
                    "onnx_sha256": "3" * 64,
                    "config_sha256": "4" * 64,
                    "engine_sha256": "5" * 64,
                },
            },
            "serial": {"enabled": False},
            "platform": {
                "device_model": "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super",
                "machine": "aarch64",
                "kernel": "6.8.12-tegra",
                "python": "3.12.3",
                "python_executable": "/usr/bin/python3",
                "jetson_linux_release": "R39 (release), REVISION: 2.1",
                "deepstream_build": "DeepStreamSDK 9.1",
                "nvpmodel": "NV Power Mode: MAXN_SUPER\n2",
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
                    "available": True,
                    "size_bytes": 1,
                    "sha256": "a" * 64,
                },
                "deepstream_yolo_parser": {
                    "path": "/models/lib/parser.so",
                    "available": True,
                    "size_bytes": 1,
                    "sha256": "b" * 64,
                },
            },
            "git": {"commit": "abc123", "dirty": False},
            "observability": {
                "preflight_skipped": False,
                "deepstream_debug_profile": False,
            },
        }
        (self.run_dir / "config").mkdir()
        (self.run_dir / "config" / "task.yaml").write_bytes(b"task_name: test\n")
        effective_runtime = {}
        for name in (
            "deepstream_config",
            "pose_sidecar",
            "class_labels",
            "keypoint_labels",
        ):
            path = self.run_dir / "config" / name
            path.write_text(f"{name}\n")
            effective_runtime[name] = qualification.stable_file_identity(path)
        effective_runtime.update(
            {
                "onnx": {
                    "path": "/models/model.onnx", "available": True,
                    "size_bytes": 10, "sha256": "3" * 64,
                },
                "engine": {
                    "path": "/models/model.engine", "available": True,
                    "size_bytes": 10, "sha256": "5" * 64,
                },
                "custom_parser": dict(
                    self.manifest["native_plugins"]["deepstream_yolo_parser"]
                ),
            }
        )
        self.manifest["inference"]["effective_runtime"] = effective_runtime
        (self.run_dir / "raw.mp4").write_bytes(b"video")
        (self.run_dir / "capture_cam0.jsonl").write_text("capture\n")
        (self.run_dir / "record_admission.csv").write_text("admission\n")
        self.status["recording_validation"] = {
            "schema_version": "2.0",
            "passed": True,
            "evidence_unchanged_during_validation": True,
            "cameras": [
                {
                    "stream_id": 0,
                    "exists": True,
                    "source_frames": 1,
                    "record_admitted_frames": 1,
                    "video_frames": 1,
                    "frame_count_method": "full_decode_gstreamer_nvv4l2decoder",
                    "nonzero_frame_count": True,
                    "source_count_matches": True,
                    "frame_count_matches": True,
                }
            ],
            "evidence": capture_recording_evidence(self.run_dir, 1),
        }
        self.status_path.write_text(json.dumps(self.status))
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))
        self._write_system()
        self._write_recording()

    def test_non_durable_or_missing_acquisition_owner_cannot_qualify(self) -> None:
        for topology in (
            {"acquisition_owner": "in_process_dev"},
            None,
        ):
            with self.subTest(topology=topology):
                if topology is None:
                    self.manifest.pop("process_topology", None)
                else:
                    self.manifest["process_topology"] = topology
                (self.run_dir / "run_manifest.json").write_text(
                    json.dumps(self.manifest)
                )
                summary = qualify_run(self.run_dir, limits_path=None)
                self.assertFalse(
                    summary["frame_integrity_gates"]["durable_acquisition_owner"]
                )
                self.assertEqual(summary["result"], "failed")

    def test_serial_alignment_requires_complete_trigger_epoch_evidence(self) -> None:
        self.manifest["serial"] = {
            "enabled": True,
            "port": "/dev/ttyACM0",
            "baud": 115200,
            "controller_protocol": "legacy",
        }
        self.status["alignment_validated"] = True
        self.status_path.write_text(json.dumps(self.status))
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))

        missing = qualify_run(self.run_dir, limits_path=None)
        self.assertFalse(
            missing["frame_integrity_gates"]["alignment_validated_when_required"]
        )

        marker_names = (
            "START_SENT", "CAPTURE_STOP_REQUESTED", "STOP_SENT", "CAPTURE_STOP_DONE"
        )
        alignment = {
            "schema_version": "2.0",
            "run_dir": str(self.run_dir.resolve()),
            "start_marker_seen": True,
            "markers": {name: None for name in marker_names},
            "counts": {"recorded_frames": 1},
            "validation": {"video_frame_count_matches_frames_csv": True},
            "frame_alignment": {
                "method": "first_recorded_frame_to_first_camera_high_after_start_sent",
                "marker_indices": {name: index for index, name in enumerate(marker_names)},
                "epoch_markers_complete": True,
                "validated": True,
                "validated_pairs": 1,
                "controller_high_events_in_epoch": 1,
                "controller_high_events_unmatched": 0,
                "shutdown_tail_high_events_unmatched": 0,
                "boundary_tail_policy": "all post-START edges, including shutdown tail",
            },
        }
        (self.run_dir / "alignment_summary.json").write_text(json.dumps(alignment))
        valid = qualify_run(self.run_dir, limits_path=None)
        self.assertTrue(
            valid["frame_integrity_gates"]["alignment_validated_when_required"]
        )

        alignment["frame_alignment"]["controller_high_events_unmatched"] = 1
        (self.run_dir / "alignment_summary.json").write_text(json.dumps(alignment))
        evidence_errors = qualification.source_evidence_errors(
            self.run_dir,
            valid["source_evidence"],
            declared_limits_path=None,
        )
        self.assertTrue(
            any("alignment_summary" in error for error in evidence_errors),
            evidence_errors,
        )
        tampered = qualify_run(self.run_dir, limits_path=None)
        self.assertFalse(
            tampered["frame_integrity_gates"]["alignment_validated_when_required"]
        )

    def test_free_running_serial_logging_does_not_require_trigger_alignment(self) -> None:
        self.manifest["capture"]["trigger_on"] = False
        self.manifest["serial"] = {
            "enabled": True,
            "port": "/dev/ttyACM0",
            "baud": 115200,
            "controller_protocol": "legacy",
        }
        self.status.pop("alignment_validated", None)
        self.status_path.write_text(json.dumps(self.status))
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))

        summary = qualify_run(self.run_dir, limits_path=None)

        self.assertTrue(
            summary["frame_integrity_gates"]["alignment_validated_when_required"]
        )
        self.assertEqual(summary["result"], "incomplete")

    def test_missing_or_modified_task_snapshot_cannot_qualify(self) -> None:
        snapshot = self.run_dir / "config" / "task.yaml"
        for mutation in ("missing", "modified"):
            with self.subTest(mutation=mutation):
                snapshot.write_bytes(b"task_name: test\n")
                if mutation == "missing":
                    snapshot.unlink()
                else:
                    snapshot.write_bytes(b"task_name: changed\n")
                summary = qualify_run(self.run_dir, limits_path=None)
                self.assertFalse(
                    summary["frame_integrity_gates"]["task_config_snapshot_valid"]
                )
                self.assertEqual(summary["result"], "failed")

    def test_missing_task_config_identity_cannot_qualify(self) -> None:
        self.manifest.pop("task_config")
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))

        summary = qualify_run(self.run_dir, limits_path=None)

        self.assertFalse(
            summary["frame_integrity_gates"]["task_config_snapshot_valid"]
        )
        self.assertEqual(summary["result"], "failed")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_publication_rejects_manifest_changed_after_it_was_parsed(self) -> None:
        manifest_path = self.run_dir / "run_manifest.json"
        original_scan = qualification._scan_system

        def mutate_after_manifest_read(*args, **kwargs):
            report = original_scan(*args, **kwargs)
            payload = json.loads(manifest_path.read_text())
            payload["run_id"] = "substituted"
            manifest_path.write_text(json.dumps(payload))
            return report

        with mock.patch.object(
            qualification, "_scan_system", side_effect=mutate_after_manifest_read
        ):
            with self.assertRaisesRegex(ValueError, "manifest/status changed"):
                qualify_run(self.run_dir, limits_path=None)

        self.assertFalse((self.run_dir / "qualification_summary.json").exists())

    def _write_system(self, rows=None) -> None:
        rows = rows or [
            ["1.0", 1_000_000_000, "ok", "False", 70, 10, 100, 40, 60, 30, 50, 66, 51, 8000],
            ["1.0", 2_000_000_000, "partial", "True", 72, 12, 100, 45, 65, 35, 55, 70, 53, 8500],
            ["1.0", 4_000_000_000, "malformed", "False", 71, 11, 100, 42, 61, 31, 52, 68, 52, 8100],
        ]
        with (self.run_dir / "diagnostics" / "system.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SYSTEM_HEADERS)
            writer.writerows(rows)

    def _write_recording(self, count: int = 2) -> None:
        with (self.run_dir / "diagnostics" / "recording.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(RECORDING_HEADERS)
            for index in range(count):
                writer.writerow(
                    [
                        index + 1,
                        index + 1,
                        0,
                        "sample",
                        index,
                        index,
                        "pts",
                        0.2,
                        12.5,
                        1,
                        8,
                        2,
                        9,
                        0,
                    ]
                )

    def _limits(self, *, validated: bool, updates: dict | None = None) -> Path:
        payload = {
            "schema_version": "1.0",
            "profile_id": "test-v1",
            "validated": validated,
            "telemetry": {
                "expected_interval_ms": 1000,
                "min_coverage_ratio": 0.5,
                "max_gap_s": 3,
                "max_malformed_rows": 1,
                "max_invalid_rows": 0,
                "max_throttle_samples": 1,
            },
            "resources": {
                "max_ram_pct": 90,
                "max_swap_pct": 90,
                "max_cpu_util_mean_pct": 100,
                "max_cpu_util_max_pct": 100,
                "max_gpu_util_pct": 100,
                "max_emc_util_pct": 100,
                "max_emc_clock_pct_of_max": 100,
                "max_temp_max_c": 60,
                "max_vdd_in_current_mw": 100000,
            },
            "recording": {
                "max_queue_wait_ms": 100,
                "max_encoder_latency_ms": 20,
                "max_waiting_for_record_admission": 100,
                "max_encoder_in_flight": 100,
                "max_pending_evictions": 0,
                "max_backpressure_fatal_events": 0,
            },
        }
        if updates:
            for section, values in updates.items():
                payload.setdefault(section, {}).update(values)
        path = Path(self.temp_dir.name) / f"limits-{validated}.yaml"
        path.write_text(yaml.safe_dump(payload))
        return path

    def test_measurement_only_profile_is_incomplete_and_preserves_status(self) -> None:
        original_status = self.status_path.read_bytes()

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=False))

        self.assertEqual(summary["result"], "incomplete")
        self.assertTrue(summary["capture_validity_unchanged"])
        self.assertEqual(self.status_path.read_bytes(), original_status)
        self.assertTrue((self.run_dir / "qualification_summary.json").is_file())
        self.assertEqual(summary["factors"]["nvpmodel"], "NV Power Mode: MAXN_SUPER\n2")
        self.assertEqual(summary["system_telemetry"]["longest_gap_s"], 2.0)
        self.assertEqual(summary["system_telemetry"]["malformed_rows"], 1)
        self.assertEqual(summary["system_telemetry"]["throttle_sample_count"], 1)
        self.assertEqual(summary["system_telemetry"]["resource_maxima"]["temp_max_c"], 53)
        self.assertEqual(summary["recording_telemetry"]["maxima"]["encoder_in_flight"], 8)

    def test_limits_loader_is_bounded_and_requires_strict_utf8(self) -> None:
        path = Path(self.temp_dir.name) / "bad-limits.yaml"
        path.write_bytes(b" " * (MAX_LIMITS_BYTES + 1))
        payload, error = _read_yaml(path)
        self.assertIsNone(payload)
        self.assertIn("exceed", error or "")

        path.write_bytes(b"schema_version: \xff")
        payload, error = _read_yaml(path)
        self.assertIsNone(payload)
        self.assertIn("strict UTF-8", error or "")

    def test_limits_loader_rejects_duplicate_keys_recursively(self) -> None:
        path = Path(self.temp_dir.name) / "duplicate-limits.yaml"
        path.write_text(
            "schema_version: '1.0'\n"
            "profile_id: test\n"
            "validated: false\n"
            "telemetry:\n"
            "  expected_interval_ms: 1000\n"
            "  expected_interval_ms: 2000\n"
        )
        payload, error = _read_yaml(path)
        self.assertIsNone(payload)
        self.assertIn("duplicate key", error or "")

    def test_limits_loader_requires_exact_root_schema_and_boolean_types(self) -> None:
        for updates, expected in (
            ({"schema_version": 1.0}, "schema_version"),
            ({"profile_id": "  "}, "profile_id"),
            ({"validated": "false"}, "validated must be boolean"),
            ({"unknown": 1}, "root has unknown keys"),
        ):
            with self.subTest(updates=updates):
                path = self._limits(validated=False)
                data = yaml.safe_load(path.read_text())
                data.update(updates)
                path.write_text(yaml.safe_dump(data))
                payload, error = _read_yaml(path)
                self.assertIsNone(payload)
                self.assertIn(expected, error or "")

    def test_limits_loader_rejects_unknown_section_keys_and_numeric_strings(self) -> None:
        path = self._limits(
            validated=False,
            updates={"telemetry": {"unexpected": 1}},
        )
        payload, error = _read_yaml(path)
        self.assertIsNone(payload)
        self.assertIn("telemetry has unknown keys", error or "")

        path = self._limits(
            validated=False,
            updates={"resources": {"max_ram_pct": "90"}},
        )
        payload, error = _read_yaml(path)
        self.assertIsNone(payload)
        self.assertIn("max_ram_pct must be numeric", error or "")

        path = self._limits(
            validated=False,
            updates={"recording": {"max_pending_evictions": 1.5}},
        )
        payload, error = _read_yaml(path)
        self.assertIsNone(payload)
        self.assertIn("max_pending_evictions must be an integer", error or "")

    def test_validated_limits_require_exact_sections_and_fields_at_load(self) -> None:
        path = self._limits(validated=True)
        data = yaml.safe_load(path.read_text())
        data.pop("resources")
        path.write_text(yaml.safe_dump(data))
        payload, error = _read_yaml(path)
        self.assertIsNone(payload)
        self.assertIn("resources must be a mapping", error or "")

        path = self._limits(validated=True)
        data = yaml.safe_load(path.read_text())
        data["recording"].pop("max_encoder_latency_ms")
        path.write_text(yaml.safe_dump(data))
        payload, error = _read_yaml(path)
        self.assertIsNone(payload)
        self.assertIn("recording is missing keys", error or "")

    def test_measurement_only_limits_allow_blank_thresholds(self) -> None:
        path = Path("qualification/limits.v1.yaml")
        payload, error = _read_yaml(path)
        self.assertIsNone(error)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertIs(payload["validated"], False)
        self.assertIsNone(payload["recording"]["max_encoder_latency_ms"])

    def test_validated_limits_pass(self) -> None:
        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "passed")
        self.assertTrue(summary["limit_checks"])
        self.assertTrue(all(check["passed"] for check in summary["limit_checks"]))

    def test_jetpack_7_2_1_rows_accept_clock_ratio_without_emc_utilization(self) -> None:
        # JetPack 7.2.1 tegrastats omits EMC_FREQ utilization. The recorder's
        # sysfs platform sampler still supplies emc_clock_pct_of_max.
        self._write_system(
            rows=[
                ["1.0", 1_000_000_000, "ok", "False", 70, 10, 100, 40, 60, 30, "", 66, 51, 8000],
                ["1.0", 2_000_000_000, "ok", "False", 72, 12, 100, 45, 65, 35, "", 70, 53, 8500],
                ["1.0", 3_000_000_000, "ok", "False", 71, 11, 100, 42, 61, 31, "", 68, 52, 8100],
            ]
        )
        limits = self._limits(validated=True)
        payload = yaml.safe_load(limits.read_text())
        payload["resources"].pop("max_emc_util_pct")
        limits.write_text(yaml.safe_dump(payload))

        summary = qualify_run(self.run_dir, limits_path=limits)

        system = summary["system_telemetry"]
        self.assertEqual(summary["result"], "passed")
        self.assertEqual(system["invalid_rows"], 0)
        self.assertEqual(system["resource_valid_counts"]["emc_util_pct"], 0)
        self.assertEqual(
            system["resource_coverage_ratios"]["emc_clock_pct_of_max"], 1.0
        )
        check_names = {check["name"] for check in summary["limit_checks"]}
        self.assertNotIn("resources.emc_util_pct", check_names)
        self.assertIn("resources.emc_clock_pct_of_max", check_names)

    def test_emc_evidence_is_still_required(self) -> None:
        self._write_system(
            rows=[
                ["1.0", 1_000_000_000, "ok", "False", 70, 10, 100, 40, 60, 30, "", "", 51, 8000],
                ["1.0", 2_000_000_000, "ok", "False", 72, 12, 100, 45, 65, 35, "", "", 53, 8500],
                ["1.0", 3_000_000_000, "ok", "False", 71, 11, 100, 42, 61, 31, "", "", 52, 8100],
            ]
        )

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["system_telemetry"]["invalid_rows"], 3)
        self.assertEqual(summary["result"], "failed")

    def test_validated_limits_allow_null_optional_emc_utilization_limit(self) -> None:
        limits = self._limits(validated=True)
        payload = yaml.safe_load(limits.read_text())
        payload["resources"]["max_emc_util_pct"] = None
        limits.write_text(yaml.safe_dump(payload))

        loaded, error = _read_yaml(limits)

        self.assertIsNone(error)
        self.assertIsNotNone(loaded)

    def test_modified_primary_recording_artifact_cannot_qualify(self) -> None:
        (self.run_dir / "raw.mp4").write_bytes(b"substituted video")

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "failed")
        self.assertFalse(
            summary["frame_integrity_gates"]["recording_evidence_unchanged"]
        )
        self.assertIn(
            "frame-integrity gate failed: recording_evidence_unchanged",
            summary["failed_reasons"],
        )

    def test_same_size_recording_mutation_during_evaluation_is_not_published(self) -> None:
        output = self.run_dir / "qualification_summary.json"
        output.write_text("previous summary\n")
        original_scan = qualification._scan_recording

        def mutate_after_initial_identity(*args, **kwargs):
            result = original_scan(*args, **kwargs)
            # Keep the size unchanged so this regression also covers filesystems
            # whose mtime/ctime granularity cannot distinguish rapid writes.
            (self.run_dir / "raw.mp4").write_bytes(b"other")
            return result

        with (
            mock.patch.object(
                qualification,
                "_scan_recording",
                side_effect=mutate_after_initial_identity,
            ),
            self.assertRaisesRegex(
                ValueError,
                "primary recording artifacts changed during qualification evaluation",
            ),
        ):
            qualify_run(
                self.run_dir,
                limits_path=self._limits(validated=True),
                output_path=output,
            )

        self.assertEqual(output.read_text(), "previous summary\n")

    def test_preview_attribution_is_required_for_preview_qualification_only(self) -> None:
        self.manifest["inference"]["preview_enabled"] = True
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))

        missing = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(missing["result"], "failed")
        self.assertFalse(
            missing["frame_integrity_gates"][
                "preview_attribution_complete_when_required"
            ]
        )
        for boundary in ("admission", "delivery"):
            with (self.run_dir / "diagnostics" / f"preview_{boundary}.csv").open(
                "w", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(PREVIEW_HEADERS)
                writer.writerow([boundary, 0, 1, 41, 1000, 2000])
        self.status["preview_attribution"] = reconcile_preview(
            self.run_dir, 1, required=True
        )
        self.status_path.write_text(json.dumps(self.status))

        present = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(present["result"], "passed")
        self.assertTrue(
            present["frame_integrity_gates"][
                "preview_attribution_complete_when_required"
            ]
        )
    def test_production_pass_requires_complete_clean_build_provenance(self) -> None:
        mutations = {
            "dirty git": lambda: self.manifest["git"].update(dirty=True),
            "missing commit": lambda: self.manifest["git"].pop("commit"),
            "missing runtime package": lambda: self.manifest["platform"][
                "packages"
            ].pop("libnvinfer10"),
            "unhashed source plugin": lambda: self.manifest["native_plugins"][
                "flir_gstreamer_source"
            ].pop("sha256"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                original = json.loads(json.dumps(self.manifest))
                mutate()
                (self.run_dir / "run_manifest.json").write_text(
                    json.dumps(self.manifest)
                )
                summary = qualify_run(
                    self.run_dir, limits_path=self._limits(validated=True)
                )
                self.assertEqual(summary["result"], "failed")
                self.manifest = original

    def test_nvidia_jetpack_meta_package_is_not_required(self) -> None:
        self.assertNotIn("nvidia-jetpack", self.manifest["platform"]["packages"])

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "passed")

    def test_backend_preflight_evidence_must_be_valid_and_match_status(self) -> None:
        self.status["preflight"] = dict(self.preflight_evidence, passed=False)
        self.status_path.write_text(json.dumps(self.status))

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "failed")
        self.assertFalse(
            summary["frame_integrity_gates"]["backend_preflight_verified"]
        )

    def test_backend_preflight_requires_hardware_path_self_test_evidence(self) -> None:
        for mutation in (
            {"video_decode_validated": False},
            {"new_streammux_validated": False},
            {"schema_version": "1.0"},
        ):
            with self.subTest(mutation=mutation):
                evidence = dict(self.preflight_evidence, **mutation)
                self.status["preflight"] = evidence
                self.manifest["preflight"] = evidence
                self.status_path.write_text(json.dumps(self.status))
                (self.run_dir / "run_manifest.json").write_text(
                    json.dumps(self.manifest)
                )

                summary = qualify_run(
                    self.run_dir,
                    limits_path=self._limits(validated=True),
                )

                self.assertFalse(
                    summary["frame_integrity_gates"]["backend_preflight_verified"]
                )

    def test_persisted_system_telemetry_failure_cannot_qualify(self) -> None:
        self.status["system_telemetry_degraded"] = True
        self.status_path.write_text(json.dumps(self.status))

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "failed")
        self.assertFalse(
            summary["frame_integrity_gates"]["system_telemetry_not_degraded"]
        )

    def test_validated_limits_require_complete_finite_nonnegative_schema(self) -> None:
        limits = self._limits(
            validated=True,
            updates={
                "telemetry": {"min_coverage_ratio": 1.1},
                "resources": {"max_ram_pct": float("nan")},
                "recording": {"max_queue_wait_ms": -1},
            },
        )

        summary = qualify_run(self.run_dir, limits_path=limits)

        self.assertEqual(summary["result"], "incomplete")
        self.assertFalse(summary["limits"]["validated"])
        self.assertIn("min_coverage_ratio", summary["limits"]["input_error"])

    def test_validated_limits_reject_boolean_thresholds(self) -> None:
        summary = qualify_run(
            self.run_dir,
            limits_path=self._limits(
                validated=True,
                updates={"resources": {"max_ram_pct": True}},
            ),
        )

        self.assertEqual(summary["result"], "incomplete")
        self.assertIn("resources.max_ram_pct", summary["limits"]["input_error"])

    def test_validated_limits_reject_a_missing_required_section(self) -> None:
        limits = self._limits(validated=True)
        payload = yaml.safe_load(limits.read_text())
        payload.pop("resources")
        limits.write_text(yaml.safe_dump(payload))

        summary = qualify_run(self.run_dir, limits_path=limits)

        self.assertEqual(summary["result"], "incomplete")
        self.assertIn("resources must be a mapping", summary["limits"]["input_error"])

    def test_unknown_or_invalid_recording_telemetry_is_fatal(self) -> None:
        with (self.run_dir / "diagnostics" / "recording.csv").open(
            "a", newline=""
        ) as handle:
            csv.writer(handle).writerow(
                [3, 3, "bad-stream", "unexpected", 3, "", "", 0, 0, 0, 0, 0, 0, 0]
            )

        summary = qualify_run(
            self.run_dir,
            limits_path=self._limits(validated=False),
        )

        self.assertEqual(summary["result"], "failed")
        self.assertTrue(any("unknown events" in item for item in summary["failed_reasons"]))
        self.assertTrue(any("invalid rows" in item for item in summary["failed_reasons"]))

    def test_nonfinite_or_negative_recording_metrics_are_fatal(self) -> None:
        with (self.run_dir / "diagnostics" / "recording.csv").open(
            "a", newline=""
        ) as handle:
            csv.writer(handle).writerow(
                [3, 3, 0, "sample", 3, "", "", "nan", -1, 0, 0, 0, 0, 0]
            )

        summary = qualify_run(
            self.run_dir, limits_path=self._limits(validated=False)
        )

        self.assertEqual(summary["result"], "failed")
        self.assertGreater(summary["recording_telemetry"]["invalid_rows"], 0)

    def test_unknown_thermal_state_cannot_satisfy_coverage(self) -> None:
        self._write_system(
            rows=[
                ["1.0", 1_000_000_000, "ok", "False", 70, 10, 100, 40, 60, 30, 50, 66, 51, 8000],
                ["1.0", 2_000_000_000, "ok", "", 70, 10, 100, 40, 60, 30, 50, 66, 51, 8000],
                ["1.0", 3_000_000_000, "ok", "False", 70, 10, 100, 40, 60, 30, 50, 66, 51, 8000],
            ]
        )
        limits = self._limits(
            validated=True,
            updates={
                "telemetry": {
                    "min_coverage_ratio": 0.8,
                    "max_invalid_rows": 1,
                }
            },
        )

        summary = qualify_run(self.run_dir, limits_path=limits)

        thermal = next(
            check
            for check in summary["limit_checks"]
            if check["name"] == "telemetry.thermal_status_coverage_ratio"
        )
        self.assertFalse(thermal["passed"])

    def test_recording_telemetry_must_cover_every_camera_and_stream(self) -> None:
        self.manifest["capture"]["num_cameras"] = 2
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))

        summary = qualify_run(
            self.run_dir,
            limits_path=self._limits(validated=False),
        )

        self.assertEqual(summary["result"], "failed")
        self.assertTrue(any("file coverage" in item for item in summary["failed_reasons"]))
        self.assertTrue(any("stream coverage" in item for item in summary["failed_reasons"]))

    def test_two_camera_recording_telemetry_coverage_is_accepted(self) -> None:
        self.manifest["capture"]["num_cameras"] = 2
        (self.run_dir / "raw_cam1.mp4").write_bytes(b"video-1")
        (self.run_dir / "capture_cam1.jsonl").write_text("capture-1\n")
        (self.run_dir / "record_admission_cam1.csv").write_text("admission-1\n")
        self.status["recording_validation"]["evidence"] = (
            capture_recording_evidence(self.run_dir, 2)
        )
        self.status["recording_validation"]["cameras"].append(
            {
                "stream_id": 1,
                "exists": True,
                "source_frames": 1,
                "record_admitted_frames": 1,
                "video_frames": 1,
                "frame_count_method": "full_decode_gstreamer_nvv4l2decoder",
                "nonzero_frame_count": True,
                "source_count_matches": True,
                "frame_count_matches": True,
            }
        )
        self.status["capture_reconciliation"] = {
            "source_frames": {"0": 1, "1": 1},
            "record_admitted_frames": {"0": 1, "1": 1},
            "source_not_recorded_frames": {"0": 0, "1": 0},
        }
        self.status_path.write_text(json.dumps(self.status))
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))
        with (self.run_dir / "diagnostics" / "recording_cam1.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(RECORDING_HEADERS)
            writer.writerow([1, 1, 1, "sample", 1, 1, "pts", 0.2, 12.5, 1, 8, 2, 9, 0])

        summary = qualify_run(
            self.run_dir,
            limits_path=self._limits(validated=False),
        )

        self.assertEqual(summary["result"], "incomplete")
        self.assertEqual(summary["recording_telemetry"]["stream_ids"], [0, 1])
        self.assertFalse(
            any("coverage failed" in item for item in summary["failed_reasons"])
        )

    def test_unexpected_stream_ids_are_counted_without_being_retained(self) -> None:
        with (self.run_dir / "diagnostics" / "recording.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(RECORDING_HEADERS)
            for stream_id in range(1, 2049):
                writer.writerow(
                    [
                        stream_id,
                        stream_id,
                        stream_id,
                        "sample",
                        stream_id,
                        stream_id,
                        "pts",
                        0.2,
                        12.5,
                        1,
                        8,
                        2,
                        9,
                        0,
                    ]
                )

        summary = qualify_run(
            self.run_dir,
            limits_path=self._limits(validated=False),
        )

        recording = summary["recording_telemetry"]
        self.assertEqual(recording["stream_ids"], [])
        self.assertEqual(recording["unexpected_stream_id_count"], 2048)
        self.assertEqual(summary["result"], "failed")
        self.assertTrue(
            any("unexpected stream IDs" in item for item in summary["failed_reasons"])
        )

    def test_recording_file_overflow_retains_only_expected_file_count(self) -> None:
        diagnostics = self.run_dir / "diagnostics"
        payload = (diagnostics / "recording.csv").read_bytes()
        for index in range(5):
            (diagnostics / f"recording_extra{index}.csv").write_bytes(payload)

        recording = _scan_recording(self.run_dir, expected_camera_count=1)
        summary = qualify_run(
            self.run_dir,
            limits_path=self._limits(validated=False),
        )

        self.assertTrue(recording["file_count_overflow"])
        self.assertEqual(recording["file_count"], 2)
        self.assertEqual(recording["retained_file_count"], 1)
        self.assertEqual(len(recording["files"]), 1)
        self.assertEqual(summary["result"], "failed")
        self.assertTrue(summary["recording_telemetry"]["file_count_overflow"])
        self.assertFalse(summary["source_evidence"]["available"])
        self.assertIn("hashing was skipped", summary["source_evidence"]["error"])

    def test_resource_or_gap_limit_failure_fails(self) -> None:
        limits = self._limits(
            validated=True,
            updates={"telemetry": {"max_gap_s": 1}, "resources": {"max_temp_max_c": 50}},
        )

        summary = qualify_run(self.run_dir, limits_path=limits)

        self.assertEqual(summary["result"], "failed")
        self.assertIn("qualification limit failed: telemetry.longest_gap_s", summary["failed_reasons"])
        self.assertIn("qualification limit failed: resources.temp_max_c", summary["failed_reasons"])

    def test_frame_integrity_failure_overrides_unvalidated_limits(self) -> None:
        self.status["recording_validation"]["passed"] = False
        self.status_path.write_text(json.dumps(self.status))

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=False))

        self.assertEqual(summary["result"], "failed")
        self.assertIn(
            "frame-integrity gate failed: recording_validation_passed",
            summary["failed_reasons"],
        )

    def test_routine_structural_validation_is_not_full_qualification(self) -> None:
        self.status["recording_validation"]["cameras"][0][
            "frame_count_method"
        ] = "mp4_sample_table"
        self.status_path.write_text(json.dumps(self.status))

        summary = qualify_run(
            self.run_dir, limits_path=self._limits(validated=False)
        )

        self.assertEqual(summary["result"], "failed")
        self.assertIn(
            "frame-integrity gate failed: recording_validation_passed",
            summary["failed_reasons"],
        )

    def test_power_interrupted_nonterminal_run_cannot_qualify(self) -> None:
        self.status["state"] = "recording"
        self.status.pop("capture_exit_code")
        self.status_path.write_text(json.dumps(self.status))

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "failed")
        self.assertFalse(summary["frame_integrity_gates"]["terminal_success_state"])

    def test_nonproduction_run_is_rejected(self) -> None:
        self.manifest["production_eligible"] = False
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "failed")
        self.assertIn(
            "frame-integrity gate failed: production_eligible",
            summary["failed_reasons"],
        )

    def test_debug_profile_requires_explicit_qualification_exception(self) -> None:
        self.manifest["production_eligible"] = False
        self.manifest["production_disqualifiers"] = ["deepstream_debug_profile"]
        self.manifest["observability"] = {
            "preflight_skipped": False,
            "deepstream_debug_profile": True,
            "deepstream_log": "diagnostics/deepstream.log",
            "deepstream_log_max_bytes": 1024,
            "deepstream_debug_probes": {
                "measure_latency_probe": {
                    "available": True, "size_bytes": 10, "sha256": "a" * 64,
                },
                "measure_fps_probe": {
                    "available": True, "size_bytes": 11, "sha256": "b" * 64,
                },
            },
        }
        self.status["production_eligible"] = False
        self.status["production_disqualifiers"] = ["deepstream_debug_profile"]
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))
        self.status_path.write_text(json.dumps(self.status))
        (self.run_dir / "diagnostics" / "deepstream.log").write_text(
            "Source id = 0 Frame_num = 1 Frame latency = 4.2 (ms)\n"
            "Comp name = nvinfer0 Component latency = 1.1\n"
            "**FPS:  30.00 (30.00)\n"
        )

        default = qualify_run(
            self.run_dir, limits_path=self._limits(validated=True)
        )
        allowed = qualify_run(
            self.run_dir,
            limits_path=self._limits(validated=True),
            allow_debug_profile=True,
        )

        self.assertEqual(default["result"], "failed")
        self.assertEqual(allowed["result"], "passed")
        self.assertEqual(
            allowed["qualification_exceptions"], ["deepstream_debug_profile"]
        )
        self.assertTrue(
            allowed["frame_integrity_gates"]["debug_latency_evidence_complete"]
        )

    def test_debug_exception_requires_complete_recognized_latency_log(self) -> None:
        self.manifest["production_eligible"] = False
        self.manifest["production_disqualifiers"] = ["deepstream_debug_profile"]
        self.manifest["observability"] = {
            "preflight_skipped": False,
            "deepstream_debug_profile": True,
            "deepstream_log": "diagnostics/deepstream.log",
            "deepstream_log_max_bytes": 1024,
            "deepstream_debug_probes": {
                "measure_latency_probe": {
                    "available": True, "size_bytes": 10, "sha256": "a" * 64,
                },
                "measure_fps_probe": {
                    "available": True, "size_bytes": 11, "sha256": "b" * 64,
                },
            },
        }
        self.status["production_eligible"] = False
        self.status["production_disqualifiers"] = ["deepstream_debug_profile"]
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))
        self.status_path.write_text(json.dumps(self.status))

        missing = qualify_run(
            self.run_dir,
            limits_path=self._limits(validated=True),
            allow_debug_profile=True,
        )
        self.assertEqual(missing["result"], "failed")
        self.assertFalse(
            missing["frame_integrity_gates"]["debug_latency_evidence_complete"]
        )
        self.assertEqual(missing["qualification_exceptions"], [])

        (self.run_dir / "diagnostics" / "deepstream.log").write_text(
            "Source id = 0 Frame_num = 1 Frame latency = 1.0 (ms)\n"
            "Comp name = nvinfer0 Component latency = 1.0\n"
            "**FPS:  30.00 (30.00)\n"
            "[SQUEAKVIEW] diagnostic log size limit reached\n"
        )
        truncated = qualify_run(
            self.run_dir,
            limits_path=self._limits(validated=True),
            allow_debug_profile=True,
        )
        self.assertEqual(truncated["result"], "failed")
        self.assertTrue(truncated["debug_latency_evidence"]["recognized"])
        self.assertTrue(truncated["debug_latency_evidence"]["truncated"])

    def test_debug_exception_cannot_override_skipped_preflight(self) -> None:
        reasons = ["preflight_skipped", "deepstream_debug_profile"]
        self.manifest["production_eligible"] = False
        self.manifest["production_disqualifiers"] = reasons
        self.manifest["observability"] = {
            "preflight_skipped": True,
            "deepstream_debug_profile": True,
        }
        self.status["production_eligible"] = False
        self.status["production_disqualifiers"] = reasons
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))
        self.status_path.write_text(json.dumps(self.status))

        summary = qualify_run(
            self.run_dir,
            limits_path=self._limits(validated=True),
            allow_debug_profile=True,
        )

        self.assertEqual(summary["result"], "failed")
        self.assertFalse(summary["frame_integrity_gates"]["production_eligible"])

    def test_missing_status_production_eligibility_is_rejected(self) -> None:
        self.status.pop("production_eligible")
        self.status_path.write_text(json.dumps(self.status))

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "failed")
        self.assertFalse(summary["frame_integrity_gates"]["production_eligible"])

    def test_failure_injected_run_is_rejected_even_if_marked_production(self) -> None:
        plan = {
            "schema_version": "1.0",
            "target": "flir_source",
            "kind": "source_read",
            "after_frames": 5,
        }
        self.manifest["failure_injection"] = plan
        self.status["failure_injection"] = plan
        self.manifest["production_eligible"] = True
        self.status["production_eligible"] = True
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))
        self.status_path.write_text(json.dumps(self.status))

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "failed")
        self.assertFalse(summary["frame_integrity_gates"]["production_eligible"])

    def test_missing_manifest_schema_is_rejected(self) -> None:
        self.manifest.pop("schema_version")
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "failed")
        self.assertIn(
            "frame-integrity gate failed: manifest_schema_supported",
            summary["failed_reasons"],
        )

    def test_unsupported_manifest_schema_is_rejected(self) -> None:
        self.manifest["schema_version"] = "99.0"
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "failed")
        self.assertIn(
            "frame-integrity gate failed: manifest_schema_supported",
            summary["failed_reasons"],
        )

    def test_legacy_engine_without_runtime_identity_cannot_qualify(self) -> None:
        model = self.manifest["inference"]["model_package"]
        model["model_manifest_schema"] = 2
        model["engine_build_identity"] = None
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "failed")
        self.assertFalse(
            summary["frame_integrity_gates"][
                "engine_identity_validated_when_required"
            ]
        )

    def test_missing_model_content_hash_cannot_qualify(self) -> None:
        model = self.manifest["inference"]["model_package"]
        model.pop("onnx_sha256")
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "failed")
        self.assertFalse(
            summary["frame_integrity_gates"][
                "model_content_identity_validated_when_required"
            ]
        )

    def test_missing_or_tampered_effective_runtime_cannot_qualify(self) -> None:
        gate = "effective_runtime_identity_validated_when_required"
        effective = self.manifest["inference"].pop("effective_runtime")
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))
        missing = qualify_run(
            self.run_dir, limits_path=self._limits(validated=True)
        )
        self.assertFalse(missing["frame_integrity_gates"][gate])
        self.assertEqual(missing["result"], "failed")

        self.manifest["inference"]["effective_runtime"] = effective
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self.manifest))
        config_path = self.run_dir / "config/deepstream_config"
        config_path.write_text("tampered\n")
        tampered = qualify_run(
            self.run_dir, limits_path=self._limits(validated=True)
        )
        self.assertFalse(tampered["frame_integrity_gates"][gate])
        self.assertEqual(tampered["result"], "failed")

    def test_missing_telemetry_is_incomplete(self) -> None:
        (self.run_dir / "diagnostics" / "system.csv").unlink()

        summary = qualify_run(self.run_dir, limits_path=self._limits(validated=True))

        self.assertEqual(summary["result"], "incomplete")
        self.assertTrue(any("system telemetry" in reason for reason in summary["incomplete_reasons"]))

    def test_recording_scan_memory_does_not_scale_with_rows(self) -> None:
        self._write_recording(count=100_000)
        tracemalloc.start()
        try:
            summary = qualify_run(self.run_dir, limits_path=self._limits(validated=False))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        self.assertEqual(summary["recording_telemetry"]["row_count"], 100_000)
        self.assertLess(peak, 8_000_000)


if __name__ == "__main__":
    unittest.main()
