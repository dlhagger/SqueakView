from __future__ import annotations

import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path

import yaml

from squeakview.common.diagnostics.qualification_matrix import (
    MAX_MATRIX_BYTES,
    MAX_MATRIX_CASES,
    MAX_ASSIGNMENT_BYTES,
    expand_cases,
    expected_case_factors,
    load_assignments,
    load_matrix,
    qualify_matrix,
    resolve_qualification_case_binding,
)
from squeakview.apps.operator.backend.contracts import RunRequest
from squeakview.common.diagnostics.evidence_identity import stable_file_identity


class QualificationMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.matrix_path = self.root / "matrix.yaml"
        self.matrix_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "1.0",
                    "matrix_id": "test",
                    "capture_profiles": [
                        {
                            "id": "capture", "width": 10, "height": 20,
                            "fps": 30, "camera_count": 1,
                            "pixel_format": "Mono8", "trigger_on": True,
                            "trigger_activation": "rising", "arduino_fps": 30,
                            "exposure_us": 10000.0, "bitrate_kbps": 4000,
                            "serial_enabled": True, "serial_port": "/dev/ttyACM0",
                            "serial_baud": 115200,
                        }
                    ],
                    "durations": [{"id": "short", "minimum_seconds": 5}],
                    "inference_enabled": [True, False],
                    "preview_enabled": [True, False],
                    "power_modes": ["25W"],
                }
            )
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _passing_summary(case: dict) -> dict:
        return {
            "result": "passed",
            "capture_duration_s": 6,
            "factors": {
                "width": 10,
                "height": 20,
                "fps": 30,
                "camera_count": 1,
                "pixel_format": "Mono8",
                "trigger_on": True,
                "trigger_activation": "rising",
                "arduino_fps": 30,
                "exposure_us": 10000.0,
                "bitrate_kbps": 4000,
                "serial_enabled": True,
                "serial_port": "/dev/ttyACM0",
                "serial_baud": 115200,
                "inference_enabled": case["inference_enabled"],
                "preview_enabled": case["preview_enabled"],
                "nvpmodel": "NV Power Mode: 25W",
            },
        }

    @staticmethod
    def _write_manifest(
        run_dir: Path,
        *,
        commit: str = "abc123",
        dirty: bool = False,
        production_eligible: bool = True,
        engine_sha256: str = "e" * 64,
    ) -> None:
        model = {
            "name": "mouse-pose",
            "model_manifest_schema": 3,
            "model_manifest_sha256": "a" * 64,
            "pose_sidecar_sha256": "b" * 64,
            "onnx_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "engine_sha256": engine_sha256,
            "engine_build_identity": {"tensorrt_version": "10.0"},
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "production_eligible": production_eligible,
                    "platform": {
                        "device_model": "NVIDIA Jetson Orin Nano Super",
                        "machine": "aarch64",
                        "kernel": "6.8.12-tegra",
                        "python": "3.12.3",
                        "python_executable": "/usr/bin/python3",
                        "jetson_linux_release": "R39.2.1",
                        "deepstream_build": "DeepStreamSDK 9.1",
                        "nvpmodel": "NV Power Mode: 25W\n1",
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
                            "size_bytes": 100,
                            "sha256": "f" * 64,
                        },
                        "deepstream_yolo_parser": {
                            "available": True,
                            "size_bytes": 200,
                            "sha256": "9" * 64,
                        },
                    },
                    "observability": {
                        "preflight_skipped": False,
                        "deepstream_debug_profile": False,
                    },
                    "git": {"commit": commit, "dirty": dirty},
                    "capture": {
                        "backend": "flir_direct", "num_cameras": 1,
                        "camera_serials": [], "width": 10, "height": 20,
                        "fps": 30, "pixel_format": "Mono8", "trigger_on": True,
                        "trigger_activation": "rising", "arduino_fps": 30,
                        "exposure_us": 10000.0,
                    },
                    "inference": {
                        "model_package": model, "bitrate_kbps": 4000,
                    },
                    "serial": {
                        "enabled": True, "port": "/dev/ttyACM0", "baud": 115200,
                        "controller_protocol": "legacy",
                    },
                    "task_config": {"size_bytes": 10, "sha256": "8" * 64},
                }
            )
        )
        diagnostics = run_dir / "diagnostics"
        diagnostics.mkdir(exist_ok=True)
        (diagnostics / "camera_runtime.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "metadata_type": "SQUEAKVIEW.FLIR.FRAME_META.v1",
                    "cameras": [
                        {
                            "camera_index": 0, "camera_serial": "TEST123",
                            "device_model": "Blackfly", "firmware_version": "1",
                            "source_width": 10, "source_height": 20,
                            "source_pixel_format": "Mono8", "actual_fps": 30,
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

    def _set_matrix(self, **updates) -> dict:
        matrix = yaml.safe_load(self.matrix_path.read_text())
        matrix.update(updates)
        self.matrix_path.write_text(yaml.safe_dump(matrix))
        return matrix

    def _bind_manifest(self, run_dir: Path, case: dict) -> None:
        manifest_path = run_dir / "run_manifest.json"
        payload = json.loads(manifest_path.read_text())
        payload["qualification"] = {
            "matrix_id": load_matrix(self.matrix_path)["matrix_id"],
            "matrix_path": str(self.matrix_path.resolve()),
            "matrix_sha256": stable_file_identity(self.matrix_path)["sha256"],
            "case_id": case["case_id"],
            "expected_factors": expected_case_factors(case),
        }
        manifest_path.write_text(json.dumps(payload))
        (run_dir / "run_status.json").write_text(
            json.dumps({"qualification": payload["qualification"]})
        )

    def test_cartesian_matrix_has_stable_unique_case_ids(self) -> None:
        matrix = load_matrix(self.matrix_path)
        cases = expand_cases(matrix)
        self.assertEqual(len(cases), 4)
        self.assertEqual(len({case["case_id"] for case in cases}), 4)
        self.assertTrue(matrix["provenance"]["require_same_device_identity"])

    def test_opt_in_case_binding_validates_effective_config_and_power_mode(self) -> None:
        case = expand_cases(load_matrix(self.matrix_path))[0]
        config = RunRequest(
            width=10,
            height=20,
            fps=30,
            num_cameras=1,
            pixel_format="Mono8",
            trigger_on=True,
            inference_enabled=case["inference_enabled"],
            preview_enabled=case["preview_enabled"],
        )
        environment = {
            "SQUEAKVIEW_QUALIFICATION_CASE_ID": case["case_id"],
            "SQUEAKVIEW_QUALIFICATION_MATRIX": str(self.matrix_path),
        }

        binding = resolve_qualification_case_binding(
            config,
            {"nvpmodel": "NV Power Mode: 25W\n0"},
            default_matrix_path=self.matrix_path,
            environ=environment,
        )

        self.assertEqual(binding["case_id"], case["case_id"])
        self.assertEqual(binding["expected_factors"], expected_case_factors(case))
        self.assertEqual(len(binding["matrix_sha256"]), 64)
        with self.assertRaisesRegex(ValueError, "preview_enabled"):
            resolve_qualification_case_binding(
                replace(config, preview_enabled=not case["preview_enabled"]),
                {"nvpmodel": "NV Power Mode: 25W\n0"},
                default_matrix_path=self.matrix_path,
                environ=environment,
            )
        with self.assertRaisesRegex(ValueError, "power_mode"):
            resolve_qualification_case_binding(
                config,
                {"nvpmodel": "current profile contains 25W"},
                default_matrix_path=self.matrix_path,
                environ=environment,
            )

    def test_unset_case_binding_preserves_ordinary_runs(self) -> None:
        self.assertIsNone(
            resolve_qualification_case_binding(
                RunRequest(),
                {},
                default_matrix_path=self.matrix_path,
                environ={},
            )
        )

    def test_matrix_override_without_case_binding_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires.*CASE_ID"):
            resolve_qualification_case_binding(
                object(),
                {},
                default_matrix_path=self.matrix_path,
                environ={"SQUEAKVIEW_QUALIFICATION_MATRIX": str(self.matrix_path)},
            )
        with self.assertRaisesRegex(ValueError, "non-empty canonical case ID"):
            resolve_qualification_case_binding(
                RunRequest(),
                {},
                default_matrix_path=self.matrix_path,
                environ={"SQUEAKVIEW_QUALIFICATION_CASE_ID": "   "},
            )

    def test_loader_rejects_oversized_or_non_utf8_input(self) -> None:
        self.matrix_path.write_bytes(b" " * (MAX_MATRIX_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            load_matrix(self.matrix_path)

        self.matrix_path.write_bytes(b"schema_version: \xff")
        with self.assertRaisesRegex(ValueError, "strict UTF-8"):
            load_matrix(self.matrix_path)

    def test_assignment_loader_is_bounded_strict_and_rejects_duplicates(self) -> None:
        path = self.root / "assignments.yaml"
        path.write_bytes(b"x" * (MAX_ASSIGNMENT_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "exceed"):
            load_assignments(path)

        path.write_bytes(b"case: \xff")
        with self.assertRaisesRegex(ValueError, "strict UTF-8"):
            load_assignments(path)

        path.write_text("case: run-a\ncase: run-b\n")
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            load_assignments(path)

        path.write_text("case: null\n")
        self.assertEqual(load_assignments(path), {"case": None})

    def test_null_checklist_entries_remain_incomplete_without_qualification(self) -> None:
        cases = expand_cases(load_matrix(self.matrix_path))
        assignments = {case["case_id"]: None for case in cases}

        report = qualify_matrix(
            self.matrix_path,
            assignments,
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=lambda *_args, **_kwargs: self.fail(
                "unassigned cells must not invoke run qualification"
            ),
        )

        self.assertEqual(report["result"], "incomplete")
        self.assertEqual(report["incomplete_count"], len(cases))

    def test_unknown_assignment_case_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown case IDs"):
            qualify_matrix(
                self.matrix_path,
                {"typo-case": self.root / "run"},
                limits_path=self.root / "limits.yaml",
                output_path=self.root / "report.json",
            )

    def test_loader_rejects_duplicate_and_unknown_keys(self) -> None:
        valid = self.matrix_path.read_text()
        self.matrix_path.write_text(valid + "matrix_id: duplicate\n")
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            load_matrix(self.matrix_path)

        self.setUp_matrix_from_valid(extra="unknown")
        with self.assertRaisesRegex(ValueError, "root has unknown keys"):
            load_matrix(self.matrix_path)

    def setUp_matrix_from_valid(self, **updates) -> None:
        payload = {
            "schema_version": "1.0",
            "matrix_id": "test",
            "capture_profiles": [
                {
                    "id": "capture",
                    "width": 10,
                    "height": 20,
                    "fps": 30,
                    "camera_count": 1,
                    "pixel_format": "Mono8",
                    "trigger_on": True,
                    "trigger_activation": "rising",
                    "arduino_fps": 30,
                    "exposure_us": 10000.0,
                    "bitrate_kbps": 4000,
                    "serial_enabled": True,
                    "serial_port": "/dev/ttyACM0",
                    "serial_baud": 115200,
                }
            ],
            "durations": [{"id": "short", "minimum_seconds": 5}],
            "inference_enabled": [True, False],
            "preview_enabled": [True, False],
            "power_modes": ["25W"],
        }
        payload.update(updates)
        self.matrix_path.write_text(yaml.safe_dump(payload))

    def test_loader_requires_exact_schema_scalar_types_and_ranges(self) -> None:
        for invalid_version in (1.0, 1, True):
            with self.subTest(schema_version=invalid_version):
                self.setUp_matrix_from_valid(schema_version=invalid_version)
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    load_matrix(self.matrix_path)

        invalid_profiles = (
            {"id": "capture", "width": True, "height": 20, "fps": 30, "camera_count": 1, "pixel_format": "Mono8"},
            {"id": "capture", "width": 0, "height": 20, "fps": 30, "camera_count": 1, "pixel_format": "Mono8"},
            {"id": "capture", "width": 10, "height": 20, "fps": 30, "camera_count": 1, "pixel_format": "Mono8", "extra": 1},
        )
        for profile in invalid_profiles:
            with self.subTest(profile=profile):
                self.setUp_matrix_from_valid(capture_profiles=[profile])
                with self.assertRaises(ValueError):
                    load_matrix(self.matrix_path)

        self.setUp_matrix_from_valid(
            durations=[{"id": "short", "minimum_seconds": 0}]
        )
        with self.assertRaisesRegex(ValueError, "minimum_seconds"):
            load_matrix(self.matrix_path)

    def test_loader_requires_exact_booleans_unique_ids_and_supported_power(self) -> None:
        self.setUp_matrix_from_valid(inference_enabled=["false"])
        with self.assertRaisesRegex(ValueError, "entries must be booleans"):
            load_matrix(self.matrix_path)

        self.setUp_matrix_from_valid(preview_enabled=[True, True])
        with self.assertRaisesRegex(ValueError, "entries must be unique"):
            load_matrix(self.matrix_path)

        duplicate = dict(yaml.safe_load(self.matrix_path.read_text())["capture_profiles"][0])
        self.setUp_matrix_from_valid(capture_profiles=[duplicate, duplicate])
        with self.assertRaisesRegex(ValueError, "capture profile ids"):
            load_matrix(self.matrix_path)

        self.setUp_matrix_from_valid(power_modes=["MAXN"])
        with self.assertRaisesRegex(ValueError, "25W, MAXN_SUPER"):
            load_matrix(self.matrix_path)

    def test_loader_rejects_unknown_or_wrong_typed_provenance(self) -> None:
        self.setUp_matrix_from_valid(provenance={"unknown": True})
        with self.assertRaisesRegex(ValueError, "provenance has unknown keys"):
            load_matrix(self.matrix_path)

        self.setUp_matrix_from_valid(
            provenance={"expected_model_identity": {"engine_sha256": 123}}
        )
        with self.assertRaisesRegex(ValueError, "engine_sha256"):
            load_matrix(self.matrix_path)

        self.setUp_matrix_from_valid(
            provenance={"expected_model_identity": {"unknown": "value"}}
        )
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            load_matrix(self.matrix_path)

    def test_loader_caps_cartesian_expansion_before_materializing_cases(self) -> None:
        template = dict(load_matrix(self.matrix_path)["capture_profiles"][0])
        profiles = [
            {**template, "id": f"capture-{index}"}
            for index in range((MAX_MATRIX_CASES // 8) + 1)
        ]
        self.setUp_matrix_from_valid(
            capture_profiles=profiles,
            durations=[{"id": "short", "minimum_seconds": 5}],
            inference_enabled=[True, False],
            preview_enabled=[True, False],
            power_modes=["25W", "MAXN_SUPER"],
        )
        with self.assertRaisesRegex(ValueError, "expands to .* cases"):
            load_matrix(self.matrix_path)

    def test_unassigned_cells_remain_incomplete(self) -> None:
        output = self.root / "report.json"
        result = qualify_matrix(
            self.matrix_path,
            {},
            limits_path=self.root / "limits.yaml",
            output_path=output,
        )
        self.assertEqual(result["result"], "incomplete")
        self.assertEqual(result["case_count"], 4)
        self.assertEqual(result["incomplete_count"], 4)
        self.assertTrue(output.is_file())

    def test_assigned_run_must_match_every_declared_factor(self) -> None:
        case = expand_cases(load_matrix(self.matrix_path))[0]
        run_dir = self.root / "run"
        run_dir.mkdir()

        def fake_qualifier(_run_dir: Path, **_kwargs):
            return {
                "result": "passed",
                "capture_duration_s": 6,
                "factors": {
                    "width": 10, "height": 20, "fps": 30,
                    "camera_count": 1, "pixel_format": "Mono8",
                    "inference_enabled": case["inference_enabled"],
                    "preview_enabled": not case["preview_enabled"],
                    "nvpmodel": "NV Power Mode: 25W",
                },
            }

        result = qualify_matrix(
            self.matrix_path,
            {case["case_id"]: run_dir},
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=fake_qualifier,
        )
        assigned = next(item for item in result["cases"] if item.get("run_directory"))
        self.assertEqual(assigned["result"], "failed")
        self.assertTrue(any("preview_enabled" in item for item in assigned["factor_mismatches"]))

    def test_power_mode_requires_an_exact_named_nvpmodel_mode(self) -> None:
        case = expand_cases(load_matrix(self.matrix_path))[0]
        run_dir = self.root / "substring-power-mode"
        run_dir.mkdir()
        self._write_manifest(run_dir)
        summary = self._passing_summary(case)
        summary["factors"]["nvpmodel"] = "NV Power Mode: 25W_EXPERIMENTAL"

        result = qualify_matrix(
            self.matrix_path,
            {case["case_id"]: run_dir},
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=lambda *_args, **_kwargs: summary,
        )

        self.assertEqual(result["cases"][0]["result"], "failed")
        self.assertTrue(
            any(
                "expected exact named mode" in mismatch
                for mismatch in result["cases"][0]["factor_mismatches"]
            )
        )

    def test_nonfinite_or_boolean_duration_cannot_satisfy_a_cell(self) -> None:
        case = expand_cases(load_matrix(self.matrix_path))[0]
        for index, duration in enumerate((float("nan"), True)):
            with self.subTest(duration=duration):
                run_dir = self.root / f"duration-{index}"
                run_dir.mkdir()
                self._write_manifest(run_dir)
                summary = self._passing_summary(case)
                summary["capture_duration_s"] = duration
                result = qualify_matrix(
                    self.matrix_path,
                    {case["case_id"]: run_dir},
                    limits_path=self.root / "limits.yaml",
                    output_path=self.root / "report.json",
                    qualifier=lambda *_args, **_kwargs: summary,
                )
                self.assertEqual(result["cases"][0]["result"], "failed")
                self.assertTrue(
                    any(
                        mismatch.startswith("duration:")
                        for mismatch in result["cases"][0]["factor_mismatches"]
                    )
                )

    def test_provenance_policy_requires_booleans(self) -> None:
        self._set_matrix(provenance={"require_clean_git": "yes"})

        with self.assertRaisesRegex(ValueError, "require_clean_git must be boolean"):
            load_matrix(self.matrix_path)

    def test_per_run_production_and_clean_git_are_fail_closed(self) -> None:
        self._set_matrix(
            inference_enabled=[True],
            preview_enabled=[True],
            provenance={
                "require_clean_git": True,
                "require_production_eligible": True,
            },
        )
        case = expand_cases(load_matrix(self.matrix_path))[0]
        run_dir = self.root / "non-production"
        run_dir.mkdir()
        self._write_manifest(
            run_dir, dirty=True, production_eligible=False
        )

        result = qualify_matrix(
            self.matrix_path,
            {case["case_id"]: run_dir},
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=lambda *_args, **_kwargs: self._passing_summary(case),
        )

        assigned = result["cases"][0]
        self.assertEqual(assigned["result"], "failed")
        self.assertTrue(
            any("production_eligible" in value for value in assigned["provenance_mismatches"])
        )
        self.assertTrue(
            any("git_dirty" in value for value in assigned["provenance_mismatches"])
        )

    def test_production_matrix_rejects_debug_or_skipped_preflight_evidence(self) -> None:
        self._set_matrix(
            inference_enabled=[False],
            preview_enabled=[True],
            provenance={"require_production_eligible": True},
        )
        case = expand_cases(load_matrix(self.matrix_path))[0]
        run_dir = self.root / "diagnostic-run"
        run_dir.mkdir()
        self._write_manifest(run_dir, production_eligible=True)
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["observability"] = {
            "preflight_skipped": True,
            "deepstream_debug_profile": True,
        }
        manifest_path.write_text(json.dumps(manifest))

        result = qualify_matrix(
            self.matrix_path,
            {case["case_id"]: run_dir},
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=lambda *_args, **_kwargs: self._passing_summary(case),
        )

        mismatches = result["cases"][0]["provenance_mismatches"]
        self.assertTrue(any("preflight_skipped" in value for value in mismatches))
        self.assertTrue(
            any("deepstream_debug_profile" in value for value in mismatches)
        )

    def test_shared_commit_mismatch_fails_all_assigned_cells(self) -> None:
        self._set_matrix(
            inference_enabled=[False],
            preview_enabled=[True, False],
            provenance={"require_same_git_commit": True},
        )
        cases = expand_cases(load_matrix(self.matrix_path))
        assignments = {}
        summaries = {}
        for index, case in enumerate(cases):
            run_dir = self.root / f"commit-{index}"
            run_dir.mkdir()
            self._write_manifest(run_dir, commit=f"commit-{index}")
            assignments[case["case_id"]] = run_dir
            summaries[run_dir] = self._passing_summary(case)

        result = qualify_matrix(
            self.matrix_path,
            assignments,
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=lambda run_dir, **_kwargs: summaries[run_dir],
        )

        self.assertEqual(result["result"], "failed")
        self.assertTrue(all(case["result"] == "failed" for case in result["cases"]))
        self.assertTrue(
            all(
                any("differs across" in value for value in case["provenance_mismatches"])
                for case in result["cases"]
            )
        )

    def test_device_identity_is_captured_for_inference_off_and_must_match(self) -> None:
        self._set_matrix(
            inference_enabled=[False],
            preview_enabled=[True, False],
            provenance={"require_same_device_identity": True},
        )
        cases = expand_cases(load_matrix(self.matrix_path))
        assignments = {}
        summaries = {}
        for index, case in enumerate(cases):
            run_dir = self.root / f"device-{index}"
            run_dir.mkdir()
            self._write_manifest(run_dir)
            if index:
                path = run_dir / "run_manifest.json"
                payload = json.loads(path.read_text())
                payload["platform"]["kernel"] = "different-tegra-kernel"
                path.write_text(json.dumps(payload))
            assignments[case["case_id"]] = run_dir
            summaries[run_dir] = self._passing_summary(case)

        result = qualify_matrix(
            self.matrix_path,
            assignments,
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=lambda run_dir, **_kwargs: summaries[run_dir],
        )

        self.assertEqual(result["result"], "failed")
        self.assertTrue(
            all(case["provenance"]["device_identity"] for case in result["cases"])
        )
        self.assertTrue(
            all(
                "device_identity differs across assigned cells"
                in case["provenance_mismatches"]
                for case in result["cases"]
            )
        )

    def test_same_device_identity_passes_for_inference_off_cells(self) -> None:
        self._set_matrix(
            inference_enabled=[False],
            preview_enabled=[True, False],
            provenance={"require_same_device_identity": True},
        )
        cases = expand_cases(load_matrix(self.matrix_path))
        assignments = {}
        summaries = {}
        for index, case in enumerate(cases):
            run_dir = self.root / f"same-device-{index}"
            run_dir.mkdir()
            self._write_manifest(run_dir)
            self._bind_manifest(run_dir, case)
            assignments[case["case_id"]] = run_dir
            summaries[run_dir] = self._passing_summary(case)

        result = qualify_matrix(
            self.matrix_path,
            assignments,
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=lambda run_dir, **_kwargs: summaries[run_dir],
        )

        self.assertEqual(result["result"], "passed")
        self.assertTrue(
            all(not case["provenance_mismatches"] for case in result["cases"])
        )

    def test_acquisition_identity_binds_task_controller_and_actual_camera(self) -> None:
        self._set_matrix(
            inference_enabled=[False],
            preview_enabled=[True, False],
            provenance={"require_same_acquisition_identity": True},
        )
        cases = expand_cases(load_matrix(self.matrix_path))
        assignments = {}
        summaries = {}
        for index, case in enumerate(cases):
            run_dir = self.root / f"acquisition-{index}"
            run_dir.mkdir()
            self._write_manifest(run_dir)
            if index:
                runtime_path = run_dir / "diagnostics/camera_runtime.json"
                runtime = json.loads(runtime_path.read_text())
                runtime["cameras"][0]["camera_serial"] = "OTHER456"
                runtime_path.write_text(json.dumps(runtime))
            assignments[case["case_id"]] = run_dir
            summaries[run_dir] = self._passing_summary(case)

        result = qualify_matrix(
            self.matrix_path,
            assignments,
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=lambda run_dir, **_kwargs: summaries[run_dir],
        )

        self.assertEqual(result["result"], "failed")
        self.assertTrue(
            all(
                "acquisition_identity differs across assigned cells"
                in case["provenance_mismatches"]
                for case in result["cases"]
            )
        )

    def test_acquisition_identity_rejects_mixed_controller_protocols(self) -> None:
        self._set_matrix(
            inference_enabled=[False],
            preview_enabled=[True, False],
            provenance={"require_same_acquisition_identity": True},
        )
        cases = expand_cases(load_matrix(self.matrix_path))
        assignments = {}
        summaries = {}
        for index, case in enumerate(cases):
            run_dir = self.root / f"controller-protocol-{index}"
            run_dir.mkdir()
            self._write_manifest(run_dir)
            if index:
                manifest_path = run_dir / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["serial"]["controller_protocol"] = (
                    "watchdog_v1_experimental"
                )
                manifest_path.write_text(json.dumps(manifest))
            assignments[case["case_id"]] = run_dir
            summaries[run_dir] = self._passing_summary(case)

        result = qualify_matrix(
            self.matrix_path,
            assignments,
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=lambda run_dir, **_kwargs: summaries[run_dir],
        )

        self.assertEqual(result["result"], "failed")
        self.assertTrue(
            all(
                "acquisition_identity differs across assigned cells"
                in case["provenance_mismatches"]
                for case in result["cases"]
            )
        )

    def test_assigned_cell_requires_exact_startup_case_binding(self) -> None:
        self._set_matrix(
            inference_enabled=[False],
            preview_enabled=[False],
            provenance={"require_case_binding": True},
        )
        case = expand_cases(load_matrix(self.matrix_path))[0]
        run_dir = self.root / "bound-run"
        run_dir.mkdir()
        self._write_manifest(run_dir)
        summary = self._passing_summary(case)

        missing = qualify_matrix(
            self.matrix_path,
            {case["case_id"]: run_dir},
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "missing.json",
            qualifier=lambda *_args, **_kwargs: summary,
        )
        self.assertEqual(missing["result"], "failed")
        self.assertIn(
            "manifest qualification case binding is missing",
            missing["cases"][0]["provenance_mismatches"],
        )

        self._bind_manifest(run_dir, case)
        passed = qualify_matrix(
            self.matrix_path,
            {case["case_id"]: run_dir},
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "passed.json",
            qualifier=lambda *_args, **_kwargs: summary,
        )
        self.assertEqual(passed["result"], "passed")

    def test_shared_model_identity_uses_hashes_not_only_name(self) -> None:
        self._set_matrix(
            inference_enabled=[True],
            preview_enabled=[True, False],
            provenance={"require_same_model_identity": True},
        )
        cases = expand_cases(load_matrix(self.matrix_path))
        assignments = {}
        summaries = {}
        for index, case in enumerate(cases):
            run_dir = self.root / f"model-{index}"
            run_dir.mkdir()
            self._write_manifest(run_dir, engine_sha256=str(index + 1) * 64)
            assignments[case["case_id"]] = run_dir
            summaries[run_dir] = self._passing_summary(case)

        result = qualify_matrix(
            self.matrix_path,
            assignments,
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=lambda run_dir, **_kwargs: summaries[run_dir],
        )

        self.assertEqual(result["result"], "failed")
        self.assertTrue(all(case["result"] == "failed" for case in result["cases"]))
        self.assertTrue(
            all(
                "model_identity differs across inference-enabled cells"
                in case["provenance_mismatches"]
                for case in result["cases"]
            )
        )

    def test_schema_two_model_cannot_enter_production_matrix(self) -> None:
        self._set_matrix(
            inference_enabled=[True],
            preview_enabled=[True],
            provenance={"require_same_model_identity": True},
        )
        case = expand_cases(load_matrix(self.matrix_path))[0]
        run_dir = self.root / "schema-two-model"
        run_dir.mkdir()
        self._write_manifest(run_dir)
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["inference"]["model_package"]["model_manifest_schema"] = 2
        manifest["inference"]["model_package"]["engine_build_identity"] = None
        manifest_path.write_text(json.dumps(manifest))

        result = qualify_matrix(
            self.matrix_path,
            {case["case_id"]: run_dir},
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=lambda *_args, **_kwargs: self._passing_summary(case),
        )

        mismatches = result["cases"][0]["provenance_mismatches"]
        self.assertEqual(result["cases"][0]["result"], "failed")
        self.assertTrue(any("model_manifest_schema" in value for value in mismatches))
        self.assertTrue(any("engine_build_identity" in value for value in mismatches))

    def test_missing_model_content_hash_cannot_enter_production_matrix(self) -> None:
        self._set_matrix(
            inference_enabled=[True],
            preview_enabled=[True],
            provenance={"require_same_model_identity": True},
        )
        case = expand_cases(load_matrix(self.matrix_path))[0]
        run_dir = self.root / "missing-onnx-identity"
        run_dir.mkdir()
        self._write_manifest(run_dir)
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["inference"]["model_package"].pop("onnx_sha256")
        manifest_path.write_text(json.dumps(manifest))

        result = qualify_matrix(
            self.matrix_path,
            {case["case_id"]: run_dir},
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=lambda *_args, **_kwargs: self._passing_summary(case),
        )

        mismatches = result["cases"][0]["provenance_mismatches"]
        self.assertEqual(result["cases"][0]["result"], "failed")
        self.assertTrue(any("onnx_sha256" in value for value in mismatches))

    def test_expected_commit_and_model_identity_are_enforced(self) -> None:
        approved_engine = "a" * 64
        other_engine = "b" * 64
        self._set_matrix(
            inference_enabled=[True],
            preview_enabled=[True],
            provenance={
                "expected_git_commit": "approved",
                "expected_model_identity": {"engine_sha256": approved_engine},
            },
        )
        case = expand_cases(load_matrix(self.matrix_path))[0]
        run_dir = self.root / "unexpected"
        run_dir.mkdir()
        self._write_manifest(run_dir, commit="other", engine_sha256=other_engine)

        result = qualify_matrix(
            self.matrix_path,
            {case["case_id"]: run_dir},
            limits_path=self.root / "limits.yaml",
            output_path=self.root / "report.json",
            qualifier=lambda *_args, **_kwargs: self._passing_summary(case),
        )

        mismatches = result["cases"][0]["provenance_mismatches"]
        self.assertTrue(any("git_commit" in value for value in mismatches))
        self.assertTrue(any("engine_sha256" in value for value in mismatches))


if __name__ == "__main__":
    unittest.main()
