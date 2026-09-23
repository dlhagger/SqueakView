from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from squeakview.apps.operator.backend.contracts import RunRequest
from squeakview.apps.operator.backend.manifest import (
    DURABLE_SUPERVISOR_OWNER,
    IN_PROCESS_DEV_OWNER,
    MAX_TASK_CONFIG_BYTES,
    RunManifestContext,
    RunManifestService,
    snapshot_task_config,
)
from squeakview.common import run_context
from squeakview.common.failure_injection import FailurePlan


class RunManifestServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.run_dir = self.root / "run-001"
        self.run_dir.mkdir()
        self.logs: list[str] = []
        self.service = RunManifestService(self.logs.append)
        self.context = RunManifestContext(
            config=RunRequest(
                fps=30,
                serial_enabled=False,
                inference_enabled=False,
                experiment_name="study",
                mouse_id="mouse-1",
            ),
            application_root=self.root,
            project_root=self.root,
            project_id="00000000-0000-0000-0000-000000000001",
            project_name="Test",
            runs_root=self.root,
            created_at="2026-09-02T10:00:00",
            storage={"free_bytes": 1234},
            model_snapshot=None,
            device_context={"jetson_model": "test-device"},
            acquisition_owner=DURABLE_SUPERVISOR_OWNER,
            preflight_evidence={
                "schema_version": "3.0",
                "passed": True,
                "skipped": False,
                "ffprobe_available": True,
                "video_decode_validated": True,
                "new_streammux_validated": True,
                "automatic_suspend_disabled": True,
                "output_size_bytes": 1,
                "output_sha256": "f" * 64,
            },
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_output_snapshot_reports_files_and_validation_evidence(self) -> None:
        (self.run_dir / "frames.csv").write_text("frame\n1\n")
        (self.run_dir / "raw.mp4").write_bytes(b"video")
        (self.run_dir / "diagnostics").mkdir()
        (self.run_dir / "diagnostics" / "recording.csv").write_text("event\nok\n")
        run_context.atomic_write_json(
            self.run_dir / run_context.RUN_STATUS_FILENAME,
            {
                "recording_validation": {"passed": True},
                "capture_reconciliation": {"passed": True},
            },
        )

        snapshot = self.service.output_snapshot(self.run_dir)

        self.assertTrue(snapshot["csv_files"]["frames"]["exists"])
        self.assertEqual(snapshot["csv_files"]["frames"]["size_bytes"], 8)
        self.assertEqual(snapshot["video_files"][0]["size_bytes"], 5)
        self.assertEqual(snapshot["recording_validation"], {"passed": True})
        self.assertEqual(snapshot["capture_reconciliation"], {"passed": True})

    def test_build_uses_supplied_provenance_and_inventory_callbacks(self) -> None:
        with mock.patch.object(
            self.service,
            "git_snapshot",
            return_value={"commit": "abc", "dirty": False},
        ):
            result = self.service.build(
                self.run_dir,
                self.context,
                output_snapshot=lambda _path: {"inventory": "snapshot"},
                bottle_snapshot=lambda _path: {"complete": True},
            )

        self.assertEqual(result["created_at"], "2026-09-02T10:00:00")
        self.assertEqual(result["schema_version"], "3.0")
        self.assertEqual(result["application"], {"root": str(self.root)})
        self.assertNotIn("workspace", result)
        self.assertEqual(
            result["project"],
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "name": "Test",
                "root": str(self.root),
            },
        )
        self.assertEqual(result["platform"], {"jetson_model": "test-device"})
        self.assertEqual(result["git"], {"commit": "abc", "dirty": False})
        self.assertEqual(result["storage"]["free_bytes"], 1234)
        self.assertEqual(
            result["storage"]["reserve_supervision"]["min_free_bytes"],
            1_000_000_000,
        )
        self.assertEqual(
            result["storage"]["reserve_supervision"]["failure_policy"],
            "fatal_graceful_capture_shutdown",
        )
        self.assertEqual(result["capture"]["fps"], 30)
        self.assertEqual(
            result["process_topology"]["acquisition_owner"],
            DURABLE_SUPERVISOR_OWNER,
        )
        self.assertTrue(result["production_eligible"])
        self.assertTrue(result["inference"]["preview_enabled"])
        self.assertEqual(result["inference"]["streammux_implementation"], "new-v2-pinned")
        self.assertEqual(
            result["recording"]["encoder"]["bitrate_kbps"],
            self.context.config.bitrate,
        )
        self.assertEqual(result["recording"]["encoder"]["pixel_fidelity"], "lossy")
        self.assertEqual(result["recording"]["encoder"]["reference_frames"], 1)
        self.assertFalse(result["recording"]["encoder"]["adaptive_quantization"])
        self.assertFalse(result["serial"]["alignment_required"])
        self.assertIsNone(result["expected_outputs"]["alignment_summary"])
        self.assertEqual(
            result["recording"]["scientific_claim"],
            "temporal_frame_completeness",
        )
        self.assertEqual(result["actual_outputs"], {"inventory": "snapshot"})
        self.assertEqual(result["bottles"], {"complete": True})

    def test_legacy_triggered_serial_capture_declares_alignment_output(self) -> None:
        context = replace(
            self.context,
            config=replace(
                self.context.config,
                serial_enabled=True,
                trigger_on=True,
                controller_protocol="legacy",
            ),
        )

        result = self.service.build(
            self.run_dir,
            context,
            output_snapshot=lambda _path: {},
            bottle_snapshot=lambda _path: {},
        )

        self.assertTrue(result["serial"]["alignment_required"])
        self.assertEqual(
            result["expected_outputs"]["alignment_summary"],
            "alignment_summary.json",
        )
        self.assertIn(
            "controller_protocol_not_v2",
            result["production_disqualifiers"],
        )

    def test_v2_manifest_embeds_final_transport_summary(self) -> None:
        diagnostics = self.run_dir / "diagnostics"
        diagnostics.mkdir()
        summary = {
            "schema_version": 1,
            "protocol": "mousehouse_v2",
            "boot_id": 99,
            "session_id": 2,
            "integrity_latched": False,
            "counts": {
                "frames_received": 12,
                "frames_stored": 12,
                "duplicates": 0,
            },
        }
        run_context.atomic_write_json(
            diagnostics / "controller_v2_summary.json",
            summary,
        )
        context = replace(
            self.context,
            config=replace(
                self.context.config,
                serial_enabled=True,
                trigger_on=True,
            ),
        )

        result = self.service.build(
            self.run_dir,
            context,
            output_snapshot=lambda _path: {},
            bottle_snapshot=lambda _path: {},
        )

        self.assertEqual(result["serial"]["controller_protocol"], "v2")
        self.assertFalse(result["serial"]["alignment_required"])
        self.assertEqual(result["serial"]["v2_transport"], summary)
        self.assertNotIn(
            "controller_protocol_not_v2",
            result["production_disqualifiers"],
        )

    def test_build_persists_qualification_case_binding(self) -> None:
        binding = {
            "matrix_id": "matrix-v1",
            "matrix_path": "/workspace/qualification/matrix.v1.yaml",
            "matrix_sha256": "a" * 64,
            "case_id": "capture--short--infer-off--preview-off--25w",
            "expected_factors": {"preview_enabled": False},
        }

        result = self.service.build(
            self.run_dir,
            replace(self.context, qualification_case=binding),
            output_snapshot=lambda _path: {},
            bottle_snapshot=lambda _path: {},
        )

        self.assertEqual(result["qualification"], binding)

    def test_build_retains_model_content_hashes_from_validated_snapshot(self) -> None:
        model_snapshot = {
            "name": "mouse-pose",
            "model_manifest_sha256": "a" * 64,
            "pose_sidecar_sha256": "b" * 64,
            "onnx_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "engine_sha256": "e" * 64,
        }
        context = RunManifestContext(
            config=RunRequest(
                fps=30,
                inference_enabled=True,
                serial_enabled=False,
            ),
            application_root=self.context.application_root,
            project_root=self.context.project_root,
            project_id=self.context.project_id,
            project_name=self.context.project_name,
            runs_root=self.context.runs_root,
            created_at=self.context.created_at,
            storage=self.context.storage,
            model_snapshot=model_snapshot,
            device_context=self.context.device_context,
            acquisition_owner=DURABLE_SUPERVISOR_OWNER,
        )

        result = self.service.build(
            self.run_dir,
            context,
            output_snapshot=lambda _path: {},
            bottle_snapshot=lambda _path: {},
        )

        self.assertEqual(result["inference"]["model_package"], model_snapshot)

    def test_effective_runtime_uses_package_parser_not_workspace_build(self) -> None:
        workspace_parser = (
            self.root
            / "native/nvdsinfer_custom_impl_yolo/libnvdsinfer_custom_impl_Yolo.so"
        )
        workspace_parser.parent.mkdir(parents=True)
        workspace_parser.write_bytes(b"workspace parser")
        package_parser = self.root / "models/mouse/lib/parser.so"
        package_parser.parent.mkdir(parents=True)
        package_parser.write_bytes(b"selected package parser")
        localized = self.run_dir / "config/model.txt"
        localized.parent.mkdir()
        localized.write_text(f"[property]\ncustom-lib-path={package_parser}\n")
        from squeakview.common.device_context import file_identity

        effective_paths = {
            "deepstream_config": localized,
            "pose_sidecar": localized.parent / "model.pose.json",
            "class_labels": localized.parent / "model.classes.txt",
            "keypoint_labels": localized.parent / "model.keypoints.txt",
            "onnx": self.root / "models/mouse/onnx/model.onnx",
            "engine": self.root / "models/mouse/engines/model.engine",
            "custom_parser": package_parser,
        }
        for name, path in effective_paths.items():
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
        effective = {
            name: file_identity(path) for name, path in effective_paths.items()
        }
        context = replace(
            self.context,
            config=replace(
                self.context.config,
                inference_enabled=True,
                ds_cfg=localized,
                exposure_us=4321.0,
            ),
            model_snapshot={
                "config": "/portable/model/config.txt",
                "config_sha256": "a" * 64,
            },
            effective_deepstream=effective,
        )

        result = self.service.build(
            self.run_dir,
            context,
            output_snapshot=lambda _path: {},
            bottle_snapshot=lambda _path: {},
        )

        self.assertEqual(result["inference"]["effective_runtime"], effective)
        self.assertEqual(
            result["native_plugins"]["deepstream_yolo_parser"]["sha256"],
            effective["custom_parser"]["sha256"],
        )
        self.assertNotEqual(
            result["native_plugins"]["deepstream_yolo_parser"]["sha256"],
            result["native_plugins"]["application_deepstream_yolo_parser_build"]["sha256"],
        )
        self.assertEqual(result["capture"]["exposure_us"], 4321.0)
        self.assertEqual(
            result["inference"]["model_package"]["config"],
            "/portable/model/config.txt",
        )
        self.assertEqual(
            result["inference"]["portable_source_config"],
            {"path": "/portable/model/config.txt", "sha256": "a" * 64},
        )

    def test_task_config_is_atomically_snapshotted_with_content_identity(self) -> None:
        source = self.root / "selected-task.yaml"
        content = b"task_name: pellet\nreward_ul: 10\n"
        source.write_bytes(content)

        identity = snapshot_task_config(self.run_dir, source)
        context = RunManifestContext(
            config=RunRequest(
                fps=30,
                serial_enabled=False,
                inference_enabled=False,
                task_cfg=source,
            ),
            application_root=self.context.application_root,
            project_root=self.context.project_root,
            project_id=self.context.project_id,
            project_name=self.context.project_name,
            runs_root=self.context.runs_root,
            created_at=self.context.created_at,
            storage=self.context.storage,
            model_snapshot=None,
            device_context=self.context.device_context,
            acquisition_owner=DURABLE_SUPERVISOR_OWNER,
            task_config_snapshot=identity,
        )
        result = self.service.build(
            self.run_dir,
            context,
            output_snapshot=lambda _path: {},
            bottle_snapshot=lambda _path: {},
        )

        self.assertEqual(source.read_bytes(), content)
        self.assertEqual((self.run_dir / "config/task.yaml").read_bytes(), content)
        self.assertEqual(result["task_config"], identity)
        self.assertEqual(identity["snapshot_path"], "config/task.yaml")
        self.assertEqual(identity["size_bytes"], len(content))

    def test_oversize_task_config_is_rejected_without_partial_snapshot(self) -> None:
        source = self.root / "oversize-task.yaml"
        source.write_bytes(b"x" * (MAX_TASK_CONFIG_BYTES + 1))

        with self.assertRaisesRegex(ValueError, "bounded snapshot limit"):
            snapshot_task_config(self.run_dir, source)

        self.assertFalse((self.run_dir / "config/task.yaml").exists())

    def test_in_process_owner_is_explicitly_nonproduction(self) -> None:
        context = RunManifestContext(
            config=self.context.config,
            application_root=self.context.application_root,
            project_root=self.context.project_root,
            project_id=self.context.project_id,
            project_name=self.context.project_name,
            runs_root=self.context.runs_root,
            created_at=self.context.created_at,
            storage=self.context.storage,
            model_snapshot=None,
            device_context=self.context.device_context,
            acquisition_owner=IN_PROCESS_DEV_OWNER,
        )

        result = self.service.build(
            self.run_dir,
            context,
            output_snapshot=lambda _path: {},
            bottle_snapshot=lambda _path: {},
        )

        self.assertEqual(
            result["process_topology"]["acquisition_owner"], IN_PROCESS_DEV_OWNER
        )
        self.assertFalse(result["production_eligible"])
        self.assertIn(
            "in_process_acquisition_owner", result["production_disqualifiers"]
        )

    def test_terminal_update_preserves_acquisition_provenance(self) -> None:
        original = {
            "schema_version": "3.0",
            "created_at": "original-time",
            "platform": {"jetson_model": "original-device"},
            "git": {"commit": "original", "dirty": False},
            "capture": {"fps": 120},
            "bottles": {"complete": False},
            "actual_outputs": {"old": True},
        }
        run_context.write_manifest(self.run_dir, original)
        run_context.atomic_write_json(
            self.run_dir / run_context.RUN_STATUS_FILENAME,
            {"state": "finalized"},
        )
        builder = mock.Mock(side_effect=AssertionError("must not rebuild terminal run"))

        self.assertTrue(
            self.service.write(
                self.run_dir,
                self.context,
                build_manifest=builder,
                output_snapshot=lambda _path: {"new": True},
                bottle_snapshot=lambda _path: {"complete": True},
            )
        )

        saved = run_context.read_json(
            self.run_dir / run_context.RUN_MANIFEST_FILENAME
        )
        builder.assert_not_called()
        self.assertEqual(saved["created_at"], "original-time")
        self.assertEqual(saved["platform"], original["platform"])
        self.assertEqual(saved["git"], original["git"])
        self.assertEqual(saved["capture"], original["capture"])
        self.assertEqual(saved["actual_outputs"], {"new": True})
        self.assertEqual(saved["bottles"], {"complete": True})

    def test_terminal_update_rejects_unsupported_manifest_schema(self) -> None:
        original = {
            "schema_version": "1.0",
            "created_at": "original-time",
            "capture": {"fps": 120},
        }
        run_context.write_manifest(self.run_dir, original)
        run_context.atomic_write_json(
            self.run_dir / run_context.RUN_STATUS_FILENAME,
            {"state": "finalized"},
        )

        with self.assertRaisesRegex(
            RuntimeError, "unsupported completed run manifest schema"
        ):
            self.service.write(
                self.run_dir,
                self.context,
                required=True,
                output_snapshot=lambda _path: {"new": True},
                bottle_snapshot=lambda _path: {"complete": True},
            )

        self.assertEqual(
            run_context.read_json(
                self.run_dir / run_context.RUN_MANIFEST_FILENAME
            ),
            original,
        )

    def test_required_write_wraps_persistence_failure(self) -> None:
        with mock.patch.object(
            run_context,
            "write_manifest",
            side_effect=OSError("ENOSPC"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "run manifest persistence failed: ENOSPC"
            ):
                self.service.write(
                    self.run_dir,
                    self.context,
                    required=True,
                    build_manifest=lambda _path: {},
                )

        self.assertIn("manifest write failed: ENOSPC", self.logs[-1])

    def test_dangling_existing_manifest_is_not_treated_as_initial_absence(self) -> None:
        manifest_path = self.run_dir / run_context.RUN_MANIFEST_FILENAME
        manifest_path.symlink_to(self.run_dir / "missing-target.json")

        with self.assertRaisesRegex(RuntimeError, "could not be read"):
            self.service.write(
                self.run_dir,
                self.context,
                required=True,
                build_manifest=lambda _path: {},
            )

        self.assertTrue(manifest_path.is_symlink())

    def test_build_labels_failure_injection_non_production(self) -> None:
        plan = FailurePlan("1.0", "serial_controller", "read_error", 10)
        context = RunManifestContext(
            config=self.context.config,
            application_root=self.context.application_root,
            project_root=self.context.project_root,
            project_id=self.context.project_id,
            project_name=self.context.project_name,
            runs_root=self.context.runs_root,
            created_at=self.context.created_at,
            storage=self.context.storage,
            model_snapshot=None,
            device_context=self.context.device_context,
            failure_plan=plan.as_manifest(),
            acquisition_owner=DURABLE_SUPERVISOR_OWNER,
        )

        result = self.service.build(
            self.run_dir,
            context,
            output_snapshot=lambda _path: {},
            bottle_snapshot=lambda _path: {},
        )

        self.assertFalse(result["production_eligible"])
        self.assertEqual(result["failure_injection"], plan.as_manifest())

    def test_build_records_deepstream_debug_profile_and_bounded_log(self) -> None:
        probes = {
            "measure_latency_probe": {
                "available": True, "size_bytes": 10, "sha256": "a" * 64,
            },
            "measure_fps_probe": {
                "available": True, "size_bytes": 11, "sha256": "b" * 64,
            },
        }
        context = replace(
            self.context,
            preflight_evidence={
                **dict(self.context.preflight_evidence or {}),
                "deepstream_debug_probes": probes,
            },
        )
        with mock.patch.dict(
            "os.environ", {"SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE": "1"}
        ):
            result = self.service.build(
                self.run_dir,
                context,
                output_snapshot=lambda _path: {},
                bottle_snapshot=lambda _path: {},
            )

        observability = result["observability"]
        self.assertTrue(observability["deepstream_debug_profile"])
        self.assertEqual(observability["deepstream_log"], "diagnostics/deepstream.log")
        self.assertEqual(observability["deepstream_log_max_bytes"], 64 * 1024 * 1024)
        self.assertEqual(observability["deepstream_debug_probes"], probes)
        self.assertFalse(result["production_eligible"])
        self.assertEqual(
            result["production_disqualifiers"], ["deepstream_debug_profile"]
        )

    def test_skipped_preflight_is_explicitly_nonproduction(self) -> None:
        with mock.patch.dict("os.environ", {"SQUEAKVIEW_SKIP_PREFLIGHT": "1"}):
            result = self.service.build(
                self.run_dir,
                self.context,
                output_snapshot=lambda _path: {},
                bottle_snapshot=lambda _path: {},
            )

        self.assertFalse(result["production_eligible"])
        self.assertEqual(result["production_disqualifiers"], ["preflight_skipped"])
        self.assertTrue(result["observability"]["preflight_skipped"])

    def test_supervisor_failure_gate_is_explicitly_nonproduction(self) -> None:
        with mock.patch.dict(
            "os.environ", {"SQUEAKVIEW_ENABLE_SUPERVISOR_FAILURE_INJECTION": "1"}
        ):
            result = self.service.build(
                self.run_dir,
                self.context,
                output_snapshot=lambda _path: {},
                bottle_snapshot=lambda _path: {},
            )

        self.assertFalse(result["production_eligible"])
        self.assertEqual(
            result["production_disqualifiers"],
            ["supervisor_failure_injection"],
        )

    def test_supervisor_failure_barrier_is_nonproduction_even_if_misconfigured(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"SQUEAKVIEW_SUPERVISOR_FAILURE_BARRIER": "pre_capture"},
        ):
            result = self.service.build(
                self.run_dir,
                self.context,
                output_snapshot=lambda _path: {},
                bottle_snapshot=lambda _path: {},
            )

        self.assertFalse(result["production_eligible"])
        self.assertIn(
            "supervisor_failure_barrier", result["production_disqualifiers"]
        )


if __name__ == "__main__":
    unittest.main()
