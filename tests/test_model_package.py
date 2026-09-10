from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from unittest import mock

from squeakview import config as squeakview_config
from squeakview import model_package
from squeakview.model_package import ModelPackageError, validate_model_package


class ModelPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.package = self.root / "models" / "test_pose_fp16"
        self.config_dir = self.package / "configs"
        for directory in (
            self.config_dir,
            self.package / "onnx",
            self.package / "engines",
            self.package / "labels",
            self.package / "validation",
            self.root / "native",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        (self.package / "model.yaml").write_text(yaml.safe_dump({
            "schema_version": 2,
            "name": "test_pose_fp16",
            "framework": "yolo26",
            "task": "pose",
            "precision": "fp16",
            "batch_size": 1,
            "classes": ["mouse"],
            "keypoints": ["nose"],
            "export": {
                "builder": "ultralytics",
                "data": "build_me/test.yaml",
                "end2end": True,
            },
        }, sort_keys=False))
        (self.package / "onnx" / "model.onnx").write_bytes(b"onnx")
        (self.package / "engines" / "model.engine").write_bytes(b"engine")
        (self.package / "labels" / "classes.txt").write_text("mouse\n")
        (self.package / "labels" / "labels.txt").write_text("nose\n")
        (self.root / "native" / "parser.so").write_bytes(b"parser")
        self.config = self.config_dir / "test_pose_fp16.txt"
        self.config.write_text(
            "[property]\n"
            "onnx-file=../onnx/model.onnx\n"
            "model-engine-file=../engines/model.engine\n"
            "labelfile-path=../labels/classes.txt\n"
            "custom-lib-path=../../../native/parser.so\n"
            "parse-bbox-func-name=NvDsInferParseYolo26Pose\n"
            "infer-dims=3;640;640\n"
            "batch-size=1\n"
            "num-detected-classes=1\n"
            "output-tensor-meta=1\n"
            "cluster-mode=4\n"
            "maintain-aspect-ratio=1\n"
            "symmetric-padding=1\n"
        )
        self.sidecar = self.config_dir / "test_pose_fp16.pose.json"
        self.sidecar.write_text(json.dumps({
            "schema_version": 2,
            "task": "pose",
            "postprocess": "pyservicemaker_yolo26_pose_v1",
            "output_layer": "output0",
            "input_width": 640,
            "input_height": 640,
            "letterbox": "symmetric",
            "end2end": True,
            "keypoint_labels_path": "../labels/labels.txt",
            "keypoint_count": 1,
            "keypoint_dims": 3,
            "keypoint_threshold": 0.5,
            "classes": [{
                "id": 0,
                "name": "mouse",
                "threshold": 0.25,
                "track": True,
                "keypoint_indices": [0],
            }],
        }, indent=2) + "\n")
        self.report = self.package / "validation" / "import_report.json"
        self.report.write_text(json.dumps({
            "onnx_input_shape": [1, 3, 640, 640],
            "onnx_output_shapes": [[1, 300, 9]],
            "checks": {"onnx": True, "raw_engine": True, "yaml_labels": True, "schema_v2": True},
        }, indent=2) + "\n")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def upgrade_to_schema_three(self, identity: dict[str, object]) -> None:
        packaged_parser = self.package / "lib" / "parser.so"
        packaged_parser.parent.mkdir()
        packaged_parser.write_bytes((self.root / "native" / "parser.so").read_bytes())
        self.config.write_text(
            self.config.read_text().replace(
                "custom-lib-path=../../../native/parser.so",
                "custom-lib-path=../lib/parser.so",
            )
        )
        report = json.loads(self.report.read_text())
        report["build_environment"] = identity
        report["checks"]["engine_execution"] = True
        report["engine_execution"] = {
            "command": ["trtexec", "--iterations=1"],
            "returncode": 0,
            "timed_out": False,
            "max_output_bytes": 65536,
            "output_truncated": False,
            "output": "PASSED TensorRT.trtexec",
            "passed": True,
        }
        self.report.write_text(json.dumps(report))
        manifest_path = self.package / "model.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest["schema_version"] = 3
        manifest["artifacts"] = {
            name: {
                "path": path.relative_to(self.package).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in {
                "config": self.config,
                "pose_schema": self.sidecar,
                "onnx": self.package / "onnx" / "model.onnx",
                "engine": self.package / "engines" / "model.engine",
                "class_labels": self.package / "labels" / "classes.txt",
                "keypoint_labels": self.package / "labels" / "labels.txt",
                "custom_parser": packaged_parser,
                "import_report": self.report,
            }.items()
        }
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    def test_complete_package_returns_identity_and_hashes(self) -> None:
        info = validate_model_package(self.config)

        self.assertEqual(info.name, "test_pose_fp16")
        self.assertEqual(info.root, self.package)
        self.assertEqual(info.import_report, self.report)
        self.assertEqual(len(info.config_sha256), 64)
        self.assertEqual(len(info.engine_sha256), 64)
        self.assertEqual(
            info.model_manifest_sha256,
            hashlib.sha256((self.package / "model.yaml").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            info.pose_sidecar_sha256,
            hashlib.sha256(self.sidecar.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            info.onnx_sha256,
            hashlib.sha256((self.package / "onnx" / "model.onnx").read_bytes()).hexdigest(),
        )
        self.assertEqual(info.model_manifest_schema, 2)
        self.assertEqual(info.manifest_snapshot()["engine"], str(self.package / "engines" / "model.engine"))
        self.assertEqual(info.manifest_snapshot()["onnx_sha256"], info.onnx_sha256)

    def test_incomplete_package_reports_every_missing_artifact(self) -> None:
        (self.package / "engines" / "model.engine").unlink()
        (self.package / "labels" / "labels.txt").unlink()

        with self.assertRaises(ModelPackageError) as raised:
            validate_model_package(self.config)

        message = str(raised.exception)
        self.assertIn("TensorRT engine", message)
        self.assertIn("keypoint labels", message)

    def test_config_outside_package_layout_is_rejected(self) -> None:
        misplaced = self.package / "model.txt"
        misplaced.write_text(self.config.read_text())

        with self.assertRaisesRegex(ModelPackageError, "configs/ directory"):
            validate_model_package(misplaced)

    def test_duplicate_model_config_key_is_rejected(self) -> None:
        with self.config.open("a") as handle:
            handle.write("batch-size=2\n")

        with self.assertRaisesRegex(ModelPackageError, "strict|already exists"):
            validate_model_package(self.config)

    def test_duplicate_manifest_key_is_rejected(self) -> None:
        with (self.package / "model.yaml").open("a") as handle:
            handle.write("name: replacement\n")

        with self.assertRaisesRegex(ModelPackageError, "duplicate key 'name'"):
            validate_model_package(self.config)

    def test_model_artifact_must_remain_inside_package(self) -> None:
        outside = self.root / "outside.engine"
        outside.write_bytes(b"engine")
        self.config.write_text(
            self.config.read_text().replace(
                "model-engine-file=../engines/model.engine",
                f"model-engine-file={outside}",
            )
        )

        with self.assertRaisesRegex(ModelPackageError, "inside model package"):
            validate_model_package(self.config)

    def test_model_config_read_is_size_bounded(self) -> None:
        with mock.patch.object(model_package, "_MAX_CONFIG_BYTES", 8):
            with self.assertRaisesRegex(ModelPackageError, "byte limit"):
                validate_model_package(self.config)

    def test_legacy_pose_schema_is_rejected(self) -> None:
        self.sidecar.write_text('{"schema_version": 1}\n')

        with self.assertRaisesRegex(ModelPackageError, "rebuild this legacy schema-1 package"):
            validate_model_package(self.config)

    def test_class_keypoint_contract_is_validated(self) -> None:
        payload = json.loads(self.sidecar.read_text())
        payload["classes"][0]["keypoint_indices"] = [2]
        self.sidecar.write_text(json.dumps(payload))

        with self.assertRaisesRegex(ModelPackageError, "out-of-range keypoint index"):
            validate_model_package(self.config)

    def test_import_report_shapes_are_validated(self) -> None:
        payload = json.loads(self.report.read_text())
        payload["onnx_output_shapes"] = [[1, 300, 8]]
        self.report.write_text(json.dumps(payload))

        with self.assertRaisesRegex(ModelPackageError, "tensor shapes"):
            validate_model_package(self.config)

    def test_model_manifest_must_be_a_mapping(self) -> None:
        (self.package / "model.yaml").write_text("- schema_version\n- 3\n")

        with self.assertRaisesRegex(ModelPackageError, "must contain a YAML mapping"):
            validate_model_package(self.config)

    def test_fractional_schema_version_is_not_silently_truncated(self) -> None:
        manifest = yaml.safe_load((self.package / "model.yaml").read_text())
        manifest["schema_version"] = 2.5
        (self.package / "model.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False)
        )

        with self.assertRaisesRegex(ModelPackageError, "schema_version must be an integer"):
            validate_model_package(self.config)

    def test_schema_three_rejects_engine_built_for_different_runtime(self) -> None:
        recorded = {
            "tensorrt_version": "10.16.2",
            "cuda_version": "13.2",
            "gpu_name": "Orin",
            "compute_capability": [8, 7],
            "device_model": "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super",
            "jetson_linux_release": "# R39 (release), REVISION: 2.1",
        }
        self.upgrade_to_schema_three(recorded)
        runtime = dict(recorded, tensorrt_version="10.17.0")

        with mock.patch.object(
            model_package, "_runtime_engine_identity", return_value=runtime
        ):
            with self.assertRaisesRegex(ModelPackageError, "different runtime/device"):
                validate_model_package(self.config)

    def test_schema_three_records_matching_engine_identity(self) -> None:
        identity = {
            "tensorrt_version": "10.16.2",
            "cuda_version": "13.2",
            "gpu_name": "Orin",
            "compute_capability": [8, 7],
            "device_model": "Jetson Orin Nano Super",
            "jetson_linux_release": "# R39, REVISION: 2.1",
        }
        self.upgrade_to_schema_three(identity)

        with mock.patch.object(
            model_package, "_runtime_engine_identity", return_value=identity
        ):
            info = validate_model_package(self.config)

        self.assertEqual(info.engine_build_identity, identity)
        self.assertEqual(info.model_manifest_schema, 3)

    def test_schema_three_rejects_artifact_changed_after_build(self) -> None:
        identity = {
            "tensorrt_version": "10.16.2",
            "cuda_version": "13.2",
            "gpu_name": "Orin",
            "compute_capability": [8, 7],
            "device_model": "Jetson Orin Nano Super",
            "jetson_linux_release": "# R39, REVISION: 2.1",
        }
        self.upgrade_to_schema_three(identity)
        (self.package / "engines" / "model.engine").write_bytes(b"mixed engine")

        with mock.patch.object(
            model_package, "_runtime_engine_identity", return_value=identity
        ):
            with self.assertRaisesRegex(ModelPackageError, "recorded SHA-256"):
                validate_model_package(self.config)

    def test_schema_three_rejects_import_report_changed_after_build(self) -> None:
        identity = {
            "tensorrt_version": "10.16.2",
            "cuda_version": "13.2",
            "gpu_name": "Orin",
            "compute_capability": [8, 7],
            "device_model": "Jetson Orin Nano Super",
            "jetson_linux_release": "# R39, REVISION: 2.1",
        }
        self.upgrade_to_schema_three(identity)
        report = json.loads(self.report.read_text())
        report["build_environment"]["tensorrt_version"] = "edited"
        self.report.write_text(json.dumps(report))

        with mock.patch.object(
            model_package, "_runtime_engine_identity", return_value=identity
        ):
            with self.assertRaisesRegex(ModelPackageError, "recorded SHA-256"):
                validate_model_package(self.config)

    def test_schema_three_requires_engine_execution_check(self) -> None:
        identity = {
            "tensorrt_version": "10.16.2",
            "cuda_version": "13.2",
            "gpu_name": "Orin",
            "compute_capability": [8, 7],
            "device_model": "Jetson Orin Nano Super",
            "jetson_linux_release": "# R39, REVISION: 2.1",
        }
        self.upgrade_to_schema_three(identity)
        report = json.loads(self.report.read_text())
        report["checks"]["engine_execution"] = False
        self.report.write_text(json.dumps(report))
        # Keep the report hash internally consistent to prove the semantic gate
        # rejects a signed report whose execution check itself did not pass.
        manifest_path = self.package / "model.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest["artifacts"]["import_report"]["sha256"] = hashlib.sha256(
            self.report.read_bytes()
        ).hexdigest()
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

        with self.assertRaisesRegex(ModelPackageError, "engine execution"):
            validate_model_package(self.config)

    def test_strict_cli_rejects_migration_only_schema_two_package(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "model_package",
                "--config",
                str(self.config),
                "--require-engine-identity",
            ],
        ):
            self.assertEqual(model_package.main(), 2)

    def test_default_model_is_not_selected_by_directory_order(self) -> None:
        self.assertEqual(squeakview_config.DEFAULT_MODEL_NAME, "")
        self.assertIsNone(squeakview_config.DEFAULT_INFER_CONFIG)


if __name__ == "__main__":
    unittest.main()
