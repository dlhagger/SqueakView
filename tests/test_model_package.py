from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from squeakview import config as squeakview_config
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

    def test_complete_package_returns_identity_and_hashes(self) -> None:
        info = validate_model_package(self.config)

        self.assertEqual(info.name, "test_pose_fp16")
        self.assertEqual(info.root, self.package)
        self.assertEqual(info.import_report, self.report)
        self.assertEqual(len(info.config_sha256), 64)
        self.assertEqual(len(info.engine_sha256), 64)
        self.assertEqual(info.manifest_snapshot()["engine"], str(self.package / "engines" / "model.engine"))

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

    def test_default_model_is_not_selected_by_directory_order(self) -> None:
        self.assertEqual(squeakview_config.DEFAULT_MODEL_NAME, "")
        self.assertIsNone(squeakview_config.DEFAULT_INFER_CONFIG)


if __name__ == "__main__":
    unittest.main()
