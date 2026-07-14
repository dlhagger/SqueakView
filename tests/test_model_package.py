from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
            self.root / "native",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        (self.package / "model.yaml").write_text("name: test_pose_fp16\n")
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
        )
        (self.config_dir / "test_pose_fp16.pose.json").write_text(
            '{"keypoint_labels_path": "../labels/labels.txt"}\n'
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_complete_package_returns_identity_and_hashes(self) -> None:
        info = validate_model_package(self.config)

        self.assertEqual(info.name, "test_pose_fp16")
        self.assertEqual(info.root, self.package)
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

    def test_default_model_is_not_selected_by_directory_order(self) -> None:
        self.assertEqual(squeakview_config.DEFAULT_MODEL_NAME, "")
        self.assertIsNone(squeakview_config.DEFAULT_INFER_CONFIG)


if __name__ == "__main__":
    unittest.main()
