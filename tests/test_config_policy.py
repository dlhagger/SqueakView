from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview import model_package
from squeakview.apps.operator.gui.config_policy import ConfigFields, collect_config


def _fields(root: Path, **updates: object) -> ConfigFields:
    values: dict[str, object] = {
        "width": "1440",
        "height": "1080",
        "fps": "30",
        "bitrate": "4000",
        "arduino_fps": "30",
        "serial_baud": "115200",
        "exposure_us": "10000",
        "pixel_format": "Mono8",
        "capture_backend": "flir_direct",
        "trigger_enabled": True,
        "serial_enabled": True,
        "allow_rtc_correction": False,
        "serial_port": "/dev/ttyACM0",
        "inference_enabled": True,
        "ds_cfg": str(root / "model.txt"),
        "task_cfg": str(root / "task.yaml"),
        "camera_count": 1,
        "mouse_id": "mouse_1",
        "experiment_mode": "existing",
        "experiment_name": "study_a",
    }
    values.update(updates)
    return ConfigFields(**values)  # type: ignore[arg-type]


class ConfigPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "model.txt").write_text("model")
        (self.root / "task.yaml").write_text("task")
        self.resolve = lambda value: Path(value) if value else None
        self.validate = mock.Mock()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_collects_typed_config_and_validates_model(self) -> None:
        result = collect_config(
            _fields(self.root),
            include_mode=True,
            resolve_path=self.resolve,
            validate_model=self.validate,
        )

        self.assertIsNone(result.error)
        self.assertEqual(result.config["width"], 1440)
        self.assertEqual(result.config["experiment_name"], "study_a")
        self.assertEqual(result.config["mouse_id"], "mouse_1")
        self.assertEqual(result.config["ds_cfg"], self.root / "model.txt")
        self.validate.assert_called_once_with(self.root / "model.txt")

    def test_inference_disabled_skips_model_requirement_and_validation(self) -> None:
        result = collect_config(
            _fields(self.root, inference_enabled=False, ds_cfg=""),
            include_mode=False,
            resolve_path=self.resolve,
            validate_model=self.validate,
        )

        self.assertIsNone(result.error)
        self.assertIsNone(result.config["ds_cfg"])
        self.assertNotIn("experiment_mode", result.config)
        self.validate.assert_not_called()

    def test_rejects_numeric_and_session_identity_errors_with_original_copy(self) -> None:
        numeric = collect_config(
            _fields(self.root, width="not-a-number"),
            include_mode=True,
            resolve_path=self.resolve,
            validate_model=self.validate,
        )
        self.assertEqual(numeric.error.title, "Invalid input")

        session = collect_config(
            _fields(self.root, experiment_name=""),
            include_mode=True,
            resolve_path=self.resolve,
            validate_model=self.validate,
        )
        self.assertEqual(session.error.title, "Experiment required")

    def test_rejects_missing_paths_and_invalid_model_package(self) -> None:
        missing_model = collect_config(
            _fields(self.root, ds_cfg=str(self.root / "missing.txt")),
            include_mode=True,
            resolve_path=self.resolve,
            validate_model=self.validate,
        )
        self.assertEqual(missing_model.error.title, "Config missing")
        self.assertIn("missing.txt", missing_model.error.message)

        self.validate.side_effect = model_package.ModelPackageError("schema mismatch")
        invalid_model = collect_config(
            _fields(self.root),
            include_mode=True,
            resolve_path=self.resolve,
            validate_model=self.validate,
        )
        self.assertEqual(invalid_model.error.title, "Invalid model package")
        self.assertEqual(invalid_model.error.message, "schema mismatch")

        missing_task = collect_config(
            _fields(self.root, inference_enabled=False, task_cfg=""),
            include_mode=True,
            resolve_path=self.resolve,
            validate_model=mock.Mock(),
        )
        self.assertEqual(missing_task.error.title, "Task config required")


if __name__ == "__main__":
    unittest.main()
