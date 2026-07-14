from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from squeakview.apps.operator.backend import process


class DeepStreamConfigLocalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.package = self.root / "model"
        self.config_dir = self.package / "configs"
        self.config_dir.mkdir(parents=True)
        for relative in ("onnx/model.onnx", "engines/model.engine", "labels/classes.txt", "labels/labels.txt"):
            path = self.package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative)
        self.parser = self.root / "native" / "parser.so"
        self.parser.parent.mkdir()
        self.parser.write_text("parser")
        self.config = self.config_dir / "model.txt"
        self.config.write_text(
            "[property]\n"
            "# preserve this comment\n"
            "onnx-file=../onnx/model.onnx\n"
            "model-engine-file=../engines/model.engine\n"
            "labelfile-path=../labels/classes.txt\n"
            "custom-lib-path=../../native/parser.so\n"
            "batch-size=1\n"
        )
        self.sidecar = self.config_dir / "model.pose.json"
        self.sidecar.write_text('{"keypoint_labels_path": "../labels/labels.txt", "draw_threshold": 0.5}\n')
        self.run_dir = self.root / "run"
        self.logs: list[str] = []

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_localization_resolves_owned_paths_and_sidecar(self) -> None:
        localized = process._localize_deepstream_config(self.config, self.run_dir, self.logs.append)

        text = localized.read_text()
        self.assertIn("# preserve this comment", text)
        self.assertIn(f"onnx-file={(self.package / 'onnx/model.onnx').resolve()}", text)
        self.assertIn(f"model-engine-file={(self.package / 'engines/model.engine').resolve()}", text)
        self.assertIn("batch-size=1", text)
        sidecar = json.loads((localized.parent / "model.pose.json").read_text())
        self.assertEqual(sidecar["keypoint_labels_path"], str((self.package / "labels/labels.txt").resolve()))
        self.assertTrue(any("localized DeepStream config" in line for line in self.logs))

    def test_missing_run_directory_returns_original_config(self) -> None:
        self.assertEqual(process._localize_deepstream_config(self.config, None, self.logs.append), self.config.resolve())
        self.assertFalse(self.logs)

    def test_invalid_sidecar_warns_without_losing_localized_config(self) -> None:
        self.sidecar.write_text("not json")

        localized = process._localize_deepstream_config(self.config, self.run_dir, self.logs.append)

        self.assertTrue(localized.exists())
        self.assertTrue(any("could not localize pose sidecar" in line for line in self.logs))


if __name__ == "__main__":
    unittest.main()
