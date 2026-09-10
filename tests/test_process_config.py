from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
        self.sidecar.write_text(json.dumps({
            "schema_version": 2,
            "postprocess": "pyservicemaker_yolo26_pose_v1",
            "keypoint_labels_path": "../labels/labels.txt",
            "keypoint_threshold": 0.5,
            "classes": [],
        }) + "\n")
        self.run_dir = self.root / "run"
        self.logs: list[str] = []

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def validated_artifacts(self, **updates: Path) -> dict[str, Path]:
        values = {
            "deepstream_config": self.config,
            "pose_sidecar": self.sidecar,
            "class_labels": self.package / "labels/classes.txt",
            "keypoint_labels": self.package / "labels/labels.txt",
            "onnx": self.package / "onnx/model.onnx",
            "engine": self.package / "engines/model.engine",
            "custom_parser": self.parser,
        }
        values.update(updates)
        return values

    def test_localization_resolves_owned_paths_and_sidecar(self) -> None:
        localized = process._localize_deepstream_config(self.config, self.run_dir, self.logs.append)

        text = localized.read_text()
        self.assertIn("# preserve this comment", text)
        self.assertIn(f"onnx-file={(self.package / 'onnx/model.onnx').resolve()}", text)
        self.assertIn(f"model-engine-file={(self.package / 'engines/model.engine').resolve()}", text)
        self.assertIn(f"labelfile-path={(localized.parent / 'model.classes.txt').resolve()}", text)
        self.assertIn("batch-size=1", text)
        sidecar = json.loads((localized.parent / "model.pose.json").read_text())
        self.assertEqual(
            sidecar["keypoint_labels_path"],
            str((localized.parent / "model.keypoints.txt").resolve()),
        )
        self.assertEqual(
            (localized.parent / "model.classes.txt").read_text(),
            (self.package / "labels/classes.txt").read_text(),
        )
        self.assertTrue(any("localized DeepStream config" in line for line in self.logs))

    def test_missing_run_directory_returns_original_config(self) -> None:
        self.assertEqual(process._localize_deepstream_config(self.config, None, self.logs.append), self.config.resolve())
        self.assertFalse(self.logs)

    def test_invalid_sidecar_fails_closed_before_localized_config_write(self) -> None:
        self.sidecar.write_text("not json")

        with self.assertRaisesRegex(RuntimeError, "safely localize pose sidecar"):
            process._localize_deepstream_config(
                self.config, self.run_dir, self.logs.append
            )

        self.assertFalse((self.run_dir / "config" / self.config.name).exists())

    def test_prepared_runtime_binds_localized_config_and_package_parser(self) -> None:
        prepared = process.prepare_effective_deepstream_config(
            self.config,
            self.run_dir,
            self.logs.append,
            validated_artifacts=self.validated_artifacts(),
        )
        cfg = process.LaunchConfig(ds_cfg=prepared.path, run_dir=self.run_dir)

        with mock.patch.object(process, "_spawn", return_value=object()) as spawn:
            process.spawn_inference(
                cfg,
                self.logs.append,
                effective_config=prepared,
            )

        args = spawn.call_args.args[1]
        self.assertEqual(args[args.index("--cfg") + 1], str(prepared.path))
        self.assertEqual(
            prepared.parser_identity["path"], str(self.parser.resolve())
        )
        self.assertEqual(
            prepared.config_identity["path"], str(prepared.path.resolve())
        )
        self.assertEqual(
            set(prepared.manifest_snapshot()),
            {
                "deepstream_config",
                "pose_sidecar",
                "class_labels",
                "keypoint_labels",
                "onnx",
                "engine",
                "custom_parser",
            },
        )
        self.assertEqual(
            prepared.artifacts["class_labels"]["path"],
            str((prepared.path.parent / "model.classes.txt").resolve()),
        )
        self.assertEqual(
            prepared.artifacts["keypoint_labels"]["path"],
            str((prepared.path.parent / "model.keypoints.txt").resolve()),
        )

    def test_tampered_localized_config_is_rejected_before_spawn(self) -> None:
        prepared = process.prepare_effective_deepstream_config(
            self.config,
            self.run_dir,
            self.logs.append,
            validated_artifacts=self.validated_artifacts(),
        )
        prepared.path.write_text(prepared.path.read_text() + "# tampered\n")
        cfg = process.LaunchConfig(ds_cfg=prepared.path, run_dir=self.run_dir)

        with (
            mock.patch.object(process, "_spawn") as spawn,
            self.assertRaisesRegex(RuntimeError, "deepstream_config"),
        ):
            process.spawn_inference(
                cfg,
                self.logs.append,
                effective_config=prepared,
            )
        spawn.assert_not_called()

    def test_tampered_package_parser_is_rejected_before_spawn(self) -> None:
        prepared = process.prepare_effective_deepstream_config(
            self.config,
            self.run_dir,
            self.logs.append,
            validated_artifacts=self.validated_artifacts(),
        )
        self.parser.write_text("replacement parser")
        cfg = process.LaunchConfig(ds_cfg=prepared.path, run_dir=self.run_dir)

        with (
            mock.patch.object(process, "_spawn") as spawn,
            self.assertRaisesRegex(RuntimeError, "custom_parser"),
        ):
            process.spawn_inference(
                cfg,
                self.logs.append,
                effective_config=prepared,
            )
        spawn.assert_not_called()

    def test_parser_outside_validated_package_selection_is_rejected(self) -> None:
        other_parser = self.root / "other" / "parser.so"
        other_parser.parent.mkdir()
        other_parser.write_text("different parser")

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            process.prepare_effective_deepstream_config(
                self.config,
                self.run_dir,
                self.logs.append,
                validated_artifacts=self.validated_artifacts(
                    custom_parser=other_parser
                ),
            )

    def test_every_effective_artifact_is_bound_against_tampering(self) -> None:
        for name in self.validated_artifacts():
            with self.subTest(name=name):
                run_dir = self.root / f"run-{name}"
                prepared = process.prepare_effective_deepstream_config(
                    self.config,
                    run_dir,
                    self.logs.append,
                    validated_artifacts=self.validated_artifacts(),
                )
                target = Path(str(prepared.artifacts[name]["path"]))
                original = target.read_bytes()
                target.write_bytes(original + b"tamper")
                try:
                    with self.assertRaisesRegex(RuntimeError, name):
                        process.verify_effective_deepstream_config(prepared)
                finally:
                    target.write_bytes(original)

    def test_every_validated_source_path_must_match_package_selection(self) -> None:
        for name in self.validated_artifacts():
            with self.subTest(name=name):
                wrong = self.root / "wrong" / f"{name}.bin"
                wrong.parent.mkdir(exist_ok=True)
                wrong.write_bytes(b"wrong")
                with self.assertRaises(RuntimeError):
                    process.prepare_effective_deepstream_config(
                        self.config,
                        self.root / f"divergence-{name}",
                        self.logs.append,
                        validated_artifacts=self.validated_artifacts(**{name: wrong}),
                    )


if __name__ == "__main__":
    unittest.main()
