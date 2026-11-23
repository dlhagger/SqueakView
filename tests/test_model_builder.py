from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps import model_builder as model_builder_cli
from squeakview.model_builder import (
    BuildSpec,
    _export_arguments,
    _prepare_export_environment,
    _require_vendor_tensorrt,
    _require_supported_tensorrt,
    _validate_export_metadata,
    BuildResult,
    discover_model_sources,
    import_model_source,
)
from squeakview.project import AppPaths, ProjectSession, create_project


class ModelBuilderTests(unittest.TestCase):
    def _project(self, root: Path):
        app_root = root / "app"
        template = app_root / "resources/project_template/tasks/default.yaml"
        template.parent.mkdir(parents=True)
        template.write_text("task_name: Test\n", encoding="utf-8")
        projects = root / "projects"
        projects.mkdir()
        return create_project(projects / "Project", name="Project", app=AppPaths.from_root(app_root))

    def test_discovers_only_complete_direct_non_symlink_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = self._project(root)
            complete = project.paths.model_sources / "mousehouse_v2"
            complete.mkdir()
            (complete / "weights.pt").write_bytes(b"checkpoint")
            (complete / "data.yaml").write_text("names: [mouse]\n", encoding="utf-8")
            incomplete = project.paths.model_sources / "incomplete"
            incomplete.mkdir()
            (incomplete / "weights.pt").write_bytes(b"checkpoint")
            (project.paths.model_sources / "linked").symlink_to(complete, target_is_directory=True)

            sources = discover_model_sources(project)

            self.assertEqual([source.name for source in sources], ["mousehouse_v2"])
            self.assertEqual(sources[0].checkpoint, complete / "weights.pt")
            self.assertEqual(sources[0].data_yaml, complete / "data.yaml")

    def test_cli_passes_explicit_project_and_emits_machine_readable_result(self) -> None:
        result = BuildResult(
            project_root=Path("/project"),
            package_root=Path("/project/models/model"),
            config=Path("/project/models/model/configs/model.txt"),
            model_name="model",
        )
        with (
            mock.patch.object(model_builder_cli, "_configure_worker_lifetime"),
            mock.patch.object(model_builder_cli, "build_model_package", return_value=result) as build,
            mock.patch("builtins.print") as output,
        ):
            code = model_builder_cli.main(
                [
                    "--project", "/project",
                    "--source", "mousehouse_v2",
                    "--model-name", "model",
                ]
            )

        self.assertEqual(code, 0)
        spec = build.call_args.args[0]
        self.assertEqual(spec.project_root, Path("/project"))
        self.assertEqual(spec.source_name, "mousehouse_v2")
        self.assertTrue(spec.set_as_default)
        self.assertEqual(spec.tensorrt_workspace_gib, 1.0)
        final = output.call_args_list[-1].args[0]
        self.assertTrue(final.startswith(model_builder_cli.EVENT_PREFIX))
        self.assertEqual(json.loads(final[len(model_builder_cli.EVENT_PREFIX):])["stage"], "result")

    def test_build_spec_bounds_tensorrt_workspace_for_jetson(self) -> None:
        spec = BuildSpec(
            project_root=Path("/project"),
            source_name="mousehouse_v2",
            model_name="mousehouse",
        )

        self.assertEqual(spec.tensorrt_workspace_gib, 1.0)

    def test_build_disables_autoinstall_before_ultralytics_import(self) -> None:
        with mock.patch.dict(os.environ, {"YOLO_AUTOINSTALL": "true"}):
            with mock.patch.dict(sys.modules, {"ultralytics.utils": None}):
                _prepare_export_environment()
            self.assertEqual(os.environ["YOLO_AUTOINSTALL"], "false")
            with mock.patch.dict(sys.modules, {"ultralytics.utils": SimpleNamespace(AUTOINSTALL=True)}):
                with self.assertRaisesRegex(RuntimeError, "fresh worker"):
                    _prepare_export_environment()
            with mock.patch.dict(sys.modules, {"ultralytics.utils": SimpleNamespace(AUTOINSTALL=False)}):
                _prepare_export_environment()

    def test_build_rejects_tensor_rt_shadowing_inside_venv(self) -> None:
        with mock.patch.object(sys, "prefix", "/project/.venv"), mock.patch.object(sys, "base_prefix", "/usr"):
            _require_vendor_tensorrt("/usr/lib/python3.12/dist-packages/tensorrt/__init__.py")
            _require_vendor_tensorrt(
                "/project/.venv/vendor/tensorrt-10.16.2.10/usr/lib/python3.12/dist-packages/tensorrt/__init__.py"
            )
            with self.assertRaisesRegex(RuntimeError, "instead of JetPack"):
                _require_vendor_tensorrt("/project/.venv/lib/python3.12/site-packages/tensorrt/__init__.py")
            with self.assertRaisesRegex(RuntimeError, "instead of JetPack"):
                _require_vendor_tensorrt("/project/.venv/vendor/tensorrt-11/usr/lib/tensorrt/__init__.py")

    def test_export_contract_is_explicit_for_both_precisions(self) -> None:
        for precision, quantize in (("fp16", 16), ("fp32", 32)):
            spec = BuildSpec(Path("/project"), "source", "model", precision=precision)
            args = _export_arguments(spec, Path("/project/data.yaml"))
            self.assertIs(args["nms"], False)
            self.assertIs(args["agnostic_nms"], False)
            self.assertIs(args["dynamic"], False)
            self.assertEqual(args["max_det"], 300)
            self.assertEqual(args["quantize"], quantize)

    def test_builder_rejects_incompatible_system_tensorrt_before_export(self) -> None:
        for version in ("10.16.1.7", "10.16.2.10"):
            _require_supported_tensorrt(version)
        for version in ("11.4.1.31", "10.15.0", "unknown"):
            with self.subTest(version=version):
                with self.assertRaisesRegex(RuntimeError, "requires TensorRT 10.16"):
                    _require_supported_tensorrt(version)

    def test_export_metadata_rejects_head_and_shape_or_precision_mismatch(self) -> None:
        import copy
        spec = BuildSpec(Path("/project"), "source", "model")
        valid = {
            "task": "pose", "end2end": True, "batch": 1,
            "imgsz": [640, 640], "kpt_shape": [8, 3],
            "args": {"nms": False, "dynamic": False, "quantize": 16},
        }
        _validate_export_metadata(valid, spec, 8, 3)
        for key, value in (("task", "detect"), ("end2end", False), ("batch", 2),
                           ("imgsz", [320, 320]), ("kpt_shape", [17, 3]), ("args", None)):
            with self.subTest(key=key):
                bad = copy.deepcopy(valid)
                bad[key] = value
                with self.assertRaises(ValueError):
                    _validate_export_metadata(bad, spec, 8, 3)
        for key, value in (("nms", None), ("nms", True), ("dynamic", True), ("quantize", 32)):
            with self.subTest(argument=key, value=value):
                bad = copy.deepcopy(valid)
                bad["args"][key] = value
                with self.assertRaises(ValueError):
                    _validate_export_metadata(bad, spec, 8, 3)

    def test_import_source_is_project_owned_private_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = self._project(root)
            checkpoint = root / "custom.pt"
            data_yaml = root / "custom.yaml"
            checkpoint.write_bytes(b"checkpoint")
            data_yaml.write_text("names: [mouse]\n", encoding="utf-8")

            with ProjectSession.open(project.paths.root) as session:
                imported = import_model_source(
                    session.project,
                    name="custom",
                    checkpoint=checkpoint,
                    data_yaml=data_yaml,
                )

            self.assertEqual(imported.root, project.paths.model_sources / "custom")
            self.assertEqual(imported.checkpoint.read_bytes(), b"checkpoint")
            self.assertEqual(imported.data_yaml.read_text(encoding="utf-8"), "names: [mouse]\n")
            self.assertEqual(imported.root.stat().st_mode & 0o777, 0o700)
            self.assertEqual(imported.checkpoint.stat().st_mode & 0o777, 0o600)
            self.assertEqual(imported.data_yaml.stat().st_mode & 0o777, 0o600)

            with ProjectSession.open(project.paths.root) as session:
                with self.assertRaises(FileExistsError):
                    import_model_source(
                        session.project,
                        name="custom",
                        checkpoint=checkpoint,
                        data_yaml=data_yaml,
                    )
            self.assertEqual(list(project.paths.model_sources.glob(".*.import-*")), [])

            linked = root / "linked.pt"
            linked.symlink_to(checkpoint)
            with ProjectSession.open(project.paths.root) as session:
                with self.assertRaisesRegex(ValueError, "symbolic links"):
                    import_model_source(
                        session.project,
                        name="linked",
                        checkpoint=linked,
                        data_yaml=data_yaml,
                    )

    def test_cli_failure_is_nonzero_and_preserves_useful_message(self) -> None:
        with (
            mock.patch.object(model_builder_cli, "_configure_worker_lifetime"),
            mock.patch.object(
                model_builder_cli,
                "build_model_package",
                side_effect=RuntimeError("CUDA unavailable"),
            ),
            mock.patch("builtins.print") as output,
            mock.patch.object(model_builder_cli.traceback, "print_exc"),
        ):
            code = model_builder_cli.main(
                [
                    "--project", "/project",
                    "--source", "mousehouse_v2",
                    "--model-name", "model",
                ]
            )

        self.assertEqual(code, 2)
        event = output.call_args_list[-1].args[0]
        payload = json.loads(event[len(model_builder_cli.EVENT_PREFIX):])
        self.assertEqual(payload, {"message": "CUDA unavailable", "stage": "failed"})


if __name__ == "__main__":
    unittest.main()
