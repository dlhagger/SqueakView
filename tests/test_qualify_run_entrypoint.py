from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import qualify_run as entrypoint
from squeakview.project import AppPaths, create_project


class QualifyRunEntrypointTests(unittest.TestCase):
    def _project(self, root: Path):
        app_root = root / "app"
        limits = app_root / "resources/project_template/qualification/limits.v1.yaml"
        task = app_root / "resources/project_template/tasks/default.yaml"
        limits.parent.mkdir(parents=True)
        task.parent.mkdir(parents=True)
        limits.write_text("schema_version: '1.0'\n", encoding="utf-8")
        task.write_text("task_name: Test\n", encoding="utf-8")
        projects = root / "projects"
        projects.mkdir()
        return create_project(
            projects / "project",
            name="Project",
            app=AppPaths.from_root(app_root),
        )

    def test_defaults_are_resolved_inside_explicit_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = self._project(Path(temp_dir))
            run_dir = project.paths.runs / "run"
            run_dir.mkdir()
            summary = {"result": "passed"}
            with mock.patch.object(entrypoint, "qualify_run", return_value=summary) as qualify:
                result = entrypoint.main(
                    [str(run_dir), "--project", str(project.paths.root)]
                )

            self.assertEqual(result, 0)
            qualify.assert_called_once_with(
                run_dir.resolve(),
                limits_path=project.paths.qualification / "limits.v1.yaml",
                output_path=None,
                allow_debug_profile=False,
            )

    def test_run_outside_project_is_rejected_before_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = self._project(root)
            outside = root / "outside-run"
            outside.mkdir()
            with mock.patch.object(entrypoint, "qualify_run") as qualify:
                result = entrypoint.main(
                    [str(outside), "--project", str(project.paths.root)]
                )

            self.assertEqual(result, 2)
            qualify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
