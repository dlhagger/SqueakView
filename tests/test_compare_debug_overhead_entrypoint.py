from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts.compare_debug_overhead import main
from squeakview.project import AppPaths, create_project


class CompareDebugOverheadEntrypointTests(unittest.TestCase):
    def test_project_is_required(self) -> None:
        output = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), redirect_stdout(output):
            result = main(["/missing/baseline", "/missing/debug"])

        self.assertEqual(result, 2)
        self.assertIn("SQUEAKVIEW_PROJECT is required", output.getvalue())

    def test_runs_and_output_must_be_owned_by_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            task = app_root / "resources/project_template/tasks/default.yaml"
            task.parent.mkdir(parents=True)
            task.write_text("task_name: Test\n", encoding="utf-8")
            projects = root / "projects"
            projects.mkdir()
            project = create_project(
                projects / "project",
                name="Project",
                app=AppPaths.from_root(app_root),
            )
            outside_run = root / "outside-run"
            outside_run.mkdir()
            valid_run = project.paths.runs / "valid"
            valid_run.mkdir()
            outside_output = root / "outside-report.json"

            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    [
                        str(outside_run),
                        str(valid_run),
                        "--project",
                        str(project.paths.root),
                        "--output",
                        str(outside_output),
                    ]
                )

            self.assertEqual(result, 2)
            self.assertIn("escapes the project root", output.getvalue())
            self.assertFalse(outside_output.exists())


if __name__ == "__main__":
    unittest.main()
