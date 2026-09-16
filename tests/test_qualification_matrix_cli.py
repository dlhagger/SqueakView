from __future__ import annotations

import json
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import yaml

from scripts.qualify_matrix import (
    _next_case_worksheet,
    _record_assignment,
    _write_new_assignment_checklist,
    main,
)
from squeakview.project import AppPaths, create_project
from squeakview.common.diagnostics.qualification_matrix import (
    expand_cases,
    load_assignments,
    load_matrix,
)


class QualificationMatrixCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        app_root = self.root / "app"
        task = app_root / "resources/project_template/tasks/default.yaml"
        task.parent.mkdir(parents=True)
        task.write_text("task_name: Test\n", encoding="utf-8")
        projects = self.root / "projects"
        projects.mkdir()
        self.project = create_project(
            projects / "qualification",
            name="Qualification",
            app=AppPaths.from_root(app_root),
        )
        self.matrix_path = self.project.paths.qualification / "matrix.yaml"
        self.matrix_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "1.0",
                    "matrix_id": "workflow-test",
                    "capture_profiles": [
                        {
                            "id": "capture",
                            "width": 10,
                            "height": 20,
                            "fps": 30,
                            "camera_count": 1,
                            "pixel_format": "Mono8",
                            "trigger_on": False,
                            "trigger_activation": "rising",
                            "arduino_fps": 30,
                            "exposure_us": 10000.0,
                            "bitrate_kbps": 4000,
                            "serial_enabled": False,
                            "serial_port": "/dev/ttyACM0",
                            "serial_baud": 115200,
                        }
                    ],
                    "durations": [{"id": "short", "minimum_seconds": 5}],
                    "inference_enabled": [False],
                    "preview_enabled": [False, True],
                    "power_modes": ["25W"],
                }
            )
        )
        self.matrix = load_matrix(self.matrix_path)
        self.case_ids = [case["case_id"] for case in expand_cases(self.matrix)]
        self.case_id = self.case_ids[0]
        self.assignments_path = self.project.paths.qualification / "assignments.yaml"
        (self.project.paths.qualification / "limits.v1.yaml").write_text(
            "schema_version: '1.0'\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_initializes_complete_checklist_without_overwriting(self) -> None:
        _write_new_assignment_checklist(self.assignments_path, self.matrix)

        self.assertEqual(
            load_assignments(self.assignments_path),
            {case_id: None for case_id in self.case_ids},
        )
        with self.assertRaises(FileExistsError):
            _write_new_assignment_checklist(self.assignments_path, self.matrix)
        self.assertEqual(
            load_assignments(self.assignments_path),
            {case_id: None for case_id in self.case_ids},
        )

    def test_records_completed_run_and_rejects_reuse(self) -> None:
        _write_new_assignment_checklist(self.assignments_path, self.matrix)
        run_dir = self.root / "run"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text(json.dumps({}))
        (run_dir / "run_status.json").write_text(
            json.dumps({"state": "finalized"})
        )

        _record_assignment(
            self.assignments_path, self.matrix, self.case_id, run_dir
        )

        self.assertEqual(
            load_assignments(self.assignments_path)[self.case_id],
            str(run_dir.resolve()),
        )
        with self.assertRaisesRegex(ValueError, "already assigned"):
            _record_assignment(
                self.assignments_path, self.matrix, self.case_ids[1], run_dir
            )

    def test_rejects_unknown_case_and_incomplete_run(self) -> None:
        _write_new_assignment_checklist(self.assignments_path, self.matrix)
        run_dir = self.root / "run"
        run_dir.mkdir()

        with self.assertRaisesRegex(ValueError, "unknown qualification case"):
            _record_assignment(
                self.assignments_path, self.matrix, "typo", run_dir
            )
        with self.assertRaisesRegex(ValueError, "missing run_manifest.json"):
            _record_assignment(
                self.assignments_path, self.matrix, self.case_id, run_dir
            )

        (run_dir / "run_manifest.json").write_text(json.dumps({}))
        (run_dir / "run_status.json").write_text(
            json.dumps({"state": "recording"})
        )
        with self.assertRaisesRegex(ValueError, "not terminal"):
            _record_assignment(
                self.assignments_path, self.matrix, self.case_id, run_dir
            )

    def test_rejects_duplicate_or_non_regular_run_metadata(self) -> None:
        _write_new_assignment_checklist(self.assignments_path, self.matrix)
        run_dir = self.root / "run"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text('{"schema": 1, "schema": 2}')
        (run_dir / "run_status.json").write_text(
            json.dumps({"state": "finalized"})
        )

        with self.assertRaisesRegex(ValueError, "duplicate field"):
            _record_assignment(
                self.assignments_path, self.matrix, self.case_id, run_dir
            )

        (run_dir / "run_manifest.json").unlink()
        (run_dir / "run_manifest.json").symlink_to("/dev/null")
        with self.assertRaisesRegex(ValueError, "regular file"):
            _record_assignment(
                self.assignments_path, self.matrix, self.case_id, run_dir
            )

    def test_next_case_worksheet_is_read_only_and_reports_all_factors(self) -> None:
        _write_new_assignment_checklist(self.assignments_path, self.matrix)
        before = self.assignments_path.read_bytes()
        assignments = load_assignments(self.assignments_path)
        assignments[self.case_ids[0]] = "/completed/run"

        worksheet = _next_case_worksheet(
            self.matrix,
            assignments,
            matrix_path=self.matrix_path,
        )

        self.assertEqual(self.assignments_path.read_bytes(), before)
        self.assertEqual(worksheet["progress"], {"assigned": 1, "remaining": 1, "total": 2})
        self.assertEqual(worksheet["next_case"]["case_id"], self.case_ids[1])
        self.assertEqual(worksheet["next_case"]["capture"]["pixel_format"], "Mono8")
        self.assertEqual(worksheet["required_minimum_seconds"], 5)
        self.assertEqual(
            worksheet["required_environment"]["SQUEAKVIEW_DISABLE_PREVIEW"], "0"
        )
        self.assertEqual(
            worksheet["required_environment"][
                "SQUEAKVIEW_VIDEO_VALIDATION_FULL_DECODE"
            ],
            "1",
        )
        self.assertIn("SQUEAKVIEW_QUALIFICATION_CASE_ID=", worksheet["launch_command"])
        self.assertIn("squeakview.sh", worksheet["launch_command"])

    def test_next_case_worksheet_reports_completed_campaign(self) -> None:
        worksheet = _next_case_worksheet(
            self.matrix,
            {case_id: f"/run/{index}" for index, case_id in enumerate(self.case_ids)},
            matrix_path=self.matrix_path,
        )

        self.assertEqual(worksheet["progress"], {"assigned": 2, "remaining": 0, "total": 2})
        self.assertIsNone(worksheet["next_case"])
        self.assertIsNone(worksheet["launch_command"])

    def test_next_case_cli_reads_checklist_without_modifying_it(self) -> None:
        _write_new_assignment_checklist(self.assignments_path, self.matrix)
        before = self.assignments_path.read_bytes()
        output = io.StringIO()

        with (
            mock.patch(
                "sys.argv",
                [
                    "qualify_matrix.py",
                    str(self.assignments_path),
                    "--matrix",
                    str(self.matrix_path),
                    "--project",
                    str(self.project.paths.root),
                    "--next-case",
                ],
            ),
            redirect_stdout(output),
        ):
            result = main()

        self.assertEqual(result, 0)
        self.assertEqual(self.assignments_path.read_bytes(), before)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["next_case"]["case_id"], self.case_ids[0])

    def test_cli_requires_project_and_rejects_state_outside_it(self) -> None:
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("sys.argv", ["qualify_matrix.py", "--list-cases"]),
            redirect_stdout(output),
        ):
            self.assertEqual(main(), 2)
        self.assertIn("SQUEAKVIEW_PROJECT is required", output.getvalue())

        outside = self.root / "outside-assignments.yaml"
        _write_new_assignment_checklist(outside, self.matrix)
        before = outside.read_bytes()
        output = io.StringIO()
        with (
            mock.patch(
                "sys.argv",
                [
                    "qualify_matrix.py",
                    str(outside),
                    "--project",
                    str(self.project.paths.root),
                    "--matrix",
                    str(self.matrix_path),
                    "--next-case",
                ],
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(main(), 2)
        self.assertIn("escapes the project root", output.getvalue())
        self.assertEqual(outside.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
