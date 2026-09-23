from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from squeakview.apps.project_launcher import ProjectChooser, ProjectSetupDialog
from squeakview.project import AppPaths, ProjectCatalog, UserPaths, create_project


class _FakeSignal:
    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)


class _FakeProcess:
    ProcessChannelMode = QtCore.QProcess.ProcessChannelMode
    UnixProcessFlag = QtCore.QProcess.UnixProcessFlag
    UnixProcessParameters = QtCore.QProcess.UnixProcessParameters

    def __init__(self, _parent=None) -> None:
        self.readyReadStandardOutput = _FakeSignal()
        self.readyReadStandardError = _FakeSignal()
        self.finished = _FakeSignal()
        self.errorOccurred = _FakeSignal()
        self.program = ""
        self.arguments = []
        self.working_directory = ""
        self.started = 0

    def setProgram(self, value: str) -> None:
        self.program = value

    def setArguments(self, value) -> None:
        self.arguments = list(value)

    def setWorkingDirectory(self, value: str) -> None:
        self.working_directory = value

    def setProcessChannelMode(self, _value) -> None:
        pass

    def setUnixProcessParameters(self, value) -> None:
        self.unix_parameters = value

    def start(self) -> None:
        self.started += 1

    def readAllStandardOutput(self):
        return b""

    def readAllStandardError(self):
        return b""


class ProjectChooserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_most_recent_valid_project_is_preselected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            task = app_root / "resources/project_template/tasks/default.yaml"
            task.parent.mkdir(parents=True)
            task.write_text("task_name: Test\n", encoding="utf-8")
            projects = root / "projects"
            projects.mkdir()
            app_paths = AppPaths.from_root(app_root)
            first = create_project(projects / "first", name="First", app=app_paths)
            second = create_project(projects / "second", name="Second", app=app_paths)
            user = UserPaths(
                config=root / "user/config",
                state=root / "user/state",
                runtime=root / "user/runtime",
                projects_parent=projects,
            )
            catalog = ProjectCatalog(app=app_paths, user=user)
            catalog.remember(first)
            catalog.remember(second)

            chooser = ProjectChooser(catalog)
            self.addCleanup(chooser.deleteLater)
            current = chooser.projects.currentItem()

            self.assertIsNotNone(current)
            assert current is not None
            self.assertEqual(
                Path(str(current.data(QtCore.Qt.ItemDataRole.UserRole))),
                second.paths.root,
            )
            self.assertTrue(chooser.open_button.isEnabled())

    def test_create_action_publishes_and_selects_project_in_chosen_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            task = app_root / "resources/project_template/tasks/default.yaml"
            task.parent.mkdir(parents=True)
            task.write_text("task_name: Test\n", encoding="utf-8")
            projects = root / "SqueakView Projects"
            user = UserPaths(
                config=root / "user/config",
                state=root / "user/state",
                runtime=root / "user/runtime",
                projects_parent=projects,
            )
            catalog = ProjectCatalog(app=AppPaths.from_root(app_root), user=user)
            catalog.ensure_projects_parent()
            chooser = ProjectChooser(catalog)
            self.addCleanup(chooser.deleteLater)

            with (
                mock.patch.object(
                    QtWidgets.QInputDialog,
                    "getText",
                    return_value=("Mouse House", True),
                ),
                mock.patch.object(
                    QtWidgets.QFileDialog,
                    "getExistingDirectory",
                    return_value=str(projects),
                ),
            ):
                chooser._create()

            self.assertIsNotNone(chooser.selected_project)
            assert chooser.selected_project is not None
            self.assertEqual(
                chooser.selected_project.paths.root,
                projects / "Mouse_House",
            )
            self.assertTrue(
                (chooser.selected_project.paths.tasks / "default.yaml").is_file()
            )
            self.assertEqual(
                chooser.result(),
                QtWidgets.QDialog.DialogCode.Accepted,
            )

    def test_project_setup_preselects_mousehouse_and_explicitly_allows_no_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            source = app_root / "resources/project_template/model_sources/mousehouse_v2"
            source.mkdir(parents=True)
            (source / "mousehouse_v2.pt").write_bytes(b"checkpoint")
            (source / "mousehouse_v2.yaml").write_text("names: [mouse]\n", encoding="utf-8")
            task = app_root / "resources/project_template/tasks/default.yaml"
            task.parent.mkdir(parents=True)
            task.write_text("task_name: Test\n", encoding="utf-8")
            projects = root / "projects"
            projects.mkdir()
            app_paths = AppPaths.from_root(app_root)
            project = create_project(projects / "project", name="Project", app=app_paths)
            user = UserPaths(
                config=root / "user/config",
                state=root / "user/state",
                runtime=root / "user/runtime",
                projects_parent=projects,
            )
            dialog = ProjectSetupDialog(ProjectCatalog(app=app_paths, user=user), project)
            self.addCleanup(dialog.deleteLater)

            self.assertEqual(dialog.source_combo.currentData(), "mousehouse_v2")
            self.assertEqual(dialog.package_name.text(), "mousehouse_v2")
            self.assertEqual(dialog.continue_button.text(), "Continue Without Inference")
            self.assertTrue(dialog.build_button.isEnabled())

    def test_project_setup_can_select_published_package_as_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            task = app_root / "resources/project_template/tasks/default.yaml"
            task.parent.mkdir(parents=True)
            task.write_text("task_name: Test\n", encoding="utf-8")
            projects = root / "projects"
            projects.mkdir()
            app_paths = AppPaths.from_root(app_root)
            project = create_project(projects / "project", name="Project", app=app_paths)
            package = project.paths.models / "mousehouse"
            (package / "configs").mkdir(parents=True)
            (package / "configs/mousehouse.txt").write_text("[property]\n", encoding="utf-8")
            (package / "model.yaml").write_text("schema_version: 3\n", encoding="utf-8")
            user = UserPaths(
                config=root / "user/config",
                state=root / "user/state",
                runtime=root / "user/runtime",
                projects_parent=projects,
            )
            dialog = ProjectSetupDialog(ProjectCatalog(app=app_paths, user=user), project)
            self.addCleanup(dialog.deleteLater)

            dialog._set_selected_default()

            self.assertEqual(dialog.project.metadata.default_model, "mousehouse")
            self.assertEqual(dialog.continue_button.text(), "Continue to SqueakView")

    def test_project_setup_starts_one_explicit_isolated_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            source = app_root / "resources/project_template/model_sources/mousehouse_v2"
            source.mkdir(parents=True)
            (source / "mousehouse_v2.pt").write_bytes(b"checkpoint")
            (source / "mousehouse_v2.yaml").write_text("names: [mouse]\n", encoding="utf-8")
            task = app_root / "resources/project_template/tasks/default.yaml"
            task.parent.mkdir(parents=True)
            task.write_text("task_name: Test\n", encoding="utf-8")
            projects = root / "projects"
            projects.mkdir()
            app_paths = AppPaths.from_root(app_root)
            project = create_project(projects / "project", name="Project", app=app_paths)
            user = UserPaths(
                config=root / "user/config",
                state=root / "user/state",
                runtime=root / "user/runtime",
                projects_parent=projects,
            )
            dialog = ProjectSetupDialog(ProjectCatalog(app=app_paths, user=user), project)
            self.addCleanup(dialog.deleteLater)

            with mock.patch(
                "squeakview.apps.project_launcher.QtCore.QProcess",
                _FakeProcess,
            ):
                dialog._start_build()
                process = dialog._process
                dialog._start_build()

            self.assertIsInstance(process, _FakeProcess)
            assert isinstance(process, _FakeProcess)
            self.assertEqual(process.program, sys.executable)
            self.assertEqual(
                process.arguments,
                [
                    "-m",
                    "squeakview.apps.model_builder",
                    "--project",
                    str(project.paths.root),
                    "--source",
                    "mousehouse_v2",
                    "--model-name",
                    "mousehouse_v2",
                ],
            )
            self.assertEqual(process.started, 1)
            self.assertTrue(
                process.unix_parameters.flags
                & QtCore.QProcess.UnixProcessFlag.CreateNewSession
            )
            dialog._process = None
            dialog._close_build_log()

    def test_project_setup_reports_export_worker_crash_as_memory_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            task = app_root / "resources/project_template/tasks/default.yaml"
            task.parent.mkdir(parents=True)
            task.write_text("task_name: Test\n", encoding="utf-8")
            projects = root / "projects"
            projects.mkdir()
            app_paths = AppPaths.from_root(app_root)
            project = create_project(projects / "project", name="Project", app=app_paths)
            user = UserPaths(
                config=root / "user/config",
                state=root / "user/state",
                runtime=root / "user/runtime",
                projects_parent=projects,
            )
            dialog = ProjectSetupDialog(ProjectCatalog(app=app_paths, user=user), project)
            self.addCleanup(dialog.deleteLater)
            dialog._process = _FakeProcess(dialog)
            dialog._build_stage = "exporting"
            dialog._open_build_log()

            with mock.patch.object(QtCore.QTimer, "singleShot"):
                dialog._build_finished(9, QtCore.QProcess.ExitStatus.CrashExit)

            self.assertIn("memory pressure", dialog.status_label.text())
            self.assertIn("[FAILED]", dialog.log.toPlainText())
            dialog._process = None
            dialog._close_build_log()


if __name__ == "__main__":
    unittest.main()
