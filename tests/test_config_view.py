from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview.apps.operator.gui.config_dialog import ConfigDialog
from squeakview.apps.operator.gui.config_view import (
    ConfigViewCallbacks,
    build_config_view,
)
from squeakview.apps.operator.gui.model_catalog import ModelChoice
from squeakview.project import Project, ProjectMetadata, ProjectPaths
from squeakview.common.profiles import ExperimentProfile, SubjectProfile
from squeakview import model_package


class ConfigViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "default.yaml").write_text("plots: []\n")
        self.dialog = QtWidgets.QDialog()
        self.callbacks = ConfigViewCallbacks(*[mock.Mock() for _ in range(7)])

    def tearDown(self) -> None:
        self.dialog.close()
        self.app.processEvents()
        self.temp_dir.cleanup()

    def test_builds_typed_form_with_profiles_defaults_and_bounded_validators(self) -> None:
        view = build_config_view(
            self.dialog,
            config={"experiment_name": "study_a", "width": 2048, "fps": 120},
            experiments=[ExperimentProfile(name="Study A", slug="study_a")],
            subjects=[SubjectProfile(name="Mouse 1", subject_id="mouse_1")],
            callbacks=self.callbacks,
            tasks_dir=self.root,
            show_session_setup=True,
            experiment_profile_name="",
            show_experiment_profile_editor=False,
        )

        self.assertEqual(view.mode_combo.currentData(), "existing")
        self.assertEqual(view.existing_experiment_combo.findData("study_a"), 1)
        self.assertEqual(view.existing_subject_combo.findData("mouse_1"), 1)
        self.assertEqual(view.width_edit.text(), "2048")
        self.assertEqual(view.fps_edit.text(), "120")
        self.assertEqual(view.task_cfg_edit.text(), str(self.root / "default.yaml"))
        self.assertIsInstance(view.width_edit.validator(), QtGui.QIntValidator)
        self.assertEqual(view.width_edit.validator().bottom(), 1)
        self.assertEqual(view.width_edit.validator().top(), 4096)
        self.assertIsNone(view.experiment_profile_name_edit)
        scroll_area = self.dialog.findChild(
            QtWidgets.QScrollArea, "configScrollArea"
        )
        self.assertIsNotNone(scroll_area)
        self.assertTrue(scroll_area.widgetResizable())
        self.assertEqual(
            scroll_area.frameShape(), QtWidgets.QFrame.Shape.NoFrame
        )

    def test_buttons_use_callbacks_and_optional_groups_preserve_visibility(self) -> None:
        view = build_config_view(
            self.dialog,
            config={"inference_enabled": False},
            experiments=[],
            subjects=[],
            callbacks=self.callbacks,
            tasks_dir=self.root,
            show_session_setup=False,
            experiment_profile_name="Study A",
            show_experiment_profile_editor=True,
        )

        self.assertTrue(view.session_group.isHidden())
        self.assertEqual(view.experiment_profile_name_edit.text(), "Study A")
        view.new_experiment_btn.click()
        view.new_subject_btn.click()
        view.add_subject_btn.click()
        view.task_browse_btn.click()
        view.cfg_browse_btn.click()
        view.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
        view.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).click()
        callbacks = (
            self.callbacks.create_experiment,
            self.callbacks.create_subject,
            self.callbacks.add_subject,
            self.callbacks.browse_task_config,
            self.callbacks.browse_model_config,
            self.callbacks.accept,
            self.callbacks.reject,
        )
        for callback in callbacks:
            callback.assert_called_once_with()


class ConfigDialogCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.project_root = self.root / "project"
        for relative in (
            "runs",
            "models",
            "tasks",
            "profiles/experiments",
            "profiles/subjects",
            "qualification",
        ):
            (self.project_root / relative).mkdir(parents=True, exist_ok=True)
        self.project = Project(
            ProjectPaths.from_existing_root(self.project_root),
            ProjectMetadata.create("Test"),
        )
        self.task = self.project.paths.tasks / "task.yaml"
        self.task.write_text("plots: []\n")
        self.store = mock.Mock()
        self.store.list_experiments.return_value = []
        self.store.list_subjects.return_value = []
        self.store_patch = mock.patch(
            "squeakview.apps.operator.gui.config_dialog.ProfileStore",
            return_value=self.store,
        )
        self.store_patch.start()
        self.catalog_patch = mock.patch(
            "squeakview.apps.operator.gui.config_dialog.enumerate_model_configs",
            return_value=(),
        )
        self.catalog_patch.start()

    def tearDown(self) -> None:
        self.catalog_patch.stop()
        self.store_patch.stop()
        self.temp_dir.cleanup()

    def _dialog(self, **updates: object) -> ConfigDialog:
        config: dict[str, object] = {
            "inference_enabled": False,
            "task_cfg": str(self.task),
            "experiment_mode": "sandbox",
        }
        config.update(updates)
        return ConfigDialog(
            config=config,
            project=self.project,
            profile_store=self.store,
        )

    def test_dialog_preserves_widget_api_and_collects_through_policy(self) -> None:
        dialog = self._dialog(width=1920, height=1080)
        try:
            result = dialog._collect_config(show_errors=False)
            self.assertEqual(dialog.width_edit.text(), "1920")
            self.assertEqual(result["width"], 1920)
            self.assertEqual(result["task_cfg"], self.task)
            self.assertEqual(result["experiment_mode"], "sandbox")
            self.assertIsNone(result["ds_cfg"])
        finally:
            dialog.close()

    def test_dialog_selects_valid_catalog_model_and_disables_invalid_choice(self) -> None:
        valid = self.project.paths.models / "mouse" / "configs" / "mouse.txt"
        broken = self.project.paths.models / "broken" / "configs" / "broken.txt"
        choices = (
            ModelChoice("Mouse", valid, True, "Schema 3; device matched"),
            ModelChoice("Broken", broken, False, "engine hash mismatch"),
        )
        with mock.patch(
            "squeakview.apps.operator.gui.config_dialog.enumerate_model_configs",
            return_value=choices,
        ):
            dialog = self._dialog()
        try:
            valid_index = dialog.model_combo.findData(str(valid))
            self.assertGreater(valid_index, 0)
            dialog.model_combo.setCurrentIndex(valid_index)
            self.assertEqual(dialog.cfg_edit.text(), str(valid))

            invalid_index = next(
                index
                for index in range(dialog.model_combo.count())
                if "Broken" in dialog.model_combo.itemText(index)
            )
            self.assertFalse(dialog.model_combo.model().item(invalid_index).isEnabled())
            self.assertIn(
                "hash mismatch",
                dialog.model_combo.itemData(
                    invalid_index, QtCore.Qt.ItemDataRole.ToolTipRole
                ),
            )
            self.assertNotIn(
                "__manual__",
                [
                    dialog.model_combo.itemData(index)
                    for index in range(dialog.model_combo.count())
                ],
            )
            self.assertTrue(dialog.cfg_edit.isHidden())
            self.assertTrue(dialog.cfg_browse_btn.isHidden())
        finally:
            dialog.close()

    def test_dialog_retains_numeric_and_model_error_dialogs(self) -> None:
        dialog = self._dialog()
        try:
            dialog.width_edit.setText("invalid")
            with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
                self.assertIsNone(dialog._collect_config(show_errors=True))
            warning.assert_called_once_with(
                dialog,
                "Invalid input",
                "Please enter valid numeric values for size, FPS, bitrate, and baud.",
            )

            model_cfg = self.project.paths.models / "model.txt"
            model_cfg.write_text("model")
            dialog.width_edit.setText("1440")
            dialog.inference_enable.setChecked(True)
            dialog.cfg_edit.setText(str(model_cfg))
            with (
                mock.patch.object(
                    model_package,
                    "validate_model_package",
                    side_effect=model_package.ModelPackageError("schema mismatch"),
                ),
                mock.patch.object(QtWidgets.QMessageBox, "warning") as warning,
            ):
                self.assertIsNone(dialog._collect_config(show_errors=True))
            warning.assert_called_once_with(
                dialog,
                "Invalid model package",
                "schema mismatch",
            )
        finally:
            dialog.close()

    def test_accept_and_experiment_editor_properties_remain_compatible(self) -> None:
        dialog = self._dialog()
        editor = ConfigDialog(
            config={"inference_enabled": False, "task_cfg": str(self.task)},
            project=self.project,
            profile_store=self.store,
            experiment_profile_name="  Study A  ",
            show_experiment_profile_editor=True,
        )
        try:
            self.assertIsNone(dialog.result_config)
            dialog.accept()
            self.assertEqual(dialog.result(), QtWidgets.QDialog.DialogCode.Accepted)
            self.assertEqual(dialog.result_config["task_cfg"], self.task)
            self.assertEqual(editor.experiment_profile_name, "Study A")
        finally:
            dialog.close()
            editor.close()

    def test_create_experiment_profile_action_keeps_saved_config_contract(self) -> None:
        new_profile = ExperimentProfile(
            name="New Study",
            slug="New_Study",
            config={},
        )
        self.store.list_experiments.side_effect = [[], [new_profile]]
        self.store.list_subjects.side_effect = [[], []]
        dialog = self._dialog()
        created = mock.Mock(experiment_name="New Study")
        created.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        try:
            with mock.patch(
                "squeakview.apps.operator.gui.config_dialog.CreateExperimentDialog",
                return_value=created,
            ):
                dialog._on_create_experiment()

            saved = self.store.save_experiment.call_args.args[0]
            self.assertEqual(saved.name, "New Study")
            self.assertEqual(saved.slug, "New Study")
            self.assertEqual(saved.config["task_cfg"], str(self.task))
            self.assertNotIn("experiment_mode", saved.config)
            self.assertEqual(dialog.mode_combo.currentData(), "existing")
            self.assertEqual(dialog.existing_experiment_combo.currentData(), "New_Study")
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
