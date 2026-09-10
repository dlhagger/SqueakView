from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from squeakview.apps.operator.gui.config_dialog import (
    CreateExperimentDialog as ConfigExperimentDialog,
    CreateSubjectDialog as ConfigSubjectDialog,
    SessionLauncherDialog as ConfigLauncherDialog,
)
from squeakview.apps.operator.gui.experiment_dialog import CreateExperimentDialog
from squeakview.apps.operator.gui.session_dialog import (
    CreateExperimentDialog as LegacyExperimentDialog,
    CreateSubjectDialog as LegacySubjectDialog,
    SessionLauncherDialog as LegacyLauncherDialog,
)
from squeakview.apps.operator.gui.session_launcher import SessionLauncherDialog
from squeakview.apps.operator.gui.subject_dialog import CreateSubjectDialog
from squeakview.common.profiles import ExperimentProfile, SubjectProfile


class FocusedSessionDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_compatibility_modules_reexport_canonical_classes(self) -> None:
        self.assertIs(LegacyExperimentDialog, CreateExperimentDialog)
        self.assertIs(LegacySubjectDialog, CreateSubjectDialog)
        self.assertIs(LegacyLauncherDialog, SessionLauncherDialog)
        self.assertIs(ConfigExperimentDialog, CreateExperimentDialog)
        self.assertIs(ConfigSubjectDialog, CreateSubjectDialog)
        self.assertIs(ConfigLauncherDialog, SessionLauncherDialog)

    def test_experiment_name_is_trimmed_and_required(self) -> None:
        dialog = CreateExperimentDialog(initial_name="  Trial A  ")
        try:
            self.assertEqual(dialog.experiment_name, "Trial A")
            dialog.name_edit.clear()
            with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
                dialog.accept()
            warning.assert_called_once()
            self.assertEqual(dialog.result(), QtWidgets.QDialog.DialogCode.Rejected)
        finally:
            dialog.close()

    def test_subject_falls_back_to_id_for_display_name(self) -> None:
        dialog = CreateSubjectDialog(experiment_name="Trial A")
        try:
            dialog.subject_id_edit.setText("  M123  ")
            self.assertEqual(dialog.subject_id, "M123")
            self.assertEqual(dialog.subject_name, "M123")
            dialog.subject_name_edit.setText("  Mouse 123  ")
            self.assertEqual(dialog.subject_name, "Mouse 123")
        finally:
            dialog.close()

    def test_launcher_accepts_an_explicit_profile_store(self) -> None:
        store = mock.Mock()
        store.list_experiments.return_value = []
        store.list_subjects.return_value = []
        dialog = SessionLauncherDialog(base_config={"fps": 30}, profile_store=store)
        try:
            self.assertIs(dialog._profile_store, store)
            store.list_experiments.assert_called()
            store.list_subjects.assert_called()
        finally:
            dialog.close()

    def test_launcher_builds_session_config_from_selected_profiles(self) -> None:
        store = mock.Mock()
        store.list_experiments.return_value = [
            ExperimentProfile(
                name="Trial A",
                slug="trial_a",
                config={"fps": 60, "mouse_id": "stale"},
                subject_ids=["M123"],
            )
        ]
        store.list_subjects.return_value = [SubjectProfile(name="Mouse 123", subject_id="M123")]
        dialog = SessionLauncherDialog(
            base_config={"fps": 30, "width": 1920},
            profile_store=store,
        )
        try:
            dialog.experiment_combo.setCurrentIndex(1)
            dialog.subject_combo.setCurrentIndex(1)
            dialog.accept()
            self.assertEqual(
                dialog.result_config,
                {
                    "fps": 60,
                    "width": 1920,
                    "mouse_id": "M123",
                    "experiment_mode": "existing",
                    "experiment_name": "trial_a",
                },
            )
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
