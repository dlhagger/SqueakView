from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from squeakview.apps.operator.gui.session_controller import (
    SessionConfigController,
    build_launch_config,
    default_config_data,
    merge_profile_selection,
    resolve_config_paths,
)
from squeakview.common.profiles import ExperimentProfile, SubjectProfile


class SessionConfigHelpersTest(unittest.TestCase):
    def test_profile_merge_copies_input_and_applies_scientific_identity(self) -> None:
        original = {"fps": 30, "experiment_name": "old", "mouse_id": "old"}
        experiment = ExperimentProfile(
            name="Study A",
            slug="study_a",
            config={
                "fps": 60,
                "trigger_on": True,
                "experiment_name": "stale_identity",
                "mouse_id": "stale_subject",
            },
        )
        subject = SubjectProfile(name="Mouse 7", subject_id="mouse_7")

        merged = merge_profile_selection(original, experiment, subject)

        self.assertEqual(original["fps"], 30)
        self.assertEqual(merged["experiment_name"], "study_a")
        self.assertEqual(merged["mouse_id"], "mouse_7")
        self.assertEqual(merged["fps"], 60)
        self.assertTrue(merged["trigger_on"])

    def test_path_resolution_does_not_mutate_dialog_result(self) -> None:
        original = {"ds_cfg": "models/mousehouse/config.txt", "task_cfg": "tasks/default.yaml"}

        resolved = resolve_config_paths(original)

        self.assertEqual(original["ds_cfg"], "models/mousehouse/config.txt")
        self.assertTrue(resolved.ds_cfg.is_absolute())
        self.assertTrue(resolved.task_cfg.is_absolute())
        self.assertEqual(resolved.data["ds_cfg"], str(resolved.ds_cfg))

    def test_launch_mapping_honors_qualification_environment(self) -> None:
        data = default_config_data()
        data["inference_enabled"] = False
        environment = {
            "SQUEAKVIEW_FAILURE_PLAN": "/tmp/fault.json",
            "SQUEAKVIEW_DISABLE_PREVIEW": "yes",
        }

        request = build_launch_config(
            data,
            bottles={"left": {"initial_weight_g": 25.0}},
            preview_window_id=42,
            environment=environment,
        )

        self.assertEqual(request.preview_window_id, 42)
        self.assertFalse(request.preview_enabled)
        self.assertEqual(request.failure_plan, Path("/tmp/fault.json"))
        self.assertEqual(request.bottles["left"]["initial_weight_g"], 25.0)

    def test_launch_mapping_rejects_missing_configuration(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Configuration not set"):
            build_launch_config(
                None,
                bottles={},
                preview_window_id=None,
                environment={},
            )


class SessionConfigControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.parent = QtWidgets.QWidget()
        self.experiment_combo = QtWidgets.QComboBox(self.parent)
        self.subject_combo = QtWidgets.QComboBox(self.parent)
        self.store = mock.Mock()
        self.experiment = ExperimentProfile(
            name="Study A",
            slug="study_a",
            config={"fps": 60},
        )
        self.subject = SubjectProfile(
            name="Mouse 7",
            subject_id="mouse_7",
            default_experiment="study_a",
        )
        self.store.list_experiments.return_value = [self.experiment]
        self.store.list_subjects.return_value = [self.subject]
        self.commits: list[dict] = []
        self.controller = SessionConfigController(
            self.parent,
            self.experiment_combo,
            self.subject_combo,
            store=self.store,
            commit=self.commits.append,
            emit=mock.Mock(),
        )

    def tearDown(self) -> None:
        self.parent.close()

    def test_reload_populates_selectors_and_preserves_no_selection(self) -> None:
        self.controller.reload()

        self.assertEqual(self.experiment_combo.count(), 2)
        self.assertEqual(self.subject_combo.count(), 2)
        self.assertEqual(self.controller.current_experiment_slug(), "")
        self.assertEqual(self.controller.current_subject_id(), "")

    def test_subject_default_selects_experiment_before_single_commit(self) -> None:
        self.controller.reload()
        self.subject_combo.setCurrentIndex(1)

        result = self.controller.subject_selected({"fps": 30})

        self.assertEqual(self.controller.current_experiment_slug(), "study_a")
        self.assertEqual(result["mouse_id"], "mouse_7")
        self.assertEqual(result["experiment_name"], "study_a")
        self.assertEqual(result["fps"], 60)
        self.assertEqual(self.commits, [result])

    def test_dialog_cancel_does_not_commit_or_replace_config(self) -> None:
        dialog = mock.Mock()
        dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Rejected
        self.controller.launcher_dialog = mock.Mock(return_value=dialog)

        result = self.controller.show_launcher({"fps": 30})

        self.assertIsNone(result)
        self.assertEqual(self.commits, [])

    def test_new_experiment_does_not_persist_previous_session_identity(self) -> None:
        dialog = mock.Mock(experiment_name="Study A")
        dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        self.controller.experiment_dialog = mock.Mock(return_value=dialog)
        self.store.save_experiment.return_value = Path("/tmp/study_a.json")

        self.controller.create_experiment(
            {"fps": 60, "experiment_name": "old_study", "mouse_id": "mouse_3"}
        )

        saved = self.store.save_experiment.call_args.args[0]
        self.assertEqual(saved.slug, "Study_A")
        self.assertEqual(saved.config["experiment_name"], "Study_A")
        self.assertEqual(saved.config["mouse_id"], "")


if __name__ == "__main__":
    unittest.main()
