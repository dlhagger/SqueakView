from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview.common.profiles import ExperimentProfile, SubjectProfile, slugify

from .dialog_style import DARK_DIALOG_STYLE, center_window


class CreateExperimentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, initial_name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Experiment")
        self.setModal(True)
        self.setMinimumWidth(380)
        self.setStyleSheet(DARK_DIALOG_STYLE)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        intro = QtWidgets.QLabel(
            "Create a reusable experiment profile. The current Configure Run settings will be saved as this experiment's defaults."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        self.name_edit = QtWidgets.QLineEdit(initial_name, self)
        self.name_edit.setPlaceholderText("Example: GoNoGo Cohort A")
        form.addRow("Experiment name:", self.name_edit)
        layout.addLayout(form)
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        center_window(self)

    @property
    def experiment_name(self) -> str:
        return self.name_edit.text().strip()

    def accept(self) -> None:
        if not self.experiment_name:
            QtWidgets.QMessageBox.warning(self, "Experiment required", "Enter an experiment name.")
            return
        super().accept()


def _create_config_dialog(*args, **kwargs):
    from .config_dialog import ConfigDialog

    return ConfigDialog(*args, **kwargs)


class ExperimentSessionMixin:
    def _find_experiment(self, slug: str) -> ExperimentProfile | None:
        for profile in self._experiments:
            if profile.slug == slug:
                return profile
        return None

    def _update_experiment_summary(self) -> None:
        experiment = self._find_experiment(str(self.experiment_combo.currentData() or ""))
        has_experiment = experiment is not None
        self.edit_experiment_btn.setEnabled(has_experiment)
        self.delete_experiment_btn.setEnabled(has_experiment)
        if experiment is None:
            self.experiment_name_value.setText("No experiment selected")
            self.experiment_camera_value.setText("-")
            self.experiment_task_value.setText("-")
            self.experiment_model_value.setText("-")
            return
        cfg = experiment.config or {}
        task_name = Path(str(cfg.get("task_cfg") or "")).name or "No task"
        ds_name = Path(str(cfg.get("ds_cfg") or "")).name if cfg.get("ds_cfg") else "Inference off"
        dims = f"{cfg.get('width', '?')}x{cfg.get('height', '?')} @ {cfg.get('fps', '?')} FPS"
        self.experiment_name_value.setText(experiment.name)
        self.experiment_camera_value.setText(dims)
        self.experiment_task_value.setText(task_name)
        self.experiment_model_value.setText(ds_name)
        self.experiment_task_value.setToolTip(str(cfg.get("task_cfg") or ""))
        self.experiment_model_value.setToolTip(str(cfg.get("ds_cfg") or ""))

    def _replace_experiment_slug_references(self, old_slug: str, new_slug: str) -> None:
        if old_slug == new_slug:
            return
        for profile in self._subjects:
            if profile.default_experiment == old_slug:
                self._profile_store.save_subject(
                    SubjectProfile(
                        name=profile.name,
                        subject_id=profile.subject_id,
                        default_experiment=new_slug,
                    )
                )

    def _remove_experiment_references(self, slug: str) -> None:
        for profile in self._subjects:
            if profile.default_experiment == slug:
                self._profile_store.save_subject(
                    SubjectProfile(
                        name=profile.name,
                        subject_id=profile.subject_id,
                        default_experiment=None,
                    )
                )

    @QtCore.Slot()
    def _on_create_experiment(self) -> None:
        name_dialog = CreateExperimentDialog(self)
        if name_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        existing = self._find_experiment(slugify(name_dialog.experiment_name))
        if existing is not None:
            QtWidgets.QMessageBox.warning(self, "Experiment exists", f"An experiment named '{existing.name}' already exists.")
            return
        draft = dict(self._base_config)
        draft["experiment_mode"] = "sandbox"
        config_dialog = _create_config_dialog(
            self,
            project=self.project,
            profile_store=self._profile_store,
            title=f"Experiment Defaults: {name_dialog.experiment_name}",
            config=draft,
            show_session_setup=False,
        )
        if config_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not config_dialog.result_config:
            return
        config = dict(config_dialog.result_config)
        config["experiment_name"] = slugify(name_dialog.experiment_name)
        config["experiment_mode"] = "existing"
        config["mouse_id"] = ""
        self._profile_store.save_experiment(
            ExperimentProfile(
                name=name_dialog.experiment_name,
                slug=name_dialog.experiment_name,
                config={key: (str(value) if isinstance(value, Path) else value) for key, value in config.items()},
                subject_ids=[],
            )
        )
        self._reload_profiles()
        idx = self.experiment_combo.findData(slugify(name_dialog.experiment_name))
        if idx >= 0:
            self.experiment_combo.setCurrentIndex(idx)

    @QtCore.Slot()
    def _on_edit_experiment(self) -> None:
        experiment = self._find_experiment(str(self.experiment_combo.currentData() or ""))
        if experiment is None:
            QtWidgets.QMessageBox.warning(self, "Experiment required", "Select an experiment to edit.")
            return
        config = dict(self._base_config)
        config.update(dict(experiment.config or {}))
        config_dialog = _create_config_dialog(
            self,
            project=self.project,
            profile_store=self._profile_store,
            title=f"Edit Experiment Defaults: {experiment.name}",
            config=config,
            show_session_setup=False,
            experiment_profile_name=experiment.name,
            show_experiment_profile_editor=True,
        )
        if config_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not config_dialog.result_config:
            return
        new_name = config_dialog.experiment_profile_name
        if not new_name:
            QtWidgets.QMessageBox.warning(self, "Experiment required", "Enter an experiment name.")
            return
        new_slug = slugify(new_name)
        conflict = self._find_experiment(new_slug)
        if conflict is not None and conflict.slug != experiment.slug:
            QtWidgets.QMessageBox.warning(self, "Experiment exists", f"An experiment named '{conflict.name}' already exists.")
            return
        new_config = dict(config_dialog.result_config)
        new_config["experiment_name"] = new_slug
        new_config["experiment_mode"] = "existing"
        new_config["mouse_id"] = ""
        self._profile_store.save_experiment(
            ExperimentProfile(
                name=new_name,
                slug=new_name,
                config={key: (str(value) if isinstance(value, Path) else value) for key, value in new_config.items()},
                subject_ids=list(experiment.subject_ids),
            )
        )
        if new_slug != experiment.slug:
            self._profile_store.delete_experiment(experiment.slug)
            self._reload_profiles()
            self._replace_experiment_slug_references(experiment.slug, new_slug)
        self._reload_profiles()
        idx = self.experiment_combo.findData(new_slug)
        if idx >= 0:
            self.experiment_combo.setCurrentIndex(idx)

    @QtCore.Slot()
    def _on_delete_experiment(self) -> None:
        experiment = self._find_experiment(str(self.experiment_combo.currentData() or ""))
        if experiment is None:
            QtWidgets.QMessageBox.warning(self, "Experiment required", "Select an experiment to delete.")
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete Experiment",
            f"Delete experiment '{experiment.name}'?\n\nThis removes the saved defaults and detaches any subjects whose default experiment points here.",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._profile_store.delete_experiment(experiment.slug)
        self._reload_profiles()
        self._remove_experiment_references(experiment.slug)
        self._reload_profiles()
