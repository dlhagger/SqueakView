from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview.common.profiles import ExperimentProfile, SubjectProfile

from .dialog_style import DARK_DIALOG_STYLE, _dark_item_dialog, center_window


class CreateSubjectDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, experiment_name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Subject")
        self.setModal(True)
        self.setMinimumWidth(380)
        self.setStyleSheet(DARK_DIALOG_STYLE)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        intro = QtWidgets.QLabel(
            "Create a subject profile and attach it to the selected experiment."
            if experiment_name
            else "Create a reusable subject profile."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        if experiment_name:
            badge = QtWidgets.QLabel(f"Experiment: {experiment_name}")
            badge.setStyleSheet("color: #9aa7cc; font-weight: 600;")
            layout.addWidget(badge)
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        self.subject_id_edit = QtWidgets.QLineEdit(self)
        self.subject_id_edit.setPlaceholderText("Example: M123")
        form.addRow("Subject ID:", self.subject_id_edit)
        self.subject_name_edit = QtWidgets.QLineEdit(self)
        self.subject_name_edit.setPlaceholderText("Optional display name")
        form.addRow("Display name:", self.subject_name_edit)
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
    def subject_id(self) -> str:
        return self.subject_id_edit.text().strip()

    @property
    def subject_name(self) -> str:
        return self.subject_name_edit.text().strip() or self.subject_id

    def accept(self) -> None:
        if not self.subject_id:
            QtWidgets.QMessageBox.warning(self, "Subject required", "Enter a subject ID.")
            return
        super().accept()


class SubjectSessionMixin:
    def _find_subject(self, subject_id: str) -> SubjectProfile | None:
        for profile in self._subjects:
            if profile.subject_id == subject_id:
                return profile
        return None

    def _refresh_subjects_for_experiment(self, selected_subject: str = "") -> None:
        experiment = self._find_experiment(str(self.experiment_combo.currentData() or ""))
        allowed = set(experiment.subject_ids if experiment else [])
        self.subject_combo.blockSignals(True)
        try:
            self.subject_combo.clear()
            self.subject_combo.addItem("Select subject…", "")
            if experiment is not None:
                for profile in self._subjects:
                    if profile.subject_id in allowed:
                        self.subject_combo.addItem(profile.name, profile.subject_id)
            idx = self.subject_combo.findData(selected_subject)
            if idx < 0:
                idx = 0
            self.subject_combo.setCurrentIndex(idx)
        finally:
            self.subject_combo.blockSignals(False)

    def _update_subject_summary(self) -> None:
        experiment = self._find_experiment(str(self.experiment_combo.currentData() or ""))
        subject = self._find_subject(str(self.subject_combo.currentData() or ""))
        has_experiment = experiment is not None
        has_subject = subject is not None
        self.create_subject_btn.setEnabled(has_experiment)
        self.add_existing_subject_btn.setEnabled(has_experiment)
        self.edit_subject_btn.setEnabled(has_subject)
        self.delete_subject_btn.setEnabled(has_subject)
        self.continue_btn.setEnabled(has_experiment and has_subject)
        if experiment is None:
            self.subject_summary_label.setText("Pick an experiment before assigning or selecting subjects.")
            return
        if subject is None:
            count = len(experiment.subject_ids)
            self.subject_summary_label.setText(
                f"{count} subject{'s' if count != 1 else ''} assigned. Select one or create a new subject."
            )
            return
        self.subject_summary_label.setText(
            f"Subject '{subject.subject_id}' selected for this session."
        )

    def _upsert_experiment_subject(self, experiment_slug: str, subject_id: str) -> None:
        experiment = self._find_experiment(experiment_slug)
        subject = self._find_subject(subject_id)
        if experiment is None or subject is None:
            return
        subject_ids = [sid for sid in experiment.subject_ids if sid.strip()]
        if subject_id not in subject_ids:
            subject_ids.append(subject_id)
        self._profile_store.save_experiment(
            ExperimentProfile(
                name=experiment.name,
                slug=experiment.slug,
                config=dict(experiment.config),
                subject_ids=subject_ids,
            )
        )
        if subject.default_experiment != experiment_slug:
            self._profile_store.save_subject(
                SubjectProfile(
                    name=subject.name,
                    subject_id=subject.subject_id,
                    default_experiment=experiment_slug,
                )
            )

    def _replace_subject_id_references(self, old_subject_id: str, new_subject_id: str) -> None:
        for experiment in self._experiments:
            subject_ids = [new_subject_id if sid == old_subject_id else sid for sid in experiment.subject_ids]
            if subject_ids != experiment.subject_ids:
                deduped: list[str] = []
                for subject_id in subject_ids:
                    if subject_id and subject_id not in deduped:
                        deduped.append(subject_id)
                self._profile_store.save_experiment(
                    ExperimentProfile(
                        name=experiment.name,
                        slug=experiment.slug,
                        config=dict(experiment.config),
                        subject_ids=deduped,
                    )
                )

    def _remove_subject_references(self, subject_id: str) -> None:
        for experiment in self._experiments:
            subject_ids = [sid for sid in experiment.subject_ids if sid != subject_id]
            if subject_ids != experiment.subject_ids:
                self._profile_store.save_experiment(
                    ExperimentProfile(
                        name=experiment.name,
                        slug=experiment.slug,
                        config=dict(experiment.config),
                        subject_ids=subject_ids,
                    )
                )

    @QtCore.Slot()
    def _on_create_subject(self) -> None:
        experiment = self._find_experiment(str(self.experiment_combo.currentData() or ""))
        if experiment is None:
            QtWidgets.QMessageBox.warning(self, "Experiment required", "Create or select an experiment first.")
            return
        dialog = CreateSubjectDialog(self, experiment_name=experiment.name)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        existing_subject = self._find_subject(dialog.subject_id)
        if existing_subject is not None:
            QtWidgets.QMessageBox.warning(self, "Subject exists", f"A subject with ID '{existing_subject.subject_id}' already exists.")
            return
        self._profile_store.save_subject(
            SubjectProfile(
                name=dialog.subject_name,
                subject_id=dialog.subject_id,
                default_experiment=experiment.slug,
            )
        )
        self._reload_profiles()
        self._upsert_experiment_subject(experiment.slug, dialog.subject_id)
        self._reload_profiles()
        idx = self.subject_combo.findData(dialog.subject_id)
        if idx >= 0:
            self.subject_combo.setCurrentIndex(idx)

    @QtCore.Slot()
    def _on_edit_subject(self) -> None:
        subject = self._find_subject(str(self.subject_combo.currentData() or ""))
        if subject is None:
            QtWidgets.QMessageBox.warning(self, "Subject required", "Select a subject to edit.")
            return
        experiment = self._find_experiment(str(self.experiment_combo.currentData() or ""))
        dialog = CreateSubjectDialog(self, experiment_name=experiment.name if experiment else "")
        dialog.subject_id_edit.setText(subject.subject_id)
        dialog.subject_name_edit.setText(subject.name if subject.name != subject.subject_id else "")
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        new_subject_id = dialog.subject_id
        new_subject_name = dialog.subject_name
        conflict = self._find_subject(new_subject_id)
        if conflict is not None and conflict.subject_id != subject.subject_id:
            QtWidgets.QMessageBox.warning(self, "Subject exists", f"A subject with ID '{conflict.subject_id}' already exists.")
            return
        self._profile_store.save_subject(
            SubjectProfile(
                name=new_subject_name,
                subject_id=new_subject_id,
                default_experiment=subject.default_experiment,
            )
        )
        if new_subject_id != subject.subject_id:
            self._profile_store.delete_subject(subject.subject_id)
            self._reload_profiles()
            self._replace_subject_id_references(
                subject.subject_id,
                new_subject_id,
            )
        self._reload_profiles()
        idx = self.subject_combo.findData(new_subject_id)
        if idx >= 0:
            self.subject_combo.setCurrentIndex(idx)

    @QtCore.Slot()
    def _on_add_existing_subject(self) -> None:
        experiment = self._find_experiment(str(self.experiment_combo.currentData() or ""))
        if experiment is None:
            QtWidgets.QMessageBox.warning(self, "Experiment required", "Create or select an experiment first.")
            return
        available = [profile for profile in self._subjects if profile.subject_id not in set(experiment.subject_ids)]
        if not available:
            QtWidgets.QMessageBox.information(self, "No Subjects Available", "All saved subjects are already assigned to this experiment.")
            return
        labels = [f"{profile.name} ({profile.subject_id})" for profile in available]
        selection, ok = _dark_item_dialog(
            self,
            title="Add Subject To Experiment",
            label="Saved subject:",
            items=labels,
        )
        if not ok or not selection:
            return
        picked = available[labels.index(selection)]
        self._upsert_experiment_subject(experiment.slug, picked.subject_id)
        self._reload_profiles()
        idx = self.subject_combo.findData(picked.subject_id)
        if idx >= 0:
            self.subject_combo.setCurrentIndex(idx)

    @QtCore.Slot()
    def _on_delete_subject(self) -> None:
        subject = self._find_subject(str(self.subject_combo.currentData() or ""))
        if subject is None:
            QtWidgets.QMessageBox.warning(self, "Subject required", "Select a subject to delete.")
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete Subject",
            f"Delete subject '{subject.subject_id}'?\n\nThis removes it from all experiments.",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._profile_store.delete_subject(subject.subject_id)
        self._reload_profiles()
        self._remove_subject_references(subject.subject_id)
        self._reload_profiles()


