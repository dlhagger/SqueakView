from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview import config as squeakview_config
from squeakview.common.profiles import ExperimentProfile, ProfileStore, SubjectProfile, slugify


class SensorDetectWorker(QtCore.QObject):
    finished = QtCore.Signal(int, int, int, str)

    @QtCore.Slot()
    def run(self) -> None:
        flir_count = 0
        zed_count = 0
        warnings: list[str] = []

        # FLIR detection via PySpin.
        system = None
        cam_list = None
        try:
            import PySpin  # type: ignore
            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()
            flir_count = int(cam_list.GetSize())
        except Exception as exc:
            warnings.append(f"FLIR detection unavailable (PySpin): {exc}")
        finally:
            try:
                if cam_list is not None:
                    cam_list.Clear()
            except Exception:
                pass
            try:
                if system is not None:
                    system.ReleaseInstance()
            except Exception:
                pass

        # ZED detection via pyzed.
        try:
            import pyzed.sl as sl  # type: ignore

            try:
                devices = sl.Camera.get_device_list()
                zed_count = int(len(devices))
            except Exception:
                zed_count = 0
        except Exception as exc:
            warnings.append(f"ZED detection unavailable (pyzed): {exc}")

        # ZED stereo camera counts as two capture sources for current dual-camera UI.
        zed_sources = 2 if zed_count > 0 else 0
        total_sources = flir_count + zed_sources
        warning_text = "\n".join(warnings)
        self.finished.emit(total_sources, flir_count, zed_count, warning_text)


class CreateExperimentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, initial_name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Experiment")
        self.setModal(True)
        self.setMinimumWidth(380)
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Create a reusable experiment profile. The current Configure Run settings will be saved as this experiment's defaults."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QtWidgets.QFormLayout()
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

    @property
    def experiment_name(self) -> str:
        return self.name_edit.text().strip()

    def accept(self) -> None:
        if not self.experiment_name:
            QtWidgets.QMessageBox.warning(self, "Experiment required", "Enter an experiment name.")
            return
        super().accept()


class CreateSubjectDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, experiment_name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Subject")
        self.setModal(True)
        self.setMinimumWidth(380)
        layout = QtWidgets.QVBoxLayout(self)
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


class SessionLauncherDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, base_config: Optional[dict] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Start Session")
        self.setModal(True)
        self.setMinimumWidth(560)
        self._base_config = dict(base_config or {})
        self._profile_store = ProfileStore()
        self._experiments = self._profile_store.list_experiments()
        self._subjects = self._profile_store.list_subjects()
        self._result_config: dict | None = None

        self.setStyleSheet("""
            QDialog {
                background-color: #0f1118;
            }
            QLabel {
                color: #d7ddf5;
            }
            QGroupBox {
                border: 1px solid #24283b;
                border-radius: 10px;
                margin-top: 14px;
                padding-top: 12px;
                color: #e7ebff;
                font-weight: 700;
            }
            QComboBox, QLineEdit {
                background-color: #11162a;
                color: #e8ecff;
                border: 1px solid #2c3550;
                border-radius: 4px;
                padding: 4px;
            }
            QComboBox QAbstractItemView {
                background-color: #11162a;
                color: #e8ecff;
                selection-background-color: #283a7a;
                selection-color: #ffffff;
                border: 1px solid #2c3550;
            }
            QPushButton {
                background-color: #2f4daa;
                color: #ffffff;
                padding: 6px 14px;
                border-radius: 4px;
                border: 1px solid #3557bf;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3a5fc9;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Session Setup")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #eef1ff;")
        layout.addWidget(title)
        subtitle = QtWidgets.QLabel(
            "Choose an experiment first, then choose a subject. Create either one here before opening the main SqueakView window."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #9aa7cc;")
        layout.addWidget(subtitle)

        experiment_group = QtWidgets.QGroupBox("1. Experiment", self)
        experiment_form = QtWidgets.QFormLayout(experiment_group)
        experiment_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.experiment_combo = QtWidgets.QComboBox(self)
        self.create_experiment_btn = QtWidgets.QPushButton("Create Experiment…", self)
        self.create_experiment_btn.clicked.connect(self._on_create_experiment)
        self.edit_experiment_btn = QtWidgets.QPushButton("Edit…", self)
        self.edit_experiment_btn.clicked.connect(self._on_edit_experiment)
        self.delete_experiment_btn = QtWidgets.QPushButton("Delete", self)
        self.delete_experiment_btn.clicked.connect(self._on_delete_experiment)
        exp_row = QtWidgets.QHBoxLayout()
        exp_row.addWidget(self.experiment_combo, 1)
        exp_row.addWidget(self.create_experiment_btn, 0)
        exp_row.addWidget(self.edit_experiment_btn, 0)
        exp_row.addWidget(self.delete_experiment_btn, 0)
        experiment_form.addRow("Saved experiments:", exp_row)
        self.experiment_summary_label = QtWidgets.QLabel("")
        self.experiment_summary_label.setWordWrap(True)
        self.experiment_summary_label.setStyleSheet(
            "color: #e7ebff; background-color: #11162a; border: 1px solid #2c3550; border-radius: 6px; padding: 8px;"
        )
        experiment_form.addRow("Defaults:", self.experiment_summary_label)
        layout.addWidget(experiment_group)

        subject_group = QtWidgets.QGroupBox("2. Subject", self)
        subject_form = QtWidgets.QFormLayout(subject_group)
        subject_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.subject_combo = QtWidgets.QComboBox(self)
        self.create_subject_btn = QtWidgets.QPushButton("Create Subject…", self)
        self.create_subject_btn.clicked.connect(self._on_create_subject)
        self.edit_subject_btn = QtWidgets.QPushButton("Edit…", self)
        self.edit_subject_btn.clicked.connect(self._on_edit_subject)
        self.add_existing_subject_btn = QtWidgets.QPushButton("Add Existing…", self)
        self.add_existing_subject_btn.clicked.connect(self._on_add_existing_subject)
        self.delete_subject_btn = QtWidgets.QPushButton("Delete", self)
        self.delete_subject_btn.clicked.connect(self._on_delete_subject)
        subject_row = QtWidgets.QHBoxLayout()
        subject_row.addWidget(self.subject_combo, 1)
        subject_row.addWidget(self.create_subject_btn, 0)
        subject_row.addWidget(self.edit_subject_btn, 0)
        subject_row.addWidget(self.add_existing_subject_btn, 0)
        subject_row.addWidget(self.delete_subject_btn, 0)
        subject_form.addRow("Assigned subjects:", subject_row)
        self.subject_summary_label = QtWidgets.QLabel("")
        self.subject_summary_label.setWordWrap(True)
        self.subject_summary_label.setStyleSheet(
            "color: #e7ebff; background-color: #11162a; border: 1px solid #2c3550; border-radius: 6px; padding: 8px;"
        )
        subject_form.addRow("Selection:", self.subject_summary_label)
        layout.addWidget(subject_group)

        button_row = QtWidgets.QHBoxLayout()
        self.sandbox_btn = QtWidgets.QPushButton("Sandbox / Manual", self)
        self.sandbox_btn.clicked.connect(self._accept_sandbox)
        button_row.addWidget(self.sandbox_btn, 0)
        button_row.addStretch(1)
        self.cancel_btn = QtWidgets.QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        self.continue_btn = QtWidgets.QPushButton("Continue", self)
        self.continue_btn.clicked.connect(self.accept)
        button_row.addWidget(self.cancel_btn, 0)
        button_row.addWidget(self.continue_btn, 0)
        layout.addSpacing(8)
        layout.addLayout(button_row)

        self.experiment_combo.currentIndexChanged.connect(self._on_experiment_changed)
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        self._reload_profiles()

    @property
    def result_config(self) -> dict | None:
        return self._result_config

    def _find_experiment(self, slug: str) -> ExperimentProfile | None:
        for profile in self._experiments:
            if profile.slug == slug:
                return profile
        return None

    def _find_subject(self, subject_id: str) -> SubjectProfile | None:
        for profile in self._subjects:
            if profile.subject_id == subject_id:
                return profile
        return None

    def _reload_profiles(self) -> None:
        self._experiments = self._profile_store.list_experiments()
        self._subjects = self._profile_store.list_subjects()
        current_exp = str(self.experiment_combo.currentData() or "")
        current_subj = str(self.subject_combo.currentData() or "")
        self.experiment_combo.blockSignals(True)
        try:
            self.experiment_combo.clear()
            self.experiment_combo.addItem("Select experiment…", "")
            for profile in self._experiments:
                self.experiment_combo.addItem(profile.name, profile.slug)
            idx = self.experiment_combo.findData(current_exp)
            if idx < 0 and len(self._experiments) == 1:
                idx = 1
            self.experiment_combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self.experiment_combo.blockSignals(False)
        self._refresh_subjects_for_experiment(current_subj)
        self._update_experiment_summary()
        self._update_subject_summary()

    def _refresh_subjects_for_experiment(self, selected_subject: str = "") -> None:
        experiment = self._find_experiment(str(self.experiment_combo.currentData() or ""))
        allowed = set(experiment.subject_ids if experiment else [])
        self.subject_combo.blockSignals(True)
        try:
            self.subject_combo.clear()
            self.subject_combo.addItem("Select subject…", "")
            for profile in self._subjects:
                if allowed and profile.subject_id not in allowed:
                    continue
                self.subject_combo.addItem(profile.name, profile.subject_id)
            idx = self.subject_combo.findData(selected_subject)
            if idx < 0:
                idx = 0
            self.subject_combo.setCurrentIndex(idx)
        finally:
            self.subject_combo.blockSignals(False)

    def _update_experiment_summary(self) -> None:
        experiment = self._find_experiment(str(self.experiment_combo.currentData() or ""))
        has_experiment = experiment is not None
        self.edit_experiment_btn.setEnabled(has_experiment)
        self.delete_experiment_btn.setEnabled(has_experiment)
        if experiment is None:
            self.experiment_summary_label.setText("No experiment selected yet. Create one to save a full Configure Run profile.")
            return
        cfg = experiment.config or {}
        task_name = Path(str(cfg.get("task_cfg") or "")).name or "No task"
        ds_name = Path(str(cfg.get("ds_cfg") or "")).name if cfg.get("ds_cfg") else "Inference off"
        backend = str(cfg.get("capture_backend", "flir")).upper()
        dims = f"{cfg.get('width', '?')}x{cfg.get('height', '?')} @ {cfg.get('fps', '?')} FPS"
        self.experiment_summary_label.setText(
            f"{experiment.name}\n{backend} · {dims}\nTask: {task_name}\nConfig: {ds_name}"
        )

    def _update_subject_summary(self) -> None:
        experiment = self._find_experiment(str(self.experiment_combo.currentData() or ""))
        subject = self._find_subject(str(self.subject_combo.currentData() or ""))
        has_experiment = experiment is not None
        has_subject = subject is not None
        self.create_subject_btn.setEnabled(has_experiment)
        self.add_existing_subject_btn.setEnabled(has_experiment)
        self.edit_subject_btn.setEnabled(has_subject)
        self.delete_subject_btn.setEnabled(has_subject)
        if experiment is None:
            self.subject_summary_label.setText("Pick an experiment before assigning or selecting subjects.")
            return
        if subject is None:
            count = len(experiment.subject_ids)
            self.subject_summary_label.setText(
                f"{count} subject{'s' if count != 1 else ''} assigned to this experiment. Select one or create a new one."
            )
            return
        self.subject_summary_label.setText(
            f"Subject '{subject.subject_id}' selected. This will be used as the mouse ID for the session."
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
    def _on_experiment_changed(self) -> None:
        preferred_subject = ""
        experiment = self._find_experiment(str(self.experiment_combo.currentData() or ""))
        if experiment:
            for profile in self._subjects:
                if profile.default_experiment == experiment.slug and profile.subject_id in set(experiment.subject_ids):
                    preferred_subject = profile.subject_id
                    break
        self._refresh_subjects_for_experiment(preferred_subject)
        self._update_experiment_summary()
        self._update_subject_summary()

    @QtCore.Slot()
    def _on_subject_changed(self) -> None:
        self._update_subject_summary()

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
        config_dialog = ConfigDialog(
            self,
            title=f"Experiment Defaults: {name_dialog.experiment_name}",
            config=draft,
            show_session_setup=False,
        )
        if config_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not config_dialog.result_config:
            return
        config = dict(config_dialog.result_config)
        config["experiment_name"] = slugify(name_dialog.experiment_name)
        config["experiment_mode"] = "existing"
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
        config_dialog = ConfigDialog(
            self,
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
        selection, ok = QtWidgets.QInputDialog.getItem(self, "Add Subject To Experiment", "Saved subject:", labels, 0, False)
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

    @QtCore.Slot()
    def _accept_sandbox(self) -> None:
        config = dict(self._base_config)
        config["experiment_mode"] = "sandbox"
        config["experiment_name"] = ""
        config["mouse_id"] = ""
        self._result_config = config
        super().accept()

    def accept(self) -> None:
        experiment = self._find_experiment(str(self.experiment_combo.currentData() or ""))
        subject = self._find_subject(str(self.subject_combo.currentData() or ""))
        if experiment is None:
            QtWidgets.QMessageBox.warning(self, "Experiment required", "Select or create an experiment first.")
            return
        if subject is None:
            QtWidgets.QMessageBox.warning(self, "Subject required", "Select or create a subject for this experiment.")
            return
        config = dict(self._base_config)
        config.update(dict(experiment.config or {}))
        config["experiment_mode"] = "existing"
        config["experiment_name"] = experiment.slug
        config["mouse_id"] = subject.subject_id
        self._result_config = config
        super().accept()


class ConfigDialog(QtWidgets.QDialog):
    """Modal dialog to configure SqueakView capture + inference parameters."""

    def __init__(
        self,
        parent=None,
        *,
        title: str = "Configure SqueakView",
        config: Optional[dict] = None,
        show_session_setup: bool = True,
        experiment_profile_name: str = "",
        show_experiment_profile_editor: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(440)

        self.setStyleSheet("""
            QDialog {
                background-color: #0f1118;
            }
            QLabel {
                color: #d7ddf5;
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 1px solid #24283b;
                border-radius: 8px;
                background-color: #14192a;
                top: -1px;
            }
            QTabBar::tab {
                background: #181f31;
                color: #aeb8da;
                border: 1px solid #24283b;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 7px 12px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #1f2840;
                color: #e7ebff;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
            QLineEdit, QComboBox {
                background-color: #11162a;
                color: #e8ecff;
                border: 1px solid #2c3550;
                border-radius: 4px;
                padding: 4px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #7480ff;
            }
            QComboBox QAbstractItemView {
                background-color: #11162a;
                color: #e8ecff;
                selection-background-color: #283a7a;
                selection-color: #ffffff;
                border: 1px solid #2c3550;
            }
            QCheckBox {
                color: #d7ddf5;
            }
            QPushButton {
                background-color: #2f4daa;
                color: #ffffff;
                padding: 6px 14px;
                border-radius: 4px;
                border: 1px solid #3557bf;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3a5fc9;
            }
            QPushButton:disabled {
                background-color: #2a3248;
                color: #7f8aac;
                border-color: #313c5d;
            }
            QProgressDialog {
                background-color: #14192a;
                color: #d7ddf5;
            }
            QMessageBox {
                background-color: #14192a;
                color: #d7ddf5;
            }
        """)
        cfg = config or {}
        self._profile_store = ProfileStore()
        self._experiments = self._profile_store.list_experiments()
        self._subjects = self._profile_store.list_subjects()
        self._detected_cameras = max(0, min(2, int(cfg.get("num_cameras", 0))))
        self._detect_thread: QtCore.QThread | None = None
        self._detect_worker: SensorDetectWorker | None = None
        self._detect_progress: QtWidgets.QProgressDialog | None = None
        self._show_experiment_profile_editor = show_experiment_profile_editor

        form = QtWidgets.QFormLayout()
        self._run_form = form
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)

        self.mode_combo = QtWidgets.QComboBox(self)
        self.mode_combo.addItem("Experiment Session", "existing")
        self.mode_combo.addItem("Sandbox / Manual Mode", "sandbox")

        self.existing_experiment_combo = QtWidgets.QComboBox(self)
        self.existing_experiment_combo.addItem("Select experiment…", "")
        for profile in self._experiments:
            self.existing_experiment_combo.addItem(profile.name, profile.slug)
        self.new_experiment_btn = QtWidgets.QPushButton("Create…", self)
        self.new_experiment_btn.clicked.connect(self._on_create_experiment)
        existing_experiment_row = QtWidgets.QWidget(self)
        existing_experiment_layout = QtWidgets.QHBoxLayout(existing_experiment_row)
        existing_experiment_layout.setContentsMargins(0, 0, 0, 0)
        existing_experiment_layout.setSpacing(8)
        existing_experiment_layout.addWidget(self.existing_experiment_combo, 1)
        existing_experiment_layout.addWidget(self.new_experiment_btn, 0)

        self.existing_subject_combo = QtWidgets.QComboBox(self)
        self.existing_subject_combo.addItem("No saved subject", "")
        for profile in self._subjects:
            self.existing_subject_combo.addItem(profile.name, profile.subject_id)
        self.new_subject_btn = QtWidgets.QPushButton("Create…", self)
        self.new_subject_btn.clicked.connect(self._on_create_subject_for_selected_experiment)
        self.add_subject_btn = QtWidgets.QPushButton("Add Existing…", self)
        self.add_subject_btn.clicked.connect(self._on_add_existing_subject_to_selected_experiment)
        existing_subject_row = QtWidgets.QWidget(self)
        existing_subject_layout = QtWidgets.QHBoxLayout(existing_subject_row)
        existing_subject_layout.setContentsMargins(0, 0, 0, 0)
        existing_subject_layout.setSpacing(8)
        existing_subject_layout.addWidget(self.existing_subject_combo, 1)
        existing_subject_layout.addWidget(self.new_subject_btn, 0)
        existing_subject_layout.addWidget(self.add_subject_btn, 0)

        initial_mode = str(cfg.get("experiment_mode", "sandbox"))
        existing_slug = str(cfg.get("experiment_name", ""))
        if existing_slug and self.existing_experiment_combo.findData(existing_slug) >= 0:
            initial_mode = "existing"
        mode_idx = max(0, self.mode_combo.findData(initial_mode))
        self.mode_combo.setCurrentIndex(mode_idx)

        int_validator = QtGui.QIntValidator(1, 4096, self)

        self.width_edit = QtWidgets.QLineEdit(str(cfg.get("width", 1440)))
        self.width_edit.setValidator(int_validator)
        form.addRow("Width:", self.width_edit)

        self.height_edit = QtWidgets.QLineEdit(str(cfg.get("height", 1080)))
        self.height_edit.setValidator(int_validator)
        form.addRow("Height:", self.height_edit)

        self.fps_edit = QtWidgets.QLineEdit(str(cfg.get("fps", 30)))
        self.fps_edit.setValidator(QtGui.QIntValidator(1, 240, self))
        form.addRow("FPS:", self.fps_edit)

        self.capture_backend_combo = QtWidgets.QComboBox()
        self.capture_backend_combo.addItem("FLIR (PySpin)", "flir")
        self.capture_backend_combo.addItem("ZED (DeepStream zedsrc)", "zed")
        current_backend = str(cfg.get("capture_backend", "flir")).lower().strip()
        idx = self.capture_backend_combo.findData(current_backend)
        if idx >= 0:
            self.capture_backend_combo.setCurrentIndex(idx)
        form.addRow("Capture Backend:", self.capture_backend_combo)

        self.trigger_chk = QtWidgets.QCheckBox("Enable camera trigger")
        self.trigger_chk.setChecked(cfg.get("trigger_on", True))
        form.addRow("", self.trigger_chk)

        self.arduino_fps_edit = QtWidgets.QLineEdit(str(cfg.get("arduino_fps", 30)))
        self.arduino_fps_edit.setValidator(QtGui.QIntValidator(1, 240, self))
        form.addRow("Arduino FPS:", self.arduino_fps_edit)

        self.serial_enable = QtWidgets.QCheckBox("Enable Arduino serial logging")
        self.serial_enable.setChecked(cfg.get("serial_enabled", True))
        form.addRow("", self.serial_enable)

        self.inference_enable = QtWidgets.QCheckBox("Enable YOLO inference (DeepStream)")
        self.inference_enable.setChecked(cfg.get("inference_enabled", True))
        form.addRow("", self.inference_enable)

        self.skeleton_chk = QtWidgets.QCheckBox("Draw skeleton overlay (pose only)")
        self.skeleton_chk.setChecked(cfg.get("draw_skeleton", False))
        form.addRow("", self.skeleton_chk)

        self.mouse_id_edit = QtWidgets.QLineEdit(str(cfg.get("mouse_id", "")))
        form.addRow("Mouse ID:", self.mouse_id_edit)

        task_default = cfg.get("task_cfg", "")
        if not task_default:
            candidate = squeakview_config.TASKS_DIR / "default.yaml"
            task_default = str(candidate) if candidate.exists() else ""
        self.task_cfg_edit = QtWidgets.QLineEdit(str(task_default))
        task_browse_btn = QtWidgets.QPushButton("Browse…")
        task_browse_btn.clicked.connect(self._on_browse_task_cfg)
        task_layout = QtWidgets.QHBoxLayout()
        task_layout.addWidget(self.task_cfg_edit, 1)
        task_layout.addWidget(task_browse_btn, 0)
        form.addRow("Task config:", task_layout)

        serial_row = QtWidgets.QHBoxLayout()
        self.serial_port_edit = QtWidgets.QLineEdit(cfg.get("serial_port", "/dev/ttyACM0"))
        serial_row.addWidget(QtWidgets.QLabel("Port:"))
        serial_row.addWidget(self.serial_port_edit)
        self.serial_baud_edit = QtWidgets.QLineEdit(str(cfg.get("serial_baud", 115200)))
        serial_row.addWidget(QtWidgets.QLabel("Baud:"))
        serial_row.addWidget(self.serial_baud_edit)
        self.serial_row_widget = QtWidgets.QWidget(self)
        self.serial_row_widget.setLayout(serial_row)
        form.addRow("", self.serial_row_widget)

        default_cfg = cfg.get("ds_cfg", "")
        self.cfg_edit = QtWidgets.QLineEdit(str(default_cfg))
        self.cfg_browse_btn = QtWidgets.QPushButton("Browse…")
        self.cfg_browse_btn.clicked.connect(self._on_browse_cfg)
        cfg_layout = QtWidgets.QHBoxLayout()
        cfg_layout.addWidget(self.cfg_edit, 1)
        cfg_layout.addWidget(self.cfg_browse_btn, 0)
        self.cfg_label = QtWidgets.QLabel("DeepStream config:")
        self.cfg_row_widget = QtWidgets.QWidget(self)
        self.cfg_row_widget.setLayout(cfg_layout)
        form.addRow(self.cfg_label, self.cfg_row_widget)

        self.flir_panel = QtWidgets.QGroupBox("FLIR Capture Panel", self)
        flir_form = QtWidgets.QFormLayout(self.flir_panel)
        flir_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        flir_form.setHorizontalSpacing(12)
        flir_form.setVerticalSpacing(8)

        self.pix_combo = QtWidgets.QComboBox()
        self.pix_combo.addItems(["Mono8", "BGR8", "GRAY8"])
        current_pix = cfg.get("pixel_format", "Mono8")
        if current_pix in [self.pix_combo.itemText(i) for i in range(self.pix_combo.count())]:
            self.pix_combo.setCurrentText(current_pix)
        flir_form.addRow("Pixel Format:", self.pix_combo)

        self.exposure_edit = QtWidgets.QLineEdit(str(cfg.get("exposure_us", 10000)))
        self.exposure_edit.setValidator(QtGui.QIntValidator(10, 10_000_000, self))
        flir_form.addRow("Exposure (us):", self.exposure_edit)

        self.socket_edit = QtWidgets.QLineEdit(str(cfg.get("socket_path", "/tmp/cam.sock")))
        flir_form.addRow("Shared socket:", self.socket_edit)

        self.socket2_edit = QtWidgets.QLineEdit(str(cfg.get("socket_path_2", "/tmp/cam1.sock")))
        flir_form.addRow("Second socket:", self.socket2_edit)
        form.addRow("", self.flir_panel)

        self.zed_panel = QtWidgets.QGroupBox("ZED Capture Panel", self)
        zed_form = QtWidgets.QFormLayout(self.zed_panel)
        zed_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        zed_form.setHorizontalSpacing(12)
        zed_form.setVerticalSpacing(8)

        self.zed_depth_chk = QtWidgets.QCheckBox("Enable ZED depth stream")
        self.zed_depth_chk.setChecked(bool(cfg.get("zed_depth_enabled", False)))
        zed_form.addRow("", self.zed_depth_chk)

        self.zed_depth_mode_combo = QtWidgets.QComboBox()
        self.zed_depth_mode_combo.addItems(["NEURAL", "NEURAL_LIGHT", "NEURAL_PLUS", "PERFORMANCE", "QUALITY"])
        current_depth_mode = str(cfg.get("zed_depth_mode", "NEURAL")).upper().strip()
        if current_depth_mode in [self.zed_depth_mode_combo.itemText(i) for i in range(self.zed_depth_mode_combo.count())]:
            self.zed_depth_mode_combo.setCurrentText(current_depth_mode)
        zed_form.addRow("ZED Depth Mode:", self.zed_depth_mode_combo)

        self.zed_depth_record_chk = QtWidgets.QCheckBox("Record depth video (higher load)")
        self.zed_depth_record_chk.setChecked(bool(cfg.get("zed_depth_record", False)))
        zed_form.addRow("", self.zed_depth_record_chk)

        self.zed_confidence_95_chk = QtWidgets.QCheckBox("Apply ZED confidence threshold 100")
        self.zed_confidence_95_chk.setChecked(int(cfg.get("zed_confidence_threshold", 100) or 100) >= 100)
        zed_form.addRow("", self.zed_confidence_95_chk)

        self.zed_adv_group = QtWidgets.QGroupBox("ZED Advanced Depth", self.zed_panel)
        zed_adv_form = QtWidgets.QFormLayout(self.zed_adv_group)
        zed_adv_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        zed_adv_form.setHorizontalSpacing(10)
        zed_adv_form.setVerticalSpacing(6)

        self.zed_texture_conf_edit = QtWidgets.QLineEdit(str(cfg.get("zed_texture_confidence_threshold", 100)))
        self.zed_texture_conf_edit.setValidator(QtGui.QIntValidator(0, 100, self))
        zed_adv_form.addRow("Texture confidence:", self.zed_texture_conf_edit)

        self.zed_depth_min_mm_edit = QtWidgets.QLineEdit(str(cfg.get("zed_depth_minimum_distance_mm", 300)))
        self.zed_depth_min_mm_edit.setValidator(QtGui.QIntValidator(100, 3000, self))
        zed_adv_form.addRow("Depth min (mm):", self.zed_depth_min_mm_edit)

        self.zed_depth_max_mm_edit = QtWidgets.QLineEdit(str(cfg.get("zed_depth_maximum_distance_mm", 20000)))
        self.zed_depth_max_mm_edit.setValidator(QtGui.QIntValidator(500, 40000, self))
        zed_adv_form.addRow("Depth max (mm):", self.zed_depth_max_mm_edit)

        self.zed_fill_mode_chk = QtWidgets.QCheckBox("Enable depth fill mode")
        self.zed_fill_mode_chk.setChecked(bool(cfg.get("zed_fill_mode", False)))
        zed_adv_form.addRow("", self.zed_fill_mode_chk)

        self.zed_depth_stab_edit = QtWidgets.QLineEdit(str(cfg.get("zed_depth_stabilization", 30)))
        self.zed_depth_stab_edit.setValidator(QtGui.QIntValidator(0, 100, self))
        zed_adv_form.addRow("Depth stabilization:", self.zed_depth_stab_edit)
        zed_form.addRow("", self.zed_adv_group)
        form.addRow("", self.zed_panel)

        self.bitrate_edit = QtWidgets.QLineEdit(str(cfg.get("bitrate", 4000)))
        self.bitrate_edit.setValidator(QtGui.QIntValidator(100, 50000, self))
        form.addRow("Bitrate (kbps):", self.bitrate_edit)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel("Select an experiment and subject, or switch to sandbox for a one-off manual session.")
        header.setWordWrap(True)
        layout.addWidget(header)
        layout.addSpacing(6)

        if show_experiment_profile_editor:
            experiment_group = QtWidgets.QGroupBox("Experiment Profile", self)
            experiment_form = QtWidgets.QFormLayout(experiment_group)
            experiment_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            experiment_form.setHorizontalSpacing(14)
            experiment_form.setVerticalSpacing(10)
            experiment_help = QtWidgets.QLabel(
                "Edit both the experiment name and the saved Configure Run defaults in this window."
            )
            experiment_help.setWordWrap(True)
            experiment_help.setStyleSheet("color: #9aa7cc;")
            self.experiment_profile_name_edit = QtWidgets.QLineEdit(experiment_profile_name, self)
            experiment_form.addRow("", experiment_help)
            experiment_form.addRow("Experiment name:", self.experiment_profile_name_edit)
            layout.addWidget(experiment_group)
        else:
            self.experiment_profile_name_edit = None

        session_group = QtWidgets.QGroupBox("Session Setup", self)
        session_form = QtWidgets.QFormLayout(session_group)
        session_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        session_form.setHorizontalSpacing(14)
        session_form.setVerticalSpacing(10)
        self.session_hint_label = QtWidgets.QLabel(
            "Experiment sessions load saved Configure Run defaults. Subjects are managed separately and attached to experiments."
        )
        self.session_hint_label.setWordWrap(True)
        self.session_hint_label.setStyleSheet("color: #9aa7cc;")
        session_form.addRow("", self.session_hint_label)
        self.session_summary_label = QtWidgets.QLabel("")
        self.session_summary_label.setWordWrap(True)
        self.session_summary_label.setStyleSheet(
            "color: #e7ebff; background-color: #11162a; border: 1px solid #2c3550; border-radius: 6px; padding: 8px;"
        )
        session_form.addRow("Summary:", self.session_summary_label)
        session_form.addRow("Mode:", self.mode_combo)
        session_form.addRow("Experiment:", existing_experiment_row)
        session_form.addRow("Subject:", existing_subject_row)
        layout.addWidget(session_group)
        self._session_group = session_group
        if not show_session_setup:
            session_group.hide()

        tabs = QtWidgets.QTabWidget(self)
        sensor_tab = QtWidgets.QWidget(self)
        sensor_layout = QtWidgets.QVBoxLayout(sensor_tab)
        sensor_layout.setContentsMargins(10, 10, 10, 10)
        sensor_form = QtWidgets.QFormLayout()
        sensor_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        sensor_form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        sensor_form.setHorizontalSpacing(14)
        sensor_form.setVerticalSpacing(10)
        self.detected_sensors_label = QtWidgets.QLabel("")
        sensor_form.addRow("Detected sensors:", self.detected_sensors_label)
        self.detected_sources_label = QtWidgets.QLabel("")
        sensor_form.addRow("Sensor details:", self.detected_sources_label)
        self.detect_btn = QtWidgets.QPushButton("Detect Sensors")
        self.detect_btn.clicked.connect(self._on_detect_sensors)
        sensor_form.addRow("", self.detect_btn)
        self.continue_btn = QtWidgets.QPushButton("Continue to Run Settings")
        self.continue_btn.clicked.connect(self._on_continue_to_run_tab)
        sensor_form.addRow("", self.continue_btn)
        sensor_help = QtWidgets.QLabel(
            "Detects connected FLIR cameras (PySpin) and ZED devices (pyzed), then configures camera count automatically (0-2). "
            "Run-time ZED capture uses DeepStream zedsrc."
        )
        sensor_help.setWordWrap(True)
        sensor_help.setStyleSheet("color: #9aa7cc;")
        sensor_layout.addLayout(sensor_form)
        sensor_layout.addWidget(sensor_help)
        sensor_layout.addStretch(1)

        run_tab = QtWidgets.QWidget(self)
        run_layout = QtWidgets.QVBoxLayout(run_tab)
        run_layout.setContentsMargins(10, 10, 10, 10)
        run_layout.addLayout(form)
        run_layout.addStretch(1)

        tabs.addTab(sensor_tab, "Configure Sensor Source")
        tabs.addTab(run_tab, "Configure Run")
        tabs.setCurrentIndex(0)
        tabs.setTabEnabled(1, False)
        self.tabs = tabs
        layout.addWidget(tabs)
        layout.addSpacing(12)
        layout.addWidget(button_box)
        self.ok_button = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if self.ok_button is not None:
            self.ok_button.setEnabled(False)

        self._result: dict | None = None
        self.inference_enable.toggled.connect(self._on_inference_toggled)
        self.capture_backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.existing_experiment_combo.currentIndexChanged.connect(self._on_existing_experiment_changed)
        self.existing_subject_combo.currentIndexChanged.connect(self._on_existing_subject_changed)
        self._on_inference_toggled(self.inference_enable.isChecked())
        self._set_detected_cameras(self._detected_cameras, from_detection=True)
        self._on_backend_changed()
        if existing_slug:
            idx = self.existing_experiment_combo.findData(existing_slug)
            if idx >= 0:
                self.existing_experiment_combo.setCurrentIndex(idx)
        subj_id = str(cfg.get("mouse_id", ""))
        if subj_id:
            idx = self.existing_subject_combo.findData(subj_id)
            if idx >= 0:
                self.existing_subject_combo.setCurrentIndex(idx)
        self._on_mode_changed()

    def _apply_zed_autoconfig_defaults(self) -> None:
        """Prefill sane defaults for ZED runs; user may edit afterward."""
        self.width_edit.setText("1920")
        self.height_edit.setText("1200")
        self.fps_edit.setText("30")
        self.arduino_fps_edit.setText("30")
        self.pix_combo.setCurrentText("GRAY8")
        self.trigger_chk.setChecked(False)
        self.socket_edit.setText("/tmp/cam.sock")
        self.socket2_edit.setText("/tmp/cam1.sock")
        self.bitrate_edit.setText("4000")
        self.zed_depth_chk.setChecked(True)
        self.zed_depth_mode_combo.setCurrentText("NEURAL")
        self.zed_texture_conf_edit.setText("100")
        self.zed_depth_min_mm_edit.setText("300")
        self.zed_depth_max_mm_edit.setText("20000")
        self.zed_fill_mode_chk.setChecked(False)
        self.zed_depth_stab_edit.setText("30")
        self.zed_confidence_95_chk.setChecked(True)
        # Prefer an existing batch=2 config for dual-source ZED left/right ingest.
        preferred_cfg = squeakview_config.DEEPSTREAM_ROOT / "configs" / "26n_pose_DSFT_fp16.txt"
        if preferred_cfg.exists():
            self.cfg_edit.setText(str(preferred_cfg))

    def _set_detected_cameras(self, count: int, *, from_detection: bool) -> None:
        count = max(0, min(2, int(count)))
        self._detected_cameras = count
        prefix = "Detected" if from_detection else "Configured"
        suffix = "camera" if count == 1 else "cameras"
        self.detected_sensors_label.setText(f"{prefix}: {count} {suffix}")
        if not from_detection:
            self.detected_sources_label.setText("FLIR: unknown | ZED devices: unknown")
        self.socket2_edit.setEnabled(count > 1)

    @QtCore.Slot(bool)
    def _on_inference_toggled(self, enabled: bool) -> None:
        self.cfg_label.setVisible(enabled)
        self.cfg_row_widget.setVisible(enabled)
        if not enabled:
            self.skeleton_chk.setChecked(False)
        self.skeleton_chk.setEnabled(enabled)

    def _set_form_row_visible(self, field: QtWidgets.QWidget, visible: bool) -> None:
        field.setVisible(visible)
        label = self._run_form.labelForField(field)
        if label is not None:
            label.setVisible(visible)

    def _current_mode(self) -> str:
        return str(self.mode_combo.currentData() or "sandbox")

    def _session_form(self) -> QtWidgets.QFormLayout:
        return self._session_group.layout()  # type: ignore[return-value]

    def _find_experiment(self, slug: str) -> ExperimentProfile | None:
        for profile in self._experiments:
            if profile.slug == slug:
                return profile
        return None

    def _find_subject(self, subject_id: str) -> SubjectProfile | None:
        for profile in self._subjects:
            if profile.subject_id == subject_id:
                return profile
        return None

    def _reload_profiles(self) -> None:
        self._experiments = self._profile_store.list_experiments()
        self._subjects = self._profile_store.list_subjects()
        current_experiment = str(self.existing_experiment_combo.currentData() or "")
        current_subject = str(self.existing_subject_combo.currentData() or "")
        self.existing_experiment_combo.blockSignals(True)
        try:
            self.existing_experiment_combo.clear()
            self.existing_experiment_combo.addItem("Select experiment…", "")
            for profile in self._experiments:
                self.existing_experiment_combo.addItem(profile.name, profile.slug)
            exp_idx = self.existing_experiment_combo.findData(current_experiment)
            self.existing_experiment_combo.setCurrentIndex(exp_idx if exp_idx >= 0 else 0)
        finally:
            self.existing_experiment_combo.blockSignals(False)
        experiment = self._find_experiment(str(self.existing_experiment_combo.currentData() or ""))
        self._refresh_existing_subjects(experiment.subject_ids if experiment else [])
        if current_subject:
            idx = self.existing_subject_combo.findData(current_subject)
            if idx >= 0:
                self.existing_subject_combo.setCurrentIndex(idx)
        self._update_session_summary()

    def _upsert_experiment_subject(self, experiment_slug: str, subject_id: str) -> None:
        if not experiment_slug or not subject_id:
            return
        profile = self._find_experiment(experiment_slug)
        if not profile:
            return
        subject_ids = [sid for sid in profile.subject_ids if sid.strip()]
        if subject_id not in subject_ids:
            subject_ids.append(subject_id)
            self._profile_store.save_experiment(
                ExperimentProfile(
                    name=profile.name,
                    slug=profile.slug,
                    config=dict(profile.config),
                    subject_ids=subject_ids,
                )
            )
        subject = self._find_subject(subject_id)
        if subject is not None and subject.default_experiment != experiment_slug:
            self._profile_store.save_subject(
                SubjectProfile(
                    name=subject.name,
                    subject_id=subject.subject_id,
                    default_experiment=experiment_slug,
                )
            )
        self._reload_profiles()

    @QtCore.Slot()
    def _on_create_experiment(self) -> None:
        config = self._collect_config(show_errors=True, include_mode=False)
        if config is None:
            return
        dialog = CreateExperimentDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        exp_name = dialog.experiment_name
        exp_profile = ExperimentProfile(
            name=exp_name,
            slug=exp_name,
            config={key: (str(value) if isinstance(value, Path) else value) for key, value in config.items()},
            subject_ids=[],
        )
        self._profile_store.save_experiment(exp_profile)
        self._reload_profiles()
        idx = self.existing_experiment_combo.findData(slugify(exp_name))
        if idx >= 0:
            self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData("existing")))
            self.existing_experiment_combo.setCurrentIndex(idx)

    @QtCore.Slot()
    def _on_create_subject_for_selected_experiment(self) -> None:
        experiment_slug = str(self.existing_experiment_combo.currentData() or "")
        experiment = self._find_experiment(experiment_slug)
        if not experiment:
            QtWidgets.QMessageBox.warning(self, "Experiment required", "Select an experiment before creating a subject.")
            return
        subject_id, ok = QtWidgets.QInputDialog.getText(self, "Create Subject", "Subject ID:")
        subject_id = subject_id.strip()
        if not ok or not subject_id:
            return
        existing = self._find_subject(subject_id)
        subject_name = existing.name if existing else subject_id
        self._profile_store.save_subject(
            SubjectProfile(
                name=subject_name,
                subject_id=subject_id,
                default_experiment=experiment.slug,
            )
        )
        self._upsert_experiment_subject(experiment.slug, subject_id)
        idx = self.existing_subject_combo.findData(subject_id)
        if idx >= 0:
            self.existing_subject_combo.setCurrentIndex(idx)

    @QtCore.Slot()
    def _on_add_existing_subject_to_selected_experiment(self) -> None:
        experiment_slug = str(self.existing_experiment_combo.currentData() or "")
        experiment = self._find_experiment(experiment_slug)
        if not experiment:
            QtWidgets.QMessageBox.warning(self, "Experiment required", "Select an experiment before adding a subject.")
            return
        available = [profile for profile in self._subjects if profile.subject_id not in set(experiment.subject_ids)]
        if not available:
            QtWidgets.QMessageBox.information(self, "No Subjects Available", "All saved subjects are already assigned to this experiment.")
            return
        labels = [f"{profile.name} ({profile.subject_id})" for profile in available]
        selection, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Add Subject To Experiment",
            "Saved subject:",
            labels,
            0,
            False,
        )
        if not ok or not selection:
            return
        picked = available[labels.index(selection)]
        self._upsert_experiment_subject(experiment.slug, picked.subject_id)
        idx = self.existing_subject_combo.findData(picked.subject_id)
        if idx >= 0:
            self.existing_subject_combo.setCurrentIndex(idx)

    @QtCore.Slot()
    def _on_mode_changed(self) -> None:
        mode = self._current_mode()
        existing = mode == "existing"
        exp_row = self.existing_experiment_combo.parentWidget()
        if exp_row is not None:
            exp_row.setVisible(existing)
        exp_label = self._session_form().labelForField(exp_row)
        if exp_label:
            exp_label.setVisible(existing)
        subj_row = self.existing_subject_combo.parentWidget()
        if subj_row is not None:
            subj_row.setVisible(existing)
        subj_label = self._session_form().labelForField(subj_row)
        if subj_label:
            subj_label.setVisible(existing)
        self.new_experiment_btn.setEnabled(True)
        self.new_subject_btn.setEnabled(existing)
        self.add_subject_btn.setEnabled(existing)
        if existing:
            self._on_existing_experiment_changed()
            self._on_existing_subject_changed()
        self._update_session_summary()

    @QtCore.Slot()
    def _on_existing_experiment_changed(self) -> None:
        slug = str(self.existing_experiment_combo.currentData() or "")
        profile = self._find_experiment(slug)
        if not profile:
            self._refresh_existing_subjects([])
            self._update_session_summary()
            return
        cfg = profile.config or {}
        self._apply_experiment_config(cfg)
        self._refresh_existing_subjects(profile.subject_ids)
        self._update_session_summary()

    @QtCore.Slot()
    def _on_existing_subject_changed(self) -> None:
        subject_id = str(self.existing_subject_combo.currentData() or "")
        profile = self._find_subject(subject_id)
        if not profile:
            self._update_session_summary()
            return
        self.mouse_id_edit.setText(profile.subject_id)
        if profile.default_experiment:
            idx = self.existing_experiment_combo.findData(profile.default_experiment)
            if idx >= 0 and self._current_mode() == "existing":
                self.existing_experiment_combo.setCurrentIndex(idx)
                return
        self._update_session_summary()

    def _update_session_summary(self) -> None:
        if self._current_mode() == "sandbox":
            self.session_summary_label.setText(
                "Sandbox mode is active. The current Configure Run values will be used directly and nothing will be loaded from an experiment profile."
            )
            return
        experiment = self._find_experiment(str(self.existing_experiment_combo.currentData() or ""))
        subject = self._find_subject(str(self.existing_subject_combo.currentData() or ""))
        if experiment is None:
            count = len(self._experiments)
            self.session_summary_label.setText(
                f"No experiment selected. {count} saved experiment{'s' if count != 1 else ''} available."
            )
            return
        subject_count = len(experiment.subject_ids)
        summary = f"Experiment '{experiment.name}' selected. {subject_count} subject{'s' if subject_count != 1 else ''} assigned."
        if subject is not None:
            summary += f" Subject '{subject.subject_id}' is active for this session."
        else:
            summary += " Select a subject to auto-fill the mouse ID."
        self.session_summary_label.setText(summary)

    def _apply_experiment_config(self, cfg: dict[str, object]) -> None:
        text_fields = {
            "task_cfg": self.task_cfg_edit,
            "ds_cfg": self.cfg_edit,
            "width": self.width_edit,
            "height": self.height_edit,
            "fps": self.fps_edit,
            "arduino_fps": self.arduino_fps_edit,
            "serial_port": self.serial_port_edit,
            "serial_baud": self.serial_baud_edit,
            "socket_path": self.socket_edit,
            "socket_path_2": self.socket2_edit,
            "bitrate": self.bitrate_edit,
            "exposure_us": self.exposure_edit,
            "zed_texture_confidence_threshold": self.zed_texture_conf_edit,
            "zed_depth_minimum_distance_mm": self.zed_depth_min_mm_edit,
            "zed_depth_maximum_distance_mm": self.zed_depth_max_mm_edit,
            "zed_depth_stabilization": self.zed_depth_stab_edit,
        }
        for key, widget in text_fields.items():
            if key in cfg and cfg.get(key) is not None:
                widget.setText(str(cfg[key]))

        if "pixel_format" in cfg:
            value = str(cfg["pixel_format"])
            if value in [self.pix_combo.itemText(i) for i in range(self.pix_combo.count())]:
                self.pix_combo.setCurrentText(value)
        if "capture_backend" in cfg:
            idx = self.capture_backend_combo.findData(str(cfg["capture_backend"]))
            if idx >= 0:
                self.capture_backend_combo.setCurrentIndex(idx)
        if "zed_depth_mode" in cfg:
            value = str(cfg["zed_depth_mode"]).upper()
            if value in [self.zed_depth_mode_combo.itemText(i) for i in range(self.zed_depth_mode_combo.count())]:
                self.zed_depth_mode_combo.setCurrentText(value)

        if "trigger_on" in cfg:
            self.trigger_chk.setChecked(bool(cfg["trigger_on"]))
        if "serial_enabled" in cfg:
            self.serial_enable.setChecked(bool(cfg["serial_enabled"]))
        if "inference_enabled" in cfg:
            self.inference_enable.setChecked(bool(cfg["inference_enabled"]))
        if "draw_skeleton" in cfg:
            self.skeleton_chk.setChecked(bool(cfg["draw_skeleton"]))
        if "zed_depth_enabled" in cfg:
            self.zed_depth_chk.setChecked(bool(cfg["zed_depth_enabled"]))
        if "zed_depth_record" in cfg:
            self.zed_depth_record_chk.setChecked(bool(cfg["zed_depth_record"]))
        if "zed_fill_mode" in cfg:
            self.zed_fill_mode_chk.setChecked(bool(cfg["zed_fill_mode"]))
        if "zed_confidence_threshold" in cfg:
            self.zed_confidence_95_chk.setChecked(int(cfg["zed_confidence_threshold"] or 0) >= 100)
        if "num_cameras" in cfg:
            self._set_detected_cameras(int(cfg["num_cameras"]), from_detection=False)
        if "mouse_id" in cfg and not self.mouse_id_edit.text().strip():
            self.mouse_id_edit.setText(str(cfg["mouse_id"]))
        self._on_backend_changed()
        self._on_inference_toggled(self.inference_enable.isChecked())

    def _refresh_existing_subjects(self, subject_ids: list[str]) -> None:
        current = str(self.existing_subject_combo.currentData() or "")
        self.existing_subject_combo.blockSignals(True)
        try:
            self.existing_subject_combo.clear()
            self.existing_subject_combo.addItem("Select subject…", "")
            allowed = set(subject_ids)
            for profile in self._subjects:
                if allowed and profile.subject_id not in allowed:
                    continue
                self.existing_subject_combo.addItem(profile.name, profile.subject_id)
            idx = self.existing_subject_combo.findData(current)
            if idx < 0:
                idx = 0
            self.existing_subject_combo.setCurrentIndex(idx)
        finally:
            self.existing_subject_combo.blockSignals(False)

    @QtCore.Slot()
    def _on_backend_changed(self) -> None:
        backend = str(self.capture_backend_combo.currentData() or "flir").lower().strip()
        flir_visible = backend == "flir"
        zed_visible = backend == "zed"

        self.flir_panel.setVisible(flir_visible)
        self.zed_panel.setVisible(zed_visible)

    @QtCore.Slot()
    def _on_continue_to_run_tab(self) -> None:
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(1)
        if self.ok_button is not None:
            self.ok_button.setEnabled(True)

    def _on_detect_sensors(self) -> None:
        if self._detect_thread is not None:
            return

        self.detect_btn.setEnabled(False)
        progress = QtWidgets.QProgressDialog("Detecting sensors...", "", 0, 0, self)
        progress.setWindowTitle("Detecting Sensors")
        progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        self._detect_progress = progress

        thread = QtCore.QThread(self)
        worker = SensorDetectWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_detect_sensors_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_detect_thread_finished)
        self._detect_worker = worker
        self._detect_thread = thread
        thread.start()

    @QtCore.Slot(int, int, int, str)
    def _on_detect_sensors_finished(self, count: int, flir_count: int, zed_count: int, warning: str) -> None:
        capped = max(0, min(2, count))
        self._set_detected_cameras(capped, from_detection=True)
        zed_sources = 2 if zed_count > 0 else 0
        self.detected_sources_label.setText(
            f"FLIR: {flir_count} | ZED devices: {zed_count} | ZED usable sources: {zed_sources}"
        )
        if zed_count > 0 and flir_count == 0:
            idx = self.capture_backend_combo.findData("zed")
            if idx >= 0:
                self.capture_backend_combo.setCurrentIndex(idx)
        elif flir_count > 0 and zed_count == 0:
            idx = self.capture_backend_combo.findData("flir")
            if idx >= 0:
                self.capture_backend_combo.setCurrentIndex(idx)
        if zed_count > 0 and str(self.capture_backend_combo.currentData() or "").lower() == "zed":
            self._apply_zed_autoconfig_defaults()
            QtWidgets.QMessageBox.information(
                self,
                "ZED Defaults Applied",
                "ZED was detected, so capture defaults were prefilled:\n"
                "1920x1200 @ 30 FPS, trigger off, GRAY8, /tmp/cam.sock + /tmp/cam1.sock,\n"
                "and batch-size=2 config when available.\n\n"
                "You can change any value before starting.",
            )
        if warning:
            QtWidgets.QMessageBox.information(self, "Sensor Detection Notes", warning)
        if count > 2:
            QtWidgets.QMessageBox.information(
                self,
                "Sensor Detection",
                f"Detected {count} cameras. SqueakView currently uses up to 2 cameras.",
            )
        if count == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Sensors Detected",
                "No sensors detected, please ensure sensors are cprrectly plugged in.\n"
                "If you are attempting to use ZED sensors, please close SqueakView, run:\n\n"
                "sudo service zed_x_daemon restart\n\n"
                "Then reopen SqueakView.",
            )
        self._on_backend_changed()

    @QtCore.Slot()
    def _on_detect_thread_finished(self) -> None:
        if self._detect_progress is not None:
            self._detect_progress.close()
            self._detect_progress.deleteLater()
            self._detect_progress = None
        self.detect_btn.setEnabled(True)
        self._detect_worker = None
        self._detect_thread = None

    def _on_browse_cfg(self) -> None:
        start_dir = Path(self.cfg_edit.text()).parent if self.cfg_edit.text() else squeakview_config.DEEPSTREAM_ROOT
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select DeepStream nvinfer config",
            str(start_dir),
            "DeepStream config (*.txt *.cfg);;All files (*)",
        )
        if path:
            self.cfg_edit.setText(path)

    def _on_browse_task_cfg(self) -> None:
        start_dir = (
            Path(self.task_cfg_edit.text()).parent
            if self.task_cfg_edit.text()
            else squeakview_config.TASKS_DIR
        )
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select task config",
            str(start_dir),
            "Task config (*.yaml *.yml *.json);;All files (*)",
        )
        if path:
            self.task_cfg_edit.setText(path)

    def _collect_config(self, *, show_errors: bool, include_mode: bool = True) -> dict | None:
        try:
            width = int(self.width_edit.text()) or 1280
            height = int(self.height_edit.text()) or 720
            fps = int(self.fps_edit.text()) or 30
            bitrate = int(self.bitrate_edit.text()) or 4000
            arduino_fps = int(self.arduino_fps_edit.text()) or 30
            serial_baud = int(self.serial_baud_edit.text()) or 115200
            exposure_us = int(self.exposure_edit.text()) if self.exposure_edit.text() else 10000
            zed_texture_conf = int(self.zed_texture_conf_edit.text()) if self.zed_texture_conf_edit.text() else 100
            zed_depth_min_mm = int(self.zed_depth_min_mm_edit.text()) if self.zed_depth_min_mm_edit.text() else 300
            zed_depth_max_mm = int(self.zed_depth_max_mm_edit.text()) if self.zed_depth_max_mm_edit.text() else 20000
            zed_depth_stab = int(self.zed_depth_stab_edit.text()) if self.zed_depth_stab_edit.text() else 30
        except ValueError:
            if show_errors:
                QtWidgets.QMessageBox.warning(self, "Invalid input", "Please enter valid numeric values for size, FPS, bitrate, and baud.")
            return None

        result = {
            "width": width,
            "height": height,
            "fps": fps,
            "pixel_format": self.pix_combo.currentText() or "Mono8",
            "capture_backend": str(self.capture_backend_combo.currentData() or "flir"),
            "zed_depth_enabled": self.zed_depth_chk.isChecked(),
            "zed_depth_mode": self.zed_depth_mode_combo.currentText() or "NEURAL",
            "zed_depth_socket": "/tmp/cam_depth.sock",
            "zed_depth_record": self.zed_depth_record_chk.isChecked(),
            "zed_confidence_threshold": 100 if self.zed_confidence_95_chk.isChecked() else 0,
            "zed_texture_confidence_threshold": max(0, min(100, zed_texture_conf)),
            "zed_depth_minimum_distance_mm": max(100, min(3000, zed_depth_min_mm)),
            "zed_depth_maximum_distance_mm": max(500, min(40000, zed_depth_max_mm)),
            "zed_fill_mode": self.zed_fill_mode_chk.isChecked(),
            "zed_depth_stabilization": max(0, min(100, zed_depth_stab)),
            "trigger_on": self.trigger_chk.isChecked(),
            "exposure_us": exposure_us,
            "arduino_fps": arduino_fps,
            "serial_enabled": self.serial_enable.isChecked(),
            "serial_port": self.serial_port_edit.text().strip() or "/dev/ttyACM0",
            "serial_baud": serial_baud,
            "ds_cfg": Path(self.cfg_edit.text().strip()) if self.inference_enable.isChecked() else None,
            "inference_enabled": self.inference_enable.isChecked(),
            "draw_skeleton": self.skeleton_chk.isChecked(),
            "task_cfg": Path(self.task_cfg_edit.text().strip()),
            "socket_path": self.socket_edit.text().strip() or "/tmp/cam.sock",
            "socket_path_2": self.socket2_edit.text().strip() or "/tmp/cam1.sock",
            "num_cameras": int(self._detected_cameras),
            "bitrate": bitrate,
            "mouse_id": self.mouse_id_edit.text().strip(),
        }
        if include_mode:
            result["experiment_mode"] = self._current_mode()
            result["experiment_name"] = ""
            if self._current_mode() == "existing":
                result["experiment_name"] = str(self.existing_experiment_combo.currentData() or "")
            if self._current_mode() == "existing" and not result["experiment_name"]:
                if show_errors:
                    QtWidgets.QMessageBox.warning(self, "Experiment required", "Please select an experiment or switch to sandbox mode.")
                return None
        if result["num_cameras"] < 1:
            if show_errors:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Sensors Detected",
                    "No cameras are currently detected.\n\nOpen the 'Configure Sensor Source' tab and click 'Detect Sensors'.",
                )
            return None
        if result["inference_enabled"]:
            if not result["ds_cfg"]:
                if show_errors:
                    QtWidgets.QMessageBox.warning(self, "Config missing", "DeepStream config is required when inference is enabled.")
                return None
            if not result["ds_cfg"].exists():
                if show_errors:
                    QtWidgets.QMessageBox.warning(self, "Config missing", f"DeepStream config not found:\n{result['ds_cfg']}")
                return None
        if not str(result["task_cfg"]):
            if show_errors:
                QtWidgets.QMessageBox.warning(self, "Task config required", "Please select a task config before starting.")
            return None
        if not result["task_cfg"].exists():
            if show_errors:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Task config missing",
                    f"Task config not found:\n{result['task_cfg']}",
                )
            return None
        return result

    def accept(self) -> None:
        result = self._collect_config(show_errors=True, include_mode=True)
        if result is None:
            return
        self._result = result
        super().accept()

    @property
    def result_config(self) -> dict | None:
        return self._result

    @property
    def experiment_profile_name(self) -> str:
        if self.experiment_profile_name_edit is None:
            return ""
        return self.experiment_profile_name_edit.text().strip()
