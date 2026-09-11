from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview import config as squeakview_config
from squeakview import model_package
from squeakview.apps.operator.gui.config_policy import ConfigFields, collect_config
from squeakview.apps.operator.gui.model_catalog import (
    enumerate_model_configs,
    validate_production_model,
)
from squeakview.apps.operator.gui.config_view import (
    ConfigViewCallbacks,
    build_config_view,
)
from squeakview.apps.operator.gui.dialog_style import fit_window_to_available_area
from squeakview.common.profiles import ExperimentProfile, ProfileStore, SubjectProfile, slugify

from squeakview.apps.operator.gui.session_dialog import (
    DARK_DIALOG_STYLE,
    CreateExperimentDialog,
    CreateSubjectDialog,
    SessionLauncherDialog,
    _dark_item_dialog,
    apply_dark_combo_popups,
    center_window,
)


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
        self.setMinimumSize(560, 520)
        self.resize(700, 760)

        self.setStyleSheet(DARK_DIALOG_STYLE)
        cfg = config or {}
        self._profile_store = ProfileStore()
        self._experiments = self._profile_store.list_experiments()
        self._subjects = self._profile_store.list_subjects()
        self._camera_count = 1
        self._capture_backend = "flir_direct"
        self._mouse_id = str(cfg.get("mouse_id", "")).strip()
        self._show_experiment_profile_editor = show_experiment_profile_editor

        view = build_config_view(
            self,
            config=cfg,
            experiments=self._experiments,
            subjects=self._subjects,
            callbacks=ConfigViewCallbacks(
                create_experiment=self._on_create_experiment,
                create_subject=self._on_create_subject_for_selected_experiment,
                add_subject=self._on_add_existing_subject_to_selected_experiment,
                browse_task_config=self._on_browse_task_cfg,
                browse_model_config=self._on_browse_cfg,
                accept=self.accept,
                reject=self.reject,
            ),
            tasks_dir=squeakview_config.TASKS_DIR,
            show_session_setup=show_session_setup,
            experiment_profile_name=experiment_profile_name,
            show_experiment_profile_editor=show_experiment_profile_editor,
        )
        self._run_form = view.run_form
        self.mode_combo = view.mode_combo
        self.existing_experiment_combo = view.existing_experiment_combo
        self.new_experiment_btn = view.new_experiment_btn
        self.existing_subject_combo = view.existing_subject_combo
        self.new_subject_btn = view.new_subject_btn
        self.add_subject_btn = view.add_subject_btn
        self.width_edit = view.width_edit
        self.height_edit = view.height_edit
        self.fps_edit = view.fps_edit
        self.trigger_chk = view.trigger_chk
        self.arduino_fps_edit = view.arduino_fps_edit
        self.serial_enable = view.serial_enable
        self.inference_enable = view.inference_enable
        self.task_cfg_edit = view.task_cfg_edit
        self.serial_port_edit = view.serial_port_edit
        self.serial_baud_edit = view.serial_baud_edit
        self.serial_row_widget = view.serial_row_widget
        self.cfg_edit = view.cfg_edit
        self.cfg_browse_btn = view.cfg_browse_btn
        self.cfg_label = view.cfg_label
        self.cfg_row_widget = view.cfg_row_widget
        self.model_combo = QtWidgets.QComboBox(self.cfg_row_widget)
        self.model_combo.setMinimumContentsLength(24)
        self.cfg_row_widget.layout().insertWidget(0, self.model_combo, 1)
        self._populate_model_combo()
        self.flir_panel = view.flir_panel
        self.pix_combo = view.pix_combo
        self.exposure_edit = view.exposure_edit
        self.bitrate_edit = view.bitrate_edit
        self.experiment_profile_name_edit = view.experiment_profile_name_edit
        self.session_hint_label = view.session_hint_label
        self.session_summary_label = view.session_summary_label
        self._session_group = view.session_group
        existing_slug = str(cfg.get("experiment_name", ""))
        self._result: dict | None = None
        self.inference_enable.toggled.connect(self._on_inference_toggled)
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.existing_experiment_combo.currentIndexChanged.connect(self._on_existing_experiment_changed)
        self.existing_subject_combo.currentIndexChanged.connect(self._on_existing_subject_changed)
        self._on_inference_toggled(self.inference_enable.isChecked())
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
        apply_dark_combo_popups(self)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        fit_window_to_available_area(self)
        center_window(self)

    @QtCore.Slot(bool)
    def _on_inference_toggled(self, enabled: bool) -> None:
        self.cfg_label.setVisible(enabled)
        self.cfg_row_widget.setVisible(enabled)

    def _populate_model_combo(self) -> None:
        """Offer complete local packages without exposing build staging dirs."""

        current = squeakview_config.resolve_workspace_path(self.cfg_edit.text())
        self.model_combo.blockSignals(True)
        try:
            self.model_combo.clear()
            self.model_combo.addItem("Select validated model…", "")
            selected = -1
            for choice in enumerate_model_configs(squeakview_config.MODEL_ROOT):
                if choice.eligible:
                    label = f"{choice.name} — {choice.config.name}"
                    self.model_combo.addItem(label, str(choice.config))
                    index = self.model_combo.count() - 1
                    self.model_combo.setItemData(
                        index, choice.detail, QtCore.Qt.ItemDataRole.ToolTipRole
                    )
                    if current is not None and choice.config == current:
                        selected = index
                else:
                    self.model_combo.addItem(
                        f"⚠ {choice.name} — unavailable", ""
                    )
                    index = self.model_combo.count() - 1
                    self.model_combo.setItemData(
                        index, choice.detail, QtCore.Qt.ItemDataRole.ToolTipRole
                    )
                    item = self.model_combo.model().item(index)
                    if item is not None:
                        item.setEnabled(False)
            self.model_combo.addItem("Manual path / Browse…", "__manual__")
            if selected >= 0:
                self.model_combo.setCurrentIndex(selected)
            elif current is not None:
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
        finally:
            self.model_combo.blockSignals(False)

    @QtCore.Slot(int)
    def _on_model_selected(self, index: int) -> None:
        selected = str(self.model_combo.itemData(index) or "")
        if selected and selected != "__manual__":
            self.cfg_edit.setText(selected)

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
        dialog = CreateSubjectDialog(self, experiment_name=experiment.name)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        subject_id = dialog.subject_id
        existing = self._find_subject(subject_id)
        subject_name = existing.name if existing else dialog.subject_name
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
        self._mouse_id = profile.subject_id
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
            "bitrate": self.bitrate_edit,
            "exposure_us": self.exposure_edit,
        }
        for key, widget in text_fields.items():
            if key in cfg and cfg.get(key) is not None:
                widget.setText(str(cfg[key]))

        if "pixel_format" in cfg:
            value = str(cfg["pixel_format"])
            if value in [self.pix_combo.itemText(i) for i in range(self.pix_combo.count())]:
                self.pix_combo.setCurrentText(value)
        if "trigger_on" in cfg:
            self.trigger_chk.setChecked(bool(cfg["trigger_on"]))
        if "serial_enabled" in cfg:
            self.serial_enable.setChecked(bool(cfg["serial_enabled"]))
        if "inference_enabled" in cfg:
            self.inference_enable.setChecked(bool(cfg["inference_enabled"]))
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

    def _on_browse_cfg(self) -> None:
        cfg_path = squeakview_config.resolve_workspace_path(self.cfg_edit.text())
        start_dir = cfg_path.parent if cfg_path else squeakview_config.MODEL_ROOT
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select DeepStream nvinfer config",
            str(start_dir),
            "DeepStream config (*.txt *.cfg);;All files (*)",
        )
        if path:
            self.cfg_edit.setText(path)

    def _on_browse_task_cfg(self) -> None:
        task_path = squeakview_config.resolve_workspace_path(self.task_cfg_edit.text())
        start_dir = task_path.parent if task_path else squeakview_config.TASKS_DIR
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select task config",
            str(start_dir),
            "Task config (*.yaml *.yml *.json);;All files (*)",
        )
        if path:
            self.task_cfg_edit.setText(path)

    def _collect_config(self, *, show_errors: bool, include_mode: bool = True) -> dict | None:
        collection = collect_config(
            ConfigFields(
                width=self.width_edit.text(),
                height=self.height_edit.text(),
                fps=self.fps_edit.text(),
                bitrate=self.bitrate_edit.text(),
                arduino_fps=self.arduino_fps_edit.text(),
                serial_baud=self.serial_baud_edit.text(),
                exposure_us=self.exposure_edit.text(),
                pixel_format=self.pix_combo.currentText(),
                capture_backend=self._capture_backend,
                trigger_enabled=self.trigger_chk.isChecked(),
                serial_enabled=self.serial_enable.isChecked(),
                serial_port=self.serial_port_edit.text(),
                inference_enabled=self.inference_enable.isChecked(),
                ds_cfg=self.cfg_edit.text(),
                task_cfg=self.task_cfg_edit.text(),
                camera_count=self._camera_count,
                mouse_id=self._mouse_id,
                experiment_mode=self._current_mode(),
                experiment_name=str(self.existing_experiment_combo.currentData() or ""),
            ),
            include_mode=include_mode,
            resolve_path=squeakview_config.resolve_workspace_path,
            validate_model=lambda path: validate_production_model(
                path, validator=model_package.validate_model_package
            ),
        )
        if collection.error is not None:
            if show_errors:
                QtWidgets.QMessageBox.warning(
                    self,
                    collection.error.title,
                    collection.error.message,
                )
            return None
        return collection.config

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
