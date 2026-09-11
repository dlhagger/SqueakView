from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview.common.profiles import ProfileStore

from .dialog_style import (
    DARK_DIALOG_STYLE,
    _meta_label,
    _size_button,
    apply_dark_combo_popups,
    center_window,
    fit_window_to_available_area,
)
from .experiment_dialog import ExperimentSessionMixin
from .subject_dialog import SubjectSessionMixin


class SessionLauncherDialog(ExperimentSessionMixin, SubjectSessionMixin, QtWidgets.QDialog):
    def __init__(
        self,
        parent=None,
        *,
        base_config: Optional[dict] = None,
        profile_store: ProfileStore | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Start Session")
        self.setModal(True)
        self.setMinimumWidth(640)
        self._base_config = dict(base_config or {})
        self._profile_store = profile_store or ProfileStore()
        self._experiments = self._profile_store.list_experiments()
        self._subjects = self._profile_store.list_subjects()
        self._result_config: dict | None = None

        self.setMinimumSize(640, 520)
        self.resize(840, 780)
        self.setStyleSheet(
            DARK_DIALOG_STYLE
            + """
            QFrame#launcherShell {
                background-color: #0f1118;
            }
            QFrame#launcherCard {
                background-color: #14192a;
                border: 1px solid #27304a;
                border-radius: 12px;
            }
            QFrame#summaryPanel {
                background-color: #101526;
                border: 1px solid #2c3550;
                border-radius: 8px;
            }
            QLabel#launcherTitle {
                color: #eef1ff;
                font-size: 24px;
                font-weight: 800;
            }
            QLabel#launcherSubtitle {
                color: #9aa7cc;
                font-size: 13px;
            }
            QLabel#stepBadge {
                background-color: #2f4daa;
                color: #ffffff;
                border-radius: 10px;
                padding: 2px 8px;
                font-weight: 800;
                font-size: 12px;
            }
            QLabel#cardTitle {
                color: #eef1ff;
                font-size: 16px;
                font-weight: 800;
            }
            QLabel#cardSubtitle, QLabel#fieldLabel, QLabel#summaryKey {
                color: #9aa7cc;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#summaryValue {
                color: #e8ecff;
                font-size: 13px;
            }
            QLabel#subjectSummary {
                color: #e8ecff;
                background-color: #101526;
                border: 1px solid #2c3550;
                border-radius: 8px;
                padding: 10px 12px;
            }
            QPushButton#launcherTertiaryButton {
                background-color: #22283b;
                border-color: #343d5b;
                min-height: 34px;
                padding: 6px 14px;
            }
            QPushButton#launcherTertiaryButton:hover {
                background-color: #2c344d;
            }
            QPushButton#launcherDangerButton {
                background-color: #342637;
                border-color: #5e3b55;
                color: #f0c2d0;
                min-height: 34px;
                padding: 6px 14px;
            }
            QPushButton#launcherDangerButton:hover {
                background-color: #493047;
            }
            QPushButton#launcherDangerButton:disabled {
                background-color: #242535;
                border-color: #303247;
                color: #686d82;
            }
            QPushButton#launcherPrimaryButton {
                background-color: #5c6df5;
                border-color: #5c6df5;
                min-height: 38px;
                padding: 8px 18px;
            }
            QPushButton#launcherPrimaryButton:disabled {
                background-color: #2a3248;
                border-color: #313c5d;
                color: #7f8aac;
            }
            """
        )

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(22, 22, 22, 18)
        outer_layout.setSpacing(16)

        header = QtWidgets.QFrame(self)
        header.setObjectName("launcherShell")
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)
        title = QtWidgets.QLabel("Start Session", self)
        title.setObjectName("launcherTitle")
        subtitle = QtWidgets.QLabel("Select an experiment profile and subject before opening the recording workspace.", self)
        subtitle.setObjectName("launcherSubtitle")
        subtitle.setWordWrap(True)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        outer_layout.addWidget(header)

        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setObjectName("launcherScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll_content = QtWidgets.QWidget(scroll_area)
        scroll_content.setObjectName("launcherScrollContent")
        scroll_content.setStyleSheet(
            "QWidget#launcherScrollContent { background-color: #0f1118; }"
        )
        layout = QtWidgets.QVBoxLayout(scroll_content)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(16)

        experiment_card = QtWidgets.QFrame(self)
        experiment_card.setObjectName("launcherCard")
        experiment_card.setMinimumHeight(320)
        experiment_layout = QtWidgets.QVBoxLayout(experiment_card)
        experiment_layout.setContentsMargins(18, 16, 18, 16)
        experiment_layout.setSpacing(12)
        experiment_header = QtWidgets.QHBoxLayout()
        experiment_header.setSpacing(10)
        experiment_badge = QtWidgets.QLabel("1", self)
        experiment_badge.setObjectName("stepBadge")
        experiment_title_box = QtWidgets.QVBoxLayout()
        experiment_title_box.setContentsMargins(0, 0, 0, 0)
        experiment_title_box.setSpacing(2)
        experiment_title = QtWidgets.QLabel("Experiment", self)
        experiment_title.setObjectName("cardTitle")
        experiment_subtitle = QtWidgets.QLabel("Loads saved camera, model, task, trigger, and serial defaults.", self)
        experiment_subtitle.setObjectName("cardSubtitle")
        experiment_subtitle.setWordWrap(True)
        experiment_title_box.addWidget(experiment_title)
        experiment_title_box.addWidget(experiment_subtitle)
        experiment_header.addWidget(experiment_badge, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        experiment_header.addLayout(experiment_title_box, 1)
        experiment_layout.addLayout(experiment_header)

        experiment_label = QtWidgets.QLabel("Saved experiment", self)
        experiment_label.setObjectName("fieldLabel")
        self.experiment_combo = QtWidgets.QComboBox(self)
        self.experiment_combo.setMinimumHeight(38)
        experiment_layout.addWidget(experiment_label)
        experiment_layout.addWidget(self.experiment_combo)

        exp_actions = QtWidgets.QHBoxLayout()
        exp_actions.setSpacing(8)
        exp_actions.addStretch(1)
        self.create_experiment_btn = QtWidgets.QPushButton("New", self)
        self.create_experiment_btn.setObjectName("launcherTertiaryButton")
        self.create_experiment_btn.clicked.connect(self._on_create_experiment)
        self.edit_experiment_btn = QtWidgets.QPushButton("Edit", self)
        self.edit_experiment_btn.setObjectName("launcherTertiaryButton")
        self.edit_experiment_btn.clicked.connect(self._on_edit_experiment)
        self.delete_experiment_btn = QtWidgets.QPushButton("Delete", self)
        self.delete_experiment_btn.setObjectName("launcherDangerButton")
        self.delete_experiment_btn.clicked.connect(self._on_delete_experiment)
        _size_button(self.create_experiment_btn, min_width=86, min_height=36)
        _size_button(self.edit_experiment_btn, min_width=86, min_height=36)
        _size_button(self.delete_experiment_btn, min_width=92, min_height=36)
        exp_actions.addWidget(self.create_experiment_btn)
        exp_actions.addWidget(self.edit_experiment_btn)
        exp_actions.addWidget(self.delete_experiment_btn)
        experiment_layout.addLayout(exp_actions)

        summary_panel = QtWidgets.QFrame(self)
        summary_panel.setObjectName("summaryPanel")
        summary_panel.setMinimumHeight(112)
        summary_layout = QtWidgets.QGridLayout(summary_panel)
        summary_layout.setContentsMargins(12, 10, 12, 10)
        summary_layout.setHorizontalSpacing(10)
        summary_layout.setVerticalSpacing(4)
        self.experiment_name_value = _meta_label("No experiment selected")
        self.experiment_camera_value = _meta_label("-")
        self.experiment_task_value = _meta_label("-")
        self.experiment_model_value = _meta_label("-")
        for row, (key, value) in enumerate(
            [
                ("Profile", self.experiment_name_value),
                ("Camera", self.experiment_camera_value),
                ("Task", self.experiment_task_value),
                ("Model", self.experiment_model_value),
            ]
        ):
            key_label = QtWidgets.QLabel(key, self)
            key_label.setObjectName("summaryKey")
            value.setObjectName("summaryValue")
            summary_layout.addWidget(key_label, row, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            summary_layout.addWidget(value, row, 1)
        summary_layout.setColumnStretch(1, 1)
        self.experiment_summary_label = self.experiment_name_value
        experiment_layout.addWidget(summary_panel)
        layout.addWidget(experiment_card)

        subject_card = QtWidgets.QFrame(self)
        subject_card.setObjectName("launcherCard")
        subject_card.setMinimumHeight(245)
        subject_layout = QtWidgets.QVBoxLayout(subject_card)
        subject_layout.setContentsMargins(18, 16, 18, 16)
        subject_layout.setSpacing(12)
        subject_header = QtWidgets.QHBoxLayout()
        subject_header.setSpacing(10)
        subject_badge = QtWidgets.QLabel("2", self)
        subject_badge.setObjectName("stepBadge")
        subject_title_box = QtWidgets.QVBoxLayout()
        subject_title_box.setContentsMargins(0, 0, 0, 0)
        subject_title_box.setSpacing(2)
        subject_title = QtWidgets.QLabel("Subject", self)
        subject_title.setObjectName("cardTitle")
        subject_subtitle = QtWidgets.QLabel("Chooses the subject identity attached to this recording run.", self)
        subject_subtitle.setObjectName("cardSubtitle")
        subject_subtitle.setWordWrap(True)
        subject_title_box.addWidget(subject_title)
        subject_title_box.addWidget(subject_subtitle)
        subject_header.addWidget(subject_badge, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        subject_header.addLayout(subject_title_box, 1)
        subject_layout.addLayout(subject_header)

        subject_label = QtWidgets.QLabel("Assigned subject", self)
        subject_label.setObjectName("fieldLabel")
        self.subject_combo = QtWidgets.QComboBox(self)
        self.subject_combo.setMinimumHeight(38)
        subject_layout.addWidget(subject_label)
        subject_layout.addWidget(self.subject_combo)

        subject_actions = QtWidgets.QHBoxLayout()
        subject_actions.setSpacing(8)
        subject_actions.addStretch(1)
        self.create_subject_btn = QtWidgets.QPushButton("New", self)
        self.create_subject_btn.setObjectName("launcherTertiaryButton")
        self.create_subject_btn.clicked.connect(self._on_create_subject)
        self.add_existing_subject_btn = QtWidgets.QPushButton("Add Existing", self)
        self.add_existing_subject_btn.setObjectName("launcherTertiaryButton")
        self.add_existing_subject_btn.clicked.connect(self._on_add_existing_subject)
        self.edit_subject_btn = QtWidgets.QPushButton("Edit", self)
        self.edit_subject_btn.setObjectName("launcherTertiaryButton")
        self.edit_subject_btn.clicked.connect(self._on_edit_subject)
        self.delete_subject_btn = QtWidgets.QPushButton("Delete", self)
        self.delete_subject_btn.setObjectName("launcherDangerButton")
        self.delete_subject_btn.clicked.connect(self._on_delete_subject)
        _size_button(self.create_subject_btn, min_width=86, min_height=36)
        _size_button(self.add_existing_subject_btn, min_width=126, min_height=36)
        _size_button(self.edit_subject_btn, min_width=86, min_height=36)
        _size_button(self.delete_subject_btn, min_width=92, min_height=36)
        subject_actions.addWidget(self.create_subject_btn)
        subject_actions.addWidget(self.add_existing_subject_btn)
        subject_actions.addWidget(self.edit_subject_btn)
        subject_actions.addWidget(self.delete_subject_btn)
        subject_layout.addLayout(subject_actions)

        self.subject_summary_label = QtWidgets.QLabel("", self)
        self.subject_summary_label.setObjectName("subjectSummary")
        self.subject_summary_label.setWordWrap(True)
        self.subject_summary_label.setMinimumHeight(42)
        subject_layout.addWidget(self.subject_summary_label)
        layout.addWidget(subject_card)
        layout.addStretch(1)
        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area, 1)

        footer = QtWidgets.QHBoxLayout()
        footer.setSpacing(10)
        footer.addStretch(1)
        self.cancel_btn = QtWidgets.QPushButton("Cancel", self)
        self.cancel_btn.setObjectName("launcherTertiaryButton")
        self.cancel_btn.clicked.connect(self.reject)
        self.continue_btn = QtWidgets.QPushButton("Continue", self)
        self.continue_btn.setObjectName("launcherPrimaryButton")
        self.continue_btn.clicked.connect(self.accept)
        _size_button(self.cancel_btn, min_width=112, min_height=40)
        _size_button(self.continue_btn, min_width=124, min_height=40)
        footer.addWidget(self.cancel_btn)
        footer.addWidget(self.continue_btn)
        outer_layout.addLayout(footer)

        self.experiment_combo.currentIndexChanged.connect(self._on_experiment_changed)
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        self._reload_profiles()
        apply_dark_combo_popups(self)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        fit_window_to_available_area(self)
        center_window(self)

    @property
    def result_config(self) -> dict | None:
        return self._result_config

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
