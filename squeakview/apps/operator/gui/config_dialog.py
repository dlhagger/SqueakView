from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview import config as squeakview_config
from squeakview import model_package
from squeakview.common.profiles import ExperimentProfile, ProfileStore, SubjectProfile, slugify


DARK_DIALOG_STYLE = """
    QDialog {
        background-color: #0f1118;
    }
    QWidget {
        color: #d7ddf5;
        selection-background-color: #5967d8;
        selection-color: #ffffff;
    }
    QLabel {
        color: #d7ddf5;
        font-size: 13px;
        background: transparent;
    }
    QGroupBox {
        border: 1px solid #24283b;
        border-radius: 10px;
        background-color: #14192a;
        margin-top: 16px;
        padding: 16px 12px 12px 12px;
        color: #e7ebff;
        font-weight: 700;
    }
    QGroupBox::title {
        color: #aeb8ff;
        subcontrol-origin: margin;
        left: 12px;
        top: 8px;
        padding: 0 6px;
        background-color: #0f1118;
    }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
        background-color: #12172a;
        color: #e8ecff;
        border: 1px solid #333a55;
        border-radius: 6px;
        padding: 6px 8px;
        min-height: 24px;
        selection-background-color: #5967d8;
        selection-color: #ffffff;
    }
    QLineEdit {
        placeholder-text-color: #7f8aac;
    }
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QTextEdit:focus, QPlainTextEdit:focus {
        border-color: #6f7dff;
        background-color: #151b31;
    }
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
    QTextEdit:disabled, QPlainTextEdit:disabled {
        background-color: #171b29;
        color: #7f8aac;
        border-color: #2a3046;
    }
    QComboBox {
        padding-right: 30px;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 28px;
        border-left: 1px solid #333a55;
        border-top-right-radius: 6px;
        border-bottom-right-radius: 6px;
        background-color: #192039;
    }
    QComboBox::down-arrow {
        width: 10px;
        height: 10px;
    }
    QComboBox QAbstractItemView {
        background-color: #11162a;
        color: #e8ecff;
        border: 1px solid #333a55;
        selection-background-color: #5967d8;
        selection-color: #ffffff;
        outline: 0;
        padding: 4px;
    }
    QComboBox QAbstractItemView::item {
        min-height: 26px;
        padding: 4px 8px;
    }
    QCheckBox {
        color: #d7ddf5;
        spacing: 8px;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 1px solid #46506d;
        background-color: #12172a;
    }
    QCheckBox::indicator:checked {
        background-color: #5c6df5;
        border-color: #5c6df5;
    }
    QPushButton {
        background-color: #2f4daa;
        color: #ffffff;
        padding: 7px 14px;
        border-radius: 6px;
        border: 1px solid #3557bf;
        font-weight: 600;
        min-height: 24px;
    }
    QPushButton:hover {
        background-color: #3a5fc9;
    }
    QPushButton:pressed {
        background-color: #293f90;
    }
    QPushButton:disabled {
        background-color: #2a3248;
        color: #7f8aac;
        border-color: #313c5d;
    }
    QDialogButtonBox QPushButton {
        min-width: 84px;
    }
    QScrollArea, QAbstractScrollArea {
        background-color: #0f1118;
        border: none;
    }
    QMenu {
        background-color: #11162a;
        color: #e8ecff;
        border: 1px solid #333a55;
        padding: 4px;
    }
    QMenu::item {
        padding: 6px 18px;
    }
    QMenu::item:selected {
        background-color: #5967d8;
        color: #ffffff;
    }
    QMessageBox {
        background-color: #14192a;
        color: #d7ddf5;
    }
    QToolTip {
        background-color: #11162a;
        color: #eef1ff;
        border: 1px solid #333a55;
        padding: 4px 6px;
    }
"""


COMBO_POPUP_STYLE = """
    QAbstractScrollArea {
        background-color: #11162a;
        border: 0;
    }
    QAbstractScrollArea::viewport {
        background-color: #11162a;
        border: 0;
    }
    QListView {
        background-color: #11162a;
        color: #e8ecff;
        border: 0;
        outline: 0;
        padding: 4px;
        selection-background-color: #5967d8;
        selection-color: #ffffff;
    }
    QListView::item {
        min-height: 26px;
        padding: 4px 8px;
        border: 0;
    }
    QListView::item:selected {
        background-color: #5967d8;
        color: #ffffff;
    }
    QListView::item:hover {
        background-color: #283a7a;
        color: #ffffff;
    }
"""


def apply_dark_combo_popups(widget: QtWidgets.QWidget) -> None:
    for combo in widget.findChildren(QtWidgets.QComboBox):
        view = QtWidgets.QListView(combo)
        view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        view.setLineWidth(0)
        view.setMidLineWidth(0)
        view.setStyleSheet(COMBO_POPUP_STYLE)
        view.setUniformItemSizes(True)
        view.setSpacing(0)
        view.setAutoFillBackground(True)
        view.viewport().setAutoFillBackground(True)
        palette = view.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#11162a"))
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#11162a"))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#e8ecff"))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#5967d8"))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
        view.setPalette(palette)
        view.viewport().setPalette(palette)
        combo.setView(view)


def center_window(widget: QtWidgets.QWidget) -> None:
    parent = widget.parentWidget()
    if parent is not None and parent.isVisible():
        target = parent.frameGeometry()
    else:
        screen = widget.screen() or QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        target = screen.availableGeometry()
    frame = widget.frameGeometry()
    frame.moveCenter(target.center())
    widget.move(frame.topLeft())


def _size_button(button: QtWidgets.QPushButton, *, min_width: int = 96, min_height: int = 38) -> None:
    button.setMinimumWidth(min_width)
    button.setMinimumHeight(min_height)
    button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)


def _meta_label(text: str = "") -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
    return label


def _dark_item_dialog(
    parent: QtWidgets.QWidget,
    *,
    title: str,
    label: str,
    items: list[str],
) -> tuple[str, bool]:
    dialog = QtWidgets.QInputDialog(parent)
    dialog.setStyleSheet(DARK_DIALOG_STYLE)
    dialog.setWindowTitle(title)
    dialog.setLabelText(label)
    dialog.setComboBoxItems(items)
    dialog.setComboBoxEditable(False)
    dialog.setMinimumWidth(420)
    apply_dark_combo_popups(dialog)
    ok = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
    return dialog.textValue(), ok


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


class SessionLauncherDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, base_config: Optional[dict] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Start Session")
        self.setModal(True)
        self.setMinimumWidth(720)
        self._base_config = dict(base_config or {})
        self._profile_store = ProfileStore()
        self._experiments = self._profile_store.list_experiments()
        self._subjects = self._profile_store.list_subjects()
        self._result_config: dict | None = None

        self.setMinimumSize(840, 780)
        self.resize(860, 800)
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
            QPushButton#launcherPrimaryButton {
                background-color: #5c6df5;
                border-color: #5c6df5;
                min-height: 38px;
                padding: 8px 18px;
            }
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 18)
        layout.setSpacing(16)

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
        layout.addWidget(header)

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
        layout.addLayout(footer)

        self.experiment_combo.currentIndexChanged.connect(self._on_experiment_changed)
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        self._reload_profiles()
        apply_dark_combo_popups(self)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        center_window(self)

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

        self.setStyleSheet(DARK_DIALOG_STYLE)
        cfg = config or {}
        self._profile_store = ProfileStore()
        self._experiments = self._profile_store.list_experiments()
        self._subjects = self._profile_store.list_subjects()
        self._camera_count = 1
        self._capture_backend = "flir_direct"
        self._mouse_id = str(cfg.get("mouse_id", "")).strip()
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

        self.trigger_chk = QtWidgets.QCheckBox("Enable camera trigger")
        self.trigger_chk.setChecked(cfg.get("trigger_on", False))
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

        form.addRow("", self.flir_panel)

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
        if show_experiment_profile_editor:
            header_text = "Edit the experiment profile and the saved run defaults used when that experiment starts."
        elif show_session_setup:
            header_text = "Choose the session mode and set model, camera, trigger, serial, and task settings."
        else:
            header_text = "Set model, camera, trigger, serial, and task settings for this run or experiment profile."
        header = QtWidgets.QLabel(header_text)
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

        run_group = QtWidgets.QGroupBox("Run Configuration", self)
        run_layout = QtWidgets.QVBoxLayout(run_group)
        run_layout.setContentsMargins(12, 12, 12, 12)
        run_layout.addLayout(form)
        layout.addWidget(run_group)
        layout.addSpacing(12)
        layout.addWidget(button_box)

        self._result: dict | None = None
        self.inference_enable.toggled.connect(self._on_inference_toggled)
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
        center_window(self)

    @QtCore.Slot(bool)
    def _on_inference_toggled(self, enabled: bool) -> None:
        self.cfg_label.setVisible(enabled)
        self.cfg_row_widget.setVisible(enabled)

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
        try:
            width = int(self.width_edit.text()) or 1280
            height = int(self.height_edit.text()) or 720
            fps = int(self.fps_edit.text()) or 30
            bitrate = int(self.bitrate_edit.text()) or 4000
            arduino_fps = int(self.arduino_fps_edit.text()) or 30
            serial_baud = int(self.serial_baud_edit.text()) or 115200
            exposure_us = int(self.exposure_edit.text()) if self.exposure_edit.text() else 10000
        except ValueError:
            if show_errors:
                QtWidgets.QMessageBox.warning(self, "Invalid input", "Please enter valid numeric values for size, FPS, bitrate, and baud.")
            return None

        ds_cfg = (
            squeakview_config.resolve_workspace_path(self.cfg_edit.text().strip())
            if self.inference_enable.isChecked()
            else None
        )
        task_cfg = squeakview_config.resolve_workspace_path(self.task_cfg_edit.text().strip())

        result = {
            "width": width,
            "height": height,
            "fps": fps,
            "pixel_format": self.pix_combo.currentText() or "Mono8",
            "capture_backend": self._capture_backend,
            "trigger_on": self.trigger_chk.isChecked(),
            "exposure_us": exposure_us,
            "arduino_fps": arduino_fps,
            "serial_enabled": self.serial_enable.isChecked(),
            "serial_port": self.serial_port_edit.text().strip() or "/dev/ttyACM0",
            "serial_baud": serial_baud,
            "ds_cfg": ds_cfg,
            "inference_enabled": self.inference_enable.isChecked(),
            "task_cfg": task_cfg,
            "num_cameras": self._camera_count,
            "bitrate": bitrate,
            "mouse_id": self._mouse_id,
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
        if result["inference_enabled"]:
            if not result["ds_cfg"]:
                if show_errors:
                    QtWidgets.QMessageBox.warning(self, "Config missing", "DeepStream config is required when inference is enabled.")
                return None
            if not result["ds_cfg"].exists():
                if show_errors:
                    QtWidgets.QMessageBox.warning(self, "Config missing", f"DeepStream config not found:\n{result['ds_cfg']}")
                return None
            try:
                model_package.validate_model_package(result["ds_cfg"])
            except model_package.ModelPackageError as exc:
                if show_errors:
                    QtWidgets.QMessageBox.warning(self, "Invalid model package", str(exc))
                return None
        if not result["task_cfg"]:
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
