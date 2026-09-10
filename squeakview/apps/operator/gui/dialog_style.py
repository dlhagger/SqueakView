from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

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

