"""Shared brand theme for the operator main window."""

from __future__ import annotations

from PySide6 import QtWidgets


MAIN_WINDOW_STYLESHEET = """
            QMainWindow {
                background-color: #171821;
                color: #e8ebf4;
            }
            QStatusBar {
                background-color: #171821;
                color: #e8ebf4;
            }
            QStatusBar QLabel {
                color: #e8ebf4;
            }
            QWidget {
                selection-background-color: #5967d8;
                selection-color: #ffffff;
            }
            QLabel {
                color: #e8ebf4;
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
            QGroupBox {
                border: 1px solid #2a2d3d;
                border-radius: 10px;
                background-color: #1a1d2a;
                margin-top: 16px;
                padding: 16px 12px 12px 12px;
            }
            QGroupBox::title {
                color: #aeb8ff;
                subcontrol-origin: margin;
                left: 14px;
                top: 10px;
                padding: 0 6px;
                background-color: #171821;
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
            QToolTip {
                background-color: #11162a;
                color: #eef1ff;
                border: 1px solid #333a55;
                padding: 4px 6px;
            }
            QFrame#runControlBar {
                background-color: #14192a;
                border: 1px solid #303751;
                border-radius: 10px;
            }
            QLabel#runStateBadge {
                background-color: #4357bd;
                color: #ffffff;
                border-radius: 8px;
                padding: 5px 10px;
                font-size: 11px;
                font-weight: 800;
            }
            QLabel#runIdentity {
                color: #eef1ff;
                font-size: 13px;
                font-weight: 700;
            }
            QLabel#runElapsed {
                color: #aeb8ff;
                font-family: monospace;
                font-size: 14px;
                font-weight: 800;
            }
            QLabel#captureHealth {
                color: #a7d9c2;
                font-size: 11px;
            }
            QDockWidget#eventDock {
                color: #eef1ff;
                background-color: #14192a;
                font-weight: 700;
            }
            QFrame#brandHeader {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2b2f46, stop:1 #202336);
                border: 1px solid #333650;
                border-radius: 10px;
            }
            QLabel#brandTitle {
                font-size: 24px;
                font-weight: 700;
                color: #ffffff;
            }
            QLabel#brandSubtitle {
                color: #aeb8ff;
                font-size: 13px;
            }
            QLabel#summaryBanner {
                background-color: rgba(46, 52, 80, 0.6);
                border: 1px solid #38405d;
                border-radius: 8px;
                padding: 10px;
                color: #e0e5ff;
            }
            QLabel#bottleStatus {
                color: #9aa7cc;
                font-size: 11px;
            }
            QLabel#taskTitle {
                color: #eef1ff;
                font-size: 14px;
                font-weight: 800;
            }
            QLabel#taskPath {
                color: #9aa7cc;
                font-size: 11px;
            }
            QTextBrowser#taskSummary {
                background-color: #101526;
                color: #d7ddf5;
                border: 1px solid #2c3550;
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton {
                padding: 8px 18px;
                border-radius: 6px;
                font-weight: 600;
                color: #e8ebf4;
                background-color: #2c3146;
                border: 1px solid #404663;
            }
            QPushButton:hover {
                background-color: #353b55;
            }
            QPushButton:disabled {
                background-color: #3b3f4f;
                color: #7d8299;
                border-color: #3b3f4f;
            }
            QPushButton#primaryButton {
                background-color: #5c6df5;
                border: 1px solid #5c6df5;
                color: white;
            }
            QPushButton#primaryButton:hover {
                background-color: #4959e6;
            }
            QPushButton#primaryButton:disabled {
                background-color: #2b3043;
                border-color: #343a50;
                color: #737b96;
            }
            QPushButton#dangerButton {
                background-color: #d9536f;
                border: 1px solid #d9536f;
            }
            QPushButton#dangerButton:hover {
                background-color: #c13d59;
            }
            QPushButton#dangerButton:disabled {
                background-color: #342a33;
                border-color: #44313b;
                color: #78636c;
            }
            QPushButton#secondaryButton {
                background-color: #353a4d;
            }
            QProgressBar {
                background-color: #11162a;
                border: 1px solid #333a55;
                border-radius: 6px;
                min-height: 10px;
            }
            QProgressBar::chunk {
                background-color: #5c6df5;
                border-radius: 5px;
            }
            QFrame#stopOverlay {
                background-color: rgba(10, 12, 20, 230);
                border: 1px solid #2a2d3d;
                border-radius: 12px;
            }
            QLabel#stopOverlayTitle {
                color: #eef1ff;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#stopOverlayMsg {
                color: #b8c0de;
                font-size: 13px;
            }
"""


def apply_main_window_theme(window: QtWidgets.QWidget) -> None:
    """Apply the canonical operator theme to a window-like widget."""

    window.setStyleSheet(MAIN_WINDOW_STYLESHEET)
