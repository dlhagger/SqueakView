from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from squeakview.apps.operator.gui.main_theme import (
    MAIN_WINDOW_STYLESHEET,
    apply_main_window_theme,
)
from squeakview.apps.operator.gui.main_window import MainWindow


class MainThemeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_theme_keeps_critical_scientific_run_state_selectors(self) -> None:
        self.assertIn("QLabel#runStateBadge", MAIN_WINDOW_STYLESHEET)
        self.assertIn("QLabel#captureHealth", MAIN_WINDOW_STYLESHEET)
        self.assertIn("QFrame#stopOverlay", MAIN_WINDOW_STYLESHEET)
        self.assertIn("QPlainTextEdit#eventLog", MAIN_WINDOW_STYLESHEET)
        self.assertIn('QDockWidget[workspaceCard="true"]', MAIN_WINDOW_STYLESHEET)

    def test_theme_applies_to_plain_window_and_legacy_adapter(self) -> None:
        window = QtWidgets.QMainWindow()
        try:
            apply_main_window_theme(window)
            self.assertEqual(window.styleSheet(), MAIN_WINDOW_STYLESHEET)

            window.setStyleSheet("")
            MainWindow._apply_brand_theme(window)
            self.assertEqual(window.styleSheet(), MAIN_WINDOW_STYLESHEET)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
