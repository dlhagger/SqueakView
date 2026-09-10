from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from squeakview.apps.operator.gui.main_window import PreviewWidget as LegacyPreviewWidget
from squeakview.apps.operator.gui.preview import PreviewWidget


class PreviewWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.widget = PreviewWidget()

    def tearDown(self) -> None:
        self.widget.close()
        self.widget.deleteLater()
        self.app.processEvents()

    def test_main_window_keeps_legacy_import_surface(self) -> None:
        self.assertIs(LegacyPreviewWidget, PreviewWidget)

    def test_status_and_info_overlays_preserve_public_behavior(self) -> None:
        self.widget.set_status("Recording")
        self.widget.set_info("Camera 1")

        self.assertEqual(self.widget.status_badge.text(), "Recording")
        self.assertIn("#c4425f", self.widget.status_badge.styleSheet())
        self.assertEqual(self.widget.info_label.text(), "Camera 1")
        self.assertFalse(self.widget.info_label.isHidden())

        self.widget.set_info(None)
        self.assertTrue(self.widget.info_label.isHidden())

    def test_preview_enablement_switches_hint_and_logo_state(self) -> None:
        self.widget.set_preview_enabled(False)
        self.assertFalse(self.widget._preview_enabled)
        self.assertEqual(self.widget.label.text(), "Preview disabled")
        self.assertFalse(self.widget.label.isHidden())
        self.assertFalse(self.widget.logo_label.isHidden())

        self.widget.set_preview_enabled(True)
        self.assertTrue(self.widget._preview_enabled)
        self.assertTrue(self.widget.label.isHidden())
        self.assertTrue(self.widget.logo_label.isHidden())


if __name__ == "__main__":
    unittest.main()
