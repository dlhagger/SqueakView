from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from squeakview.apps.operator.gui.main_window import PreviewWidget as LegacyPreviewWidget
from squeakview.apps.operator.gui.preview import AspectRatioPreviewHost, PreviewWidget


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

    def test_sink_overlays_are_independent_native_child_windows(self) -> None:
        overlays = (
            self.widget.label,
            self.widget.logo_label,
            self.widget.status_badge,
            self.widget.info_label,
        )

        for overlay in overlays:
            self.assertTrue(
                overlay.testAttribute(QtCore.Qt.WidgetAttribute.WA_NativeWindow)
            )
            self.assertTrue(
                overlay.testAttribute(
                    QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
                )
            )

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

    def test_aspect_host_uses_largest_centered_four_three_surface(self) -> None:
        host = AspectRatioPreviewHost(self.widget)

        host.resize(1000, 400)
        host._fit_preview()
        self.assertEqual(self.widget.geometry(), QtCore.QRect(233, 0, 533, 400))

        host.resize(400, 600)
        host._fit_preview()
        self.assertEqual(self.widget.geometry(), QtCore.QRect(0, 150, 400, 300))

        host.close()
        host.deleteLater()

    def test_aspect_host_can_follow_a_different_camera_ratio(self) -> None:
        host = AspectRatioPreviewHost(self.widget)
        host.resize(1000, 400)

        host.set_aspect_ratio(16.0 / 9.0)

        self.assertEqual(self.widget.geometry(), QtCore.QRect(144, 0, 711, 400))
        host.close()
        host.deleteLater()

    def test_resizing_host_preserves_native_preview_window(self) -> None:
        host = AspectRatioPreviewHost(self.widget)
        host.resize(640, 480)
        host.show()
        self.app.processEvents()
        native_window_id = self.widget.window_id()

        host.resize(1000, 400)
        self.app.processEvents()

        self.assertEqual(self.widget.window_id(), native_window_id)
        self.assertEqual(self.widget.geometry(), QtCore.QRect(233, 0, 533, 400))
        host.close()
        host.deleteLater()


if __name__ == "__main__":
    unittest.main()
