from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from squeakview.apps.operator.gui.dock_workspace import (
    DockWorkspace,
    LAYOUT_STATE_KEY,
    build_layout_menu,
)


class _MemorySettings:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.sync_count = 0

    def value(self, key: str, defaultValue=None):  # noqa: N803
        return self.values.get(key, defaultValue)

    def setValue(self, key: str, value) -> None:  # noqa: N802
        self.values[key] = value

    def sync(self) -> None:
        self.sync_count += 1


class DockWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.settings = _MemorySettings()
        self.workspace = self._workspace(self.settings)

    def tearDown(self) -> None:
        self.workspace._host.close()
        self.workspace.deleteLater()
        self.app.processEvents()

    @staticmethod
    def _workspace(settings: _MemorySettings) -> DockWorkspace:
        host = QtWidgets.QMainWindow()
        workspace = DockWorkspace(host, settings=settings)
        for card_id in (
            "preview",
            "system",
            "clock",
            "task",
            "bottles",
            "behavior",
            "events",
        ):
            workspace.add_card(card_id, card_id.title(), QtWidgets.QWidget())
        workspace.establish_default_layout()
        return workspace

    def test_default_is_locked_complete_and_hides_optional_events(self) -> None:
        self.assertFalse(self.workspace.layout_editable)
        self.assertEqual(len(self.workspace.cards), 7)
        self.assertTrue(self.workspace.cards["events"].isHidden())
        self.assertTrue(self.workspace.cards["clock"].isHidden())
        for dock in self.workspace.cards.values():
            self.assertEqual(
                dock.features(),
                QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable,
            )

    def test_user_unlock_is_overridden_only_while_runtime_is_locked(self) -> None:
        movable = QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        self.workspace.set_user_editable(True)
        self.assertTrue(self.workspace.layout_editable)
        self.assertTrue(self.workspace.cards["preview"].features() & movable)

        self.workspace.set_runtime_locked(True)
        self.assertFalse(self.workspace.layout_editable)
        self.assertFalse(self.workspace.cards["preview"].features() & movable)

        self.workspace.set_runtime_locked(False)
        self.assertTrue(self.workspace.layout_editable)

    def test_state_round_trip_and_corrupt_state_fallback(self) -> None:
        self.workspace.cards["events"].show()
        preview = self.workspace.cards["preview"]
        self.workspace._host.removeDockWidget(preview)
        self.workspace._host.addDockWidget(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea, preview
        )
        self.workspace.save_layout()
        self.assertIn(LAYOUT_STATE_KEY, self.settings.values)

        restored = self._workspace(self.settings)
        try:
            self.assertTrue(restored.restore_layout())
            self.assertFalse(restored.cards["events"].isHidden())
            self.assertEqual(
                restored._host.dockWidgetArea(restored.cards["preview"]),
                QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
            )
            restored.reset_layout()
            self.assertEqual(
                restored._host.dockWidgetArea(restored.cards["preview"]),
                QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
            )
            self.assertTrue(restored.cards["events"].isHidden())
            self.assertTrue(restored.cards["clock"].isHidden())
        finally:
            restored._host.close()
            restored.deleteLater()

        self.settings.values[LAYOUT_STATE_KEY] = QtCore.QByteArray(b"invalid")
        fallback = self._workspace(self.settings)
        try:
            self.assertFalse(fallback.restore_layout())
            self.assertTrue(fallback.cards["events"].isHidden())
            self.assertTrue(fallback.cards["clock"].isHidden())
        finally:
            fallback._host.close()
            fallback.deleteLater()

    def test_duplicate_card_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate workspace card id"):
            self.workspace.add_card("preview", "Duplicate", QtWidgets.QWidget())

    def test_layout_menu_saves_explicitly_while_runtime_is_locked(self) -> None:
        button = QtWidgets.QPushButton("Layout")
        actions = build_layout_menu(button, self.workspace)

        actions["save"].trigger()
        self.assertIn(LAYOUT_STATE_KEY, self.settings.values)
        self.assertEqual(self.settings.sync_count, 1)

        self.workspace.set_runtime_locked(True)
        self.assertFalse(actions["unlock"].isEnabled())
        self.assertFalse(actions["reset"].isEnabled())
        self.assertTrue(actions["save"].isEnabled())


if __name__ == "__main__":
    unittest.main()
