from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from squeakview.apps.operator.gui.main_window import MainWindow
from squeakview.apps.operator.gui.main_view import MainViewCallbacks, build_main_view


class MainViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.which_patcher = mock.patch(
            "squeakview.apps.operator.gui.system_meters.shutil.which",
            return_value=None,
        )
        self.which_patcher.start()
        self.window = QtWidgets.QMainWindow()
        self.callbacks = MainViewCallbacks(
            configure=mock.Mock(),
            start_run=mock.Mock(),
            stop_run=mock.Mock(),
            select_experiment=mock.Mock(),
            create_experiment=mock.Mock(),
            select_subject=mock.Mock(),
            create_subject=mock.Mock(),
            save_bottles=mock.Mock(),
            copy_events=mock.Mock(),
            open_run_folder=mock.Mock(),
        )
        self.view = build_main_view(self.window, self.callbacks)

    def tearDown(self) -> None:
        self.view.dashboard.close()
        self.window.close()
        self.app.processEvents()
        self.which_patcher.stop()

    def test_builds_stable_operator_widget_contract(self) -> None:
        self.assertIs(self.window.centralWidget(), self.view.central)
        self.assertEqual(self.view.run_state_label.text(), "READY")
        self.assertEqual(self.view.run_identity_label.text(), "No session selected")
        self.assertEqual(self.view.run_elapsed_label.text(), "00:00:00")
        self.assertFalse(self.view.run_btn.isEnabled())
        self.assertFalse(self.view.stop_btn.isEnabled())
        self.assertTrue(self.view.event_log.isReadOnly())
        self.assertEqual(self.view.event_log.document().maximumBlockCount(), 500)
        self.assertIs(self.view.stop_overlay.parentWidget(), self.window)
        self.assertEqual(self.view.bottle_panel.save_button.text(), "Save Bottle Info")
        self.assertIs(self.view.workspace.parent(), self.window)
        self.assertEqual(
            set(self.view.workspace.cards),
            {"preview", "system", "clock", "task", "bottles", "behavior", "events"},
        )
        self.assertEqual(self.view.event_log.objectName(), "eventLog")
        self.assertIsInstance(self.view.event_dock, QtWidgets.QDockWidget)
        self.assertFalse(self.view.workspace.layout_editable)
        event_buttons = {
            button.text(): button
            for button in self.view.event_dock.findChildren(QtWidgets.QPushButton)
        }
        self.assertEqual(event_buttons["Copy"].size().toTuple(), (160, 36))
        self.assertEqual(
            event_buttons["Open Run Folder"].size().toTuple(), (160, 36)
        )

    def test_connects_behavior_through_explicit_callbacks(self) -> None:
        self.view.configure_btn.click()
        self.view.run_btn.setEnabled(True)
        self.view.run_btn.click()
        self.view.stop_btn.setEnabled(True)
        self.view.stop_btn.click()
        self.view.new_experiment_btn.click()
        self.view.new_subject_btn.click()
        self.view.bottle_panel.save_requested.emit()

        self.callbacks.configure.assert_called_once_with()
        self.callbacks.start_run.assert_called_once_with()
        self.callbacks.stop_run.assert_called_once_with()
        self.callbacks.create_experiment.assert_called_once_with()
        self.callbacks.create_subject.assert_called_once_with()
        self.callbacks.save_bottles.assert_called_once_with()

    def test_native_title_menu_toggles_all_card_visibility(self) -> None:
        self.window.show()
        self.app.processEvents()
        self.assertFalse(self.view.event_dock.isVisible())

        menu = self.window.createPopupMenu()
        self.assertIsNotNone(menu)
        assert menu is not None
        actions = {action.text(): action for action in menu.actions()}
        self.assertEqual(
            {
                "Live Preview",
                "System Load",
                "Clock Preflight",
                "Live Task State",
                "Bottles",
                "Behavior Dashboard",
                "Operator Events",
            }.difference(actions),
            set(),
        )
        self.assertTrue(all(actions[title].isEnabled() for title in actions))

        actions["Operator Events"].trigger()
        self.assertTrue(self.view.event_dock.isVisible())
        actions["Operator Events"].trigger()
        self.assertFalse(self.view.event_dock.isVisible())

    def test_programmatic_event_panel_open_updates_native_visibility_action(self) -> None:
        self.window.show()
        self.view.event_dock.show()
        self.app.processEvents()

        self.assertTrue(self.view.event_dock.isVisible())
        self.assertTrue(self.view.event_dock.toggleViewAction().isChecked())

    def test_layout_menu_unlocks_and_resets_native_dock_cards(self) -> None:
        actions = {
            action.text(): action
            for action in self.view.layout_btn.menu().actions()
        }
        unlock = actions["Unlock card layout"]
        reset = actions["Reset to default layout"]

        unlock.setChecked(True)
        self.assertTrue(self.view.workspace.layout_editable)
        for dock in self.view.workspace.cards.values():
            self.assertTrue(
                dock.features()
                & QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            )

        self.view.event_dock.show()
        reset.trigger()
        self.assertFalse(self.view.event_dock.isVisible())

    def test_main_window_adapter_preserves_legacy_widget_aliases(self) -> None:
        host = QtWidgets.QMainWindow()
        callback_names = (
            "_on_configure",
            "_on_run",
            "_on_stop",
            "_on_experiment_selected",
            "_on_new_experiment",
            "_on_subject_selected",
            "_on_new_subject",
            "_on_save_bottles",
            "_copy_event_log",
            "_open_run_folder",
        )
        for name in callback_names:
            setattr(host, name, mock.Mock())

        try:
            MainWindow._build_ui(host)
            self.assertIs(host.bottle_group, host.bottle_panel)
            self.assertIs(host.left_fluid_combo, host.bottle_panel.left_fluid_combo)
            self.assertIs(host.right_final_weight_edit, host.bottle_panel.right_final_weight_edit)
            self.assertIs(host.save_bottles_btn, host.bottle_panel.save_button)
            self.assertIs(host.stop_overlay_title, host.stop_overlay.title_label)
            self.assertIs(host.stop_overlay_bar, host.stop_overlay.progress_bar)
            self.assertIs(host.workspace.parent(), host)
        finally:
            host.dashboard.close()
            host.close()


if __name__ == "__main__":
    unittest.main()
