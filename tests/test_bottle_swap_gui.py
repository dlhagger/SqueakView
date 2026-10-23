from __future__ import annotations

import os
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtCore, QtTest, QtWidgets
from squeakview.apps.operator.gui.bottle_measurements import (
    BottleMeasurementPanel, BottleSwapDialog, BottleSwapController,
)


def snapshot(state="ACTIVE", **kwargs):
    data = dict(available=True, state=state, selected_sides=[], swap_id=None,
        global_counts={"left": 32, "right": 47}, history=[], gaps=[],
        current={"left": dict(bottle_id="l1", side="left", fluid="water", initial_weight_g=100, licks=32),
                 "right": dict(bottle_id="r1", side="right", fluid="saline", initial_weight_g=110, licks=47)})
    data.update(kwargs)
    return data


class BottleSwapGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_stop_refresh_waits_for_final_controller_counts(self):
        panel = BottleMeasurementPanel()
        controller = BottleSwapController(panel, object(), lambda: False)
        controller.timer.stop()
        try:
            controller._finished("snapshot", snapshot("ACTIVE"))
            self.assertTrue(controller._terminal_refresh_pending)
            controller._terminal_refresh_pending = False
            controller._finished("snapshot", snapshot("STOPPED", available=False))
            self.assertFalse(controller._terminal_refresh_pending)
        finally:
            controller.deleteLater()
            panel.close()

    def test_empty_history_avoids_blank_table_and_unneeded_navigation(self):
        panel = BottleMeasurementPanel()
        try:
            panel.apply_swap_snapshot({}, recording=False, busy=False)
            panel.history_toggle.setChecked(True)
            self.assertFalse(panel.history_empty.isHidden())
            self.assertTrue(panel.history_table.isHidden())
            self.assertTrue(panel.edit_history_button.isHidden())
            self.assertTrue(panel.history_navigation.isHidden())
            panel.apply_swap_snapshot(snapshot(), recording=True, busy=False)
            self.assertTrue(panel.history_empty.isHidden())
            self.assertFalse(panel.history_table.isHidden())
            self.assertTrue(panel.history_navigation.isHidden())
        finally:
            panel.close()

    def test_status_poll_keeps_fluid_popup_open_and_preserves_edits(self):
        panel = BottleMeasurementPanel()
        entered, release = threading.Event(), threading.Event()
        paused = snapshot("PAUSED", selected_sides=["right"], swap_id="s1")
        class Backend:
            def bottle_swap_snapshot(self):
                entered.set()
                release.wait(2)
                return paused
        controller = BottleSwapController(panel, Backend(), lambda: True)
        controller.timer.stop()
        controller.snapshot = paused
        dialog = BottleSwapDialog(panel)
        controller.dialog = dialog
        try:
            dialog.apply_snapshot(paused, busy=False)
            dialog.show()
            fluid = dialog.forms["right"]["fluid"]
            dialog.forms["right"]["initial"].setText("125.5")
            fluid.setCurrentText("custom drink")
            fluid.showPopup()
            self.app.processEvents()
            self.assertTrue(fluid.view().isVisible())
            controller.refresh()
            self.assertTrue(entered.wait(1))
            self.assertTrue(dialog.forms["right"]["group"].isEnabled())
            self.assertTrue(fluid.view().isVisible())
            release.set()
            deadline = time.monotonic() + 2
            while controller.busy and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.assertFalse(controller.busy)
            self.assertTrue(fluid.view().isVisible())
            self.assertEqual(fluid.currentText(), "custom drink")
            self.assertEqual(dialog.forms["right"]["initial"].text(), "125.5")
            index = fluid.model().index(fluid.findText("sucrose"), 0)
            QtTest.QTest.mouseClick(fluid.view().viewport(), QtCore.Qt.MouseButton.LeftButton,
                                   pos=fluid.view().visualRect(index).center())
            self.assertEqual(fluid.currentText(), "sucrose")
            self.assertIn("#0f1118", dialog.styleSheet())
        finally:
            release.set()
            fluid.hidePopup()
            dialog.close()
            panel.close()

    def test_resume_clicked_during_poll_is_sent_after_read_completes(self):
        panel = BottleMeasurementPanel()
        entered, release = threading.Event(), threading.Event()
        calls = []
        paused = snapshot("PAUSED", selected_sides=["left"], swap_id="s1")
        class Backend:
            def bottle_swap_snapshot(self):
                entered.set()
                release.wait(2)
                return paused
            def resume_bottle_swap(self, payload):
                calls.append(payload)
                return snapshot()
        controller = BottleSwapController(panel, Backend(), lambda: True)
        controller.timer.stop()
        controller.snapshot = paused
        controller.dialog = BottleSwapDialog(panel)
        controller.dialog.resume_requested.connect(lambda payload: controller._request("resume", payload))
        try:
            controller.dialog.apply_snapshot(paused, busy=False)
            controller.dialog.forms["left"]["initial"].setText("100")
            controller.refresh()
            self.assertTrue(entered.wait(1))
            self.assertTrue(controller.dialog.resume_button.isEnabled())
            controller.dialog.resume_button.click()
            self.assertIsNotNone(controller._queued_request)
            self.assertFalse(controller.dialog.resume_button.isEnabled())
            self.assertEqual(calls, [])
            release.set()
            deadline = time.monotonic() + 2
            while controller.busy and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["left"]["initial_weight_g"], 100)
        finally:
            release.set()
            controller.dialog.close()
            panel.close()

    def test_pause_confirmation_gates_handling_and_resume(self):
        dialog = BottleSwapDialog()
        try:
            requests = []
            dialog.begin_requested.connect(requests.append)
            dialog.apply_snapshot(snapshot(), busy=False)
            self.assertFalse(dialog.resume_button.isEnabled())
            dialog.side_combo.setCurrentIndex(1)
            dialog.pause_button.click()
            self.assertEqual(requests, [["right"]])
            dialog.apply_snapshot(snapshot("PAUSING", selected_sides=["right"], swap_id="s1"), busy=False)
            self.assertFalse(dialog.forms["right"]["group"].isEnabled())
            dialog.apply_snapshot(snapshot("PAUSED", selected_sides=["right"], swap_id="s1"), busy=False)
            self.assertTrue(dialog.forms["right"]["group"].isEnabled())
            self.assertTrue(dialog.forms["left"]["group"].isHidden())
            self.assertTrue(dialog.resume_button.isEnabled())
        finally:
            dialog.close()

    def test_replacement_requires_fluid_and_weight_and_retry_retains_values(self):
        dialog = BottleSwapDialog()
        try:
            paused = snapshot("PAUSED", selected_sides=["left"], swap_id="s1")
            dialog.apply_snapshot(paused, busy=False)
            submitted = []
            dialog.resume_requested.connect(submitted.append)
            dialog.resume_button.click()
            self.assertFalse(submitted)
            self.assertIn("requires", dialog.error_label.text())
            fields = dialog.forms["left"]
            fields["fluid"].setCurrentText("custom solution")
            fields["initial"].setText("125.5")
            fields["final"].setText("90")
            dialog.apply_snapshot(dict(paused, error="Calibration failed"), busy=False)
            dialog.resume_button.click()
            self.assertEqual(submitted, [{"left": {"fluid": "custom solution", "initial_weight_g": 125.5, "final_weight_g": 90.0}}])
            dialog.apply_snapshot(dict(paused, state="UNCONFIRMED"), busy=False)
            self.assertFalse(dialog.resume_button.isEnabled())
        finally:
            dialog.close()

    def test_history_keeps_unchanged_bottle_counts_and_mouse_save_separate(self):
        panel = BottleMeasurementPanel()
        try:
            data = snapshot(swap_id="s1", history=[dict(side="left", fluid="water", licks=20)], gaps=[{}])
            panel.apply_swap_snapshot(data, recording=True, busy=False)
            self.assertFalse(panel.right_initial_weight_edit.isEnabled())
            self.assertEqual(panel.history_table.rowCount(), 3)
            self.assertIn("Right bottle saline: 47 licks", panel.lick_status_label.text())
            panel.mouse_weight_edit.setText("26.5")
            self.assertEqual(panel.collect_payload(include_final=True, strict=True), {"mouse_weight_g": 26.5})
            panel.apply_swap_snapshot(data, recording=False, busy=False)
            self.assertFalse(panel.swap_button.isEnabled())
            self.assertTrue(panel.right_final_weight_edit.isEnabled())
            panel.right_final_weight_edit.setText("101")
            panel.apply_swap_snapshot(data, recording=False, busy=False)
            payload = panel.collect_payload(include_final=True, strict=True)
            self.assertEqual(payload["current_bottle_ids"], {"left": "l1", "right": "r1"})
            self.assertEqual(payload["right"]["final_weight_g"], 101)
            self.assertEqual(panel.history_table.item(0, 0).data(256), None)
        finally:
            panel.close()

    def test_unconfirmed_pause_can_retry_but_unconfirmed_resume_checks_status(self):
        dialog = BottleSwapDialog()
        try:
            pending = snapshot("UNCONFIRMED", selected_sides=["right"], swap_id="s1")
            requests = []
            dialog.begin_requested.connect(requests.append)
            dialog.apply_snapshot(pending, busy=False)
            self.assertTrue(dialog.pause_button.isEnabled())
            dialog.pause_button.click()
            self.assertEqual(requests, [["right"]])
            self.assertFalse(dialog.forms["right"]["group"].isEnabled())
            self.assertFalse(dialog.resume_button.isEnabled())
            dialog.apply_snapshot(dict(pending, resume_pending=True), busy=False)
            self.assertFalse(dialog.pause_button.isEnabled())
            self.assertTrue(dialog.check_status_button.isEnabled())
        finally:
            dialog.close()

    def test_history_browses_older_entries_and_returns_to_latest(self):
        panel = BottleMeasurementPanel()
        class Backend:
            def bottle_history_page(self, offset, limit=50):
                return dict(history=[dict(bottle_id=f"old-{i}", side="left", fluid="water", licks=i) for i in range(offset, min(offset + limit, 120))], history_count=120)
            def bottle_swap_snapshot(self, *, force_probe=False):
                return snapshot(available=force_probe)
        controller = BottleSwapController(panel, Backend(), lambda: True)
        controller.timer.stop()
        try:
            data = snapshot(history=[dict(bottle_id=f"old-{i}", side="left", fluid="water", licks=i) for i in range(70, 120)], history_count=120)
            controller._finished("snapshot", data)
            self.assertIn("71–120 of 120", panel.history_range.text())
            panel.history_previous.click()
            deadline = time.monotonic() + 2
            while controller.busy and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.assertIn("21–70 of 120", panel.history_range.text())
            self.assertEqual(panel.history_table.item(0, 0).data(256), "old-20")
            panel.history_latest.click()
            self.assertIn("71–120 of 120", panel.history_range.text())
            panel.check_status_button.click()
            deadline = time.monotonic() + 2
            while controller.busy and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.assertTrue(controller.snapshot["available"])
        finally:
            panel.close()

    def test_closed_bottle_final_weight_edit_uses_bottle_id(self):
        from unittest.mock import patch
        panel = BottleMeasurementPanel()
        calls = []
        class Backend:
            def update_bottle_history_weight(self, bottle_id, weight):
                calls.append((bottle_id, weight))
                return snapshot()
        controller = BottleSwapController(panel, Backend(), lambda: False)
        controller.timer.stop()
        try:
            panel.apply_swap_snapshot(snapshot(history=[dict(bottle_id="old-left", side="left", fluid="water", licks=12)]), recording=False, busy=False)
            panel.history_table.setCurrentCell(0, 0)
            with patch.object(QtWidgets.QInputDialog, "getText", return_value=("92.5", True)):
                controller.edit_history_weight()
            deadline = time.monotonic() + 2
            while controller.busy and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.assertEqual(calls, [("old-left", 92.5)])
        finally:
            panel.close()

    def test_backend_requests_run_outside_gui_thread(self):
        panel = BottleMeasurementPanel()
        worker_threads = []
        class Backend:
            def bottle_swap_snapshot(self):
                worker_threads.append(threading.get_ident())
                return snapshot()
        controller = BottleSwapController(panel, Backend(), lambda: True)
        controller.timer.stop()
        try:
            gui_thread = threading.get_ident()
            controller.refresh()
            deadline = time.monotonic() + 2
            while controller.busy and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.assertFalse(controller.busy)
            self.assertEqual(len(worker_threads), 1)
            self.assertNotEqual(worker_threads[0], gui_thread)
            self.assertTrue(panel.swap_button.isEnabled())
        finally:
            panel.close()


if __name__ == "__main__":
    unittest.main()
