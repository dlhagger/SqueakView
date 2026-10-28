from __future__ import annotations

import os
import unittest
from types import MappingProxyType, SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtGui, QtWidgets

from squeakview.apps.operator.gui.bottle_measurements import (
    BOTTLE_FLUID_PRESETS,
    BottleMeasurementPanel,
    BottleSideInput,
    build_bottle_payload,
    parse_weight,
    present_saved_bottles,
)
from squeakview.apps.operator.gui.main_window import (
    BOTTLE_FLUID_PRESETS as LEGACY_BOTTLE_FLUID_PRESETS,
    MainWindow,
)
from squeakview.common.run_context import normalize_bottle_measurements


class BottlePayloadTests(unittest.TestCase):
    def test_close_save_discard_cancel_and_failed_save(self):
        buttons = QtWidgets.QMessageBox.StandardButton
        for answer, save_result, expected in ((buttons.Save, True, True), (buttons.Save, False, False),
                                              (buttons.Discard, False, True), (buttons.Cancel, True, False)):
            with self.subTest(answer=answer, save_result=save_result):
                view = SimpleNamespace(_on_save_bottles=mock.Mock(return_value=save_result))
                with mock.patch("squeakview.apps.operator.gui.main_window.QtWidgets.QMessageBox.question", return_value=answer):
                    self.assertEqual(MainWindow._confirm_measurement_close(view), expected)
                self.assertEqual(view._on_save_bottles.called, answer == buttons.Save)

    def test_normalization_preserves_supervisor_frozen_side_mappings(self) -> None:
        bottles = MappingProxyType(
            {
                "left": MappingProxyType(
                    {
                        "fluid": "water",
                        "initial_weight_g": 443.0,
                        "final_weight_g": 422.0,
                    }
                ),
                "right": MappingProxyType(
                    {
                        "fluid": "ethanol",
                        "initial_weight_g": 321.0,
                        "final_weight_g": 232.0,
                    }
                ),
            }
        )

        normalized = normalize_bottle_measurements(bottles)

        self.assertEqual(normalized["left"]["fluid"], "water")
        self.assertEqual(normalized["left"]["initial_weight_g"], 443.0)
        self.assertEqual(normalized["left"]["final_weight_g"], 422.0)
        self.assertEqual(normalized["right"]["fluid"], "ethanol")
        self.assertEqual(normalized["right"]["initial_weight_g"], 321.0)
        self.assertEqual(normalized["right"]["final_weight_g"], 232.0)

    def test_parse_weight_normalizes_valid_values(self) -> None:
        self.assertEqual(parse_weight(" 12.3456789 ", "Left", strict=True), 12.345679)
        self.assertIsNone(parse_weight("", "Left", strict=True))

    def test_non_strict_payload_ignores_invalid_and_final_values(self) -> None:
        payload = build_bottle_payload(
            {
                "left": BottleSideInput(" water ", "bad", "4.0"),
                "right": BottleSideInput("", "-1", "3.0"),
            },
            include_final=False,
            strict=False,
        )

        self.assertEqual(
            payload,
            {
                "mouse_weight_g": None,
                "left": {
                    "fluid": "water",
                    "initial_weight_g": None,
                    "final_weight_g": None,
                },
                "right": {
                    "fluid": "",
                    "initial_weight_g": None,
                    "final_weight_g": None,
                },
            },
        )

    def test_strict_payload_requires_fluid_for_any_entered_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "Right fluid is required"):
            build_bottle_payload(
                {
                    "left": BottleSideInput("", "", ""),
                    "right": BottleSideInput("", "10", "9"),
                },
                include_final=True,
                strict=True,
            )

    def test_strict_payload_reports_invalid_and_negative_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "Left initial weight must be a number"):
            parse_weight("not-a-weight", "Left initial weight", strict=True)
        with self.assertRaisesRegex(ValueError, "Left final weight cannot be negative"):
            parse_weight("-0.1", "Left final weight", strict=True)
        with self.assertRaisesRegex(ValueError, "Left final weight must be a finite number"):
            parse_weight("nan", "Left final weight", strict=True)

    def test_saved_summary_presentation_preserves_warnings_and_pending_state(self) -> None:
        presentation = present_saved_bottles(
            {"complete": False, "warnings": ["left intake is negative", ""]}
        )

        self.assertTrue(presentation.completion_pending)
        self.assertIn("check the weight warning", presentation.status_text)
        self.assertIn("left intake is negative", presentation.warning_message or "")

        complete = present_saved_bottles({"complete": True, "warnings": []})
        self.assertFalse(complete.completion_pending)
        self.assertEqual(complete.status_text, "Measurements saved.")
        self.assertIsNone(complete.warning_message)


class BottleMeasurementPanelTests(unittest.TestCase):
    def test_recovery_warning_is_visible_without_opening_bottle_history(self):
        snapshot = {"available": True, "state": "ACTIVE", "lick_recovery": {
            "state": "RECOVERING", "available": False, "attempts": 1}}
        self.panel.apply_swap_snapshot(snapshot, recording=True, busy=False)
        self.assertFalse(self.panel.lick_status_label.isHidden())
        self.assertIn("recovering", self.panel.lick_status_label.text())
        self.assertFalse(self.panel.swap_button.isEnabled())
        snapshot["lick_recovery"]["state"] = "FAILED"
        self.panel.apply_swap_snapshot(snapshot, recording=True, busy=False)
        self.assertIn("unavailable", self.panel.lick_status_label.text())
        self.assertTrue(self.panel.swap_button.isEnabled())
        snapshot["lick_recovery"]["available"] = True
        self.panel.apply_swap_snapshot(snapshot, recording=True, busy=False)
        self.assertTrue(self.panel.lick_status_label.isHidden())

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.panel = BottleMeasurementPanel()

    def test_panel_does_not_cap_the_height_of_its_dock_row(self) -> None:
        self.assertEqual(
            self.panel.sizePolicy().verticalPolicy(),
            QtWidgets.QSizePolicy.Policy.Preferred,
        )

    def tearDown(self) -> None:
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()

    def test_main_window_preserves_fluid_preset_import(self) -> None:
        self.assertIs(LEGACY_BOTTLE_FLUID_PRESETS, BOTTLE_FLUID_PRESETS)

    def test_form_uses_bounded_numeric_validators_and_collects_payload(self) -> None:
        validator = self.panel.left_initial_weight_edit.validator()
        self.assertIsInstance(validator, QtGui.QDoubleValidator)
        self.assertEqual(validator.bottom(), 0.0)
        self.assertEqual(validator.top(), 100000.0)
        self.assertEqual(validator.decimals(), 4)

        self.panel.left_fluid_combo.setCurrentText("water")
        self.panel.left_initial_weight_edit.setText("12.5")
        self.panel.left_final_weight_edit.setText("10.25")
        payload = self.panel.collect_payload(include_final=True, strict=True)

        self.assertEqual(
            payload["left"],
            {
                "fluid": "water",
                "initial_weight_g": 12.5,
                "final_weight_g": 10.25,
            },
        )

    def test_mouse_weight_is_collected_at_start_and_on_explicit_save(self) -> None:
        self.assertEqual(self.panel.title(), "Weigh-In")
        self.panel.mouse_weight_edit.setText("25.375")
        for include_final in (False, True):
            payload = self.panel.collect_payload(include_final=include_final, strict=True)
            self.assertEqual(payload["mouse_weight_g"], 25.375)
        self.panel.mouse_weight_edit.clear()
        self.assertIsNone(self.panel.collect_payload(include_final=True, strict=True)["mouse_weight_g"])

    def test_invalid_mouse_weight_is_rejected_on_save(self) -> None:
        for value in ("bad", "nan", "inf", "-1", "0"):
            with self.subTest(value=value):
                self.panel.mouse_weight_edit.setText(value)
                with self.assertRaisesRegex(ValueError, "Mouse weight"):
                    self.panel.collect_payload(include_final=True, strict=True)
                self.assertIsNone(
                    self.panel.collect_payload(include_final=False, strict=False)["mouse_weight_g"]
                )

    def test_save_signal_and_completion_state_are_presented_by_panel(self) -> None:
        requested: list[bool] = []
        self.panel.save_requested.connect(lambda: requested.append(True))
        self.panel.save_button.click()
        self.assertEqual(requested, [True])

        self.panel.set_completion_pending(True)
        self.assertEqual(self.panel.save_button.text(), "Save weights")
        self.assertEqual(self.panel.styleSheet(), "")
        self.panel.set_completion_pending(False)
        self.assertEqual(self.panel.save_button.text(), "Save weights")
        self.panel.mark_saved()
        self.assertEqual(self.panel.save_button.text(), "Saved")
        self.assertFalse(self.panel.save_button.isEnabled())

    def test_workflow_has_one_instruction_and_only_the_relevant_action(self):
        self.panel.set_run_state("ready")
        self.assertEqual(self.panel.phase_label.text(), "Before recording")
        self.assertIn("automatically", self.panel.status_label.text())
        self.assertTrue(self.panel.save_button.isHidden())
        self.assertTrue(self.panel.history_toggle.isHidden())
        self.panel.set_run_state("recording")
        self.assertEqual(self.panel.phase_label.text(), "Recording")
        self.assertTrue(self.panel.save_button.isHidden())
        self.assertTrue(self.panel.lick_status_label.isHidden())
        self.assertTrue(self.panel.swap_button.isHidden())
        self.panel.set_run_state("complete")
        self.panel.set_completion_pending(True)
        self.assertEqual(self.panel.phase_label.text(), "Final weights")
        self.assertFalse(self.panel.save_button.isHidden())
        self.assertEqual(self.panel.styleSheet(), "")
        self.panel.set_completion_pending(False)
        self.panel.mark_saved()
        self.assertEqual(self.panel.phase_label.text(), "Final weights")
        self.assertEqual(self.panel.save_button.text(), "Saved")
        self.assertEqual(self.panel.styleSheet(), "")

    def test_editing_any_measurement_restores_save_button_without_changing_heading(self):
        self.panel.set_run_state("complete")
        for widget, change in (
            (self.panel.left_fluid_combo, lambda widget: widget.setCurrentText("water")),
            (self.panel.right_initial_weight_edit, lambda widget: widget.setText("100")),
            (self.panel.left_final_weight_edit, lambda widget: widget.setText("95")),
            (self.panel.mouse_weight_edit, lambda widget: widget.setText("25")),
        ):
            self.panel.mark_saved()
            self.panel.apply_swap_snapshot({}, recording=False, busy=False)
            self.assertEqual(self.panel.save_button.text(), "Saved")
            change(widget)
            self.assertEqual(self.panel.save_button.text(), "Save weights")
            self.assertTrue(self.panel.save_button.isEnabled())
            self.assertEqual(self.panel.phase_label.text(), "Final weights")

    def test_saving_refreshes_the_secondary_controller_details(self):
        self.panel.left_fluid_combo.setCurrentText("water")
        self.panel.left_initial_weight_edit.setText("33")
        self.panel.left_final_weight_edit.setText("30")
        controller = SimpleNamespace(_terminal_refresh_pending=False, refresh=mock.Mock())
        backend = SimpleNamespace(
            current_snapshot=SimpleNamespace(run_dir="run"),
            save_bottle_measurements=mock.Mock(return_value={"complete": True}),
        )
        view = SimpleNamespace(
            bottle_panel=self.panel, backend=backend, _bottle_swap_controller=controller,
            _collect_bottle_payload=self.panel.collect_payload,
            _set_bottle_completion_pending=self.panel.set_completion_pending,
            _set_bottle_status=self.panel.set_status, _emit_log=mock.Mock(),
        )
        MainWindow._on_save_bottles(view)
        self.assertTrue(controller._terminal_refresh_pending)
        controller.refresh.assert_called_once_with()
        self.assertEqual(self.panel.status_label.text(), "Measurements saved.")

    def test_clear_final_fields_does_not_clear_initial_measurements(self) -> None:
        self.panel.left_initial_weight_edit.setText("10")
        self.panel.left_final_weight_edit.setText("9")
        self.panel.right_final_weight_edit.setText("8")

        self.panel.clear_final_fields()

        self.assertEqual(self.panel.left_initial_weight_edit.text(), "10")
        self.assertEqual(self.panel.left_final_weight_edit.text(), "")
        self.assertEqual(self.panel.right_final_weight_edit.text(), "")

    def test_start_validates_initial_entries_and_rejects_unsaved_previous_finals(self):
        view = SimpleNamespace(
            bottle_panel=self.panel, _config_data={}, _preview_window_id=None,
            project=object(), _collect_bottle_payload=self.panel.collect_payload,
        )
        self.panel.mouse_weight_edit.setText("bad")
        with mock.patch("squeakview.apps.operator.gui.main_window.build_launch_config") as build:
            with self.assertRaisesRegex(ValueError, "Mouse weight must be a number"):
                MainWindow._build_launch_config(view)
            build.assert_not_called()
            self.panel.mouse_weight_edit.setText("25.5")
            self.panel.left_fluid_combo.setCurrentText("water")
            self.panel.left_initial_weight_edit.setText("100")
            MainWindow._build_launch_config(view)
            self.assertEqual(build.call_args.kwargs['bottles']['left']['initial_weight_g'], 100)
            self.assertEqual(build.call_args.kwargs['bottles']['mouse_weight_g'], 25.5)
            self.panel._current_bottle_ids = {'left': 'l1'}
            self.panel.left_final_weight_edit.setText("95")
            with self.assertRaisesRegex(ValueError, "Save the previous run"):
                MainWindow._build_launch_config(view)

    def test_reconfiguring_next_session_keeps_previous_final_weights_saveable(self):
        view = SimpleNamespace(
            bottle_panel=self.panel, run_state_label=mock.Mock(),
            backend=SimpleNamespace(current_snapshot=SimpleNamespace(run_dir=None)),
        )
        MainWindow._set_run_state(view, "ready")
        self.assertFalse(self.panel.save_button.isEnabled())
        view.backend.current_snapshot.run_dir = "previous-run"
        MainWindow._set_run_state(view, "ready")
        self.assertTrue(self.panel.left_final_weight_edit.isEnabled())
        self.assertTrue(self.panel.save_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
