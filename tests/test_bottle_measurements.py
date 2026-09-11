from __future__ import annotations

import os
import unittest

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
)


class BottlePayloadTests(unittest.TestCase):
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
        self.assertEqual(complete.status_text, "Bottle info complete.")
        self.assertIsNone(complete.warning_message)


class BottleMeasurementPanelTests(unittest.TestCase):
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

    def test_save_signal_and_completion_state_are_presented_by_panel(self) -> None:
        requested: list[bool] = []
        self.panel.save_requested.connect(lambda: requested.append(True))
        self.panel.save_button.click()
        self.assertEqual(requested, [True])

        self.panel.set_completion_pending(True)
        self.assertEqual(self.panel.save_button.text(), "Save Final Weights")
        self.assertIn("#b98335", self.panel.styleSheet())
        self.panel.set_completion_pending(False)
        self.assertEqual(self.panel.save_button.text(), "Save Bottle Info")
        self.assertEqual(self.panel.styleSheet(), "")

    def test_clear_final_fields_does_not_clear_initial_measurements(self) -> None:
        self.panel.left_initial_weight_edit.setText("10")
        self.panel.left_final_weight_edit.setText("9")
        self.panel.right_final_weight_edit.setText("8")

        self.panel.clear_final_fields()

        self.assertEqual(self.panel.left_initial_weight_edit.text(), "10")
        self.assertEqual(self.panel.left_final_weight_edit.text(), "")
        self.assertEqual(self.panel.right_final_weight_edit.text(), "")


if __name__ == "__main__":
    unittest.main()
