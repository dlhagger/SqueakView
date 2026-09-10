from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from squeakview.apps.operator.gui.dashboard import MatrixWidget as DashboardMatrixWidget
from squeakview.apps.operator.gui.dashboard_matrix import MatrixWidget


class MatrixWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_dashboard_preserves_matrix_widget_re_export(self) -> None:
        self.assertIs(DashboardMatrixWidget, MatrixWidget)

    def test_configured_series_order_and_updates_are_preserved(self) -> None:
        widget = MatrixWidget(
            "Outcomes",
            {
                "rows": [["go_correct", "go_incorrect"], ["nogo_correct"]],
                "style_map": {
                    "go_correct": "correct",
                    "go_incorrect": "incorrect",
                },
            },
        )
        try:
            self.assertEqual(
                widget.series_keys(),
                ["GO_CORRECT", "GO_INCORRECT", "NOGO_CORRECT"],
            )
            widget.update_values({"GO_CORRECT": 7, "GO_INCORRECT": 2})
            labels = {
                key: label.text() for key, label, _cell in widget._cells.values()
            }
            self.assertEqual(labels["GO_CORRECT"], "7")
            self.assertEqual(labels["GO_INCORRECT"], "2")
            self.assertEqual(labels["NOGO_CORRECT"], "0")
        finally:
            widget.close()

    def test_invalid_layout_fragments_fall_back_safely(self) -> None:
        widget = MatrixWidget(
            "Fallback",
            {"rows": "not-a-row-list", "style_map": "not-a-mapping"},
        )
        try:
            self.assertEqual(widget.series_keys(), ["A", "B", "C", "D"])
        finally:
            widget.close()


if __name__ == "__main__":
    unittest.main()
