from __future__ import annotations

"""Matrix presentation widget used by the behavior dashboard.

Keeping this view independent from dashboard ingestion/state makes its layout
and styling testable without constructing the full live dashboard.
"""

from collections.abc import Mapping

from PySide6 import QtWidgets


class MatrixWidget(QtWidgets.QWidget):
    """Display named counters in a configurable, color-coded matrix."""

    def __init__(self, title: str, layout_cfg: Mapping[str, object], parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._cells: dict[
            tuple[int, int], tuple[str, QtWidgets.QLabel, QtWidgets.QFrame]
        ] = {}
        self._series_keys: list[str] = []
        self._style_map = self._build_style_map(layout_cfg)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet(
            "color: #cfd4ea; font-size: 12px; font-weight: 600;"
        )
        outer.addWidget(title_label)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        outer.addLayout(grid)

        rows = layout_cfg.get("rows") or []
        if not isinstance(rows, list) or not rows:
            rows = [["A", "B"], ["C", "D"]]

        for row_index, row in enumerate(rows):
            if not isinstance(row, list):
                continue
            grid.setRowStretch(row_index, 1)
            for column_index, series in enumerate(row):
                if row_index == 0:
                    grid.setColumnStretch(column_index, 1)
                key = str(series).strip().upper()
                self._series_keys.append(key)
                cell = QtWidgets.QFrame()
                cell.setStyleSheet(self._style_for_key(key))
                cell_layout = QtWidgets.QVBoxLayout(cell)
                cell_layout.setContentsMargins(10, 8, 10, 8)
                cell_layout.setSpacing(4)
                name_label = QtWidgets.QLabel(key.replace("_", " ").title())
                name_label.setStyleSheet("color: #aeb8ff; font-size: 11px;")
                value_label = QtWidgets.QLabel("0")
                value_label.setStyleSheet(
                    "color: #e8ebf4; font-size: 18px; font-weight: 700;"
                )
                cell_layout.addWidget(name_label)
                cell_layout.addWidget(value_label)
                grid.addWidget(cell, row_index, column_index)
                self._cells[(row_index, column_index)] = (key, value_label, cell)

        self.setStyleSheet("QLabel { color: #e8ebf4; }")

    def series_keys(self) -> list[str]:
        """Return a copy of the configured keys in visual traversal order."""
        return [key for key in self._series_keys if key]

    def update_values(self, values: Mapping[str, int]) -> None:
        """Refresh cell labels without retaining the caller's value mapping."""
        for key, label, _cell in self._cells.values():
            label.setText(str(values.get(key, 0)))

    @staticmethod
    def _build_style_map(layout_cfg: Mapping[str, object]) -> dict[str, str]:
        raw = layout_cfg.get("style_map") or {}
        if not isinstance(raw, Mapping):
            return {}
        return {
            str(key).upper(): str(value)
            for key, value in raw.items()
            if key is not None and value is not None
        }

    def _style_for_key(self, key: str) -> str:
        role = self._style_map.get(key.upper(), "neutral")
        if role == "correct":
            background = "#1f8f5a"
            border = "#23a166"
        elif role == "incorrect":
            background = "#8f2f3b"
            border = "#a53644"
        else:
            background = "#0f1118"
            border = "#2a2d3d"
        return (
            "QFrame { "
            f"background-color: {background}; border: 1px solid {border}; "
            "border-radius: 8px; }"
        )
