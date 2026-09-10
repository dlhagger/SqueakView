from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from PySide6 import QtCore, QtGui, QtWidgets


BOTTLE_FLUID_PRESETS = [
    "",
    "water",
    "sucrose",
    "quinine",
    "ethanol",
    "saline",
    "custom",
]

_PENDING_STYLE = (
    "QGroupBox { border: 2px solid #b98335; } "
    "QGroupBox::title { color: #ffd28b; }"
)


@dataclass(frozen=True)
class BottleSideInput:
    fluid: str
    initial_weight: str
    final_weight: str


@dataclass(frozen=True)
class BottleSavePresentation:
    status_text: str
    completion_pending: bool
    warning_message: str | None = None


def parse_weight(text: str, label: str, *, strict: bool) -> float | None:
    """Normalize an operator-entered bottle weight.

    Non-strict collection is used to snapshot the initial form at launch, where
    an unfinished field should remain absent rather than prevent recording.
    Explicit saves use strict validation and surface actionable field errors.
    """

    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        if strict:
            raise ValueError(f"{label} must be a number.") from None
        return None
    if not math.isfinite(value):
        if strict:
            raise ValueError(f"{label} must be a finite number.")
        return None
    if value < 0:
        if strict:
            raise ValueError(f"{label} cannot be negative.")
        return None
    return round(value, 6)


def build_bottle_payload(
    sides: Mapping[str, BottleSideInput],
    *,
    include_final: bool,
    strict: bool,
) -> dict[str, object]:
    """Build the immutable-shaped payload handed to the run backend."""

    payload: dict[str, object] = {}
    for side in ("left", "right"):
        values = sides[side]
        label = side.title()
        fluid = values.fluid.strip()
        initial = parse_weight(
            values.initial_weight,
            f"{label} initial weight",
            strict=strict,
        )
        final = (
            parse_weight(
                values.final_weight,
                f"{label} final weight",
                strict=strict,
            )
            if include_final
            else None
        )
        if strict and (initial is not None or final is not None) and not fluid:
            raise ValueError(
                f"{label} fluid is required when saving bottle weights."
            )
        payload[side] = {
            "fluid": fluid,
            "initial_weight_g": initial,
            "final_weight_g": final,
        }
    return payload


def present_saved_bottles(summary: Mapping[str, object]) -> BottleSavePresentation:
    """Translate a backend bottle summary into deterministic operator feedback."""

    complete = bool(summary.get("complete"))
    raw_warnings = summary.get("warnings", [])
    warnings = (
        [str(item) for item in raw_warnings if str(item)]
        if isinstance(raw_warnings, (list, tuple))
        else []
    )
    if warnings:
        return BottleSavePresentation(
            status_text="Bottle info saved; check the weight warning.",
            completion_pending=not complete,
            warning_message=(
                "Bottle info saved with a plausibility warning:\n\n"
                + "\n".join(warnings)
            ),
        )
    state = "complete" if complete else "saved; missing one or more weights"
    return BottleSavePresentation(
        status_text=f"Bottle info {state}.",
        completion_pending=not complete,
    )


class BottleMeasurementPanel(QtWidgets.QGroupBox):
    """Bottle entry form with validation and presentation kept out of the shell."""

    save_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Bottles", parent)
        layout = QtWidgets.QGridLayout(self)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Side", self), 0, 0)
        layout.addWidget(QtWidgets.QLabel("Fluid", self), 0, 1)
        layout.addWidget(QtWidgets.QLabel("Initial g", self), 0, 2)
        layout.addWidget(QtWidgets.QLabel("Final g", self), 0, 3)

        self.left_fluid_combo = self._make_fluid_combo()
        self.left_initial_weight_edit = self._make_weight_edit("initial")
        self.left_final_weight_edit = self._make_weight_edit("final")
        self.right_fluid_combo = self._make_fluid_combo()
        self.right_initial_weight_edit = self._make_weight_edit("initial")
        self.right_final_weight_edit = self._make_weight_edit("final")

        layout.addWidget(QtWidgets.QLabel("Left", self), 1, 0)
        layout.addWidget(self.left_fluid_combo, 1, 1)
        layout.addWidget(self.left_initial_weight_edit, 1, 2)
        layout.addWidget(self.left_final_weight_edit, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Right", self), 2, 0)
        layout.addWidget(self.right_fluid_combo, 2, 1)
        layout.addWidget(self.right_initial_weight_edit, 2, 2)
        layout.addWidget(self.right_final_weight_edit, 2, 3)

        self.status_label = QtWidgets.QLabel(
            "Bottle info pending for next run.", self
        )
        self.status_label.setObjectName("bottleStatus")
        self.status_label.setWordWrap(True)
        self.save_button = QtWidgets.QPushButton("Save Bottle Info", self)
        self.save_button.setObjectName("secondaryButton")
        self.save_button.clicked.connect(self.save_requested.emit)
        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(self.status_label, 1)
        action_row.addWidget(self.save_button, 0)
        layout.addLayout(action_row, 3, 0, 1, 4)
        layout.setColumnStretch(1, 1)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )

    def _make_fluid_combo(self) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(self)
        combo.setEditable(True)
        combo.addItems(BOTTLE_FLUID_PRESETS)
        combo.setMinimumWidth(96)
        return combo

    def _make_weight_edit(self, phase: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(self)
        edit.setPlaceholderText(phase)
        edit.setMaximumWidth(82)
        validator = QtGui.QDoubleValidator(0.0, 100000.0, 4, edit)
        validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        edit.setValidator(validator)
        return edit

    def collect_payload(
        self, *, include_final: bool, strict: bool
    ) -> dict[str, object]:
        sides = {
            "left": BottleSideInput(
                fluid=self.left_fluid_combo.currentText(),
                initial_weight=self.left_initial_weight_edit.text(),
                final_weight=self.left_final_weight_edit.text(),
            ),
            "right": BottleSideInput(
                fluid=self.right_fluid_combo.currentText(),
                initial_weight=self.right_initial_weight_edit.text(),
                final_weight=self.right_final_weight_edit.text(),
            ),
        }
        return build_bottle_payload(
            sides,
            include_final=include_final,
            strict=strict,
        )

    def clear_final_fields(self) -> None:
        self.left_final_weight_edit.clear()
        self.right_final_weight_edit.clear()

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_completion_pending(self, pending: bool) -> None:
        self.setStyleSheet(_PENDING_STYLE if pending else "")
        self.save_button.setText(
            "Save Final Weights" if pending else "Save Bottle Info"
        )
