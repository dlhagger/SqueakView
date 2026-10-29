from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from PySide6 import QtCore, QtGui, QtWidgets

from .dialog_style import DARK_DIALOG_STYLE, apply_dark_combo_popups, center_window, fit_window_to_available_area
from .main_theme import BOTTLE_HISTORY_HEADER_STYLE
from squeakview.common.measurements import MAX_WEIGHT_G, weight_value


BOTTLE_FLUID_PRESETS = [
    "",
    "water",
    "sucrose",
    "quinine",
    "ethanol",
    "saline",
    "custom",
]

def describe_bottle_error(reason: str) -> str:
    messages = {
        "CALIBRATION_FAILED": "Calibration failed. Both lick channels remain paused; you can retry.",
        "CAL_STOP_VERIFY_FAILED": "The lick sensor did not confirm stopping for calibration. Both channels remain paused; you can retry.",
        "CAL_REGISTER_VERIFY_FAILED": "The lick sensor did not confirm a calibration setting. Both channels remain paused; you can retry.",
        "CAL_READINESS_TIMEOUT": "The lick sensor did not become ready in time. Both channels remain paused.",
        "CAL_SENSOR_HEALTH_TIMEOUT": "The lick sensor did not pass its health checks in time. Both channels remain paused.",
        "CAL_CHANNELS_NOT_RELEASED": "A lick channel stayed touched during calibration. Clear hands from both spouts, then retry.",
    }
    return messages.get(reason, reason)


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

    Strict collection at Start and Save surfaces invalid entries rather than
    silently omitting an operator's measurement. Empty fields remain optional.
    """

    try:
        return weight_value(text, label)
    except ValueError:
        if strict:
            raise
        return None


def build_bottle_payload(
    sides: Mapping[str, BottleSideInput],
    *,
    include_final: bool,
    strict: bool,
    mouse_weight: str = "",
) -> dict[str, object]:
    """Build the immutable-shaped payload handed to the run backend."""

    weight = parse_weight(mouse_weight, "Mouse weight", strict=strict)
    if weight == 0:
        if strict:
            raise ValueError("Mouse weight must be greater than zero.")
        weight = None
    payload: dict[str, object] = {"mouse_weight_g": weight}
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
            status_text="Measurements saved; check the weight warning.",
            completion_pending=not complete,
            warning_message=(
                "Measurements saved with a plausibility warning:\n\n"
                + "\n".join(warnings)
            ),
        )
    return BottleSavePresentation(
        status_text="Measurements saved." if complete else "Measurements saved. Complete the remaining bottle fields, then save again.",
        completion_pending=not complete,
    )


class BottleMeasurementPanel(QtWidgets.QGroupBox):
    """Bottle entry form with validation and presentation kept out of the shell."""

    save_requested = QtCore.Signal()
    details_expansion_changed = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Weigh-In", parent)
        layout = QtWidgets.QGridLayout(self)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(10)
        self.phase_label = QtWidgets.QLabel("Before recording", self)
        self.phase_label.setObjectName("bottlePhase")
        layout.addWidget(self.phase_label, 0, 0, 1, 4)
        self.status_label = QtWidgets.QLabel("Enter starting weights. Start Recording saves them automatically.", self)
        self.status_label.setObjectName("bottleStatus")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(44)
        layout.addWidget(self.status_label, 1, 0, 1, 4)
        layout.addWidget(QtWidgets.QLabel("Bottle", self), 2, 0)
        layout.addWidget(QtWidgets.QLabel("Fluid", self), 2, 1)
        layout.addWidget(QtWidgets.QLabel("Initial (g)", self), 2, 2)
        layout.addWidget(QtWidgets.QLabel("Final (g)", self), 2, 3)

        self.left_fluid_combo = self._make_fluid_combo()
        self.left_initial_weight_edit = self._make_weight_edit("initial")
        self.left_final_weight_edit = self._make_weight_edit("final")
        self.right_fluid_combo = self._make_fluid_combo()
        self.right_initial_weight_edit = self._make_weight_edit("initial")
        self.right_final_weight_edit = self._make_weight_edit("final")

        layout.addWidget(QtWidgets.QLabel("Left", self), 3, 0)
        layout.addWidget(self.left_fluid_combo, 3, 1)
        layout.addWidget(self.left_initial_weight_edit, 3, 2)
        layout.addWidget(self.left_final_weight_edit, 3, 3)
        layout.addWidget(QtWidgets.QLabel("Right", self), 4, 0)
        layout.addWidget(self.right_fluid_combo, 4, 1)
        layout.addWidget(self.right_initial_weight_edit, 4, 2)
        layout.addWidget(self.right_final_weight_edit, 4, 3)

        self.mouse_weight_edit = self._make_weight_edit("optional")
        self.mouse_weight_edit.setToolTip("Mouse body weight in grams for this run (optional).")
        layout.addWidget(QtWidgets.QLabel("Mouse weight (g)", self), 5, 0, 1, 2)
        layout.addWidget(self.mouse_weight_edit, 5, 2)

        self.save_button = QtWidgets.QPushButton("Save weights", self)
        self.save_button.setObjectName("primaryButton")
        self.save_button.clicked.connect(self.save_requested.emit)
        layout.addWidget(self.save_button, 6, 0, 1, 4)
        self.lick_status_label = QtWidgets.QLabel("Lickometer: unavailable until recording", self)
        self.lick_status_label.setObjectName("bottleLickStatus")
        self.lick_status_label.setWordWrap(True)
        layout.addWidget(self.lick_status_label, 8, 0, 1, 4)
        self.swap_button = QtWidgets.QPushButton("Swap Bottles…", self)
        self.swap_button.setEnabled(False)
        swap_row = QtWidgets.QHBoxLayout()
        swap_row.addWidget(self.swap_button)
        self.check_status_button = QtWidgets.QPushButton("Check Status", self)
        self.check_status_button.setEnabled(False)
        swap_row.addWidget(self.check_status_button)
        swap_row.addStretch()
        layout.addLayout(swap_row, 9, 0, 1, 4)
        self.history_toggle = QtWidgets.QToolButton(self)
        self.history_toggle.setObjectName("bottleHistoryToggle")
        self.history_toggle.setText("Bottle tools and history")
        self.history_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.history_toggle.setCheckable(True)
        self.history_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.history_table = QtWidgets.QTableWidget(0, 9, self)
        self.history_table.setObjectName("bottleHistory")
        self.history_table.setHorizontalHeaderLabels(["Side", "Fluid", "Initial g", "Final g", "Intake g", "Licks", "Status", "Installed at", "Removed at"])
        self.history_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.history_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.verticalHeader().hide()
        self.history_table.verticalHeader().setDefaultSectionSize(32)
        self.history_table.horizontalHeader().setMinimumSectionSize(52)
        self.history_table.horizontalHeader().setObjectName("bottleHistoryHeader")
        self.history_table.horizontalHeader().setStyleSheet(BOTTLE_HISTORY_HEADER_STYLE)
        self.history_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.history_table.setMinimumHeight(132)
        self.history_table.setMaximumHeight(220)
        self.history_table.hide()
        self.history_empty = QtWidgets.QLabel("No bottles recorded yet.\nBottle history appears when a run starts.", self)
        self.history_empty.setObjectName("bottleHistoryEmpty")
        self.history_empty.setWordWrap(True)
        self.history_empty.hide()
        self.edit_history_button = QtWidgets.QPushButton("Edit Selected Final Weight…", self)
        self.edit_history_button.hide()
        self.edit_history_button.setEnabled(False)
        self.edit_history_button.setProperty("compact", True)
        self.history_toggle.toggled.connect(self._toggle_history)
        layout.addWidget(self.history_toggle, 7, 0, 1, 4)
        layout.addWidget(self.history_table, 10, 0, 1, 4)
        layout.addWidget(self.history_empty, 10, 0, 1, 4)
        layout.addWidget(self.edit_history_button, 11, 0, 1, 4)
        self.history_navigation = QtWidgets.QWidget(self)
        navigation = QtWidgets.QHBoxLayout(self.history_navigation)
        navigation.setContentsMargins(0, 0, 0, 0)
        self.history_previous = QtWidgets.QPushButton("Older", self)
        self.history_next = QtWidgets.QPushButton("Newer", self)
        self.history_latest = QtWidgets.QPushButton("Latest", self)
        self.history_range = QtWidgets.QLabel(self)
        self.history_range.setWordWrap(True)
        navigation.addWidget(self.history_range, 1)
        for widget in (self.history_previous, self.history_next, self.history_latest, self.history_range):
            if widget is not self.history_range:
                widget.setProperty("compact", True)
                navigation.addWidget(widget)
        self.history_navigation.hide()
        layout.addWidget(self.history_navigation, 12, 0, 1, 4)
        self._history_count = 0
        self._history_page = None
        self._history_offset = 0
        self._bottle_history_locked = False
        self._current_bottle_ids = {}
        self._swap_recording = False
        self._lick_recovery_warning = False
        self._run_state = None
        self._completion_pending = True
        self._save_acknowledged = False
        self._saved_final_weights = {"left": "", "right": ""}
        self._form_edits = set()
        for side in ("left", "right"):
            combo = getattr(self, f"{side}_fluid_combo")
            combo.currentTextChanged.connect(lambda _text, side=side: self._measurement_edited(side, "fluid"))
            for phase in ("initial", "final"):
                edit = getattr(self, f"{side}_{phase}_weight_edit")
                edit.textChanged.connect(lambda _text, side=side, phase=phase: self._measurement_edited(side, phase))
        self.mouse_weight_edit.textChanged.connect(lambda _text: self._measurement_edited("mouse", "weight"))
        layout.setColumnStretch(1, 1)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._update_history_visibility()

    def _toggle_history(self, expanded: bool) -> None:
        # Let the workspace remember dock proportions before extra widgets
        # increase this card's minimum size.
        self.details_expansion_changed.emit(expanded)
        self._update_history_visibility()
        self.layout().invalidate()
        self.updateGeometry()
        self.history_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)

    def _update_history_visibility(self) -> None:
        expanded = self.history_toggle.isChecked()
        self.lick_status_label.setVisible(expanded or self._lick_recovery_warning)
        self.swap_button.setVisible(expanded and self._swap_recording)
        self.check_status_button.setVisible(expanded and self._swap_recording)
        has_rows = self.history_table.rowCount() > 0
        self.history_table.setVisible(expanded and has_rows)
        self.history_empty.setVisible(expanded and not has_rows)
        self.edit_history_button.setVisible(expanded and self._history_count > 0)
        self.history_navigation.setVisible(expanded and self._history_count > 50)

    def apply_swap_snapshot(self, snapshot: dict, *, recording: bool, busy: bool) -> None:
        self._swap_recording = recording
        state = str(snapshot.get("state", "UNKNOWN")).upper()
        available = bool(snapshot.get("available"))
        self.swap_button.setEnabled(recording and available and not busy and state in {"ACTIVE", "PAUSED", "FAILED", "UNCONFIRMED"})
        self.check_status_button.setEnabled(recording and not busy)
        self.swap_button.setText("Manage Bottle Swap…" if state != "ACTIVE" and recording else "Swap Bottles…")
        totals = snapshot.get("global_counts", {})
        current = snapshot.get("current", {})
        text = f"Lick channels: {state.lower()}" if recording else "Recording stopped"
        recovery = snapshot.get("lick_recovery") or {}
        self._lick_recovery_warning = bool(recording and recovery and not recovery.get("available"))
        if self._lick_recovery_warning:
            failed = recovery.get("state") == "FAILED"
            text = "Lick channels: unavailable" if failed else "Lick channels: recovering"
            text += " · recording continues"
            if recovery.get("attempts"):
                text += f"\nCalibration attempt {recovery['attempts']}/3"
            if not failed:
                self.swap_button.setEnabled(False)
        if recording and not available:
            text += "\nBottle swapping is unavailable."
        if totals:
            text += f"\nSession licks · Left {totals.get('left', 0)} · Right {totals.get('right', 0)}"
        gaps = snapshot.get("gaps", [])
        if gaps:
            text += f"\nObservation gaps: {snapshot.get('gap_count', len(gaps))}"
        if snapshot.get("error"):
            text += f"\n{describe_bottle_error(str(snapshot['error']))}"
        self.lick_status_label.setText(text)
        history_count = int(snapshot.get("history_count", len(snapshot.get("history", []))))
        if self._history_page is None:
            history = snapshot.get("history", [])
            offset = max(0, history_count - len(history))
        else:
            history = self._history_page
            offset = self._history_offset
        self.history_range.setText(f"Completed bottles {offset + 1 if history else 0}–{offset + len(history)} of {history_count}")
        self.history_previous.setEnabled(not busy and offset > 0)
        self.history_next.setEnabled(not busy and offset + len(history) < history_count)
        self.history_latest.setEnabled(not busy and self._history_page is not None)
        self._displayed_history_offset = offset
        self._history_count = history_count
        self._bottle_history_locked = bool(recording and (history or snapshot.get("swap_id") or gaps))
        for side in ("left", "right"):
            bottle = current.get(side) or {}
            bottle_id = bottle.get("bottle_id", bottle.get("id"))
            # Only a changed bottle identity owns new form values. A stop or
            # status refresh must never replace the operator's unsaved edits.
            if bottle_id and self._current_bottle_ids.get(side) != bottle_id:
                first_snapshot = side not in self._current_bottle_ids
                self._current_bottle_ids[side] = bottle_id
                for field in ("fluid", "initial", "final"):
                    # A first recovery poll can finish after the user has
                    # started typing. Final weights also belong to the user
                    # while the terminal controller snapshot is still pending.
                    if not recording and (first_snapshot or field == "final") and (side, field) in self._form_edits:
                        continue
                    widget = getattr(self, f"{side}_fluid_combo" if field == "fluid" else f"{side}_{field}_weight_edit")
                    value = bottle.get("fluid") if field == "fluid" else bottle.get(f"{field}_weight_g")
                    with QtCore.QSignalBlocker(widget):
                        if field == "fluid":
                            widget.setCurrentText(str(value or ""))
                        else:
                            widget.setText("" if value is None else str(value))
                    self._form_edits.discard((side, field))
                saved_final = bottle.get("final_weight_g")
                self._saved_final_weights[side] = "" if saved_final is None else str(saved_final)
        self._update_form_controls()
        rows = [(bottle, "Closed" if bottle.get("final_weight_g") is not None else "Final weight pending") for bottle in history]
        rows += [(bottle, "Current") for bottle in current.values() if bottle]
        if self.history_table.rowCount() != len(rows):
            self.history_table.setRowCount(len(rows))
        self._update_history_visibility()
        self.edit_history_button.setEnabled(bool(history) and not busy)
        for row, (bottle, status) in enumerate(rows):
            def timestamp(key):
                value = bottle.get(key)
                if value is None:
                    return None
                seconds = max(0, int(value) // 1_000_000)
                return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"
            values = [bottle.get("side", ""), bottle.get("fluid", ""), bottle.get("initial_weight_g"),
                      bottle.get("final_weight_g"), bottle.get("intake_g"), bottle.get("licks", bottle.get("lick_count", 0)), status, timestamp("started_monotonic_us"), timestamp("ended_monotonic_us")]
            for column, value in enumerate(values):
                text = "—" if value is None else str(value)
                item = self.history_table.item(row, column)
                if item is None:
                    item = QtWidgets.QTableWidgetItem(text)
                    self.history_table.setItem(row, column, item)
                elif item.text() != text:
                    item.setText(text)
                bottle_id = bottle.get("bottle_id") if status != "Current" else None
                if item.data(QtCore.Qt.ItemDataRole.UserRole) != bottle_id:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, bottle_id)

    def _make_fluid_combo(self) -> QtWidgets.QComboBox:
        combo = _SwapComboBox(self)
        combo.setEditable(True)
        combo.addItems(BOTTLE_FLUID_PRESETS)
        combo.setMinimumWidth(108)
        return combo

    def _make_weight_edit(self, phase: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(self)
        edit.setPlaceholderText(phase)
        edit.setMinimumWidth(88)
        edit.setMaximumWidth(120)
        validator = QtGui.QDoubleValidator(0.0, MAX_WEIGHT_G, 4, edit)
        validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        edit.setValidator(validator)
        return edit

    def collect_payload(
        self, *, include_final: bool, strict: bool
    ) -> dict[str, object]:
        if include_final and self._bottle_history_locked:
            weight = parse_weight(self.mouse_weight_edit.text(), "Mouse weight", strict=strict)
            if weight == 0:
                if strict:
                    raise ValueError("Mouse weight must be greater than zero.")
                weight = None
            return {"mouse_weight_g": weight}
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
        payload = build_bottle_payload(
            sides,
            include_final=include_final,
            strict=strict,
            mouse_weight=self.mouse_weight_edit.text(),
        )
        if include_final and self._current_bottle_ids:
            payload["current_bottle_ids"] = dict(self._current_bottle_ids)
        return payload

    def set_run_state(self, state: str) -> None:
        changed = self._run_state != state
        if changed and state in {"ready", "starting"}:
            self._save_acknowledged = False
        self._run_state = state
        if changed:
            self.set_status({
                "ready": "Enter fluids and starting weights. Start Recording saves them automatically.",
                "starting": "Saving starting measurements with this run…",
                "recording": "Starting measurements saved. Enter final weights after Stop Recording.",
                "finalizing": "Wait for recording to finish before entering final weights.",
                "validating": "Capture stopped. Enter final bottle weights now; Save becomes available when validation finishes.",
                "complete": "Enter final bottle weights, then save.",
                "failed": "Recording stopped. Measurements can still be saved; check Events for the run error.",
            }.get(state, ""))
        self._update_form_controls()

    def _update_form_controls(self) -> None:
        validating = self._run_state == "validating"
        active = self._run_state in {"starting", "recording", "finalizing", "validating"}
        if self._run_state is None:
            active = self._swap_recording
        for side in ("left", "right"):
            for suffix in ("fluid_combo", "initial_weight_edit"):
                widget = getattr(self, f"{side}_{suffix}")
                widget.setEnabled(not active)
                widget.setToolTip("Use Swap Bottles to preserve bottle history." if active else "")
            getattr(self, f"{side}_final_weight_edit").setEnabled(
                (not active or validating) and self._run_state != "ready")
            getattr(self, f"{side}_final_weight_edit").setPlaceholderText(
                "After Stop" if (active and not validating) or self._run_state == "ready" else "final")
        self.mouse_weight_edit.setEnabled(not active)
        self.save_button.setText("Saved" if self._save_acknowledged else "Save weights")
        self.save_button.setEnabled(not active and self._run_state != "ready" and not self._save_acknowledged)
        self.save_button.setVisible((not active or validating) and self._run_state != "ready")
        self.history_toggle.setVisible(self._run_state != "ready")
        self._update_phase_heading()
        self.save_button.setToolTip(
            "Initial measurements are saved automatically when recording starts."
            if self._run_state == "ready" else
            "Save becomes available when validation finishes." if validating else
            "Stop recording before saving final measurements." if active else "")

    def clear_final_fields(self) -> None:
        self.left_final_weight_edit.clear()
        self.right_final_weight_edit.clear()
        self._saved_final_weights = {"left": "", "right": ""}
        self._form_edits.difference_update({("left", "final"), ("right", "final")})

    def mark_saved(self) -> None:
        self._form_edits.clear()
        self._saved_final_weights = {
            side: getattr(self, f"{side}_final_weight_edit").text()
            for side in ("left", "right")
        }
        self._save_acknowledged = True
        self._update_form_controls()

    def _measurement_edited(self, side: str, field: str) -> None:
        self._form_edits.add((side, field))
        if self._save_acknowledged:
            self._save_acknowledged = False
            self.set_status("Unsaved changes.")
            self._update_form_controls()

    def has_unsaved_final_weights(self) -> bool:
        return bool(self._current_bottle_ids) and any(
            getattr(self, f"{side}_final_weight_edit").text().strip() != self._saved_final_weights[side].strip()
            for side in ("left", "right")
        )

    def has_unsaved_measurements(self) -> bool:
        return bool(self._form_edits) and (
            bool(self._current_bottle_ids) or self._run_state in {"complete", "failed"}
        )

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_completion_pending(self, pending: bool) -> None:
        self._completion_pending = pending
        self._update_phase_heading()

    def _update_phase_heading(self) -> None:
        self.phase_label.setText({
            "ready": "Before recording",
            "starting": "Starting recording",
            "recording": "Recording",
            "finalizing": "Finishing recording",
            "validating": "Final weights",
            "complete": "Final weights",
            "failed": "Recording stopped",
        }.get(self._run_state, "Before recording"))


class _SwapComboBox(QtWidgets.QComboBox):
    """Keep the dropdown affordance visible with the operator's dark theme."""

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor("#d7ddf5" if self.isEnabled() else "#7f8aac"), 2))
        x, y = self.width() - 15, self.height() // 2
        painter.drawPolyline(QtGui.QPolygon([
            QtCore.QPoint(x - 4, y - 2), QtCore.QPoint(x, y + 2), QtCore.QPoint(x + 4, y - 2),
        ]))


class BottleSwapDialog(QtWidgets.QDialog):
    """Keep handling instructions behind an acknowledged controller pause."""

    check_requested = QtCore.Signal()
    begin_requested = QtCore.Signal(object)
    resume_requested = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Swap Bottles")
        self.setMinimumWidth(560)
        self.resize(680, 620)
        self.setStyleSheet(DARK_DIALOG_STYLE + """
            QLabel#swapTitle { color: #f3f5ff; font-size: 21px; font-weight: 700; }
            QLabel#swapState { color: #ffe0a3; background-color: #30281c;
                border: 1px solid #75603b; border-radius: 6px; padding: 10px; font-weight: 700; }
            QLabel#swapInstructions { color: #c9d2eb; }
            QLabel#swapError { color: #ffb4b4; }
            QPushButton#swapSecondary { background-color: #242d46; border-color: #46516e; }
        """)
        self._snapshot: dict = {}
        self._loaded_swap = None
        self._busy = False
        self._last_backend_error = ""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        title = QtWidgets.QLabel("Bottle swap", self)
        title.setObjectName("swapTitle")
        layout.addWidget(title)
        self.state_label = QtWidgets.QLabel(self)
        self.state_label.setObjectName("swapState")
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)
        self.instructions = QtWidgets.QLabel(
            "Select the bottles to replace. Pause both lick channels before handling either bottle.", self
        )
        self.instructions.setWordWrap(True)
        self.instructions.setObjectName("swapInstructions")
        layout.addWidget(self.instructions)
        choice_row = QtWidgets.QHBoxLayout()
        choice_row.addWidget(QtWidgets.QLabel("Replace", self))
        self.side_combo = _SwapComboBox(self)
        for label, sides in (("Left", ["left"]), ("Right", ["right"]), ("Both", ["left", "right"])):
            self.side_combo.addItem(label, sides)
        choice_row.addWidget(self.side_combo)
        self.pause_button = QtWidgets.QPushButton("Pause Both Lick Channels", self)
        self.pause_button.clicked.connect(self._begin)
        choice_row.addWidget(self.pause_button)
        self.choice_group = QtWidgets.QWidget(self)
        self.choice_group.setLayout(choice_row)
        choice_row.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.choice_group)
        self.check_status_button = QtWidgets.QPushButton("Check Controller Status", self)
        self.check_status_button.setObjectName("swapSecondary")
        self.check_status_button.clicked.connect(self.check_requested.emit)
        status_row = QtWidgets.QHBoxLayout()
        status_row.addStretch()
        status_row.addWidget(self.check_status_button)
        layout.addLayout(status_row)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.form_container = QtWidgets.QWidget(scroll)
        form_layout = QtWidgets.QVBoxLayout(self.form_container)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(12)
        form_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.form_container)
        layout.addWidget(scroll, 1)
        self.forms: dict[str, dict] = {}
        for side in ("left", "right"):
            group = QtWidgets.QGroupBox(f"{side.title()} bottle", self.form_container)
            form = QtWidgets.QFormLayout(group)
            form.setHorizontalSpacing(18)
            form.setVerticalSpacing(10)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            old = QtWidgets.QLabel(group)
            old.setWordWrap(True)
            final = QtWidgets.QLineEdit(group)
            final.setPlaceholderText("optional; complete later if unavailable")
            fluid = _SwapComboBox(group)
            fluid.setEditable(True)
            fluid.addItems(BOTTLE_FLUID_PRESETS)
            fluid.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
            fluid.setToolTip("Choose a fluid or type a custom fluid name.")
            fluid.setMaxVisibleItems(8)
            initial = QtWidgets.QLineEdit(group)
            for edit in (final, initial):
                edit.setValidator(QtGui.QDoubleValidator(0, 100000, 4, edit))
            form.addRow("Current bottle", old)
            form.addRow("Old bottle final (g)", final)
            form.addRow("Replacement fluid", fluid)
            form.addRow("Replacement initial (g)", initial)
            form_layout.addWidget(group)
            self.forms[side] = dict(group=group, old=old, final=final, fluid=fluid, initial=initial)
        self.error_label = QtWidgets.QLabel(self)
        self.error_label.setObjectName("swapError")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        self.resume_button = QtWidgets.QPushButton("Recalibrate & Resume", self)
        self.resume_button.clicked.connect(self._resume)
        self.resume_button.setMinimumWidth(220)
        footer = QtWidgets.QHBoxLayout()
        close = QtWidgets.QPushButton("Close", self)
        close.setObjectName("swapSecondary")
        close.setToolTip("Closing this window does not resume paused lick channels.")
        close.clicked.connect(self.close)
        footer.addWidget(close)
        footer.addStretch()
        footer.addWidget(self.resume_button)
        layout.addLayout(footer)
        apply_dark_combo_popups(self)
        self.apply_snapshot({}, busy=False)

    def _begin(self):
        sides = self._snapshot.get("selected_sides") if self._snapshot.get("state") == "UNCONFIRMED" else None
        self.begin_requested.emit(sides or self.side_combo.currentData())

    def _resume(self):
        try:
            payload = {}
            for side in self._snapshot.get("selected_sides", []):
                fields = self.forms[side]
                fluid = fields["fluid"].currentText().strip()
                initial = parse_weight(fields["initial"].text(), f"{side.title()} replacement initial weight", strict=True)
                if not fluid or initial is None:
                    raise ValueError(f"{side.title()} replacement requires a fluid and initial weight.")
                payload[side] = dict(fluid=fluid, initial_weight_g=initial,
                    final_weight_g=parse_weight(fields["final"].text(), f"{side.title()} final weight", strict=True))
            self.error_label.clear()
            self.resume_requested.emit(payload)
        except ValueError as exc:
            self.error_label.setText(str(exc))

    def apply_snapshot(self, snapshot: dict, *, busy: bool):
        self._snapshot = snapshot
        self._busy = busy
        state = str(snapshot.get("state", "UNKNOWN")).upper()
        selected = snapshot.get("selected_sides", [])
        paused = state in {"PAUSED", "FAILED"}
        active = state == "ACTIVE"
        if selected and not active:
            for index in range(self.side_combo.count()):
                if self.side_combo.itemData(index) == list(selected):
                    if self.side_combo.currentIndex() != index:
                        self.side_combo.setCurrentIndex(index)
                    break
        self.choice_group.setVisible(not paused and state != "CALIBRATING")
        self.side_combo.setEnabled(active and not busy)
        retry_pause = state == "UNCONFIRMED" and bool(selected) and not snapshot.get("resume_pending")
        self.pause_button.setText("Check / Retry Pause" if retry_pause else "Pause Both Lick Channels")
        self.pause_button.setEnabled((active or retry_pause) and not busy and bool(snapshot.get("available")))
        self.check_status_button.setEnabled(not busy)
        self.resume_button.setEnabled(paused and not busy and bool(selected))
        swap_id = snapshot.get("swap_id")
        new_swap = swap_id is not None and swap_id != self._loaded_swap
        if new_swap:
            self._loaded_swap = swap_id
        for side, fields in self.forms.items():
            fields["group"].setVisible(side in selected and not active)
            fields["group"].setEnabled(paused and not busy)
            bottle = snapshot.get("current", {}).get(side) or {}
            fields["old"].setText(
                f"{bottle.get('fluid') or 'Unspecified fluid'} · Initial: {bottle.get('initial_weight_g', '—')} g · "
                f"Licks: {bottle.get('licks', bottle.get('lick_count', 0))}"
            )
            if new_swap:
                fields["fluid"].setCurrentText(str(bottle.get("fluid") or ""))
                fields["initial"].clear()
                fields["final"].clear()
        if paused:
            self.state_label.setText("Both lick channels paused · Recording and other tasks continue")
            text = "Enter the old final weight, then the replacement fluid and initial weight. Keep hands clear of the spouts before recalibrating."
        elif state == "CALIBRATING":
            self.state_label.setText("Recalibrating both lick channels")
            text = "Calibrating both lick channels. Keep hands clear of the spouts. Waiting for controller confirmation."
        elif active:
            self.state_label.setText("Lick recording active")
            text = "Lick recording active. Select bottles, then pause both channels before handling."
        else:
            self.state_label.setText("Waiting for confirmed controller state")
            text = "Waiting for confirmed lickometer state. Keep bottles untouched until both channels are confirmed paused."
        self.instructions.setText(text)
        backend_error = str(snapshot.get("error") or "")
        if backend_error != self._last_backend_error:
            self.error_label.setText(describe_bottle_error(backend_error))
            self._last_backend_error = backend_error


class BottleSwapController(QtCore.QObject):
    """Perform all supervisor swap requests off the GUI heartbeat thread."""

    finished = QtCore.Signal(str, object, int)

    def __init__(self, panel, backend, recording, parent=None):
        super().__init__(parent)
        self.panel = panel
        self.backend = backend
        self.recording = recording
        self.snapshot: dict = {}
        self.busy = False
        self._operation = None
        self._generation = 0
        self._queued_request = None
        self._was_recording = False
        self._terminal_refresh_pending = False
        self.dialog = None
        self.finished.connect(self._finished)
        panel.swap_button.clicked.connect(self.open_dialog)
        panel.edit_history_button.clicked.connect(self.edit_history_weight)
        panel.check_status_button.clicked.connect(lambda: self._request("probe"))
        panel.history_previous.clicked.connect(lambda: self._history_page(-1))
        panel.history_next.clicked.connect(lambda: self._history_page(1))
        panel.history_latest.clicked.connect(self._history_latest)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def _dialog_is_busy(self):
        changing = self.busy and self._operation in {"begin", "resume"}
        queued = self._queued_request is not None and self._queued_request[0] in {"begin", "resume"}
        return not self.recording() or changing or queued

    def _panel_is_busy(self):
        return self.busy and self._operation in {"begin", "resume", "history_weight"}

    def begin_run(self):
        """Retire the preceding run's polls before starting a new request."""
        self._generation += 1
        self.busy = False
        self._operation = None
        self._queued_request = None
        self._terminal_refresh_pending = False
        self._was_recording = False
        self.snapshot = {}
        self.panel._history_page = None
        self.panel._history_offset = 0
        self.panel._current_bottle_ids = {}
        self.panel._bottle_history_locked = False
        self.panel._form_edits.clear()
        self.panel.apply_swap_snapshot({}, recording=False, busy=False)
        if self.dialog:
            self.dialog.close()

    def refresh(self):
        recording = self.recording()
        self.panel.apply_swap_snapshot(self.snapshot, recording=recording, busy=self._panel_is_busy())
        if self._was_recording and not recording:
            self._terminal_refresh_pending = True
        self._was_recording = recording
        if self.dialog:
            self.dialog.apply_snapshot(self.snapshot, busy=self._dialog_is_busy())
        if (recording or self._terminal_refresh_pending) and not self.busy and callable(getattr(self.backend, "bottle_swap_snapshot", None)):
            self._terminal_refresh_pending = False
            self._request("snapshot")

    def _request(self, operation, payload=None):
        if self.busy:
            # Polling never owns the form. Retain a user action that arrives
            # during a read and dispatch it as soon as that read completes.
            if operation not in {"snapshot", "history_page"} and self._operation in {"snapshot", "history_page", "probe"}:
                if self._queued_request is None:
                    self._queued_request = (operation, payload)
                if self.dialog:
                    self.dialog.apply_snapshot(self.snapshot, busy=self._dialog_is_busy())
            return
        if operation not in {"snapshot", "history_weight", "history_page"} and not self.recording():
            return
        self.busy = True
        self._operation = operation
        if operation == "begin":
            self.snapshot = dict(self.snapshot, selected_sides=list(payload), state="PAUSING")
        elif operation == "resume":
            self.snapshot = dict(self.snapshot, resume_pending=True, state="CALIBRATING")
        # Background reads do not disable/re-enable visible actions every poll.
        changing = operation in {"begin", "resume", "history_weight"}
        self.panel.apply_swap_snapshot(self.snapshot, recording=self.recording(), busy=changing)
        if self.dialog:
            self.dialog.apply_snapshot(self.snapshot, busy=self._dialog_is_busy())
        import threading
        generation = self._generation
        def work():
            try:
                if operation == "snapshot":
                    result = self.backend.bottle_swap_snapshot()
                elif operation == "probe":
                    result = self.backend.bottle_swap_snapshot(force_probe=True)
                elif operation == "history_page":
                    result = self.backend.bottle_history_page(offset=payload, limit=50)
                    result["offset"] = payload
                elif operation == "begin":
                    result = self.backend.begin_bottle_swap(payload)
                elif operation == "history_weight":
                    result = self.backend.update_bottle_history_weight(*payload)
                else:
                    result = self.backend.resume_bottle_swap(payload)
            except Exception as exc:
                result = dict(self.snapshot, state="UNCONFIRMED", error=str(exc))
            try:
                self.finished.emit(operation, result, generation)
            except RuntimeError:
                pass  # Window was destroyed while a supervisor request completed.
        threading.Thread(target=work, name="bottle-swap-gui", daemon=True).start()

    @QtCore.Slot(str, object, int)
    def _finished(self, operation, snapshot, generation=None):
        if generation is not None and generation != self._generation:
            return
        self.busy = False
        self._operation = None
        if operation != "history_page" and self.snapshot.get("run_dir") != (snapshot or {}).get("run_dir"):
            self.panel._history_page = None
            self.panel._history_offset = 0
        if operation == "history_page" and "history_count" in snapshot:
            self.panel._history_page = snapshot.get("history", [])
            self.panel._history_offset = snapshot.get("offset", 0)
            self.snapshot["history_count"] = snapshot["history_count"]
        else:
            self.snapshot = dict(snapshot or {})
        if not self.recording() and self.snapshot.get("current") and self.snapshot.get("state") != "STOPPED":
            self._terminal_refresh_pending = True
        queued = self._queued_request
        self._queued_request = None
        if queued is not None and (self.recording() or queued[0] == "history_weight"):
            self._request(*queued)
            return
        self.panel.apply_swap_snapshot(self.snapshot, recording=self.recording(), busy=False)
        if self.dialog:
            self.dialog.apply_snapshot(self.snapshot, busy=self._dialog_is_busy())
        if operation == "history_weight" and self.panel._history_page is not None and not self.snapshot.get("error"):
            self._request("history_page", self.panel._history_offset)

    def _history_page(self, direction):
        offset = self.panel._displayed_history_offset
        count = self.panel._history_count
        target = max(0, offset - 50) if direction < 0 else min(max(0, count - 50), offset + 50)
        self._request("history_page", target)

    def _history_latest(self):
        self.panel._history_page = None
        self.panel.apply_swap_snapshot(self.snapshot, recording=self.recording(), busy=self._panel_is_busy())

    def edit_history_weight(self):
        row = self.panel.history_table.currentRow()
        item = self.panel.history_table.item(row, 0) if row >= 0 else None
        bottle_id = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
        if not bottle_id:
            self.panel.set_status("Select a completed bottle in history to edit its final weight.")
            return
        text, accepted = QtWidgets.QInputDialog.getText(self.panel, "Bottle final weight", "Final weight (g)")
        if not accepted:
            return
        try:
            weight = parse_weight(text, "Final weight", strict=True)
            if weight is None:
                raise ValueError("Enter a final weight.")
        except ValueError as exc:
            self.panel.set_status(str(exc))
            return
        self._request("history_weight", (bottle_id, weight))

    def open_dialog(self):
        if self.dialog is None:
            self.dialog = BottleSwapDialog(self.panel.window())
            self.dialog.check_requested.connect(lambda: self._request("probe"))
            self.dialog.begin_requested.connect(lambda sides: self._request("begin", sides))
            self.dialog.resume_requested.connect(lambda replacements: self._request("resume", replacements))
        was_visible = self.dialog.isVisible()
        self.dialog.apply_snapshot(self.snapshot, busy=self._dialog_is_busy())
        self.dialog.show()
        if not was_visible:
            fit_window_to_available_area(self.dialog)
            center_window(self.dialog)
        self.dialog.raise_()
        self.dialog.activateWindow()
