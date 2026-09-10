"""Finalization progress presentation for the operator window.

The widget owns only the blocking overlay surface.  Translating backend status
and post-run progress into operator-facing copy is kept in a pure function so
every persisted finalization stage can be tested without constructing the main
window or starting Qt's event loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from PySide6 import QtCore, QtWidgets


@dataclass(frozen=True, slots=True)
class FinalizationPresentation:
    """Text presented while capture shutdown and validation are in progress."""

    stage: str
    title: str
    message: str
    progress_percent: int | None = None


_STAGE_COPY: dict[str, tuple[str, str]] = {
    "stopping": (
        "Stopping Capture…",
        "Stopping the controller and closing the capture pipeline.",
    ),
    "capture_draining": (
        "Draining Capture…",
        "Waiting for camera and recording buffers to become quiet.",
    ),
    "capture_drained": (
        "Closing Capture…",
        "All captured buffers are accounted for; closing the MP4.",
    ),
    "capture_closed": (
        "Capture Saved — Validating…",
        "The MP4 is safely closed. Starting post-run validation.",
    ),
    "capture_reconciliation": (
        "Validating Capture…",
        "Reconciling source frames with the non-leaky recording branch.",
    ),
    "inference_admission": (
        "Validating Inference…",
        "Checking which captured frames entered inference.",
    ),
    "recording_validation": (
        "Validating Video…",
        "Comparing MP4 frame counts with recorded source frames.",
    ),
    "recording_validation_complete": (
        "Video Validated…",
        "Recording frame counts passed. Preparing timing alignment.",
    ),
    "streaming_alignment": (
        "Aligning Timing…",
        "Aligning camera and controller timestamps.",
    ),
    "complete": (
        "Validation Complete…",
        "Capture validation passed; finishing the run manifest.",
    ),
}

_CAPTURE_STATES = {
    "stopping",
    "capture_draining",
    "capture_drained",
    "capture_closed",
}


def _nonnegative_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0.0 else None


def _duration_text(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def present_finalization(
    status: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> FinalizationPresentation:
    """Resolve persisted run state into deterministic operator-facing copy."""

    state = str(status.get("state") or "stopping")
    if state in {"finalizing", "analyzing"}:
        stage = str(status.get("stage") or progress.get("stage") or state)
    elif state in _CAPTURE_STATES:
        stage = state
    else:
        stage = str(progress.get("stage") or status.get("stage") or state)

    title, message = _STAGE_COPY.get(
        stage,
        _STAGE_COPY.get(state, ("Finalizing Run…", f"Post-run stage: {stage}")),
    )
    progress_percent = None
    decoded = _nonnegative_number(progress.get("video_frames_decoded"))
    expected = _nonnegative_number(progress.get("video_frames_expected"))
    if stage == "recording_validation" and decoded is not None and expected:
        percentage = min(100.0, decoded * 100.0 / expected)
        progress_percent = int(round(percentage))
        message = (
            f"Validated {int(decoded):,} of {int(expected):,} video samples "
            f"({percentage:.1f}%)."
        )
        rate = _nonnegative_number(progress.get("video_validation_rate_fps"))
        if rate:
            message += f"  {rate:.1f} frames/s."
        eta = _nonnegative_number(progress.get("video_validation_eta_s"))
        elapsed = _nonnegative_number(progress.get("video_validation_elapsed_s"))
        if eta is not None and decoded > 0:
            message += f"  About {_duration_text(eta)} remaining."
        elif elapsed is not None:
            message += f"  Elapsed {_duration_text(elapsed)}."
    else:
        processed = int(progress.get("frames_processed") or 0)
        if processed:
            message += f"  {processed:,} frames processed."
    return FinalizationPresentation(
        stage=stage,
        title=title,
        message=message,
        progress_percent=progress_percent,
    )


class FinalizationOverlay(QtWidgets.QFrame):
    """Full-central-widget overlay shown during bounded run finalization."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("stopOverlay")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.title_label = QtWidgets.QLabel("Finalizing Run…", self)
        self.title_label.setObjectName("stopOverlayTitle")
        self.message_label = QtWidgets.QLabel(
            "Finalizing capture/inference, closing files, and stopping serial control.",
            self,
        )
        self.message_label.setObjectName("stopOverlayMsg")
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(360)

        layout.addWidget(self.title_label, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message_label, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_bar, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        self.hide()
        self.raise_()
        self.resize_to_parent()

    def resize_to_parent(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(0, 0, parent.width(), parent.height())

    def show_overlay(self) -> None:
        self.resize_to_parent()
        self.raise_()
        self.show()

    def hide_overlay(self) -> None:
        self.hide()

    def apply_presentation(self, presentation: FinalizationPresentation) -> None:
        self.title_label.setText(presentation.title)
        self.message_label.setText(presentation.message)
        if presentation.progress_percent is None:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setTextVisible(False)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(presentation.progress_percent)
            self.progress_bar.setTextVisible(True)

    def update_progress(
        self,
        status: Mapping[str, Any],
        progress: Mapping[str, Any],
    ) -> FinalizationPresentation:
        presentation = present_finalization(status, progress)
        self.apply_presentation(presentation)
        return presentation
