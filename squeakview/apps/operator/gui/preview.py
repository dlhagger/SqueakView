"""Live-preview widget used by the operator window.

Keeping the preview surface in its own module prevents the main window from
owning rendering, overlay layout, and disabled-preview animation details.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview import config as squeakview_config


class PreviewWidget(QtWidgets.QWidget):
    """Native video target with status overlays and a disabled-state logo."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NativeWindow)
        self.setMinimumHeight(260)
        self.setMinimumWidth(320)
        self.setStyleSheet("background-color: #0f1118; border: 1px solid #24283b; border-radius: 10px;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel("Live preview will appear here once DeepStream starts…")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #8088a6; letter-spacing: 0.2px;")
        layout.addWidget(self.label, 1)

        self.logo_label = QtWidgets.QLabel(self)
        self.logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setStyleSheet("background: rgba(15,17,24,0.85);")
        self.logo_label.hide()
        self._logo_pixmap = self._load_logo()
        self._logo_scale = 1.0
        self._logo_anim: QtCore.QVariantAnimation | None = None
        self._logo_opacity = QtWidgets.QGraphicsOpacityEffect(self.logo_label)
        self.logo_label.setGraphicsEffect(self._logo_opacity)
        self._logo_opacity.setOpacity(1.0)
        self._target_aspect = 4.0 / 3.0

        self.status_badge = QtWidgets.QLabel("Idle", self)
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setStyleSheet("""
            QLabel#statusBadge {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3a3f5c, stop:1 #2d3046);
                color: #e8ebf4;
                padding: 4px 10px;
                border-radius: 8px;
                font-weight: 700;
                font-size: 11px;
            }
        """)
        self.info_label = QtWidgets.QLabel("", self)
        self.info_label.setStyleSheet(
            "color: #a5adc8; background: rgba(15,17,24,0.6); padding: 4px 8px; border-radius: 8px; font-size: 11px;"
        )
        self.info_label.hide()
        self._preview_enabled = True

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_aspect_constraint()
        self._reposition_overlays()
        self._update_logo_scale()

    def _update_aspect_constraint(self) -> None:
        height = max(1, self.height())
        target_width = max(360, int(round(height * self._target_aspect)))
        if self.maximumWidth() != target_width:
            self.setMaximumWidth(target_width)

    def _reposition_overlays(self) -> None:
        margin = 12
        self.status_badge.adjustSize()
        self.status_badge.move(margin, margin)
        if self.info_label.isVisible():
            self.info_label.adjustSize()
            info_y = margin + self.status_badge.height() + 6
            self.info_label.move(margin, info_y)
        self.logo_label.setGeometry(0, 0, self.width(), self.height())

    def set_status(self, text: str, *, color: str | None = None) -> None:
        self.status_badge.setText(text)
        normalized = text.lower()
        if color is None:
            if "fail" in normalized or "ended" in normalized or "unavailable" in normalized:
                color = "#a93750"
            elif "live" in normalized or "recording" in normalized:
                color = "#c4425f"
            elif "start" in normalized or "waiting" in normalized:
                color = "#a66a20"
            elif "final" in normalized or "stopping" in normalized:
                color = "#6855c7"
            elif "complete" in normalized:
                color = "#267a58"
            elif "ready" in normalized:
                color = "#4357bd"
            else:
                color = "#34394f"
        self.status_badge.setStyleSheet(
            "QLabel#statusBadge {"
            f"background-color: {color}; color: #ffffff; padding: 4px 10px; "
            "border-radius: 8px; font-weight: 700; font-size: 11px; }"
        )
        self._reposition_overlays()

    def set_info(self, text: str | None) -> None:
        if text:
            self.info_label.setText(text)
            self.info_label.show()
        else:
            self.info_label.hide()
        self._reposition_overlays()

    def window_id(self) -> int:
        return int(self.winId())

    def show_hint(self, visible: bool) -> None:
        self.label.setVisible(visible)

    def set_preview_enabled(self, enabled: bool) -> None:
        self._preview_enabled = enabled
        if enabled:
            self._stop_logo_anim()
            self.logo_label.hide()
            self.label.hide()
        else:
            self.label.setText("Preview disabled")
            self.label.show()
            self.logo_label.show()
            self._start_logo_anim()
            self._update_logo_scale()
        self._reposition_overlays()

    def _load_logo(self) -> QtGui.QPixmap | None:
        try:
            logo_path = squeakview_config.WORKSPACE / "SqueakView_logo.png"
            if logo_path.exists():
                pix = QtGui.QPixmap(str(logo_path))
                return pix if not pix.isNull() else None
        except Exception:
            return None
        return None

    def _update_logo_scale(self) -> None:
        if not self.logo_label.isVisible():
            return
        if self._logo_pixmap is None:
            self.logo_label.setText("Preview disabled")
            return
        pix = self._logo_pixmap.scaled(
            self.logo_label.size() * (0.6 * self._logo_scale),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.logo_label.setPixmap(pix)

    def _start_logo_anim(self) -> None:
        self._stop_logo_anim()
        self._logo_scale = 0.75
        self._logo_opacity.setOpacity(0.0)
        self._update_logo_scale()
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(180)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        def _on_value(value: object) -> None:
            t = float(value)
            self._logo_scale = 0.75 + (0.25 * t)
            self._logo_opacity.setOpacity(t)
            self._update_logo_scale()

        anim.valueChanged.connect(_on_value)
        self._logo_anim = anim
        anim.start()

    def _stop_logo_anim(self) -> None:
        if self._logo_anim is not None:
            self._logo_anim.stop()
            self._logo_anim.deleteLater()
            self._logo_anim = None
        self._logo_scale = 1.0
        self._logo_opacity.setOpacity(1.0)
