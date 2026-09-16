"""Live-preview widget used by the operator window.

Keeping the preview surface in its own module prevents the main window from
owning rendering, overlay layout, and disabled-preview animation details.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview import config as squeakview_config


class AspectRatioPreviewHost(QtWidgets.QWidget):
    """Center a native preview surface at the largest size that fits the card."""

    def __init__(
        self,
        preview: "PreviewWidget",
        *,
        aspect_ratio: float = 4.0 / 3.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be positive")
        self._aspect_ratio = float(aspect_ratio)
        self.preview = preview
        self.preview.setParent(self)
        self.preview.setMinimumSize(0, 0)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_preview()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self._fit_preview()

    def set_aspect_ratio(self, aspect_ratio: float) -> None:
        """Update the source aspect ratio without recreating the native window."""

        if aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be positive")
        self._aspect_ratio = float(aspect_ratio)
        self._fit_preview()

    def _fit_preview(self) -> None:
        bounds = self.contentsRect()
        available_width = bounds.width()
        available_height = bounds.height()
        if available_width <= 0 or available_height <= 0:
            self.preview.setGeometry(bounds.x(), bounds.y(), 0, 0)
            return

        if available_width / available_height > self._aspect_ratio:
            preview_height = available_height
            preview_width = int(round(preview_height * self._aspect_ratio))
        else:
            preview_width = available_width
            preview_height = int(round(preview_width / self._aspect_ratio))

        preview_width = min(available_width, max(1, preview_width))
        preview_height = min(available_height, max(1, preview_height))
        preview_x = bounds.x() + (available_width - preview_width) // 2
        preview_y = bounds.y() + (available_height - preview_height) // 2
        self.preview.setGeometry(
            preview_x,
            preview_y,
            preview_width,
            preview_height,
        )


class PreviewWidget(QtWidgets.QWidget):
    """Native video target with status overlays and a disabled-state logo."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NativeWindow)
        self.setStyleSheet("background-color: #0f1118; border: 1px solid #24283b; border-radius: 10px;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel("Live preview will appear here once DeepStream starts…")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setMargin(18)
        self.label.setStyleSheet("color: #9ba5c7; letter-spacing: 0.2px; font-size: 12px;")
        self._protect_overlay_from_video_sink(self.label)
        layout.addWidget(self.label, 1)

        self.logo_label = QtWidgets.QLabel(self)
        self.logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setStyleSheet("background: rgba(15,17,24,0.85);")
        self._protect_overlay_from_video_sink(self.logo_label)
        self.logo_label.hide()
        self._logo_pixmap = self._load_logo()
        self._logo_scale = 1.0
        self._logo_anim: QtCore.QVariantAnimation | None = None
        self._logo_opacity = QtWidgets.QGraphicsOpacityEffect(self.logo_label)
        self.logo_label.setGraphicsEffect(self._logo_opacity)
        self._logo_opacity.setOpacity(1.0)
        self.status_badge = QtWidgets.QLabel("Idle", self)
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setStyleSheet("""
            QLabel#statusBadge {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3a3f5c, stop:1 #2d3046);
                color: #e8ebf4;
                padding: 4px 10px;
                border-radius: 8px;
                font-weight: 700;
                font-size: 12px;
            }
        """)
        self._protect_overlay_from_video_sink(self.status_badge)
        self.info_label = QtWidgets.QLabel("", self)
        self.info_label.setStyleSheet(
            "color: #c2cae5; background: rgba(15,17,24,0.82); padding: 5px 9px; border-radius: 8px; font-size: 12px;"
        )
        self._protect_overlay_from_video_sink(self.info_label)
        self.info_label.hide()
        self._preview_enabled = True

    @staticmethod
    def _protect_overlay_from_video_sink(widget: QtWidgets.QWidget) -> None:
        """Give an overlay its own child window above the sink's paint target."""

        widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_NativeWindow)
        widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reposition_overlays()
        self._update_logo_scale()

    def _reposition_overlays(self) -> None:
        margin = 12
        self.status_badge.adjustSize()
        self.status_badge.move(margin, margin)
        if self.info_label.isVisible():
            self.info_label.adjustSize()
            info_y = margin + self.status_badge.height() + 6
            self.info_label.move(margin, info_y)
        self.logo_label.setGeometry(0, 0, self.width(), self.height())
        # The video sink paints continuously into this widget's native window.
        # Native child overlays stay above that paint target; raising here also
        # restores their order after a resize or visibility transition.
        self.label.raise_()
        self.logo_label.raise_()
        self.status_badge.raise_()
        self.info_label.raise_()

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
            "border-radius: 8px; font-weight: 700; font-size: 12px; }"
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
            logo_path = squeakview_config.APP_ROOT / "SqueakView_logo.png"
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
