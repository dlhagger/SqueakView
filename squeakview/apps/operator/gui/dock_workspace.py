from __future__ import annotations

"""Versioned native-Qt card workspace for the operator application."""

from collections.abc import Mapping
from typing import Protocol

from PySide6 import QtCore, QtGui, QtWidgets


LAYOUT_SCHEMA_VERSION = 1
LAYOUT_STATE_KEY = f"operator_workspace/state_v{LAYOUT_SCHEMA_VERSION}"


class SettingsStore(Protocol):
    def value(self, key: str, defaultValue=None): ...  # noqa: N803

    def setValue(self, key: str, value) -> None: ...  # noqa: N802


class DockWorkspace(QtCore.QObject):
    """Own movable cards, a known-good default, and bounded layout restore."""

    layout_editable_changed = QtCore.Signal(bool)
    runtime_lock_changed = QtCore.Signal(bool)

    def __init__(
        self,
        host: QtWidgets.QMainWindow,
        *,
        settings: SettingsStore | None = None,
    ) -> None:
        super().__init__(host)
        self._host = host
        host.setObjectName("workspaceDockHost")
        host.setDockNestingEnabled(True)
        host.setAnimated(False)
        self._settings = settings or QtCore.QSettings("SqueakView", "SqueakView")
        self._cards: dict[str, QtWidgets.QDockWidget] = {}
        self._default_state = QtCore.QByteArray()
        self._user_editable = False
        self._runtime_locked = False
        self._content_expansions = {}
        self._pending_content_restores = {}
        self._content_restore_timer = QtCore.QTimer(self)
        self._content_restore_timer.setSingleShot(True)
        self._content_restore_timer.timeout.connect(self._restore_content_sizes)

    @property
    def cards(self) -> Mapping[str, QtWidgets.QDockWidget]:
        return self._cards

    @property
    def layout_editable(self) -> bool:
        return self._user_editable and not self._runtime_locked

    def add_card(
        self,
        card_id: str,
        title: str,
        content: QtWidgets.QWidget,
    ) -> QtWidgets.QDockWidget:
        if card_id in self._cards:
            raise ValueError(f"Duplicate workspace card id: {card_id}")
        dock = QtWidgets.QDockWidget(title, self._host)
        dock.setObjectName(f"workspaceCard.{card_id}")
        dock.setProperty("workspaceCard", True)
        dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setWidget(content)
        self._host.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self._cards[card_id] = dock
        self._apply_card_features(dock)
        return dock

    def establish_default_layout(self) -> None:
        """Arrange the canonical operator workspace and capture it for reset."""

        required = {
            "preview",
            "system",
            "clock",
            "task",
            "bottles",
            "behavior",
            "events",
        }
        missing = required.difference(self._cards)
        if missing:
            raise RuntimeError(
                "Cannot establish workspace layout; missing cards: "
                + ", ".join(sorted(missing))
            )

        preview = self._cards["preview"]
        system = self._cards["system"]
        clock = self._cards["clock"]
        task = self._cards["task"]
        bottles = self._cards["bottles"]
        behavior = self._cards["behavior"]
        events = self._cards["events"]

        self._host.splitDockWidget(preview, behavior, QtCore.Qt.Orientation.Vertical)
        self._host.splitDockWidget(preview, system, QtCore.Qt.Orientation.Horizontal)
        self._host.splitDockWidget(system, clock, QtCore.Qt.Orientation.Vertical)
        self._host.splitDockWidget(system, task, QtCore.Qt.Orientation.Horizontal)
        self._host.splitDockWidget(task, bottles, QtCore.Qt.Orientation.Vertical)
        self._host.splitDockWidget(behavior, events, QtCore.Qt.Orientation.Vertical)

        self._host.resizeDocks(
            [preview, system, task],
            [760, 620, 540],
            QtCore.Qt.Orientation.Horizontal,
        )
        self._host.resizeDocks(
            [preview, behavior],
            [600, 400],
            QtCore.Qt.Orientation.Vertical,
        )
        self._host.resizeDocks(
            [task, bottles],
            [360, 240],
            QtCore.Qt.Orientation.Vertical,
        )
        self._host.resizeDocks(
            [behavior, events],
            [400, 260],
            QtCore.Qt.Orientation.Vertical,
        )
        events.hide()
        clock.hide()
        for card_id, dock in self._cards.items():
            if card_id not in {"events", "clock"}:
                dock.show()
        self._default_state = self._host.saveState(LAYOUT_SCHEMA_VERSION)

    def reset_layout(self) -> None:
        self._content_expansions.clear()
        self._pending_content_restores.clear()
        if self._default_state.isEmpty() or not self._host.restoreState(
            self._default_state, LAYOUT_SCHEMA_VERSION
        ):
            self.establish_default_layout()
        self._recover_offscreen_floating_cards()

    def restore_layout(self) -> bool:
        """Restore a compatible saved state or atomically use the default."""

        self._content_expansions.clear()
        self._pending_content_restores.clear()
        raw_state = self._settings.value(LAYOUT_STATE_KEY)
        if isinstance(raw_state, bytes):
            raw_state = QtCore.QByteArray(raw_state)
        restored = isinstance(raw_state, QtCore.QByteArray) and not raw_state.isEmpty()
        if restored:
            restored = self._host.restoreState(raw_state, LAYOUT_SCHEMA_VERSION)
        if not restored:
            self.reset_layout()
        self._recover_offscreen_floating_cards()
        return bool(restored)

    def save_layout(self) -> None:
        self._settings.setValue(
            LAYOUT_STATE_KEY, self._host.saveState(LAYOUT_SCHEMA_VERSION)
        )
        sync = getattr(self._settings, "sync", None)
        if callable(sync):
            sync()
        status = getattr(self._settings, "status", None)
        if callable(status) and status() != QtCore.QSettings.Status.NoError:
            raise OSError("Qt could not persist the operator workspace layout")

    def set_user_editable(self, editable: bool) -> None:
        self._user_editable = bool(editable)
        self._apply_all_card_features()

    def set_card_content_expanded(self, card_id: str, expanded: bool) -> None:
        """Return space borrowed by temporary card details after collapse.

        Qt grows neighboring dock rows to satisfy an expanded child's minimum
        size, but keeps those splitter sizes when the child is hidden again.
        Remember sizes rather than the full dock state so visibility, card
        placement, and recording controls are never reset.
        """
        if expanded:
            self._pending_content_restores.pop(card_id, None)
            self._content_expansions.setdefault(card_id, {
                key: QtCore.QSize(dock.size()) for key, dock in self._cards.items()
                if dock.isVisible()
            })
        else:
            sizes = self._content_expansions.pop(card_id, None)
            if sizes is not None:
                self._pending_content_restores[card_id] = sizes
                # Wait until the hidden children's minimum sizes propagate.
                self._content_restore_timer.start(0)

    def _restore_content_sizes(self) -> None:
        pending = self._pending_content_restores
        self._pending_content_restores = {}
        for card_id, sizes in pending.items():
            dock = self._cards[card_id]
            if dock.isFloating():
                if card_id in sizes:
                    dock.resize(sizes[card_id])
                continue
            docked = [(self._cards[key], size) for key, size in sizes.items()
                      if self._cards[key].isVisible() and not self._cards[key].isFloating()]
            for orientation in (QtCore.Qt.Orientation.Horizontal, QtCore.Qt.Orientation.Vertical):
                self._host.resizeDocks([widget for widget, _size in docked], [
                    size.width() if orientation == QtCore.Qt.Orientation.Horizontal else size.height()
                    for _widget, size in docked
                ], orientation)

    def set_runtime_locked(self, locked: bool) -> None:
        self._runtime_locked = bool(locked)
        self._apply_all_card_features()
        self.runtime_lock_changed.emit(self._runtime_locked)

    def _apply_all_card_features(self) -> None:
        for dock in self._cards.values():
            self._apply_card_features(dock)
        self.layout_editable_changed.emit(self.layout_editable)

    def _apply_card_features(self, dock: QtWidgets.QDockWidget) -> None:
        # Closable enables Qt's native, checkable card-visibility actions in
        # the title-bar context menu. Closing a card only hides its widget.
        features = QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        if self.layout_editable:
            features |= (
                QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            )
        dock.setFeatures(features)

    def _recover_offscreen_floating_cards(self) -> None:
        screens = QtGui.QGuiApplication.screens()
        available = [screen.availableGeometry() for screen in screens]
        for dock in self._cards.values():
            if not dock.isFloating():
                continue
            frame = dock.frameGeometry()
            if not any(rect.intersects(frame) for rect in available):
                dock.setFloating(False)


def build_layout_menu(
    button: QtWidgets.QPushButton,
    workspace: DockWorkspace,
) -> dict[str, QtGui.QAction]:
    """Attach the small, explicit editing surface to the pinned run bar."""

    menu = QtWidgets.QMenu(button)
    unlock_action = menu.addAction("Unlock card layout")
    unlock_action.setCheckable(True)
    save_action = menu.addAction("Save current layout")
    reset_action = menu.addAction("Reset to default layout")
    unlock_action.toggled.connect(workspace.set_user_editable)
    save_action.triggered.connect(workspace.save_layout)
    reset_action.triggered.connect(workspace.reset_layout)
    workspace.layout_editable_changed.connect(unlock_action.setChecked)

    def apply_runtime_lock(locked: bool) -> None:
        unlock_action.setEnabled(not locked)
        reset_action.setEnabled(not locked)

    workspace.runtime_lock_changed.connect(apply_runtime_lock)
    button.setMenu(menu)
    return {
        "unlock": unlock_action,
        "save": save_action,
        "reset": reset_action,
    }


__all__ = [
    "DockWorkspace",
    "LAYOUT_SCHEMA_VERSION",
    "LAYOUT_STATE_KEY",
    "build_layout_menu",
]
