"""Keep cyclic destruction of Qt wrappers on their owning GUI thread.

Python can otherwise initiate cyclic GC in an IPC/console worker, including
collecting unreferenced QObject graphs whose timers belong to the Qt thread.
This is process-local presentation policy, independent of acquisition workers.
"""
from __future__ import annotations

import gc

from PySide6 import QtCore


class GuiGarbageCollector(QtCore.QObject):
    def __init__(self, app, *, interval_ms=5000):
        super().__init__(app)
        self._closed = False
        self._collecting = False
        self._previously_enabled = gc.isenabled()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(max(1000, int(interval_ms)))
        self._timer.timeout.connect(self.collect)
        app.aboutToQuit.connect(self.close)
        gc.disable()
        self._timer.start()

    def _require_gui_thread(self):
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("GUI cyclic garbage collection must run on the Qt thread")

    @QtCore.Slot()
    def collect(self):
        self._require_gui_thread()
        if self._closed or self._collecting:
            return
        self._collecting = True
        try:
            gc.collect()
        finally:
            self._collecting = False

    @QtCore.Slot()
    def close(self):
        self._require_gui_thread()
        if self._closed:
            return
        self._timer.stop()
        try:
            self.collect()
        finally:
            self._closed = True
            if self._previously_enabled:
                gc.enable()


def install_gui_gc(app):
    """Install once per QApplication and preserve the caller's prior GC state."""
    existing = getattr(app, "_squeakview_gc_guard", None)
    if existing is None or existing._closed:
        existing = GuiGarbageCollector(app)
        app._squeakview_gc_guard = existing
    return existing
