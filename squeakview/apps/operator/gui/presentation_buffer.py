"""Bound display traffic before Qt's otherwise unbounded cross-thread queue."""
from collections import deque
import queue
import threading

from PySide6 import QtCore

from .dashboard_model import feeder_jam_event


class GuiPresentationBuffer(QtCore.QObject):
    wake = QtCore.Signal()
    MAX_ITEMS = 2048
    BATCH_ITEMS = 64

    def __init__(self, deliver_log, deliver_dashboard, parent=None, *, console=True):
        super().__init__(parent)
        self._log = deliver_log
        self._dashboard = deliver_dashboard
        self._items = deque(maxlen=self.MAX_ITEMS)
        self._feeder = None
        self._lock = threading.Lock()
        self._scheduled = False
        self._closed = False
        self._dropped = 0
        self._console = queue.Queue(maxsize=512) if console else None
        self._console_state = {"dropped": 0, "lock": threading.Lock()}
        self._drain_timer = QtCore.QTimer(self)
        self._drain_timer.setSingleShot(True)
        self._drain_timer.timeout.connect(self._drain)
        if console:
            threading.Thread(target=type(self)._write_console, args=(self._console, self._console_state),
                             name="gui-console", daemon=True).start()
        self.wake.connect(self._drain, QtCore.Qt.ConnectionType.QueuedConnection)

    def post_log(self, message):
        message = str(message)[:8192]
        if self._console is not None:
            try:
                self._console.put_nowait(message)
            except queue.Full:
                with self._console_state["lock"]:
                    self._console_state["dropped"] += 1
        self._post("log", message)

    def post_dashboard(self, event):
        self._post("dashboard", event)

    def _post(self, kind, value):
        with self._lock:
            if self._closed:
                return
            if kind == "dashboard" and feeder_jam_event(value) is not None:
                # Preserve the latest authoritative latch state even under
                # display shedding. Scientific records remain in the journal.
                self._feeder = value
            else:
                if len(self._items) == self.MAX_ITEMS:
                    self._dropped += 1
                self._items.append((kind, value))
            if self._scheduled:
                return
            self._scheduled = True
        self.wake.emit()

    @QtCore.Slot()
    def _drain(self):
        with self._lock:
            if self._closed:
                self._scheduled = False
                return
            batch = [self._items.popleft() for _ in range(min(len(self._items), self.BATCH_ITEMS))]
            feeder, dropped = self._feeder, self._dropped
            self._feeder, self._dropped = None, 0
            more = bool(self._items)
            if not more:
                self._scheduled = False
        if dropped:
            self._log(f"[GUI] Display backlog shed {dropped} presentation events; scientific logging continues independently.")
        for kind, value in batch:
            (self._log if kind == "log" else self._dashboard)(value)
        if feeder is not None:
            self._dashboard(feeder)
        if more:
            self._drain_timer.start(0)

    @staticmethod
    def _write_console(messages, state):
        # The daemon owns plain Python state only. Holding a QObject here can
        # destroy its timers on the worker thread when the GUI is collected.
        while True:
            message = messages.get()
            if message is None:
                return
            try:
                with state["lock"]:
                    dropped, state["dropped"] = state["dropped"], 0
                if dropped:
                    print(f"[GUI] Terminal backlog shed {dropped} presentation messages.", flush=True)
                print(message, flush=True)
            except (BrokenPipeError, OSError, ValueError):
                return

    def close(self):
        self._drain_timer.stop()
        with self._lock:
            self._closed = True
            self._items.clear()
            self._feeder = None
            self._log = self._dashboard = lambda _value: None
        if self._console is not None:
            while True:
                try:
                    self._console.get_nowait()
                except queue.Empty:
                    break
            self._console.put_nowait(None)
