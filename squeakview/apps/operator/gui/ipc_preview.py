from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Callable

from PySide6 import QtCore


class IpcPreviewController(QtCore.QObject):
    """Own a disposable renderer process; native video failures stay outside Qt."""

    ready = QtCore.Signal()
    failed = QtCore.Signal(str)
    ended = QtCore.Signal()

    def __init__(self, emit: Callable[[str], None], parent=None) -> None:
        super().__init__(parent)
        self._emit = emit
        self._process = None
        self._retiring = set()
        self._kill_deadlines = {}
        self._retire_timer = QtCore.QTimer(self)
        self._retire_timer.setInterval(100)
        self._retire_timer.timeout.connect(self._reap_retiring)
        self._output = bytearray()
        self._stderr = bytearray()
        self._ready = False

    @property
    def running(self) -> bool:
        return self._process is not None

    def start(self, socket_path: Path, window_id: int) -> bool:
        self.stop()
        if int(window_id) <= 0:
            self._fail("Qt preview window is not available")
            return False
        if len(self._retiring) >= 2:
            self._fail("previous preview renderers are still stopping; retry shortly")
            return False
        process = QtCore.QProcess(self)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.SeparateChannels)
        self._process = process
        self._output.clear()
        self._stderr.clear()
        self._ready = False
        process.readyReadStandardOutput.connect(self._output_ready)
        process.readyReadStandardError.connect(self._errors_ready)
        process.finished.connect(self._worker_finished)
        process.errorOccurred.connect(self._worker_error)
        program, arguments = self._renderer_command(socket_path, window_id)
        process.start(program, arguments)
        self._emit(f"[PREVIEW] waiting for IPC frames on {socket_path}")
        return True

    @staticmethod
    def _renderer_command(socket_path, window_id):
        return sys.executable, ["-m", "squeakview.apps.operator.gui.preview_worker",
                                "--socket", str(socket_path), "--window-id", str(int(window_id)),
                                "--parent-pid", str(os.getpid())]

    def stop(self) -> None:
        process = self._process
        self._process = None
        self._ready = False
        if process is None:
            return
        self._retiring.add(process)
        # All waiting happens in the worker. Even a driver teardown that holds
        # the GIL cannot block the GUI event loop or its heartbeat timer.
        process.terminate()
        self._kill_deadlines[process] = time.monotonic() + 1.5
        self._retire_timer.start()

    @QtCore.Slot()
    def _reap_retiring(self):
        now = time.monotonic()
        for process in tuple(self._retiring):
            if now >= self._kill_deadlines.get(process, float("inf")):
                self._kill_retiring(process)
                self._kill_deadlines[process] = float("inf")
        if not self._retiring:
            self._retire_timer.stop()

    @QtCore.Slot()
    def _output_ready(self):
        self._read_output(self.sender())

    @QtCore.Slot()
    def _errors_ready(self):
        self._read_errors(self.sender())

    @QtCore.Slot(int, QtCore.QProcess.ExitStatus)
    def _worker_finished(self, code, status):
        self._finished(self.sender(), code, status)

    @QtCore.Slot(QtCore.QProcess.ProcessError)
    def _worker_error(self, error):
        self._process_error(self.sender(), error)

    def _kill_retiring(self, process) -> None:
        if process in self._retiring:
            process.kill()

    def _read_errors(self, process) -> None:
        data = bytes(process.readAllStandardError())
        if process is self._process:
            self._stderr.extend(data)
            del self._stderr[:-8192]

    def _read_output(self, process) -> None:
        data = bytes(process.readAllStandardOutput())
        if process is not self._process:
            return
        self._output.extend(data)
        if len(self._output) > 65536:
            self.stop()
            self._fail("renderer produced an oversized status message")
            return
        while b"\n" in self._output:
            line, _, remaining = self._output.partition(b"\n")
            self._output[:] = remaining
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                continue  # Native libraries may write their own diagnostics.
            if not isinstance(event, dict):
                continue
            if event.get("event") == "ready" and not self._ready:
                self._ready = True
                self._emit("[PREVIEW] embedded IPC preview is playing")
                self.ready.emit()
            elif event.get("event") == "error":
                detail = str(event.get("detail", "renderer failed"))[:4096]
                self.stop()
                self._fail(detail)
                return
            elif event.get("event") == "ended":
                self.stop()
                self.ended.emit()
                return

    def _release_process(self, process):
        self._retiring.discard(process)
        self._kill_deadlines.pop(process, None)
        if not self._retiring:
            self._retire_timer.stop()
        for signal in (process.readyReadStandardOutput, process.readyReadStandardError,
                       process.finished, process.errorOccurred):
            signal.disconnect()
        process.deleteLater()

    def _process_error(self, process, error) -> None:
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            current = process is self._process
            detail = process.errorString()
            if current:
                self._process = None
            self._release_process(process)
            if current:
                self._fail(detail)

    def _finished(self, process, code, status) -> None:
        self._read_errors(process)
        current = process is self._process
        self._retiring.discard(process)
        if current:
            self._process = None
            self._ready = False
            detail = self._stderr.decode("utf-8", errors="replace").strip()[-2048:]
            suffix = f": {detail}" if detail else ""
            self._fail(f"renderer exited (code {code}, status {status.name}){suffix}")
        self._release_process(process)

    def _fail(self, detail: str) -> None:
        message = f"Preview unavailable: {detail}"
        self._emit(f"[PREVIEW] {message}")
        self.failed.emit(message)
