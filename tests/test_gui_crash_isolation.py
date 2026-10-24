from __future__ import annotations

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import threading
import unittest
from pathlib import Path
from unittest import mock

from PySide6 import QtCore, QtWidgets
from squeakview.apps.operator.gui.ipc_preview import IpcPreviewController
from squeakview.apps.operator.gui.presentation_buffer import GuiPresentationBuffer
from squeakview.common.dashboard import DashboardEvent

PROCESS_TYPE = QtCore.QProcess


class FakeProcess(QtCore.QObject):
    readyReadStandardOutput = QtCore.Signal()
    readyReadStandardError = QtCore.Signal()
    finished = QtCore.Signal(int, QtCore.QProcess.ExitStatus)
    errorOccurred = QtCore.Signal(QtCore.QProcess.ProcessError)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.output = b''
        self.errors = b''
        self.terminated = False
        self.killed = False
    def setProcessChannelMode(self, _mode): pass
    def start(self, program, arguments): self.command = (program, arguments)
    def terminate(self): self.terminated = True
    def kill(self): self.killed = True
    def readAllStandardOutput(self):
        output, self.output = self.output, b''
        return output
    def readAllStandardError(self):
        output, self.errors = self.errors, b''
        return output
    def errorString(self): return 'failed start'


class GuiCrashIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def process_factory(self, parent=None):
        process = FakeProcess(parent)
        self.processes.append(process)
        return process

    def setUp(self):
        self.processes = []
        self.controllers = []
        self.buffers = []

    def make_controller(self, *args, **kwargs):
        controller = IpcPreviewController(*args, **kwargs)
        self.controllers.append(controller)
        return controller

    def tearDown(self):
        import shiboken6
        for controller in self.controllers:
            if not shiboken6.isValid(controller):
                continue
            process = controller._process
            controller._process = None
            for pending in set(controller._retiring) | ({process} if process is not None else set()):
                if shiboken6.isValid(pending):
                    pending.kill()
                    controller._release_process(pending)
            controller._retire_timer.stop()
            controller.deleteLater()
        for buffer in self.buffers:
            if shiboken6.isValid(buffer):
                buffer.close()
                buffer.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def test_renderer_crash_only_emits_preview_failure_and_never_loads_gi(self):
        controller = self.make_controller(lambda _m: None)
        failures = []
        controller.failed.connect(failures.append)
        with mock.patch('squeakview.apps.operator.gui.ipc_preview.QtCore.QProcess', side_effect=self.process_factory) as process_class:
            process_class.ProcessChannelMode = PROCESS_TYPE.ProcessChannelMode
            self.assertTrue(controller.start(Path('/tmp/preview.sock'), 42))
        process = self.processes[0]
        self.assertIn('squeakview.apps.operator.gui.preview_worker', process.command[1])
        self.assertIn('--parent-pid', process.command[1])
        process.finished.emit(11, QtCore.QProcess.ExitStatus.CrashExit)
        self.assertFalse(controller.running)
        self.assertIn('renderer exited', failures[0])

    def test_stop_is_nonblocking_and_kills_stalled_child_without_affecting_replacement(self):
        controller = self.make_controller(lambda _m: None)
        with mock.patch('squeakview.apps.operator.gui.ipc_preview.QtCore.QProcess', side_effect=self.process_factory) as process_class:
            process_class.ProcessChannelMode = PROCESS_TYPE.ProcessChannelMode
            controller.start(Path('/tmp/a'), 42)
            old = self.processes[0]
            controller.stop()
            self.assertTrue(old.terminated)
            self.assertFalse(controller.running)
            controller.start(Path('/tmp/b'), 42)
        replacement = self.processes[1]
        controller._kill_retiring(old)
        self.assertTrue(old.killed)
        old.finished.emit(9, QtCore.QProcess.ExitStatus.CrashExit)
        self.assertTrue(controller.running)
        replacement.output = b'{"event":"ready"}\n'
        replacement.readyReadStandardOutput.emit()
        self.assertTrue(controller._ready)
        controller.stop()

    def test_renderer_status_buffers_are_bounded(self):
        controller = self.make_controller(lambda _m: None)
        failures = []
        controller.failed.connect(failures.append)
        with mock.patch('squeakview.apps.operator.gui.ipc_preview.QtCore.QProcess', side_effect=self.process_factory) as process_class:
            process_class.ProcessChannelMode = PROCESS_TYPE.ProcessChannelMode
            controller.start(Path('/tmp/a'), 42)
        process = self.processes[0]
        process.errors = b'x' * 100000
        process.readyReadStandardError.emit()
        self.assertEqual(len(controller._stderr), 8192)
        process.output = b'x' * 65537
        process.readyReadStandardOutput.emit()
        self.assertTrue(process.terminated)
        self.assertIn('oversized', failures[0])

    def test_presentation_flood_is_bounded_and_latest_feeder_state_survives(self):
        logs, events = [], []
        buffer = GuiPresentationBuffer(logs.append, events.append, console=False)
        self.buffers.append(buffer)
        def flood():
            for index in range(10000): buffer.post_log(str(index))
            buffer.post_dashboard(DashboardEvent.parse('FEED_JAM,1,2,nan,1,0,0,0,Eligible,JAM'))
            buffer.post_dashboard(DashboardEvent.parse('ACK_CLEAR_JAM'))
        thread = threading.Thread(target=flood)
        thread.start(); thread.join()
        self.assertEqual(len(buffer._items), buffer.MAX_ITEMS)
        self.assertTrue(buffer._scheduled)
        for _ in range(100): self.app.processEvents()
        self.assertFalse(buffer._scheduled)
        self.assertTrue(any('7952' in value and 'shed' in value for value in logs))
        self.assertEqual(events[-1].raw_line, 'ACK_CLEAR_JAM')
        self.assertIn('9999', logs)
        buffer.close()
        buffer.post_log('closed')
        self.app.processEvents()
        self.assertNotIn('closed', logs)

    def test_real_renderer_process_death_leaves_qt_event_loop_running(self):
        import sys
        controller = self.make_controller(lambda _message: None)
        loop = QtCore.QEventLoop()
        failures, ticks = [], []
        controller.failed.connect(lambda message: (failures.append(message), loop.quit()))
        timer = QtCore.QTimer()
        timer.setInterval(1)
        timer.timeout.connect(lambda: ticks.append(True))
        timer.start()
        with mock.patch.object(controller, '_renderer_command', return_value=(
                sys.executable, ['-c', 'import os,signal;os.kill(os.getpid(),signal.SIGKILL)'])):
            controller.start(Path('/tmp/unused'), 42)
        deadline = QtCore.QTimer(loop)
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        deadline.start(3000)
        loop.exec()
        deadline.stop()
        timer.stop()
        self.assertTrue(failures)
        self.assertTrue(ticks)
        self.assertIn('CrashExit', failures[0])
        self.assertFalse(controller.running)

    def test_recovery_restores_measurements_and_existing_phase_without_start(self):
        import tempfile
        from types import SimpleNamespace
        from squeakview.apps.operator.backend.contracts import LaunchConfig
        from squeakview.apps.operator.backend.events import RunPhase, RunSnapshot
        from squeakview.apps.operator.gui.main_window import MainWindow
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'bottle_setup.json').write_text(json.dumps({
                'mouse_weight_g': 23.4,
                'sides': {'left': {'fluid': 'ethanol', 'initial_weight_g': 23.4,
                                   'final_weight_g': None},
                          'right': {'fluid': 'water', 'initial_weight_g': 45,
                                    'final_weight_g': None}}}))
            config = LaunchConfig(run_dir=root, experiment_name='existing', mouse_id='mouse-1')
            snapshot = RunSnapshot(phase=RunPhase.RECORDING, run_dir=root, error=None,
                                   capture_running=False, finalization_in_progress=False)
            backend = mock.Mock()
            jam = DashboardEvent.parse('FEED_JAM,100,200,nan,3,69420,69420,69420,Feeding,Jam')
            backend.recover_runtime_state.return_value = {'config': config, 'snapshot': snapshot,
                'started_at': None, 'dashboard_events': (jam,)}
            view = SimpleNamespace(backend=backend, _config_data={}, run_identity_label=mock.Mock(),
                summary_label=mock.Mock(), dashboard=mock.Mock(), bottle_panel=mock.Mock(),
                _run_controller=mock.Mock(), _bottle_swap_controller=mock.Mock(), _emit_log=mock.Mock(), _on_dashboard_event=mock.Mock())
            MainWindow._recover_supervised_run(view)
            view._on_dashboard_event.assert_called_once_with(jam)
            backend.start_run.assert_not_called()
            backend.save_bottle_measurements.assert_not_called()
            view.bottle_panel.mouse_weight_edit.setText.assert_called_once_with('23.4')
            view.bottle_panel.left_fluid_combo.setCurrentText.assert_called_once_with('ethanol')
            view.bottle_panel.right_initial_weight_edit.setText.assert_called_once_with('45')
            view._run_controller.on_backend_run_started.assert_called_once_with()
            self.assertEqual(view._run_controller.handle_backend_event.call_args.args[0].phase, RunPhase.RECORDING)

    def test_renderer_finished_and_pending_kill_timer_survive_controller_destruction(self):
        import shiboken6
        for _ in range(20):
            parent = QtCore.QObject()
            controller = self.make_controller(lambda _message: None, parent)
            with mock.patch('squeakview.apps.operator.gui.ipc_preview.QtCore.QProcess', side_effect=self.process_factory) as process_class:
                process_class.ProcessChannelMode = PROCESS_TYPE.ProcessChannelMode
                controller.start(Path('/tmp/a'), 42)
            process = self.processes[-1]
            controller.stop()
            self.assertTrue(controller._retire_timer.isActive())
            process.finished.emit(9, QtCore.QProcess.ExitStatus.CrashExit)
            parent.deleteLater()
            self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            self.app.processEvents()
            self.assertFalse(shiboken6.isValid(controller))

    def test_failed_start_cleans_up_retiring_renderer(self):
        controller = self.make_controller(lambda _message: None)
        with mock.patch('squeakview.apps.operator.gui.ipc_preview.QtCore.QProcess', side_effect=self.process_factory) as process_class:
            process_class.ProcessChannelMode = PROCESS_TYPE.ProcessChannelMode
            process_class.ProcessError = PROCESS_TYPE.ProcessError
            controller.start(Path('/tmp/a'), 42)
            process = self.processes[-1]
            controller.stop()
            process.errorOccurred.emit(PROCESS_TYPE.ProcessError.FailedToStart)
        self.assertFalse(controller._retiring)
        self.assertFalse(controller._kill_deadlines)

    def test_console_writer_never_retains_qobject_when_terminal_is_blocked(self):
        import gc
        import weakref
        from squeakview.apps.operator.gui import presentation_buffer
        entered, release = threading.Event(), threading.Event()
        def blocked_print(*_args, **_kwargs):
            entered.set()
            release.wait(3.0)
        with mock.patch.object(presentation_buffer, 'print', side_effect=blocked_print, create=True):
            buffer = GuiPresentationBuffer(lambda _value: None, lambda _event: None)
            reference = weakref.ref(buffer)
            buffer.post_log('blocked terminal')
            self.assertTrue(entered.wait(1.0))
            buffer.close()
            buffer.deleteLater()
            self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            del buffer
            gc.collect()
            try:
                self.assertIsNone(reference())
            finally:
                release.set()

    def test_worker_gc_cannot_destroy_live_gui_timers(self):
        import gc
        errors = []
        previous = QtCore.qInstallMessageHandler(lambda _kind, _context, text: errors.append(text))
        stop = threading.Event()
        def collect():
            while not stop.wait(0.001): gc.collect()
        collector = threading.Thread(target=collect)
        collector.start()
        try:
            for _ in range(30):
                parent = QtCore.QObject()
                controller = self.make_controller(lambda _value: None, parent)
                controller._retire_timer.start()
                self.app.processEvents()
                parent.deleteLater()
                self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
                self.app.processEvents()
        finally:
            stop.set()
            collector.join()
            QtCore.qInstallMessageHandler(previous)
        self.assertFalse([text for text in errors if 'another thread' in text])


if __name__ == '__main__': unittest.main()
