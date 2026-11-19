import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from squeakview.apps.inference.contracts import InferenceConfig
from squeakview.apps.inference.optional_live import OptionalLiveSession


class LiveDiagnosticsTests(unittest.TestCase):
    def test_stalled_native_call_dumps_threads_before_worker_is_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = OptionalLiveSession(InferenceConfig(run_dir=root), emit=mock.Mock())
            directory = session.status_path.parent
            code = f'''
import ctypes
import threading
from pathlib import Path
from squeakview.apps.inference.live_diagnostics import LiveWorkerDiagnostics
diagnostics = LiveWorkerDiagnostics(Path({str(directory)!r}))
assert diagnostics.ready, diagnostics.error
def waiting_thread():
    threading.Event().wait(30)
threading.Thread(target=waiting_thread, daemon=True).start()
def blocked_native():
    native = ctypes.PyDLL(None)
    mutex = (ctypes.c_long * 16)()
    assert native.pthread_mutex_init(ctypes.byref(mutex), None) == 0
    assert native.pthread_mutex_lock(ctypes.byref(mutex)) == 0
    print('ready', flush=True)
    # PyDLL retains the GIL: a normal Python signal callback cannot run here.
    native.pthread_mutex_lock(ctypes.byref(mutex))
blocked_native()
'''
            child = subprocess.Popen([sys.executable, '-u', '-c', code],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, start_new_session=True)
            signals = []

            def send(sig):
                signals.append(sig)
                os.killpg(child.pid, sig)

            try:
                self.assertTrue(select.select([child.stdout], [], [], 5)[0])
                self.assertEqual(child.stdout.readline().strip(), 'ready')
                session.process = SimpleNamespace(p=child, wait=child.wait,
                    is_running=lambda: child.poll() is None, send_signal_group=send)
                status = {'state': 'running', 'heartbeat_monotonic': 123,
                          'frames_processed': [42], 'stack_dump_ready': True}
                session._kill_stalled('live processing stopped responding', status)
                self.assertEqual(child.wait(timeout=3), -signal.SIGKILL)
                self.assertEqual(signals, [signal.SIGUSR1, signal.SIGKILL])
                stacks = (directory / 'live_stacks.log').read_text()
                self.assertIn('blocked_native', stacks)
                self.assertIn('waiting_thread', stacks)
                self.assertEqual((directory / 'live_stacks.log').stat().st_mode & 0o777, 0o600)
                evidence = json.loads((directory / 'live_hang.json').read_text())
                self.assertEqual(evidence['last_worker_status'], status)
                self.assertEqual(evidence['pid'], child.pid)
                self.assertTrue(evidence['threads'])
                self.assertEqual(session.state, 'degraded')
            finally:
                if child.poll() is None:
                    child.kill()
                child.communicate(timeout=3)

    def test_blocked_evidence_writer_cannot_prevent_worker_termination(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = OptionalLiveSession(InferenceConfig(run_dir=Path(tmp)), emit=mock.Mock())
            session.process = mock.Mock()
            release = threading.Event()
            entered = threading.Event()

            def blocked(*_args):
                entered.set()
                release.wait(3)

            try:
                with mock.patch('squeakview.apps.inference.live_diagnostics.save_hang_evidence',
                                side_effect=blocked):
                    session._kill_stalled('stalled', {})
                self.assertTrue(entered.is_set())
                session.process.send_signal_group.assert_called_once_with(signal.SIGKILL)
                self.assertFalse(release.is_set())
            finally:
                release.set()

    def test_diagnostic_failure_does_not_prevent_worker_termination(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = OptionalLiveSession(InferenceConfig(run_dir=Path(tmp)), emit=mock.Mock())
            session.process = mock.Mock()
            with mock.patch('squeakview.apps.inference.live_diagnostics.save_hang_evidence',
                            side_effect=OSError('disk full')):
                session._kill_stalled('stalled', {})
            session.process.send_signal_group.assert_called_once_with(signal.SIGKILL)


if __name__ == '__main__':
    unittest.main()
