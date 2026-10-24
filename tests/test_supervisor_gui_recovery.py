"""Kill/stall real presentation subprocesses while a backend writer continues."""
from __future__ import annotations

import json
import errno
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from test_supervisor_server import FakeBackend, _runtime_context
from squeakview.apps.operator.backend.events import RunPhase
from squeakview.apps.operator.backend.supervisor.protocol import CommandEnvelope
from squeakview.apps.operator.backend.supervisor.server import SupervisorServer

GUI = r'''
import json, os, socket, sys, time
from pathlib import Path
root=Path(sys.argv[1]); mode=sys.argv[2]
marker=root/'launches'
number=int(marker.read_text())+1 if marker.exists() else 1
marker.write_text(str(number))
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
s.connect(os.environ['SQUEAKVIEW_SUPERVISOR_SOCKET'])
f=s.makefile('r')
def send(name, request_id, payload=None):
    s.sendall((json.dumps(dict(version=1,type='command',name=name,request_id=request_id,payload=payload or {}))+'\n').encode())
def result(name, request_id):
    send(name,request_id)
    while True:
        event=json.loads(f.readline())
        if event.get('request_id')==request_id:
            assert event['payload']['ok'],event
            return event['payload']['result']
send('heartbeat','gui-heartbeat')
if number==1:
    if mode=='kill':
        time.sleep(.15)
        os.kill(os.getpid(),9)
    time.sleep(10) # Server detects expiry; default SIGTERM ends this process.
else:
    runtime=result('runtime_state','recover')
    assert runtime['snapshot']['phase']=='recording',runtime
    assert os.environ['SQUEAKVIEW_GUI_RECOVERY']=='1'
    (root/'recovered.json').write_text(json.dumps(runtime))
    result('stop_run','operator-stop')
    result('shutdown_server','operator-shutdown')
'''


@unittest.skipUnless(sys.platform == 'linux', 'Unix GUI authentication uses Linux peer credentials')
class SupervisorGuiRecoveryTests(unittest.TestCase):
    def _exercise(self, mode: str, fault: str | None = None) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = _runtime_context(root)
            gui = root / 'fake_gui.py'
            gui.write_text(GUI)
            server = SupervisorServer(root/'operator.sock', runtime_context=context,
                backend_factory=FakeBackend, gui_lease_timeout_s=1)
            backend = server.backend
            backend.phase = RunPhase.RECORDING
            backend.state.run_dir = root/'run'
            backend.state.run_dir.mkdir()
            done = threading.Event()
            ticks = []
            def write_recording():
                with (backend.state.run_dir/'recording.txt').open('a') as stream:
                    while not done.wait(.01):
                        if backend.phase == RunPhase.RECORDING:
                            ticks.append(time.monotonic())
                            stream.write('frame\n')
                            stream.flush()
            writer = threading.Thread(target=write_recording)
            writer.start()
            timeout = threading.Timer(12, server.request_shutdown)
            timeout.start()
            original_prepare = server._prepare_listener
            original_start = threading.Thread.start
            injected = [False]
            def prepare():
                listener = original_prepare()
                class Listener:
                    def accept(self):
                        if fault == "accept" and not injected[0]:
                            injected[0] = True
                            raise OSError(errno.EMFILE, "injected descriptor exhaustion")
                        return listener.accept()
                    def close(self):
                        listener.close()
                return Listener()
            def start(thread):
                if fault == "sender" and thread.name == "squeakview-supervisor-events" and not injected[0]:
                    injected[0] = True
                    raise RuntimeError("injected thread exhaustion")
                return original_start(thread)
            try:
                with mock.patch.object(server, "_prepare_listener", side_effect=prepare), mock.patch.object(threading.Thread, "start", start):
                    result = server.serve([sys.executable,str(gui),str(root),mode])
            finally:
                timeout.cancel()
                done.set()
                writer.join(2)
            self.assertEqual(result,0)
            self.assertEqual(backend.abort_calls,[])
            self.assertFalse(backend.cancelled.is_set())
            self.assertEqual(backend.phase,RunPhase.FINALIZED)
            self.assertGreater(len(ticks),50)
            self.assertLess(max(b-a for a,b in zip(ticks,ticks[1:])),.8)
            runtime = json.loads((root/'recovered.json').read_text())
            self.assertEqual(runtime['snapshot']['run_dir'],str(backend.state.run_dir))
            self.assertEqual(runtime['config']['fps'],30)
            recovery = [json.loads(line) for line in (backend.state.run_dir/'diagnostics/supervisor_recovery.jsonl').read_text().splitlines()]
            self.assertTrue(any(event['event']=='gui_connected' and event['recovered'] for event in recovery))
            if mode == 'kill':
                self.assertTrue(any(event['event']=='gui_exit' and event['returncode']==-9 for event in recovery))
            elif mode == 'stall':
                self.assertTrue(any('heartbeat lease expired' in event.get('reason','') for event in recovery))
            if fault:
                self.assertTrue(injected[0])
                self.assertTrue(any('failed' in event.get('reason','') for event in recovery))

    def test_killed_gui_is_replaced_without_stopping_writer_then_explicit_stop_works(self):
        self._exercise('kill')

    def test_expired_gui_heartbeat_is_replaced_without_stopping_writer(self):
        self._exercise('stall')

    def test_old_sender_failure_cannot_detach_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            server=SupervisorServer(root/'operator.sock',runtime_context=_runtime_context(root),backend_factory=FakeBackend)
            old, replacement=mock.Mock(),mock.Mock()
            server._client=replacement
            server._signal_client_loss('old sender failed',client=old)
            self.assertIs(server._client,replacement)
            self.assertFalse(server._client_lost.is_set())
            self.assertFalse(server._stopping.is_set())

    def test_unresponsive_gui_kill_timeout_does_not_escape_into_acquisition_shutdown(self):
        gui=mock.Mock()
        gui.poll.return_value=None
        gui.wait.side_effect=subprocess.TimeoutExpired('gui',2)
        SupervisorServer._retire_gui(gui)
        gui.kill.assert_called_once_with()

    def test_unreapable_gui_does_not_accumulate_replacement_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = SupervisorServer(root/'operator.sock', runtime_context=_runtime_context(root),
                backend_factory=FakeBackend, gui_connect_timeout_s=0.01)
            server.backend.phase = RunPhase.RECORDING
            gui = mock.Mock()
            gui.poll.return_value = None
            # End only the presentation-loop test; no user stop/abort request.
            timer = threading.Timer(5, server._stopping.set)
            def stop_after_pending(event, **_details):
                if event == 'gui_retirement_pending':
                    server._stopping.set()
            with mock.patch('squeakview.apps.operator.backend.supervisor.server.subprocess.Popen', return_value=gui) as launch, \
                 mock.patch.object(server, '_retire_gui'), \
                 mock.patch.object(server, '_record_gui_event', side_effect=stop_after_pending) as record:
                timer.start()
                try:
                    server.serve([sys.executable, '-c', 'pass'])
                finally:
                    timer.cancel()
                launch.assert_called_once()
                self.assertTrue(any(call.args[0] == 'gui_retirement_pending' for call in record.call_args_list))
                self.assertEqual(server.backend.abort_calls, [])

    def test_private_socket_is_unlinked_after_permission_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            server=SupervisorServer(root/'operator.sock',runtime_context=_runtime_context(root),backend_factory=FakeBackend)
            listener=server._prepare_listener()
            listener.close()
            server._unlink_owned_socket()
            self.assertFalse(server.socket_path.exists())

    def test_recovery_diagnostic_writer_does_not_block_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            server=SupervisorServer(root/'operator.sock',runtime_context=_runtime_context(root),backend_factory=FakeBackend)
            entered,release=threading.Event(),threading.Event()
            def blocked_writer():
                entered.set()
                release.wait(2)
            with mock.patch.object(server,'_write_gui_diagnostics',side_effect=blocked_writer):
                server._record_gui_event('gui_connection_lost',reason='test')
                self.assertTrue(entered.wait(1))
                for _ in range(1000):
                    server._record_gui_event('gui_connection_lost',reason='test')
                self.assertLessEqual(server._diagnostic_events.qsize(),64)
                release.set()
                server._diagnostic_thread.join(2)

    def test_accept_descriptor_exhaustion_preserves_writer_and_recovers(self):
        self._exercise('accept-fault', fault='accept')

    def test_sender_thread_exhaustion_preserves_writer_and_recovers(self):
        self._exercise('sender-fault', fault='sender')

    def test_delayed_old_socket_close_cannot_close_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            server=SupervisorServer(root/'operator.sock',runtime_context=_runtime_context(root),backend_factory=FakeBackend)
            old,replacement=mock.Mock(),mock.Mock()
            server._client=old
            entered,release=threading.Event(),threading.Event()
            original_close=server._close_detached_client
            def delayed_close(client):
                if client is old:
                    entered.set()
                    release.wait(2)
                original_close(client)
            with mock.patch.object(server,'_close_detached_client',side_effect=delayed_close):
                worker=threading.Thread(target=lambda:server._signal_client_loss('old lost',client=old))
                worker.start()
                self.assertTrue(entered.wait(1))
                with server._client_lock:
                    server._client=replacement
                    server._client_lost.clear()
                release.set()
                worker.join(2)
            self.assertIs(server._client,replacement)
            replacement.shutdown.assert_not_called()
            replacement.close.assert_not_called()
            self.assertFalse(server._client_lost.is_set())
            server._stopping.set()
            if server._diagnostic_thread:
                server._diagnostic_thread.join(2)

    def test_diagnostic_thread_exhaustion_cannot_stop_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            server=SupervisorServer(root/'operator.sock',runtime_context=_runtime_context(root),backend_factory=FakeBackend)
            server.backend.phase=RunPhase.RECORDING
            with mock.patch.object(threading.Thread,'start',side_effect=RuntimeError('no thread resources')):
                server._record_gui_event('gui_connection_lost',reason='test')
            self.assertIsNone(server._diagnostic_thread)
            self.assertFalse(server._stopping.is_set())
            self.assertEqual(server.backend.phase,RunPhase.RECORDING)
            self.assertFalse(server.backend.cancelled.is_set())

    def test_old_sender_cannot_consume_replacement_sequence(self):
        from types import SimpleNamespace
        import queue
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            server=SupervisorServer(root/'operator.sock',runtime_context=_runtime_context(root),backend_factory=FakeBackend)
            old,replacement=mock.Mock(),mock.Mock()
            server._client=old
            entered,release=threading.Event(),threading.Event()
            class DelayedQueue(queue.Queue):
                def get(self,timeout=None):
                    entered.set()
                    release.wait(2)
                    return SimpleNamespace(name='log',payload={'message':'stale'},request_id=None,sent=None)
            server._presentation_events=DelayedQueue()
            with mock.patch('squeakview.apps.operator.backend.supervisor.server.SocketEnvelopeWriter') as writer:
                worker=threading.Thread(target=server._sender,args=(old,))
                worker.start()
                self.assertTrue(entered.wait(1))
                with server._client_lock:
                    server._client=replacement
                    server._sequence=17
                release.set()
                worker.join(2)
                self.assertEqual(server._sequence,17)
                writer.return_value.send.assert_not_called()
