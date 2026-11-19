"""Optional process failures must never escape into acquisition control."""
import json
from contextlib import contextmanager
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from squeakview.apps.inference.contracts import InferenceConfig
from squeakview.apps.inference.optional_live import OptionalLiveSession


class OptionalLiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def session(self, **changes):
        session = OptionalLiveSession(InferenceConfig(run_dir=self.root, width=2, height=2, **changes), emit=mock.Mock())
        self.addCleanup(session.close)
        return session

    def test_disabled_session_has_no_process_or_mailbox(self):
        session = self.session(enable_infer=False)
        self.assertEqual(session.prepare(), ())
        session.start(); session.close()
        self.assertIsNone(session.process)

    def test_preallocation_failure_degrades_without_launching(self):
        session = self.session()
        with mock.patch('os.posix_fallocate', side_effect=OSError('no shared memory')):
            self.assertEqual(session.prepare(), ())
        session.start()
        self.assertEqual(session.state, 'degraded')
        self.assertIsNone(session.process)

    def test_child_configuration_owns_only_live_sources(self):
        session = self.session(preview_sockets=(self.root / 'preview.sock',))
        paths = session.prepare()
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].stat().st_size, 4096 + 65536 + 4)
        config = json.loads((paths[0].parent / 'config.json').read_text())
        self.assertFalse(config['raw_only'])
        self.assertEqual(config['live_mailboxes'], [str(paths[0])])
        self.assertIsNone(config['failure_plan'])

    def test_spawn_failure_is_recoverable_and_does_not_raise(self):
        session = self.session()
        session.prepare()
        with mock.patch('squeakview.apps.operator.backend.supervision.spawn', side_effect=OSError('cannot spawn')):
            session.start()
        self.assertEqual(session.state, 'degraded')

    def test_crash_and_dead_heartbeat_are_degraded_and_bounded(self):
        for running in (False, True):
            with self.subTest(running=running):
                session = self.session()
                session.process = mock.Mock()
                session.process.is_running.return_value = running
                session.process.wait.return_value = -11
                session._stop = mock.Mock()
                session._stop.wait.return_value = False
                session._started = 1
                with mock.patch('time.monotonic', return_value=100):
                    session._watch()
                self.assertEqual(session.state, 'degraded')
                if running:
                    session.process.send_signal_group.assert_called_once_with(signal.SIGKILL)

    def test_cleanup_and_status_errors_do_not_override_capture(self):
        session = self.session()
        session.process = mock.Mock()
        session.process.wait.side_effect = OSError('broken process handle')
        session.emit.side_effect = OSError('broken log')
        with mock.patch('squeakview.common.run_context.update_status', side_effect=OSError('disk full')):
            session.close()
        self.assertEqual(session.state, 'degraded')

    def test_transient_empty_status_read_does_not_kill_healthy_worker(self):
        session = self.session()
        session.prepare()
        session.state = 'running'
        session._started = 0
        session.process = mock.Mock()
        session.process.is_running.return_value = True
        session._stop = mock.Mock()
        session._stop.wait.side_effect = [False, False, False, True]
        snapshots = [
            {'state': 'running', 'heartbeat_monotonic': 100,
             'frames_processed': [10], 'stack_dump_ready': True},
            {},  # Atomic replacement raced the strict status-file reader.
            {'state': 'running', 'heartbeat_monotonic': 101,
             'frames_processed': [11], 'stack_dump_ready': True},
        ]
        with mock.patch('squeakview.common.run_context.read_live_snapshot', side_effect=snapshots), \
             mock.patch('squeakview.apps.inference.optional_live.time.monotonic',
                        side_effect=[100, 100.5, 101]):
            session._watch()
        session.process.send_signal_group.assert_not_called()
        self.assertEqual(session.state, 'running')
        session._stop.wait.side_effect = None

    def test_persistently_unreadable_status_expires_and_retains_dump_handshake(self):
        session = self.session()
        session.prepare()
        session._started = 0
        session.process = mock.Mock()
        session.process.is_running.return_value = True
        session._stop = mock.Mock()
        session._stop.wait.return_value = False
        last_good = {'state': 'running', 'heartbeat_monotonic': 100,
                     'frames_processed': [10], 'stack_dump_ready': True}
        with mock.patch('squeakview.common.run_context.read_live_snapshot',
                        side_effect=[last_good, {}, {}]), \
             mock.patch('squeakview.apps.inference.optional_live.time.monotonic',
                        side_effect=[100, 105, 111]), \
             mock.patch.object(session, '_kill_stalled') as kill:
            session._watch()
        kill.assert_called_once_with('live processing stopped responding', last_good)

    def test_atomic_status_replacement_can_return_empty_read(self):
        session = self.session()
        path = session.status_path
        path.parent.mkdir(parents=True)
        first = {'state': 'running', 'heartbeat_monotonic': 100}
        latest = {'state': 'running', 'heartbeat_monotonic': 101}
        path.write_text(json.dumps(first))
        replacement = path.with_suffix('.next')
        replacement.write_text(json.dumps(latest))
        original_open = Path.open

        @contextmanager
        def replace_after_read(target, *args, **kwargs):
            with original_open(target, *args, **kwargs) as handle:
                yield handle
            if target == path:
                replacement.replace(path)

        from squeakview.common import run_context
        with mock.patch.object(Path, 'open', replace_after_read):
            self.assertEqual(run_context.read_json(path), {})
        self.assertEqual(run_context.read_json(path), latest)

    def test_completed_status_does_not_hide_native_teardown_crash(self):
        for code, expected in ((0, 'completed'), (-6, 'degraded')):
            with self.subTest(code=code):
                session = self.session()
                session.process = mock.Mock()
                session.process.is_running.return_value = False
                session.process.wait.return_value = code
                session._stop = mock.Mock()
                session._stop.wait.return_value = False
                with mock.patch('squeakview.common.run_context.read_live_snapshot', return_value={'state': 'completed'}):
                    session._watch()
                self.assertEqual(session.state, expected)

    def test_capture_stop_then_cleanup_sends_only_one_interrupt(self):
        session = self.session()
        session.state = 'running'
        session.process = mock.Mock()
        session.process.is_running.return_value = True
        session.process.wait.return_value = 0
        session._temporary = mock.Mock()

        # Capture requests a stop first. Cleanup runs after capture drains,
        # while the live worker may still be releasing native resources.
        session.request_stop()
        session.close()
        session.close()

        session.process.send_signal_group.assert_called_once_with(signal.SIGINT)
        self.assertEqual(session.state, 'completed')

    def test_fresh_heartbeat_but_stalled_output_kills_only_optional_worker(self):
        session = self.session()
        session.prepare()
        session.process = mock.Mock()
        session.process.is_running.return_value = True
        session.process.wait.return_value = -9
        session._stop = mock.Mock()
        session._stop.wait.return_value = False
        statuses = [{'state': 'running', 'heartbeat_monotonic': now, 'frames_processed': [0]}
                    for now in (0, 5, 10)]
        with mock.patch('squeakview.common.run_context.read_live_snapshot', side_effect=statuses), \
             mock.patch('squeakview.apps.inference.optional_live.mailbox_sequence', side_effect=[10, 20, 30]), \
             mock.patch('squeakview.apps.inference.optional_live.time.monotonic', side_effect=[0, 5, 10]):
            session._watch()
        self.assertEqual(session.state, 'degraded')
        session.process.send_signal_group.assert_called_once_with(signal.SIGKILL)

    def test_capture_progress_detects_stall_even_when_mailbox_publication_is_stuck(self):
        session = self.session()
        session.prepare()
        source = mock.Mock()
        source.try_activity.side_effect = [SimpleNamespace(source_count=n) for n in (10, 20)]
        session.recording_sources = (source,)
        session.process = mock.Mock()
        session.process.is_running.return_value = True
        session.process.wait.return_value = -9
        session._stop = mock.Mock()
        session._stop.wait.return_value = False
        with mock.patch('squeakview.common.run_context.read_live_snapshot', side_effect=[
                {'state': 'running', 'heartbeat_monotonic': now, 'frames_processed': [0]}
                for now in (0, 10)]), \
             mock.patch('squeakview.apps.inference.optional_live.mailbox_sequence', return_value=0), \
             mock.patch('squeakview.apps.inference.optional_live.time.monotonic', side_effect=[0, 10]):
            session._watch()
        self.assertEqual(session.state, 'degraded')
        session.process.send_signal_group.assert_called_once_with(signal.SIGKILL)

    def test_cleanup_preserves_signal_exit_failure_after_completed_status(self):
        for code in (-signal.SIGINT, -signal.SIGABRT):
            with self.subTest(code=code):
                session = self.session()
                session.state = 'running'
                session.process = mock.Mock()
                session.process.is_running.return_value = False
                session.process.wait.return_value = code
                with mock.patch('squeakview.common.run_context.read_live_snapshot',
                                return_value={'state': 'completed'}):
                    session.close()
                self.assertEqual(session.state, 'degraded')

    def test_stuck_shutdown_escalates_without_unlinking_live_mapping(self):
        session = self.session()
        session.process = mock.Mock()
        session.process.wait.side_effect = subprocess.TimeoutExpired('live worker', 1)
        session._temporary = mock.Mock()
        session.close()
        session._temporary.cleanup.assert_not_called()
        self.assertEqual([call.args[0] for call in session.process.send_signal_group.call_args_list],
                         [signal.SIGINT, signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(session.state, 'degraded')


if __name__ == '__main__':
    unittest.main()
