"""Optional process failures must never escape into acquisition control."""
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

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

    def test_completed_status_does_not_hide_native_teardown_crash(self):
        for code, expected in ((0, 'completed'), (-6, 'degraded')):
            with self.subTest(code=code):
                session = self.session()
                session.process = mock.Mock()
                session.process.is_running.return_value = False
                session.process.wait.return_value = code
                session._stop = mock.Mock()
                session._stop.wait.return_value = False
                with mock.patch('squeakview.common.run_context.read_json', return_value={'state': 'completed'}):
                    session._watch()
                self.assertEqual(session.state, expected)

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
