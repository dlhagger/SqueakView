import tempfile
import unittest
from pathlib import Path

from squeakview.apps.operator.backend.first_frames import wait_for_first_frames
from squeakview.common.capture_health import health_path, COUNTERS
from squeakview.common.run_context import atomic_write_json


class FirstFrameTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.now = 100.0
        self.cancelled = False
        self.running = True

    def health(self, index, count, **updates):
        atomic_write_json(health_path(self.run, index), dict(
            schema_version=1, camera_index=index, state='recording',
            native_timing_validated=count > 0, source_frames=count,
            host_unix_ns=1, host_monotonic_ns=1,
            **dict.fromkeys(COUNTERS, 0), **updates))

    def wait(self, seconds):
        self.now += seconds

    def confirm(self, count=2, wait=None):
        return wait_for_first_frames(
            self.run, count, deadline=110.0,
            cancelled=lambda: self.cancelled, capture_running=lambda: self.running,
            wait=wait or self.wait, monotonic=lambda: self.now)

    def test_all_cameras_required_and_missing_camera_identified(self):
        self.health(0, 20)
        self.health(1, 0)
        with self.assertRaisesRegex(TimeoutError, 'camera.s.: 1'):
            self.confirm()
        self.assertAlmostEqual(self.now, 110)

    def test_missing_health_files_timeout_instead_of_accepting_ttl(self):
        with self.assertRaisesRegex(TimeoutError, '0, 1'):
            self.confirm()

    def test_ready_frames_return_without_added_delay(self):
        self.health(0, 2)
        self.health(1, 3)
        self.assertEqual(self.confirm(), {0: 2, 1: 3})
        self.assertEqual(self.now, 100)

    def test_camera_arriving_during_wait_succeeds(self):
        self.health(0, 1)
        def arrive(seconds):
            self.wait(seconds)
            self.health(1, 1)
        self.assertEqual(self.confirm(wait=arrive), {0: 1, 1: 1})
        self.assertLess(self.now, 101)

    def test_cancellation_and_capture_exit_abort_promptly(self):
        for flag, message in [('cancelled', 'cancelled'), ('running', 'exited')]:
            with self.subTest(flag=flag):
                self.cancelled, self.running = False, True
                def stop(seconds):
                    self.wait(seconds)
                    setattr(self, flag, flag == 'cancelled')
                with self.assertRaisesRegex(RuntimeError, message):
                    self.confirm(wait=stop)
                self.assertLess(self.now, 101)

    def test_invalid_or_closed_evidence_fails(self):
        self.health(0, 1)
        path = health_path(self.run, 0)
        import json
        payload = json.loads(path.read_text())
        for field, value in [('camera_index', 1), ('state', 'closed'),
                             ('native_timing_validated', False), ('source_frames', True)]:
            with self.subTest(field=field):
                atomic_write_json(path, dict(payload, **{field: value}))
                with self.assertRaises((ValueError, RuntimeError)):
                    self.confirm(count=1)

    def test_ready_evidence_after_deadline_is_rejected(self):
        self.health(0, 1)
        self.now = 110
        with self.assertRaises(TimeoutError):
            self.confirm(count=1)
