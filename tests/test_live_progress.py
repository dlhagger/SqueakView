import fcntl
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from squeakview.apps.inference.live_progress import LIVE_MAGIC, LiveProgressWatchdog, mailbox_sequence
from tests.support.vendor_module import load_inference_module


class LiveProgressTests(unittest.TestCase):
    def test_fresh_heartbeat_without_output_progress_is_not_enough(self):
        guard = LiveProgressWatchdog(1, 10)
        self.assertIsNone(guard.check([10], [0], 0))
        self.assertIsNone(guard.check([20], [0], 9))
        self.assertIn('stopped advancing', guard.check([30], [0], 10))

    def test_waiting_for_triggers_or_paused_input_does_not_trigger_false_alarm(self):
        guard = LiveProgressWatchdog(1, 10)
        for now in (0, 30, 60):
            self.assertIsNone(guard.check([0], [0], now))
        self.assertIsNone(guard.check([10], [1], 61))
        self.assertIsNone(guard.check([10], [1], 1000))

    def test_delivered_output_refreshes_pending_work_timer(self):
        guard = LiveProgressWatchdog(1, 10)
        guard.check([10], [0], 0)
        self.assertIsNone(guard.check([20], [1], 9))
        self.assertIsNone(guard.check([21], [1], 10))
        self.assertIsNone(guard.check([30], [1], 19))
        self.assertIn('stopped advancing', guard.check([31], [1], 20))

    def test_healthy_camera_cannot_hide_other_camera_stall(self):
        guard = LiveProgressWatchdog(2, 10)
        guard.check([10, 10], [1, 0], 0)
        self.assertIn('stream 1', guard.check([20, 20], [10, 0], 10))

    def test_busy_mailbox_does_not_clear_pending_work(self):
        guard = LiveProgressWatchdog(1, 10)
        guard.check([10], [0], 0)
        self.assertIn('stopped advancing', guard.check([None], [0], 10))

    def test_invalid_progress_is_rejected(self):
        for outputs in (None, [], [True], [-1], ['1']):
            with self.subTest(outputs=outputs):
                self.assertIn('invalid', LiveProgressWatchdog(1).check([1], outputs, 0))

    def test_mailbox_header_read_is_nonblocking_and_checks_magic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mailbox'
            path.write_bytes(struct.pack('=QQQ', LIVE_MAGIC, 12, 0))
            self.assertEqual(mailbox_sequence(path), 12)
            with path.open('rb') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertIsNone(mailbox_sequence(path))
            for header in (struct.pack('=QQQ', LIVE_MAGIC, 12, 1), b'bad', b'\0' * 24):
                path.write_bytes(header)
                self.assertIsNone(mailbox_sequence(path))

    def test_probes_count_per_camera_without_retaining_buffers(self):
        module = load_inference_module('live_progress_probe')
        counts = [0, 0]
        faults = []
        batch = module.LiveBatchProgress(counts, faults.append)
        batch.handle_metadata(SimpleNamespace(frame_items=iter([
            SimpleNamespace(source_id=1), SimpleNamespace(pad_index=0)])))
        self.assertEqual(counts, [1, 1])
        self.assertTrue(module.LiveBufferProgress(counts, 1, faults.append).handle_buffer(object()))
        self.assertEqual(counts, [1, 2])
        self.assertFalse(faults)

    def test_probe_error_is_contained_and_reports_optional_failure(self):
        module = load_inference_module('live_progress_probe')
        faults = []
        probe = module.LiveBatchProgress([0], faults.append)
        probe.handle_metadata(SimpleNamespace(frame_items=iter([SimpleNamespace(source_id=9)])))
        self.assertIn('unexpected live source', faults[0])
