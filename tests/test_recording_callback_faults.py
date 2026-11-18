"""Python callback boundary tests; inert SDK bases are used off Jetson."""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.support.vendor_module import load_inference_module
from squeakview.apps.inference.recording_liveness import RecordingLivenessMonitor, RecordingLivenessPolicy

recording = load_inference_module('recording')


class RecordingCallbackFaultTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.failures = []

    def test_admission_write_fault_is_reported_once_without_unwinding_native_callback(self):
        operator = recording.RecordingAdmissionOperator(self.root / 'frames.csv', 0,
                                                        on_fatal=self.failures.append)
        self.addCleanup(operator.close)
        operator._writer = mock.Mock()
        operator._writer.writerow.side_effect = OSError('disk full')
        for _ in range(3):
            self.assertTrue(operator.handle_buffer(SimpleNamespace(timestamp=1)))
        operator._writer.writerow.assert_called_once()
        self.assertEqual(len(self.failures), 1)
        self.assertIn('disk full', self.failures[0])
        with self.assertRaisesRegex(RuntimeError, 'disk full'):
            operator.close()

    def test_fault_reporter_exception_cannot_escape_callback(self):
        operator = recording.RecordingAdmissionOperator(self.root / 'frames.csv', 0,
            on_fatal=mock.Mock(side_effect=RuntimeError('reporter failed')))
        self.addCleanup(operator.close)
        self.assertTrue(operator.handle_buffer(SimpleNamespace(timestamp='bad')))
        with self.assertRaisesRegex(RuntimeError, 'recording admission'):
            operator.close()

    def test_normal_admission_and_closed_writer_keep_existing_semantics(self):
        operator = recording.RecordingAdmissionOperator(self.root / 'frames.csv', 0,
                                                        on_fatal=self.failures.append)
        self.assertTrue(operator.handle_buffer(SimpleNamespace(timestamp=123)))
        operator.close()
        self.assertFalse(operator.handle_buffer(SimpleNamespace(timestamp=124)))
        self.assertIn('0,0,123,', operator.path.read_text())
        self.assertFalse(self.failures)

    def telemetry(self):
        telemetry = recording.RecordingPathTelemetry(self.root / 'telemetry.csv', 0,
            sample_interval_s=.000000001, on_fatal=self.failures.append)
        self.addCleanup(telemetry.close)
        return telemetry

    def test_diagnostic_write_failure_preserves_recording_counters(self):
        telemetry = self.telemetry()
        telemetry._writer = mock.Mock()
        telemetry._writer.writerow.side_effect = OSError('diagnostics unavailable')
        for pts in (1, 2, 3):
            self.assertTrue(recording.RecordingIngressOperator(telemetry).handle_buffer(SimpleNamespace(timestamp=pts)))
            telemetry.admit(pts)
            self.assertTrue(recording.RecordingEgressOperator(telemetry).handle_buffer(SimpleNamespace(timestamp=pts)))
        self.assertEqual(telemetry.shutdown_progress(), (3, 3, 3))
        self.assertIn('diagnostics unavailable', telemetry.diagnostic_error)
        telemetry._writer.writerow.assert_called_once()
        self.assertFalse(self.failures)

    def test_ingress_and_egress_contain_unexpected_python_faults(self):
        for cls in (recording.RecordingIngressOperator, recording.RecordingEgressOperator):
            with self.subTest(cls=cls.__name__):
                telemetry = self.telemetry()
                operator = cls(telemetry)
                self.assertTrue(operator.handle_buffer(SimpleNamespace(timestamp='bad')))
                with self.assertRaisesRegex(RuntimeError, 'recording telemetry'):
                    telemetry.close()
        self.assertEqual(len(self.failures), 2)

    def test_busy_telemetry_does_not_block_watchdog_or_hide_other_streams(self):
        telemetry = self.telemetry()
        fatal = threading.Event()
        stop = threading.Event()
        def failure(reason):
            self.failures.append(reason)
            fatal.set()
        monitor = RecordingLivenessMonitor([telemetry], stop, failure,
            RecordingLivenessPolicy(.02, 20_000_000, .001))
        worker = threading.Thread(target=monitor.run, daemon=True)
        telemetry._lock.acquire()
        try:
            worker.start()
            observed = fatal.wait(1)
        finally:
            stop.set()
            telemetry._lock.release()
            worker.join(1)
        self.assertTrue(observed)
        self.assertFalse(worker.is_alive())
        self.assertIn('snapshot unavailable', self.failures[0])

    def test_diagnostic_close_failure_is_recorded_as_optional(self):
        telemetry = self.telemetry()
        real_file = telemetry._file
        telemetry._file = mock.Mock()
        telemetry._file.flush.side_effect = OSError('flush failed')
        try:
            telemetry.close()
            telemetry._file.close.assert_called_once()
            self.assertIn('flush failed', telemetry.diagnostic_error)
        finally:
            real_file.close()
