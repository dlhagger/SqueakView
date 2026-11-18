from __future__ import annotations

import signal
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from squeakview.apps.inference.capture_shutdown import stop_capture
from squeakview.apps.operator.backend.lifecycle import (
    FinalizationHooks, FinalizationRequest, finalize_run,
)
from squeakview.apps.operator.backend.events import RunPhase
from squeakview.apps.operator.gui.run_presentation import present_finalization
from squeakview.common import run_context, shutdown_policy
from squeakview.common.run_layout import artifact_path, initialize_layout


class CaptureClosePolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.app = SimpleNamespace(
            _stopped=False, _stop_event=threading.Event(), live_session=None,
            config=SimpleNamespace(failure_plan=None), exit_code=0,
            _force_process_exit=False, _explicit_stop_requested=True,
            pipeline=mock.Mock(closure_confirmed=True), _pipeline_wait_thread=None,
            _pipeline_wait_error=None, _recording_liveness_thread=None,
            _storage_reserve_thread=None, observations=None, frames=mock.Mock(),
            record_admissions=[mock.Mock()], record_telemetry=[mock.Mock()],
            preview_boundaries=[], run_dir=self.root,
            system_telemetry=SimpleNamespace(
                stop=mock.Mock(), path=self.root / 'system.csv',
                sample_count=1, last_error=None,
            ),
        )
        self.logs = []

    def stop(self):
        stop_capture(self.app, emit=self.logs.append)

    def status(self):
        return run_context.read_json_required(artifact_path(self.root, 'run_status.json'))

    def test_receipt_follows_all_writers_and_precedes_optional_cleanup(self):
        order = []
        for name, call in (
            ('stop', self.app.pipeline.stop), ('wait', self.app.pipeline.wait),
            ('admission', self.app.record_admissions[0].close),
            ('telemetry', self.app.record_telemetry[0].close),
            ('frames', self.app.frames.close),
        ):
            def closed(label=name):
                self.assertFalse(shutdown_policy.has_clean_capture_closure(self.root))
                order.append(label)
            call.side_effect = closed

        def optional_close():
            self.assertTrue(shutdown_policy.has_clean_capture_closure(self.root))
            order.append('optional')
        self.app.system_telemetry.stop.side_effect = optional_close
        self.stop()
        self.stop()
        self.assertEqual(order, ['stop', 'wait', 'admission', 'telemetry', 'frames', 'optional'])
        self.assertEqual(self.app.exit_code, 0)

    def test_system_cleanup_exception_is_warning(self):
        self.app.system_telemetry.stop.side_effect = OSError('diagnostic cleanup failed')
        self.stop()
        self.assertEqual(self.app.exit_code, 0)
        self.assertTrue(shutdown_policy.has_clean_capture_closure(self.root))
        self.assertEqual(self.status()['capture_close_errors'], [])
        self.assertIn('diagnostic cleanup failed', self.status()['shutdown_warnings'][0])

    def test_recording_diagnostic_error_is_a_warning_after_clean_capture(self):
        self.app.record_telemetry[0].diagnostic_error = 'diagnostic CSV write failed'
        self.stop()
        self.assertEqual(self.app.exit_code, 0)
        self.assertTrue(shutdown_policy.has_clean_capture_closure(self.root))
        self.assertIn('diagnostic CSV write failed', self.status()['shutdown_warnings'][0])

    def test_hung_optional_cleanup_is_bounded_and_keeps_closure_evidence(self):
        release = threading.Event()
        self.addCleanup(release.set)
        self.app.system_telemetry.stop.side_effect = release.wait
        with mock.patch.dict('os.environ', {'SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S': '0.1'}):
            self.stop()
        self.assertEqual(self.app.exit_code, 0)
        self.assertTrue(self.app._force_process_exit)
        self.assertTrue(shutdown_policy.has_clean_capture_closure(self.root))
        self.assertTrue(self.status()['shutdown_degraded'])

    def test_failed_native_drain_never_closes_callback_writers_or_publishes_receipt(self):
        self.app.pipeline.wait.side_effect = RuntimeError('drain failed')
        self.stop()
        self.app.frames.close.assert_not_called()
        self.assertFalse(shutdown_policy.has_clean_capture_closure(self.root))
        self.assertNotEqual(self.app.exit_code, 0)
        self.assertTrue(self.app._force_process_exit)

    def test_hung_native_wait_never_publishes_receipt(self):
        release = threading.Event()
        thread = threading.Thread(target=release.wait)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(release.set)
        self.app._pipeline_wait_thread = thread
        with mock.patch.dict('os.environ', {'SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S': '0.1'}):
            self.stop()
        self.assertFalse(shutdown_policy.has_clean_capture_closure(self.root))
        self.app.frames.close.assert_not_called()
        self.assertNotEqual(self.app.exit_code, 0)

    def test_failed_frame_flush_never_publishes_receipt(self):
        self.app.frames.close.side_effect = OSError('disk full')
        self.stop()
        self.assertFalse(shutdown_policy.has_clean_capture_closure(self.root))
        self.assertNotEqual(self.app.exit_code, 0)

    def test_wait_return_without_confirmed_eos_and_null_does_not_close_writers(self):
        self.app.pipeline.closure_confirmed = False
        self.stop()
        self.app.frames.close.assert_not_called()
        self.assertNotEqual(self.app.exit_code, 0)
        self.assertFalse(shutdown_policy.has_clean_capture_closure(self.root))

    def test_prior_capture_fault_is_not_reclassified_by_clean_teardown(self):
        self.app.exit_code = 4
        self.stop()
        self.assertFalse(shutdown_policy.has_clean_capture_closure(self.root))
        self.assertEqual(self.app.exit_code, 4)

    def test_unsolicited_pipeline_end_does_not_get_clean_receipt(self):
        self.app._explicit_stop_requested = False
        self.stop()
        self.assertFalse(shutdown_policy.has_clean_capture_closure(self.root))

    def test_receipt_write_failure_does_not_claim_clean_shutdown(self):
        with mock.patch('squeakview.apps.inference.capture_shutdown.write_capture_closure',
                        side_effect=OSError('disk full')):
            self.stop()
        self.assertFalse(shutdown_policy.has_clean_capture_closure(self.root))
        self.assertNotEqual(self.app.exit_code, 0)

    def _slow_drain_with_progress(self, *, encoded):
        count = [0]
        self.app.record_telemetry[0].shutdown_progress.side_effect = (
            lambda: (0, 0, count[0]) if encoded else None)

        def drain():
            with (self.root / 'raw.mp4').open('wb', buffering=0) as video:
                for _ in range(12):
                    count[0] += 1
                    if not encoded:
                        video.write(b'encoded-frame')
                    time.sleep(.05)
        self.app.pipeline.wait.side_effect = drain
        with mock.patch.dict('os.environ', {
            'SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S': '.25',
            'SQUEAKVIEW_PIPELINE_SHUTDOWN_MAX_S': '2',
        }):
            self.stop()
        self.assertEqual(self.app.exit_code, 0)
        self.assertTrue(shutdown_policy.has_clean_capture_closure(self.root))
        report = self.status()['shutdown_stages']['capture pipeline wait']
        self.assertTrue(report['completed'])
        self.assertGreater(report['elapsed_s'], .25)
        key = 'camera_0_encoded' if encoded else 'camera_0_video_bytes'
        self.assertGreater(report['counters'][key], 0)

    def test_encoder_progress_allows_slow_native_drain(self):
        self._slow_drain_with_progress(encoded=True)

    def test_video_growth_allows_slow_muxer_drain(self):
        self._slow_drain_with_progress(encoded=False)


class FinalizationExitPolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        initialize_layout(self.root, compact_metadata=True)
        run_context.write_status(self.root, 'recording',
            recording_validation={'passed': True}, acquisition_integrity={'passed': True},
            alignment_validated=True)
        self.capture = mock.Mock()
        self.capture.wait.return_value = -signal.SIGABRT
        self.serial = mock.Mock(stop_ack_count=10, fatal_error=None)
        self.hooks = FinalizationHooks(
            log=mock.Mock(), transition=mock.Mock(), wait_for_capture_drain=mock.Mock(return_value=True),
            run_capture_finalizer=mock.Mock(return_value=0), write_run_manifest=mock.Mock(return_value=True),
            run_output_snapshot=mock.Mock(return_value={}), sleep=lambda _: None,
        )
        self.request = FinalizationRequest(
            final_state='finalized', error=None, terminate_capture=True,
            known_capture_returncode=None, run_dir=self.root, capture=self.capture,
            serial=self.serial, capture_running=True, controller_started=True, trigger_on=True,
            phase=RunPhase.RECORDING, alignment_required=True,
        )

    def status(self):
        return run_context.read_json_required(artifact_path(self.root, 'run_status.json'))

    def test_late_abort_with_proven_closure_and_validated_data_finishes_with_warning(self):
        shutdown_policy.write_capture_closure(self.root)
        result = finalize_run(self.request, self.hooks)
        self.assertEqual(result.final_state, 'finalized')
        status = self.status()
        self.assertEqual(status['capture_returncode'], -signal.SIGABRT)
        self.assertTrue(status['shutdown_degraded'])
        self.assertIn('confirmed recording closure', status['shutdown_warnings'][0])
        presentation = present_finalization(status)
        self.assertTrue(presentation.passed)
        self.assertEqual(presentation.preview_status, 'Complete with warnings')

    def test_nonzero_exit_without_receipt_still_fails(self):
        self.assertEqual(finalize_run(self.request, self.hooks).final_state, 'failed')

    def test_receipt_never_overrides_failed_video_validation(self):
        shutdown_policy.write_capture_closure(self.root)
        run_context.update_status(self.root, recording_validation={'passed': False})
        self.assertEqual(finalize_run(self.request, self.hooks).final_state, 'failed')
        self.assertNotIn('saved data passed validation', str(self.status()))

    def test_receipt_never_overrides_failed_integrity_validation(self):
        shutdown_policy.write_capture_closure(self.root)
        run_context.update_status(self.root, acquisition_integrity={'passed': False})
        self.assertEqual(finalize_run(self.request, self.hooks).final_state, 'failed')

    def test_receipt_never_overrides_failed_controller_alignment(self):
        shutdown_policy.write_capture_closure(self.root)
        run_context.update_status(self.root, alignment_validated=False)
        self.assertEqual(finalize_run(self.request, self.hooks).final_state, 'failed')

    def test_receipt_never_overrides_unconfirmed_process_exit(self):
        shutdown_policy.write_capture_closure(self.root)
        self.capture.wait.side_effect = subprocess.TimeoutExpired('capture', 2)
        self.assertEqual(finalize_run(self.request, self.hooks).final_state, 'failed')
        self.hooks.run_capture_finalizer.assert_not_called()

    def test_receipt_never_overrides_controller_stop_failure(self):
        shutdown_policy.write_capture_closure(self.root)
        self.serial.wait_for_stop_ack.return_value = False
        self.assertEqual(finalize_run(self.request, self.hooks).final_state, 'failed')

    def test_receipt_never_overrides_missing_frames_at_drain(self):
        shutdown_policy.write_capture_closure(self.root)
        self.hooks.wait_for_capture_drain.return_value = False
        self.assertEqual(finalize_run(self.request, self.hooks).final_state, 'failed')

    def test_receipt_never_overrides_sensor_failure(self):
        shutdown_policy.write_capture_closure(self.root)
        self.serial.fatal_error = 'truncated sensor record'
        self.assertEqual(finalize_run(self.request, self.hooks).final_state, 'failed')

    def test_receipt_never_overrides_finalizer_timeout(self):
        shutdown_policy.write_capture_closure(self.root)
        self.hooks.run_capture_finalizer.return_value = 124
        self.assertEqual(finalize_run(self.request, self.hooks).final_state, 'failed')

    def test_warning_survives_successful_process_exit(self):
        self.capture.wait.return_value = 0
        run_context.update_status(self.root, shutdown_warnings=['system diagnostics did not close'])
        self.assertEqual(finalize_run(self.request, self.hooks).final_state, 'finalized')
        self.assertTrue(self.status()['shutdown_degraded'])


class ShutdownEvidenceTests(unittest.TestCase):
    def test_receipt_requires_exact_types_and_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutdown_policy.write_capture_closure(root)
            path = artifact_path(root, shutdown_policy.CLOSURE_FILENAME)
            original = run_context.read_json_required(path)
            for key, value in [('schema_version', True), ('schema_version', 1), ('run_dir', '/another-run'),
                               ('pipeline_drained', 1), ('writers_closed', False),
                               ('eos_observed', False), ('pipeline_null_confirmed', False),
                               ('capture_exit_code', False), ('explicit_stop', False)]:
                with self.subTest(key=key):
                    run_context.atomic_write_json(path, {**original, key: value})
                    self.assertFalse(shutdown_policy.has_clean_capture_closure(root))
            for malformed in ('[]', '{', 'x' * 5000):
                path.write_text(malformed)
                self.assertFalse(shutdown_policy.has_clean_capture_closure(root))

    def test_supervisor_budget_covers_all_stages_including_custom_drain_timeout(self):
        for value in ('', '0.1', '20', '60', '120', 'inf', 'nan', 'bad', '121'):
            with self.subTest(value=value), mock.patch.dict('os.environ', {
                'SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S': value,
            }):
                pipeline = shutdown_policy.pipeline_shutdown_timeout_s()
                grace = shutdown_policy.capture_exit_grace_s()
                self.assertGreater(grace, pipeline + min(5, pipeline) + 1 + 5 + 1 + 3.5)
                self.assertLessEqual(pipeline, 120)
