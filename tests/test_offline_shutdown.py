"""Native teardown ownership and cancellation without decoder/GPU use."""
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from squeakview.apps.inference import offline


class OfflineShutdownTests(unittest.TestCase):
    def app(self, pipeline):
        app = object.__new__(offline.OfflineInferenceApp)
        app.pipeline = pipeline
        app.observations = mock.Mock()
        app.ledger = mock.Mock()
        app.stop_event = threading.Event()
        app.exit_code = 0
        app.eos_received = False
        app.audit = SimpleNamespace(count=0)
        app._write_manifest = mock.Mock()
        return app

    def test_partial_start_failure_waits_for_native_stop_before_closing(self):
        ended = threading.Event()
        pipeline = mock.Mock()
        pipeline.start.side_effect = RuntimeError('partial start')
        pipeline.wait.side_effect = ended.wait
        pipeline.stop.side_effect = ended.set
        app = self.app(pipeline)
        self.assertEqual(app.run(), 1)
        pipeline.wait.assert_called_once()
        pipeline.stop.assert_called_once()
        app.observations.close.assert_called_once()
        app.ledger.close.assert_called_once()

    def test_wait_error_wakes_owner_and_preserves_failure(self):
        pipeline = mock.Mock()
        pipeline.wait.side_effect = RuntimeError('native bus failure')
        app = self.app(pipeline)
        self.assertEqual(app.run(), 1)
        self.assertIn('native bus failure', str(app._write_manifest.call_args_list))
        app.ledger.close.assert_called_once()

    def test_explicit_cancel_stops_pipeline_and_cannot_claim_completion(self):
        ended = threading.Event()
        pipeline = mock.Mock()
        pipeline.wait.side_effect = ended.wait
        pipeline.stop.side_effect = ended.set
        app = self.app(pipeline)
        app.stop_event.set()
        self.assertEqual(app.run(), 1)
        pipeline.stop.assert_called_once()
        app.observations.close.assert_called_once()

    def test_stalled_native_wait_does_not_close_callback_owned_writers(self):
        release = threading.Event()
        waiting = threading.Event()
        finished = threading.Event()
        pipeline = mock.Mock()
        def wait():
            waiting.set()
            release.wait(2)
            finished.set()
        pipeline.wait.side_effect = wait
        app = self.app(pipeline)
        app.stop_event.set()
        try:
            with mock.patch.object(offline, 'OFFLINE_SHUTDOWN_TIMEOUT_S', .05):
                self.assertEqual(app.run(), 1)
            self.assertTrue(waiting.is_set())
            self.assertTrue(app._force_process_exit)
            app.observations.close.assert_not_called()
            app.ledger.close.assert_not_called()
        finally:
            release.set()
            self.assertTrue(finished.wait(2))
