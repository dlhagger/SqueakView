"""Shutdown remains ordered for healthy writers and bounded for broken ones."""
import csv
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from squeakview.apps.inference import service_maker_runner as runner
from squeakview.apps.inference.recording import RecordingAdmissionOperator
from squeakview.common import run_context


class CaptureShutdownTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.app = runner.ServiceMakerApp(runner.InferenceConfig(
            run_dir=Path(self.temp.name), enable_infer=False,
        ))

    def test_capture_crash_diagnostics_precede_native_app_construction(self):
        config = runner.InferenceConfig(run_dir=Path(self.temp.name), enable_infer=False)
        order = []
        def install(*args, **kwargs):
            order.append(("diagnostics", args, kwargs))
        def construct(_config):
            order.append(("construct",))
            raise RuntimeError("native construction failed")
        with mock.patch.object(runner, "install_crash_diagnostics", side_effect=install), mock.patch.object(runner, "ServiceMakerApp", side_effect=construct):
            with self.assertRaisesRegex(RuntimeError, "native construction failed"):
                runner.run(config)
        self.assertEqual(order[0], ("diagnostics", ("capture",), {"log_path": Path(self.temp.name) / "diagnostics" / "capture_crash.log"}))
        self.assertEqual(order[1], ("construct",))

    def test_normal_drain_finishes_inflight_callback_before_closing_ledger(self):
        entered, release, finished = (threading.Event() for _ in range(3))
        def admit(_timestamp):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test callback was not drained")
        admission = RecordingAdmissionOperator(
            self.app.run_dir / "record_admission.csv", 0,
            telemetry=SimpleNamespace(admit=admit),
        )
        self.app.record_admissions = [admission]
        def callback():
            admission.handle_buffer(SimpleNamespace(timestamp=123))
            finished.set()
        worker = threading.Thread(target=callback, daemon=True)
        worker.start()
        self.assertTrue(entered.wait(1))
        self.app.pipeline = SimpleNamespace(stop=release.set, wait=lambda: worker.join(2))
        original_close = admission.close
        def checked_close():
            self.assertTrue(finished.is_set(), "writer closed before in-flight callback drained")
            original_close()
        with mock.patch.object(admission, "close", side_effect=checked_close):
            self.app.stop()
        with admission.path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["pts_ns"] for row in rows], ["123"])
        self.assertEqual(self.app.exit_code, 0)
        self.assertFalse(self.app._force_process_exit)

    def test_native_timeout_and_locked_writer_do_not_deadlock_cleanup(self):
        entered, release = threading.Event(), threading.Event()
        def admit(_timestamp):
            entered.set()
            release.wait(2)
        admission = RecordingAdmissionOperator(
            self.app.run_dir / "record_admission.csv", 0,
            telemetry=SimpleNamespace(admit=admit),
        )
        self.app.record_admissions = [admission]
        worker = threading.Thread(target=lambda: admission.handle_buffer(SimpleNamespace(timestamp=123)), daemon=True)
        worker.start()
        self.assertTrue(entered.wait(1))
        self.app.pipeline = SimpleNamespace(stop=lambda: release.wait(2), wait=lambda: worker.join(2))
        started = time.monotonic()
        try:
            with mock.patch.dict("os.environ", {"SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S": "0.1"}):
                self.app.stop()
            self.assertLess(time.monotonic() - started, 1)
            self.assertTrue(self.app._force_process_exit)
            status = run_context.read_json(self.app.run_dir / "run_status.json")
            errors = status["capture_close_errors"]
            self.assertTrue(any("record admission 0" in error and "timed out" in error for error in errors))
            self.assertNotEqual(status["capture_exit_code"], 0)
        finally:
            release.set()
            worker.join(2)
            admission.close()

    def test_stalled_monitor_join_is_bounded_and_reported(self):
        release = threading.Event()
        monitor = threading.Thread(target=release.wait, daemon=True)
        monitor.start()
        self.app._storage_reserve_thread = monitor
        try:
            with mock.patch.dict("os.environ", {"SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S": "0.1"}):
                self.app.stop()
            self.assertTrue(self.app._force_process_exit)
            status = run_context.read_json(self.app.run_dir / "run_status.json")
            self.assertIn("storage reserve monitor: timed out", status["capture_close_error"])
        finally:
            release.set()
            monitor.join(2)

    def test_nonfinite_or_unrepresentable_timeout_cannot_break_cleanup(self):
        for value in ("inf", "nan", "1e300"):
            with self.subTest(value=value):
                app = runner.ServiceMakerApp(runner.InferenceConfig(run_dir=Path(self.temp.name) / value, enable_infer=False))
                app.pipeline = SimpleNamespace(stop=lambda: time.sleep(0.005), wait=lambda: None)
                with mock.patch.dict("os.environ", {"SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S": value}):
                    app.stop()
                self.assertEqual(app.exit_code, 0)
                self.assertTrue((app.run_dir / "run_status.json").exists())

    def test_stalled_status_write_still_reports_terminal_event(self):
        release = threading.Event()
        try:
            with mock.patch.object(runner.run_context, "write_status", side_effect=lambda *_a, **_k: release.wait(2)), mock.patch.object(runner, "_safe_print") as emit, mock.patch.dict("os.environ", {"SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S": "0.1"}):
                self.app.stop()
            self.assertTrue(self.app._force_process_exit)
            self.assertTrue(any('"type":"capture_closed"' in str(call) for call in emit.call_args_list))
            self.assertNotEqual(self.app.exit_code, 0)
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
