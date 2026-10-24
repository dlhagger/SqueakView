from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from squeakview.apps.operator.backend.manager import OperatorBackend
from squeakview.apps.operator.backend.supervision import ProcessHandle
from squeakview.common.child_events import encode_child_event
from squeakview.common.serial import SerialHandle


def broken_presentation(_message):
    raise RuntimeError("Qt receiver has been destroyed")


class FailureIsolationTests(unittest.TestCase):
    def test_capture_messages_never_reopen_live_serial_writers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            serial = SerialHandle("test", 115200, broken_presentation)
            try:
                self.assertTrue(serial.set_csv_path(path))
                csv_file, journal = serial._csv_file, serial._v2_journal
                self.assertTrue(serial.set_csv_path(path))
                self.assertIs(serial._csv_file, csv_file)
                self.assertIs(serial._v2_journal, journal)
                backend = OperatorBackend.__new__(OperatorBackend)
                backend.emit = broken_presentation
                backend.state = SimpleNamespace(run_dir=path, serial=serial)
                backend._recording_started = False
                backend._ensure_metadata = mock.Mock()
                backend._mark_inference_ready = mock.Mock()
                backend._on_capture_protocol_fatal = mock.Mock()
                # The capture worker emits these at both ends of the same run;
                # neither is an instruction to replace the serial ledger.
                for kind in ("pipeline_ready", "capture_closed", "pipeline_ready"):
                    backend._inference_emit(encode_child_event(kind, run_dir=str(path)))
                    self.assertIs(serial._csv_file, csv_file)
                    self.assertIs(serial._v2_journal, journal)
                    self.assertFalse(csv_file.closed)
                serial._emit_serial_logs = True
                serial._handle_semantic_line(
                    "DRINK_START,10,20,LD,1,0,0,0,Eligible,nan", received_ns=30
                )
                self.assertIsNone(serial.fatal_error)
                serial._csv_file.flush()
                with (path / "serial.csv").open() as handle:
                    self.assertEqual(list(csv.DictReader(handle))[0]["eventType"], "DRINK_START")
                backend._mark_inference_ready.assert_called()
            finally:
                serial.close()

    def test_failed_dashboard_callback_does_not_escape_into_serial_persistence(self):
        backend = OperatorBackend.__new__(OperatorBackend)
        backend.emit = broken_presentation
        backend.ingest = mock.Mock(side_effect=RuntimeError("display unavailable"))
        backend._serial_emit("【SER】 DRINK_START,10,20,LD,1,0,0,0,Eligible,nan")
        backend.ingest.assert_called_once()

    def _run_noisy_child(self, emit, output_path=None):
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdout.write('x'*1024+'\\n'); sys.stdout.flush(); sys.stdout.write(('frame status\\n')*200000)"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
        )
        self.addCleanup(lambda: child.kill() if child.poll() is None else None)
        exited = threading.Event()
        handle = ProcessHandle("capture", child, emit, lambda _code: exited.set(), output_log_path=output_path)
        self.assertTrue(exited.wait(8), "child stdout stopped draining after a presentation/log failure")
        handle._thread.join(1)
        self.assertEqual(child.returncode, 0)

    def test_presentation_exception_cannot_back_up_capture_stdout(self):
        self._run_noisy_child(broken_presentation)

    def test_diagnostic_file_failure_cannot_back_up_capture_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = mock.Mock()
            fake.write.side_effect = OSError("diagnostic disk unavailable")
            fake.close.side_effect = OSError("flush unavailable")
            with mock.patch.object(Path, "open", return_value=fake):
                self._run_noisy_child(lambda _line: None, Path(tmp) / "capture.log")


if __name__ == "__main__":
    unittest.main()
