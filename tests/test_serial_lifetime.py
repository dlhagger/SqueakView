"""Scientific writers outlive every serial producer, including delayed shutdown."""
import csv
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from squeakview.common import serial as serial_util
from squeakview.common import controller_protocol_v2 as v2


class Port:
    def __init__(self):
        self.is_open = True
        self.close_count = 0
    def close(self):
        self.close_count += 1
        self.is_open = False


class SerialLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.handle = serial_util.SerialHandle("/dev/test", 115200, lambda _message: None)
        self.handle._close_wait_timeout_s = 0.05
        self.handle.set_csv_path(self.root)
        self.handle._closed = False
        self.handle.ser = Port()
        self.releases = []

    def tearDown(self):
        for release in self.releases:
            release.set()
        self.handle.close()
        self.handle._close_done.wait(2)
        self.temp.cleanup()

    def frame(self):
        return v2.decode_frame(v2.encode_frame(
            message_type=v2.MessageType.EVENT, flags=0, boot_id=9,
            session_id=1, sequence=1, monotonic_us=7,
            payload=b"DRINK_START,10,20,L,1,30,40,50,Eligible,nan",
        )[:-1])

    def start_persistence(self):
        self.handle._v2_worker = threading.Thread(target=self.handle._v2_persistence_pump, daemon=True)
        self.handle._v2_worker.start()

    def blocked_reader(self, after_release=None):
        release = threading.Event()
        self.releases.append(release)
        def read():
            release.wait()
            if after_release:
                after_release()
        self.handle._thread = threading.Thread(target=read, daemon=True)
        self.handle._thread.start()
        return release

    def test_late_reader_record_is_persisted_before_journal_and_csv_close(self):
        self.start_persistence()
        release = self.blocked_reader(lambda: self.handle._v2_queue.put((self.frame(), 1, 2)))
        journal = self.handle._v2_journal
        csv_file = self.handle._csv_file
        self.handle.close()
        self.assertIn("serial reader did not stop", self.handle.fatal_error)
        self.assertFalse(csv_file.closed)
        self.assertFalse(journal._file.closed)
        self.assertFalse(self.handle._v2_worker_stop.is_set())
        self.assertFalse(self.handle._closed)
        self.assertFalse(self.handle.open(self.root))
        self.assertIn("still pending", self.handle.last_error)
        release.set()
        self.assertTrue(self.handle._close_done.wait(2))
        self.assertTrue(self.handle._closed)
        self.assertTrue(csv_file.closed)
        self.assertTrue(journal._file.closed)
        saved = [json.loads(line) for line in (self.root / "diagnostics/controller_v2.jsonl").read_text().splitlines()]
        self.assertEqual([record["sequence"] for record in saved], [1])
        with (self.root / "serial.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["eventType"] for row in rows], ["DRINK_START"])

    def test_blocked_persistence_retains_writers_until_store_finishes(self):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        journal = self.handle._v2_journal
        csv_file = self.handle._csv_file
        store = journal.store
        def delayed_store(*args, **kwargs):
            entered.set()
            release.wait()
            return store(*args, **kwargs)
        with mock.patch.object(journal, "store", side_effect=delayed_store):
            self.start_persistence()
            self.handle._v2_queue.put((self.frame(), 1, 2))
            self.assertTrue(entered.wait(1))
            self.handle.close()
            self.assertIn("persistence worker did not stop", self.handle.fatal_error)
            self.assertFalse(csv_file.closed)
            self.assertFalse(journal._file.closed)
            release.set()
            self.assertTrue(self.handle._close_done.wait(2))
        self.assertTrue(journal._file.closed)
        self.assertTrue(csv_file.closed)
        self.assertEqual(self.handle._v2_counts["frames_stored"], 1)

    def test_concurrent_close_has_one_owner_and_closes_each_writer_once(self):
        port = self.handle.ser
        journal = self.handle._v2_journal
        release = self.blocked_reader()
        with mock.patch.object(journal, "close", wraps=journal.close) as close_journal:
            callers = [threading.Thread(target=self.handle.close) for _ in range(8)]
            for caller in callers:
                caller.start()
            for caller in callers:
                caller.join(1)
                self.assertFalse(caller.is_alive())
            self.assertEqual(port.close_count, 1)
            self.assertEqual(close_journal.call_count, 0)
            release.set()
            self.assertTrue(self.handle._close_done.wait(2))
            self.handle.close()
            self.assertEqual(close_journal.call_count, 1)
        self.assertFalse(self.handle._atexit_registered)

    def test_worker_fatal_callback_can_close_without_joining_itself(self):
        callback_returned = threading.Event()
        csv_file = self.handle._csv_file
        def fatal(_message):
            self.handle.close()
            self.assertFalse(csv_file.closed)
            callback_returned.set()
        self.handle.on_fatal = fatal
        self.handle._thread = threading.Thread(target=lambda: self.handle._report_fatal("reader failed"), daemon=True)
        self.handle._thread.start()
        self.assertTrue(callback_returned.wait(1))
        self.assertTrue(self.handle._close_done.wait(2))
        self.assertTrue(csv_file.closed)
        self.assertTrue(self.handle._closed)

    def test_partial_thread_start_failure_retires_started_workers_and_files(self):
        self.handle.close()
        handle = serial_util.SerialHandle("/dev/test", 115200, lambda _message: None)
        port = Port()
        started = []
        files = []
        original_start = threading.Thread.start
        original_set_csv = handle.set_csv_path
        calls = 0
        def start(thread):
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise RuntimeError("injected thread start failure")
            original_start(thread)
            started.append(thread)
        def set_csv(path):
            result = original_set_csv(path)
            files.extend([handle._csv_file, handle._v2_journal._file])
            return result
        with mock.patch.object(serial_util, "serial", mock.Mock(Serial=mock.Mock(return_value=port))), mock.patch.object(threading.Thread, "start", new=start), mock.patch.object(handle, "set_csv_path", side_effect=set_csv):
            self.assertFalse(handle.open(self.root / "partial"))
        for worker in started:
            worker.join(1)
            self.assertFalse(worker.is_alive())
        self.assertTrue(all(handle.closed for handle in files))
        self.assertEqual(handle.last_error, "injected thread start failure")
        self.assertTrue(handle._closed)
        self.assertFalse(handle._atexit_registered)
        self.assertFalse(port.is_open)

    def test_exit_cleanup_does_not_start_a_thread(self):
        csv_file = self.handle._csv_file
        journal = self.handle._v2_journal
        with mock.patch.object(threading.Thread, "start", side_effect=RuntimeError("cannot start thread at shutdown")) as start:
            self.handle._close_at_exit()
        start.assert_not_called()
        self.assertTrue(csv_file.closed)
        self.assertTrue(journal._file.closed)
        self.assertTrue(self.handle._closed)

    def test_reconnect_published_during_shutdown_is_closed_after_reader_exit(self):
        replacement = Port()
        release = self.blocked_reader(lambda: setattr(self.handle, "ser", replacement))
        caller = threading.Thread(target=self.handle.close)
        caller.start()
        self.assertTrue(self.handle._stop.wait(1))
        release.set()
        caller.join(2)
        self.assertFalse(caller.is_alive())
        self.assertTrue(self.handle._close_done.wait(2))
        self.assertFalse(replacement.is_open)
        self.assertEqual(replacement.close_count, 1)
        self.assertTrue(self.handle._closed)

    def test_python_exit_callback_closes_writers_without_thread_start(self):
        marker = self.root / "exit_result.txt"
        run_dir = self.root / "exit-run"
        program = "\n".join([
            "import atexit",
            "from pathlib import Path",
            "from squeakview.common.serial import SerialHandle",
            f"marker = Path({str(marker)!r})",
            "def verify(): marker.write_text(str(handle._closed))",
            "atexit.register(verify)",
            "handle = SerialHandle('/dev/test', 115200, lambda message: None)",
            f"assert handle.set_csv_path(Path({str(run_dir)!r}))",
        ])
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Exception ignored in atexit", result.stderr)
        self.assertEqual(marker.read_text(), "True")

    def test_blocked_command_writer_keeps_csv_open_until_write_returns(self):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        csv_file = self.handle._csv_file
        def write(payload):
            entered.set()
            release.wait()
            return len(payload)
        self.handle.ser.write = write
        self.handle._writer_thread = threading.Thread(target=self.handle._writer_pump, daemon=True)
        self.handle._writer_thread.start()
        self.handle._queue_line("PING", priority=1, wait=False, marker_before="TEST_SEND")
        self.assertTrue(entered.wait(1))
        self.handle.close()
        self.assertIn("serial command writer did not stop", self.handle.fatal_error)
        self.assertFalse(csv_file.closed)
        release.set()
        self.assertTrue(self.handle._close_done.wait(2))
        self.assertTrue(csv_file.closed)
        with (self.root / "serial.csv").open() as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]["reason"], "TEST_SEND")

    def test_timeout_fatal_callback_does_not_reenter_its_own_wait(self):
        release = self.blocked_reader()
        self.handle._close_wait_timeout_s = 0.2
        self.handle.on_fatal = lambda _message: self.handle.close()
        started = time.monotonic()
        self.handle.close()
        self.assertLess(time.monotonic() - started, 0.35)
        release.set()
        self.assertTrue(self.handle._close_done.wait(2))


if __name__ == "__main__":
    unittest.main()
