from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps.operator.gui.stall_diagnostics import GuiStallDiagnostics


class GuiStallDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.now = 0.0
        self.monitor = GuiStallDiagnostics(monotonic=lambda: self.now, start=False)
        self.monitor.set_run(self.run, True)
        self.path = self.run / "diagnostics" / "gui_stalls.jsonl"

    def records(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_stall_captures_blocked_thread_and_recovery(self):
        entered, release = threading.Event(), threading.Event()

        def blocked_gui_operation():
            entered.set()
            release.wait()

        worker = threading.Thread(target=blocked_gui_operation, name="simulated-gui")
        worker.start()
        self.assertTrue(entered.wait(1))
        try:
            self.monitor.sent()
            self.now = 2.9
            self.monitor.poll()
            self.assertFalse(self.path.exists())
            for self.now in (3.0, 6.0, 9.0, 12.0, 15.0):
                self.monitor.poll()
            records = self.records()
            self.assertEqual(len(records), 3)
            blocked = next(t for t in records[0]["threads"] if t["name"] == "simulated-gui")
            self.assertIn("blocked_gui_operation", "".join(blocked["stack"]))
            self.monitor.generated()
            self.monitor.sent()
            self.monitor.poll()
            self.assertEqual(self.records()[-1]["event"], "recovered")
        finally:
            release.set()
            worker.join(1)

    def test_responsive_gui_with_stalled_sender_is_distinguished(self):
        self.monitor.sent()
        self.now = 4.0
        self.monitor.generated()
        self.monitor.poll()
        self.assertEqual(self.records()[0]["event"], "heartbeat_send_stall")

    def test_recovered_tick_between_polls_is_recorded(self):
        self.now = 4.0
        self.monitor.generated()
        self.monitor.sent()
        self.monitor.poll()
        self.assertEqual(self.records()[0]["event"], "recovered")
        self.assertEqual(self.records()[0]["last_tick_gap_s"], 4.0)

    def test_file_cap_and_storage_failure_are_best_effort(self):
        self.monitor.MAX_BYTES = 1
        self.now = 3.0
        self.monitor.poll()
        self.assertFalse(self.path.exists())
        self.monitor.MAX_BYTES = 2 * 1024 * 1024
        self.monitor._written.clear()
        with mock.patch("os.open", side_effect=OSError("disk full")):
            self.now = 6.0
            self.monitor.poll()
        self.assertFalse(self.path.exists())

    def test_callbacks_do_no_io_and_close_never_joins_writer(self):
        with mock.patch.object(self.monitor, "_capture") as capture:
            self.monitor.generated()
            self.monitor.sent()
            self.monitor.close("IPC disconnected")
            capture.assert_not_called()

    def test_background_monitor_works_without_gui_ticks(self):
        monitor = GuiStallDiagnostics()
        self.addCleanup(monitor.close)
        captured = threading.Event()
        with mock.patch.object(monitor, "_capture", side_effect=lambda *a, **k: captured.set()):
            monitor.set_run(self.run, True)
            monitor._generated -= 4
            self.assertTrue(captured.wait(2))

    def test_blocked_diagnostic_writer_does_not_block_heartbeat_or_close(self):
        entered, release = threading.Event(), threading.Event()
        monitor = GuiStallDiagnostics(start=False)

        def blocked_storage(*args, **kwargs):
            entered.set()
            release.wait()

        with mock.patch.object(monitor, "_capture", side_effect=blocked_storage):
            monitor.set_run(self.run, True)
            monitor._generated -= 4
            monitor._thread.start()
            try:
                self.assertTrue(entered.wait(2))
                completed = threading.Event()

                def gui_callbacks():
                    monitor.generated()
                    monitor.sent()
                    monitor.close()
                    completed.set()

                caller = threading.Thread(target=gui_callbacks, daemon=True)
                caller.start()
                self.assertTrue(completed.wait(1), "GUI waited for diagnostic storage")
            finally:
                release.set()
                monitor.close()
                monitor._thread.join(2)

    def test_idle_does_not_write_and_loss_is_saved_by_monitor(self):
        self.monitor.set_run(self.run, False)
        self.now = 20
        self.monitor.poll()
        self.assertFalse(self.path.exists())
        self.monitor.close("peer closed")
        self.monitor._watch()
        record = self.records()[0]
        self.assertEqual(record["event"], "connection_lost")
        self.assertEqual(record["failure"], "peer closed")


if __name__ == "__main__":
    unittest.main()
