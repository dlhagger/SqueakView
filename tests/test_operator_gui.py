from __future__ import annotations

import os
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from squeakview.apps.operator.gui.config_dialog import SessionLauncherDialog
from squeakview.apps.operator.gui.session_dialog import (
    SessionLauncherDialog as ExtractedSessionLauncherDialog,
)
from squeakview.apps.operator.gui.main_window import (
    MainWindow,
    _create_supervisor_heartbeat_timer,
    _elapsed_text,
    _is_serial_log_message,
    _last_csv_row,
    _tail_text_line,
)
from squeakview.common.log_mirror import LineBufferedLogMirror
from squeakview.common.profiles import ExperimentProfile, SubjectProfile


class RuntimeFileHelpersTest(unittest.TestCase):
    def test_supervisor_heartbeat_originates_on_qt_main_loop_thread(self) -> None:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        owner = QtCore.QObject()
        backend = mock.Mock()
        window = mock.Mock(backend=backend)
        observed_threads = []
        loop = QtCore.QEventLoop()

        def heartbeat() -> None:
            observed_threads.append(QtCore.QThread.currentThread())
            MainWindow._send_supervisor_heartbeat(window)
            loop.quit()

        timer = _create_supervisor_heartbeat_timer(owner, heartbeat)
        timer.setInterval(1)
        QtCore.QTimer.singleShot(1_000, loop.quit)
        loop.exec()
        timer.stop()

        self.assertIsNotNone(app)
        self.assertEqual(observed_threads, [app.thread()])
        backend.heartbeat.assert_called_once_with()

    def test_gui_log_mirror_bounds_file_and_partial_line_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gui.log"
            console = io.StringIO()
            mirror = LineBufferedLogMirror(
                path, console, max_bytes=80, max_pending_chars=16
            )
            self.addCleanup(mirror.close)

            mirror.write("first line\n")
            mirror.write("x" * 100)
            mirror.write("\nlast line\n")
            mirror.flush()

            self.assertLessEqual(path.stat().st_size, 80)
            self.assertIn("first line", path.read_text())
            self.assertIn("line truncated", path.read_text())
            self.assertIn("x" * 100, console.getvalue())

    def test_gui_log_mirror_stops_file_writes_at_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gui.log"
            console = io.StringIO()
            mirror = LineBufferedLogMirror(path, console, max_bytes=20)
            self.addCleanup(mirror.close)

            mirror.write("first\n")
            mirror.write("this line cannot fit\n")
            mirror.write("later\n")
            mirror.flush()

            self.assertLessEqual(path.stat().st_size, 20)
            self.assertEqual(path.read_text(), "first\n")
            self.assertIn("file log capped", console.getvalue())
            self.assertIn("later", console.getvalue())

    def test_gui_log_mirror_filters_fragmented_camera_lines_without_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "gui.log"
            console = io.StringIO()
            mirror = LineBufferedLogMirror(log_path, console)
            self.addCleanup(mirror.close)

            mirror.write("[SER] CAMERA_HIGH,1,2")
            mirror.write("\n")
            mirror.write("capture started")
            mirror.write("\n")
            mirror.write("[SER] CAMERA_LOW,1,2\n")
            mirror.flush()

            self.assertEqual(log_path.read_text(), "capture started\n")
            self.assertEqual(
                console.getvalue(),
                "[SER] CAMERA_HIGH,1,2\ncapture started\n[SER] CAMERA_LOW,1,2\n",
            )

    def test_gui_log_mirror_flushes_an_unterminated_normal_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "gui.log"
            mirror = LineBufferedLogMirror(log_path, io.StringIO())
            self.addCleanup(mirror.close)

            mirror.write("partial diagnostic")
            mirror.flush()

            self.assertEqual(log_path.read_text(), "partial diagnostic")

    def test_gui_log_mirror_flush_does_not_orphan_camera_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "gui.log"
            mirror = LineBufferedLogMirror(log_path, io.StringIO())
            self.addCleanup(mirror.close)

            mirror.write("[SER] CAMERA_HIGH,1,2")
            mirror.flush()
            mirror.write("\nnormal\n")
            mirror.flush()

            self.assertEqual(log_path.read_text(), "normal\n")

    def test_gui_log_mirror_keeps_file_log_after_terminal_disconnect(self) -> None:
        class BrokenTerminal(io.StringIO):
            def write(self, _data: str) -> int:
                raise BrokenPipeError("terminal disconnected")

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "gui.log"
            mirror = LineBufferedLogMirror(log_path, BrokenTerminal())
            self.addCleanup(mirror.close)

            self.assertEqual(mirror.write("shutdown started\n"), 17)
            mirror.flush()

            self.assertEqual(log_path.read_text(), "shutdown started\n")

    def test_gui_log_mirror_close_is_idempotent_and_does_not_close_console(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "gui.log"
            console = io.StringIO()
            with (
                mock.patch("squeakview.common.log_mirror.atexit.register") as register,
                mock.patch("squeakview.common.log_mirror.atexit.unregister") as unregister,
            ):
                mirror = LineBufferedLogMirror(log_path, console)
                mirror.write("final partial")
                mirror.close()
                mirror.close()

            self.assertTrue(mirror.closed)
            self.assertFalse(console.closed)
            self.assertEqual(log_path.read_text(), "final partial")
            register.assert_called_once_with(mirror.close)
            unregister.assert_called_once_with(mirror.close)
            self.assertEqual(mirror.write("console only\n"), 13)
            self.assertIn("console only", console.getvalue())
            self.assertEqual(log_path.read_text(), "final partial")

    def test_elapsed_text_supports_long_runs(self) -> None:
        self.assertEqual(_elapsed_text(16 * 3600 + 2 * 60 + 9), "16:02:09")

    def test_serial_messages_are_identified_for_status_bar_filtering(self) -> None:
        self.assertTrue(_is_serial_log_message("[17:01:37] 【SER】 CAMERA_HIGH,..."))
        self.assertTrue(_is_serial_log_message("[17:01:37] 【SER→】 STOP"))
        self.assertTrue(_is_serial_log_message("[17:01:37] [SER] ACK_STOP received."))
        self.assertFalse(_is_serial_log_message("[GUI] Run stopped and validation passed"))

    def test_broken_stdout_cannot_abort_gui_lifecycle_logging(self) -> None:
        window = mock.Mock()
        window.event_log = mock.Mock()
        with mock.patch("builtins.print", side_effect=BrokenPipeError("closed")):
            MainWindow._append_log(window, "[GUI] stopping")

        window.event_log.appendPlainText.assert_called_once_with("[GUI] stopping")
        window.statusBar.return_value.showMessage.assert_called_once_with(
            "[GUI] stopping", 5000
        )

    def test_tail_and_csv_reader_return_latest_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recording.csv"
            path.write_text("event,waiting\nsample,1\nsample,7\n")
            self.assertEqual(_tail_text_line(path), "sample,7")
            self.assertEqual(
                _last_csv_row(path),
                {"event": "sample", "waiting": "7"},
            )


class SessionLauncherSubjectScopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _dialog(self, subject_ids: list[str]) -> SessionLauncherDialog:
        store = mock.Mock()
        store.list_experiments.return_value = [
            ExperimentProfile(
                name="Experiment A",
                slug="experiment_a",
                subject_ids=subject_ids,
            )
        ]
        store.list_subjects.return_value = [
            SubjectProfile(name="Assigned", subject_id="assigned"),
            SubjectProfile(name="Unassigned", subject_id="unassigned"),
        ]
        dialog = SessionLauncherDialog(base_config={}, profile_store=store)
        dialog.experiment_combo.setCurrentIndex(1)
        return dialog

    def test_legacy_import_reexports_extracted_launcher(self) -> None:
        self.assertIs(SessionLauncherDialog, ExtractedSessionLauncherDialog)

    def test_empty_assignment_does_not_expose_all_subjects(self) -> None:
        dialog = self._dialog([])
        try:
            self.assertEqual(dialog.subject_combo.count(), 1)
            self.assertEqual(dialog.subject_combo.currentData(), "")
        finally:
            dialog.close()

    def test_only_assigned_subjects_are_selectable(self) -> None:
        dialog = self._dialog(["assigned"])
        try:
            values = [
                dialog.subject_combo.itemData(index)
                for index in range(dialog.subject_combo.count())
            ]
            self.assertEqual(values, ["", "assigned"])
        finally:
            dialog.close()


class BehaviorDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_auto_pellet_mode_detects_arrival_events(self) -> None:
        from squeakview.apps.operator.gui.dashboard import BehaviorDashboard

        dashboard = BehaviorDashboard(window_sec=30.0, pellet_mode="auto")
        dashboard.ingest("PELLET_ARRIVAL")
        self.assertEqual(dashboard._observed_pellet_mode, "arrival")
        self.assertEqual(dashboard.counters.get("PELLET"), 1)

    def test_auto_pellet_mode_detects_retrieval_events(self) -> None:
        from squeakview.apps.operator.gui.dashboard import BehaviorDashboard

        dashboard = BehaviorDashboard(window_sec=30.0, pellet_mode="auto")
        dashboard.ingest("PELLET_RETRIEVAL")
        self.assertEqual(dashboard._observed_pellet_mode, "retrieval")
        self.assertEqual(dashboard.counters.get("PELLET"), 1)

    def test_auto_pellet_mode_switches_to_both_when_both_events_seen(self) -> None:
        from squeakview.apps.operator.gui.dashboard import BehaviorDashboard

        dashboard = BehaviorDashboard(window_sec=30.0, pellet_mode="auto")
        dashboard.ingest("PELLET_ARRIVAL")
        dashboard.ingest("PELLET_RETRIEVAL")
        self.assertEqual(dashboard._observed_pellet_mode, "both")
        self.assertEqual(dashboard.counters.get("PELLET"), 2)


if __name__ == "__main__":
    unittest.main()
