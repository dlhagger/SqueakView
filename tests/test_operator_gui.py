from __future__ import annotations

import os
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from squeakview.apps.operator.gui.config_dialog import SessionLauncherDialog
from squeakview.apps.operator.gui.main_window import (
    _elapsed_text,
    _is_serial_log_message,
    _last_csv_row,
    _preflight_failure_message,
    _tail_text_line,
)
from squeakview.common.log_mirror import LineBufferedLogMirror
from squeakview.common.profiles import ExperimentProfile, SubjectProfile


class RuntimeFileHelpersTest(unittest.TestCase):
    def test_gui_log_mirror_filters_fragmented_camera_lines_without_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "gui.log"
            console = io.StringIO()
            mirror = LineBufferedLogMirror(log_path, console)

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

            mirror.write("partial diagnostic")
            mirror.flush()

            self.assertEqual(log_path.read_text(), "partial diagnostic")

    def test_gui_log_mirror_flush_does_not_orphan_camera_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "gui.log"
            mirror = LineBufferedLogMirror(log_path, io.StringIO())

            mirror.write("[SER] CAMERA_HIGH,1,2")
            mirror.flush()
            mirror.write("\nnormal\n")
            mirror.flush()

            self.assertEqual(log_path.read_text(), "normal\n")

    def test_ffprobe_preflight_failure_has_explicit_install_instructions(self) -> None:
        message = _preflight_failure_message("[FAIL] FFmpeg/ffprobe is not installed.")
        self.assertIn("FFmpeg is not installed", message)
        self.assertIn("sudo apt install ffmpeg", message)

    def test_unknown_preflight_failure_uses_generic_message(self) -> None:
        message = _preflight_failure_message("[FAIL] something else")
        self.assertIn("Open Operator Events", message)

    def test_elapsed_text_supports_long_runs(self) -> None:
        self.assertEqual(_elapsed_text(16 * 3600 + 2 * 60 + 9), "16:02:09")

    def test_serial_messages_are_identified_for_status_bar_filtering(self) -> None:
        self.assertTrue(_is_serial_log_message("[17:01:37] 【SER】 CAMERA_HIGH,..."))
        self.assertTrue(_is_serial_log_message("[17:01:37] 【SER→】 STOP"))
        self.assertTrue(_is_serial_log_message("[17:01:37] [SER] ACK_STOP received."))
        self.assertFalse(_is_serial_log_message("[GUI] Run stopped and validation passed"))

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
        with mock.patch(
            "squeakview.apps.operator.gui.config_dialog.ProfileStore",
            return_value=store,
        ):
            dialog = SessionLauncherDialog(base_config={})
        dialog.experiment_combo.setCurrentIndex(1)
        return dialog

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
